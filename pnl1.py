"""PNL1: a small network trained directly on a differentiable P&L criterion (docs/ml/PROTOCOL_PNL1.md).
usage: python pnl1.py sweep [--jobs 4]      16 points x 3 seeds per market x bar, train/val only -> docs/ml/results/pnl1_runs/*.json
       python pnl1.py select                selection by validation; test of the selected points; walk-forward; placebo"""
import os,sys,glob,json,argparse,numpy as np,pandas as pd,torch
import trend,ml_data as md,data_m1

RUNS='docs/ml/results/pnl1_runs';TRAIN_END='2022-12-31 23:59';VAL=('2023-01-01','2023-12-31 23:59');TEST0='2024-01-01'
MB=[('BTCUSDT','1d'),('BTCUSDT','4h'),('ETHUSDT','1d'),('ETHUSDT','4h'),('XAUUSD','1h'),('XAUUSD','4h'),('XAUUSD','1d')]
LOSSES=['L1','L2a','L2b','L3'];MODELS=[('mlp',16),('mlp',64),('gru',16),('gru',64)]
CH,BATCH,EPOCH_BATCHES,MAX_EP,PAT=256,64,50,150,15

def bars(m,b):
    """Close series and per-bar funding (crypto: daily funding, /6 per 4h bar)."""
    if m=='XAUUSD':
        g=pd.read_parquet(f'{data_m1.ROOT}/ml/gold/xauusd_m1.parquet');c=pd.Series(g.close.values,index=pd.to_datetime(g.timestamp).dt.tz_localize(None)).sort_index()
        c=c[~c.index.duplicated()].resample({'1h':'1h','4h':'4h','1d':'1D'}[b]).last().dropna();return c,pd.Series(0.,index=c.index)
    c=md.klines(m,b).close;f=trend.daily_funding(m).reindex(c.index.floor('D')).fillna(0).values/(1 if b=='1d' else 6)
    return c,pd.Series(f,index=c.index)

def features(c,intraday):
    lp=np.log(c);r=lp.diff();s=r.rolling(100,min_periods=50).std();X=pd.DataFrame(index=c.index)
    for n in (1,2,4,8,16,32,64,120):X[f'r{n}']=(lp-lp.shift(n))/(s*np.sqrt(n))
    for n in (20,50,100,200):X[f'sma{n}']=(lp-np.log(c.rolling(n).mean()))/(s*np.sqrt(n))
    X['v20_100']=np.log(r.rolling(20).std()/s);X['v100_500']=np.log(s/r.rolling(500,min_periods=200).std())
    for n in (20,100):hi=c.rolling(n).max();lo=c.rolling(n).min();X[f'rng{n}']=((c-lo)/(hi-lo)).fillna(0.5)*2-1
    d=c.index.dayofweek+c.index.hour/24;X['dow_s'],X['dow_c']=np.sin(2*np.pi*d/7),np.cos(2*np.pi*d/7)
    if intraday:h=c.index.hour;X['h_s'],X['h_c']=np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24)
    return X.replace([np.inf,-np.inf],np.nan)

class Data:
    def __init__(s,m,b,placebo_seed=None):
        c,f=bars(m,b)
        if placebo_seed is not None:                                   # block-shuffled returns (blocks of 20 bars) -> rebuilt price path
            r=np.log(c).diff().fillna(0).values;nb=len(r)//20;rng=np.random.default_rng(placebo_seed);idx=np.concatenate([np.arange(k*20,(k+1)*20) for k in rng.permutation(nb)]+[np.arange(nb*20,len(r))])
            c=pd.Series(np.exp(np.log(c.iloc[0])+np.cumsum(r[idx])),index=c.index)
        X=features(c,b!='1d');ok=X.notna().all(1).values;first=int(np.argmax(ok))
        s.t=c.index[first:];X=X.iloc[first:].fillna(0);s.r=np.r_[np.diff(np.log(c.values))[first:],0.];s.f=np.r_[f.values[first+1:],0.]  # r[i]: bar i -> i+1
        tr=s.t<=pd.Timestamp(TRAIN_END);mu,sd=X[tr].mean(),X[tr].std()+1e-9;s.X=((X-mu)/sd).clip(-10,10).values.astype(np.float32)
        s.rbar=s.r[tr].mean();s.sig=s.r[tr].std();s.itr=np.where(tr)[0];s.m,s.b=m,b
        s.cost=(1e-4 if m=='XAUUSD' else 10e-4);s.ann={'1d':365,'4h':2190,'1h':24*252}[b] if m!='XAUUSD' else {'1d':252,'4h':6*252,'1h':23*252}[b]
    def idx(s,lo,hi):return np.where((s.t>=pd.Timestamp(lo))&(s.t<=pd.Timestamp(hi)))[0]

class Net(torch.nn.Module):
    def __init__(s,nin,kind,w,drop=0.2):
        super().__init__();s.kind=kind;s.w=w;s.drop=torch.nn.Dropout(drop)
        if kind=='mlp':s.f=torch.nn.Sequential(torch.nn.Linear(nin+1,w),torch.nn.Tanh(),torch.nn.Dropout(drop),torch.nn.Linear(w,w),torch.nn.Tanh(),torch.nn.Linear(w,1))
        else:s.cell=torch.nn.GRUCell(nin+1,w);s.out=torch.nn.Linear(w,1)
    def forward(s,X):
        """X: (B, L, nin) -> positions (B, L), unrolled with the own previous position as input."""
        B,L,_=X.shape;p=X.new_zeros(B);h=X.new_zeros(B,s.w) if s.kind=='gru' else None;P=[]
        for t in range(L):
            z=torch.cat([X[:,t],p[:,None]],1)
            if s.kind=='mlp':p=torch.tanh(s.f(z))[:,0]
            else:h=s.cell(z,h);p=torch.tanh(s.out(s.drop(h)))[:,0]
            P.append(p)
        return torch.stack(P,1)

def loss_fn(name,P,R,cost):
    """P (B,L) positions; R (B,L) normalised, drift-free next-bar returns. Turnover from the chunk's own positions (start flat)."""
    dp=torch.diff(torch.cat([P.new_zeros(P.shape[0],1),P],1),dim=1).abs();pnl=P*R-cost*dp
    if name=='L1':return -(pnl.mean(1)/(pnl.std(1)+1e-6)).mean()
    if name in ('L2a','L2b'):lam=1.0 if name=='L2a' else 5.0;return (-pnl.mean(1)+lam*pnl.var(1)).mean()
    eq=pnl.cumsum(1);mdd=(torch.cummax(eq,1).values-eq).max(1).values;return (-pnl.mean(1)+mdd/pnl.shape[1]).mean()

def run_period(net,D,ix):
    """Continuous pass over the bars ix, starting flat; raw returns, cost, funding. Returns daily P&L and positions."""
    net.eval()
    with torch.no_grad():P=net(torch.as_tensor(D.X[ix])[None])[0].numpy()
    pr=np.r_[0.,P[:-1]];pnl=P*D.r[ix]-P*D.f[ix]-D.cost*np.abs(P-pr)
    s=pd.Series(pnl,index=D.t[ix]);g=s.resample('1D');d=g.sum()[g.count()>0]           # calendar days with at least one bar
    return d,pd.Series(P,index=D.t[ix])

def sharpe(d,ann_d):return float(d.mean()/d.std()*np.sqrt(ann_d)) if len(d)>5 and d.std()>0 else 0.

def train(D,kind,w,loss,seed,tr_ix,va_ix,log=None):
    torch.manual_seed(seed);np.random.seed(seed);net=Net(D.X.shape[1],kind,w);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3)
    R=torch.as_tensor(((D.r-D.rbar)/D.sig).astype(np.float32));X=torch.as_tensor(D.X);cost=D.cost/D.sig;rng=np.random.default_rng(seed)
    starts=tr_ix[tr_ix<=tr_ix.max()-CH];ad=365 if D.m!='XAUUSD' else 252;best,bs,bad,hist=-np.inf,None,0,[]
    for ep in range(MAX_EP):
        net.train()
        for _ in range(EPOCH_BATCHES):
            st=rng.choice(starts,BATCH);ii=st[:,None]+np.arange(CH);l=loss_fn(loss,net(X[ii]),R[ii],cost);opt.zero_grad();l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
        v=sharpe(run_period(net,D,va_ix)[0],ad);hist.append(round(v,3))
        if v>best:best,bs,bad=v,{k:x.clone() for k,x in net.state_dict().items()},0
        else:bad+=1
        if bad>=PAT:break
    net.load_state_dict(bs);return net,best,hist

def metrics(net,D,ix,base):
    d,P=run_period(net,D,ix);ad=365 if D.m!='XAUUSD' else 252;j=pd.concat([d.rename('s'),base.reindex(d.index).fillna(0)],axis=1);j=j[(j.bh!=0)|(j.s!=0)]
    A=np.c_[np.ones(len(j)),j.bh,j.trend];coef=np.linalg.lstsq(A,j.s.values,rcond=None)[0];res=j.s.values-A[:,1:]@coef[1:]
    yrs=max(len(P)/D.ann,1e-9);eq=d.cumsum()
    return {'sharpe':round(sharpe(d,ad),3),'alpha_sharpe':round(sharpe(pd.Series(res),ad),3),'beta_bh':round(coef[1],3),'beta_trend':round(coef[2],3),
            'bh_sharpe':round(sharpe(j.bh,ad),3),'trend_sharpe':round(sharpe(j.trend,ad),3),'ann_ret_pct':round(d.sum()/yrs*100,2),
            'max_dd_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1),'turnover_per_year':round(float(np.abs(np.diff(np.r_[0,P.values])).sum()/yrs),1),
            'time_in_market':round(float((P.abs()>0.05).mean()),3),'mean_abs_pos':round(float(P.abs().mean()),3),'mean_pos':round(float(P.mean()),3)},d,P

def baselines(m):
    """Daily buy-and-hold and trend-rule log returns for the market."""
    if m=='XAUUSD':c,_=bars(m,'1d');P=c.to_frame(m);F=None
    else:P=md.klines(m,'1d').close.to_frame(m);F=pd.DataFrame({m:trend.daily_funding(m)})
    tr,_=trend.backtest(P,F);bh=np.log(P[m]).diff()
    return pd.DataFrame({'bh':bh,'trend':tr}).fillna(0)

def tag(m,b,loss,kind,w,seed):return f'{m}_{b}_{loss}_{kind}{w}_s{seed}'

def sweep_job(job):
    torch.set_num_threads(1);m,b,loss,kind,w,seed=job;p=f'{RUNS}/{tag(*job)}.json'
    if os.path.exists(p):return
    D=Data(m,b);base=baselines(m);net,best,hist=train(D,kind,w,loss,seed,D.itr,D.idx(*VAL))
    tr,_,_=metrics(net,D,D.itr,base);va,_,_=metrics(net,D,D.idx(*VAL),base)
    torch.save(net.state_dict(),p.replace('.json','.pt'));json.dump({'market':m,'bar':b,'loss':loss,'model':kind,'width':w,'seed':seed,'epochs':len(hist),'val_curve':hist,'train':tr,'val':va},open(p,'w'),indent=1)
    print(tag(*job),'val',va['sharpe'],'train',tr['sharpe'],flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('cmd');a.add_argument('--jobs',type=int,default=4);a=a.parse_args();os.makedirs(RUNS,exist_ok=True)
    if a.cmd=='sweep':
        J=[(m,b,l,k,w,s) for m,b in MB for l in LOSSES for k,w in MODELS for s in range(3)];J.sort(key=lambda j:(j[1]!='1h',j[3]!='gru',-j[4]))   # longest first
        from multiprocessing import Pool
        with Pool(a.jobs,maxtasksperchild=4) as p:list(p.imap_unordered(sweep_job,J))
