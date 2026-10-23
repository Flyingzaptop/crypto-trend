"""PNL1-M1: direct P&L optimisation on 1-minute bars (docs/ml/PROTOCOL_PNL1.md, addendum 1). GPU-native, CPU works for tests.
Encoder (MLP / 1D-CNN on the 240-minute z window + context, or GRU over minutes) computes e_t for all minutes in parallel; a small
recurrent position head p_t = tanh(v . tanh(A e_t + b p_{t-1} + c) + d) is unrolled over the chunk (sequential in time, parallel over chunks).
usage: python pnl_m1.py run MARKET COST_BP LOSS MODEL SEED [--device cuda]   -> docs/ml/results/pnl_m1_runs/<tag>.json (train/val only)
       python pnl_m1.py bench [--device cuda]"""
import os,sys,json,time,argparse,numpy as np,pandas as pd,torch
import rlm1

RUNS='docs/ml/results/pnl_m1_runs';WIN=240;EP=10080
MKT={'BTCUSDT':('2019-01-01','2022-12-31 23:59'),'ETHUSDT':('2019-01-01','2022-12-31 23:59'),'XAUUSD':('2017-01-01','2022-12-31 23:59')}
VAL=('2023-01-01','2023-12-31 23:59')
CFG=dict(chunk=2880,batch=128,epoch_batches=20,max_epochs=60,patience=8,lr=1e-3,wd=1e-3,drop=0.2,head=16)

class TD:
    def __init__(s,m,dev):
        D=rlm1.Data(m,True,MKT[m]);s.D=D;s.dev=dev;f=lambda a,dt:torch.as_tensor(np.ascontiguousarray(a),dtype=dt,device=dev)
        s.z=f(D.z,torch.float32);s.X=f(D.X,torch.float32);s.r=f(D.r,torch.float32);s.ar=torch.arange(-WIN+1,1,device=dev)
        i0,i1=D.span(MKT[m]);i0=max(i0,D.first);s.tr=(i0,i1);s.rbar=float(D.r[i0:i1].mean());s.sig=float(D.r[i0:i1].std());s.m=m
    def window(s,t):return s.z[t[...,None]+s.ar]                                     # (..., WIN)

class Enc(torch.nn.Module):
    def __init__(s,kind,w,nctx,drop):
        super().__init__();s.kind=kind;s.out=w
        if kind=='mlp':s.f=torch.nn.Sequential(torch.nn.Linear(WIN+nctx,w),torch.nn.Tanh(),torch.nn.Dropout(drop),torch.nn.Linear(w,w),torch.nn.Tanh())
        elif kind=='cnn':
            s.c=torch.nn.Sequential(torch.nn.Conv1d(1,8,8,stride=4),torch.nn.ReLU(),torch.nn.Conv1d(8,8,4,stride=2),torch.nn.ReLU(),torch.nn.Flatten())
            n=s.c(torch.zeros(1,1,WIN)).shape[1];s.f=torch.nn.Sequential(torch.nn.Linear(n+nctx,w),torch.nn.Tanh(),torch.nn.Dropout(drop))
        else:s.g=torch.nn.GRU(1+nctx,w,batch_first=True);s.drop=torch.nn.Dropout(drop)
    def forward(s,T,t):
        """t: (B, L) minute indices -> e: (B, L, w)."""
        B,L=t.shape;ctx=T.X[t]
        if s.kind=='mlp':return s.f(torch.cat([T.window(t),ctx],-1))
        if s.kind=='cnn':
            c=s.c(T.window(t).reshape(B*L,1,WIN)).reshape(B,L,-1);return s.f(torch.cat([c,ctx],-1))
        tw=torch.cat([t[:,:1]+torch.arange(-WIN,0,device=t.device),t],1)              # 240-minute warm-up, no loss on it
        h,_=s.g(torch.cat([T.z[tw][...,None],T.X[tw]],-1));return s.drop(h[:,WIN:])

class Net(torch.nn.Module):
    def __init__(s,kind,w,nctx,cfg):
        super().__init__();s.enc=Enc(kind,w,nctx,cfg['drop']);k=cfg['head'];s.A=torch.nn.Linear(w,k);s.b=torch.nn.Parameter(torch.randn(k)*0.1);s.v=torch.nn.Linear(k,1)
    def forward(s,T,t,p0=None):
        H=s.A(s.enc(T,t));B,L,_=H.shape;p=torch.zeros(B,device=H.device) if p0 is None else p0;P=[]
        for i in range(L):p=torch.tanh(s.v(torch.tanh(H[:,i]+s.b*p[:,None])))[:,0];P.append(p)
        return torch.stack(P,1)

def loss_fn(name,P,R,cost):
    dp=torch.diff(torch.cat([P.new_zeros(P.shape[0],1),P],1),dim=1).abs();pnl=P*R-cost*dp
    if name=='L1':return -(pnl.mean(1)/(pnl.std(1)+1e-6)).mean()
    if name in ('L2a','L2b'):return (-pnl.mean(1)+(1.0 if name=='L2a' else 5.0)*pnl.var(1)).mean()
    eq=pnl.cumsum(1);return (-pnl.mean(1)+(torch.cummax(eq,1).values-eq).max(1).values/pnl.shape[1]).mean()

def evaluate(net,T,period,cost,chunk_batch=64):
    """Weekly chunks (start flat, closed with cost at the end), real returns. Returns per-minute pnl and positions."""
    D=T.D;i0,i1=D.span(period);i0=max(i0,D.first+WIN);st=np.arange(i0,i1-1,EP);L=np.minimum(EP,i1-1-st);out_p,out_x,out_t=[],[],[]
    net.eval()
    with torch.no_grad():
        for k in range(0,len(st),chunk_batch):
            s=st[k:k+chunk_batch];l=L[k:k+chunk_batch];t=torch.as_tensor(s[:,None]+np.arange(EP),device=T.dev).clamp(max=len(D.r)-2)
            P=net(T,t).cpu().numpy();r=D.r[np.minimum(t.cpu().numpy()+1,len(D.r)-1)]          # position at t earns the NEXT minute's return
            for j in range(len(s)):
                p=P[j,:l[j]];pr=np.r_[0.,p[:-1]];pnl=p*r[j,:l[j]]-cost*np.abs(p-pr);pnl[-1]-=cost*abs(p[-1])
                out_p.append(pnl);out_x.append(p);out_t.append(D.t[s[j]+1:s[j]+1+l[j]])
    return pd.DataFrame({'pnl':np.concatenate(out_p),'pos':np.concatenate(out_x)},index=np.concatenate(out_t))

def metrics(x,T,tp):
    D=T.D;d=x.pnl.resample('1D').sum()[x.pnl.resample('1D').count()>0];bh=pd.Series(D.r,index=D.t).reindex(x.index).resample('1D').sum().reindex(d.index).fillna(0)
    tr=tp['trend'].reindex(d.index).fillna(0) if tp is not None else pd.Series(0.,index=d.index);ad=365 if T.m!='XAUUSD' else 252
    sh=lambda q:float(q.mean()/q.std()*np.sqrt(ad)) if len(q)>5 and q.std()>0 else 0.
    A=np.c_[np.ones(len(d)),bh,tr];coef=np.linalg.lstsq(A,d.values,rcond=None)[0];res=d.values-A[:,1:]@coef[1:]
    p=x.pos.values;on=np.abs(p)>0.05;sg=np.sign(p)*on;chg=np.r_[False,sg[1:]!=sg[:-1]];runs=np.diff(np.r_[0,np.where(chg)[0],len(sg)]);lab=sg[np.r_[0,np.where(chg)[0]]];eq=d.cumsum()
    pd_=x.pos.resample('1D').mean().reindex(d.index)
    return {'sharpe':round(sh(d),3),'alpha_sharpe':round(sh(pd.Series(res)),3),'beta_bh':round(float(coef[1]),3),'beta_trend':round(float(coef[2]),3),
            'bh_sharpe':round(sh(bh),3),'ann_ret_pct':round(float(d.mean()*ad*100),2),'max_dd_pct':round(float((1-np.exp(-(eq.cummax()-eq).max()))*100),1),
            'trades_per_day':round(float(chg.sum()/max(len(d),1)),2),'turnover_per_day':round(float(np.abs(np.diff(p)).sum()/max(len(d),1)),2),
            'median_hold_min':float(np.median(runs[lab!=0])) if (lab!=0).any() else 0.,'time_in_market':round(float(on.mean()),3),'mean_pos':round(float(p.mean()),3),
            'corr_trend_sign':round(float(pd_.corr(np.sign(tp['pos'].reindex(d.index)))),3) if tp is not None else np.nan}

def trend_info(m):
    import trend,rlm2
    if m=='XAUUSD':
        g=pd.read_parquet(rlm1.BARS[m]);c=pd.Series(g.close.values,index=pd.to_datetime(g.timestamp).dt.tz_localize(None)).resample('1D').last().dropna().to_frame(m);r,N=trend.backtest(c)
    else:
        import ml_data as md;c=md.klines(m,'1d').close.to_frame(m);r,N=trend.backtest(c,pd.DataFrame({m:trend.daily_funding(m)}))
    return pd.DataFrame({'trend':r,'pos':N[m]})

def train(m,cost_bp,loss,kind,seed,dev,cfg=CFG,log=None,T=None):
    torch.manual_seed(seed);w={'mlp16':16,'mlp64':64,'cnn':32,'gru16':16,'gru64':64}[kind];T=T or TD(m,dev);cost=cost_bp*1e-4
    net=Net(kind[:3],w,T.X.shape[1],cfg).to(dev);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    g=torch.Generator(device=dev).manual_seed(seed);i0,i1=T.tr;L=cfg['chunk'];best,bs,bad,hist,t0=-np.inf,None,0,[],time.time();tp=trend_info(m)
    for ep in range(cfg['max_epochs']):
        net.train()
        for _ in range(cfg['epoch_batches']):
            st=torch.randint(i0+WIN,i1-L-2,(cfg['batch'],),generator=g,device=dev);t=st[:,None]+torch.arange(L,device=dev)
            R=(T.r[t+1]-T.rbar)/T.sig;l=loss_fn(loss,net(T,t),R,cost/T.sig);opt.zero_grad();l.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
        v=metrics(evaluate(net,T,VAL,cost),T,tp);hist.append({'ep':ep,'sharpe':v['sharpe'],'trades_per_day':v['trades_per_day'],'time_in_market':v['time_in_market']})
        if log:print(f"{log} ep {ep} {time.time()-t0:.0f}s val {hist[-1]}",flush=True)
        if v['sharpe']>best:best,bs,bad,vb=v['sharpe'],{k:x.detach().clone() for k,x in net.state_dict().items()},0,v
        else:bad+=1
        if bad>=cfg['patience']:break
    net.load_state_dict(bs);return net,T,tp,vb,hist,time.time()-t0

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('cmd');a.add_argument('args',nargs='*');a.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a.add_argument('--epochs',type=int,default=30);a.add_argument('--patience',type=int,default=5);a=a.parse_args();dev=torch.device(a.device);cfg=dict(CFG)
    cfg['max_epochs']=a.epochs;cfg['patience']=a.patience
    if a.cmd=='bench':
        for kind in ('mlp64','cnn','gru64'):
            c=dict(cfg,max_epochs=2,epoch_batches=5);net,T,tp,vb,h,sec=train('BTCUSDT',5,'L1',kind,0,dev,c);print(kind,f'{sec:.1f}s for 2 epochs x 5 batches (+2 val evals)',vb['sharpe'],flush=True)
    elif a.cmd=='run':
        m,cb,loss,kind,seed=a.args;cb=float(cb);seed=int(seed);tg=f'{m}_{cb:g}bp_{loss}_{kind}_s{seed}';os.makedirs(RUNS,exist_ok=True)
        net,T,tp,vb,hist,sec=train(m,cb,loss,kind,seed,dev,cfg,log=tg);tr=metrics(evaluate(net,T,MKT[m],cb*1e-4),T,tp)
        torch.save(net.state_dict(),f'{RUNS}/{tg}.pt');json.dump({'market':m,'cost_bp':cb,'loss':loss,'model':kind,'seed':seed,'sec':round(sec),'cfg':cfg,'val_curve':hist,'val':vb,'train':tr},open(f'{RUNS}/{tg}.json','w'),indent=1,default=float)
        print(tg,'done',sec,flush=True)
