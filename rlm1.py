"""RLM1: PPO on raw 1-minute bars of BTCUSDT / ETHUSDT, position {-1,0,+1} every minute (docs/ml/PROTOCOL_RLM1.md).
usage: python rlm1.py [--jobs 4] [--costs 5 2] -> docs/ml/results/rlm1_runs/*.json, rlm1_*.csv"""
import os,json,argparse,numpy as np,pandas as pd
import data_m1

OUT='docs/ml/results';RUNS=f'{OUT}/rlm1_runs';CACHE=f'{data_m1.ROOT}/ml/m1feat'
TRAIN=('2019-01-01','2022-12-31 23:59');VAL=('2023-01-01','2023-12-31 23:59');WIN=240;EP=10080;K,TAU=1e-7,1440;SCALE=1e4
QS=int(os.environ.get('RLM1_QUARTER',1_500_000))      # steps per quarter of training (6M total)
CTX=['ret15','ret60','ret240','ret1440','ret4320','ret10080','lsv60','lsv43200','rng1440','rng10080','hs','hc','ds','dc']
CTXD=['dsma20','dsma50','dsma100','dret20','dret60','dret120']     # variant C: daily trend state from completed days only
VARS={'A':dict(pen=False,daily=False,gamma=0.999,gae=0.98),'B':dict(pen=True,daily=False,gamma=0.999,gae=0.98),'C':dict(pen=False,daily=True,gamma=0.99999,gae=0.995)}

def prep(sym):
    """Minute grid with features; cached as npz. Rows = minutes; decision at t uses data up to the close of minute t."""
    path=f'{CACHE}/{sym}.npz'
    if not os.path.exists(path):
        m=data_m1.load(sym).close;m.index=m.index.floor('min');m=m[~m.index.duplicated()]
        idx=pd.date_range('2018-11-01',m.index[-1],freq='min');c=m.reindex(idx).ffill();lp=np.log(c);r=lp.diff().fillna(0)
        s1440=r.rolling(1440).std();s60=r.rolling(60).std();s43200=r.rolling(43200,min_periods=20000).std()
        X=pd.DataFrame(index=idx)
        for n in (15,60,240,1440,4320,10080):X[f'ret{n}']=(lp-lp.shift(n))/(s1440*np.sqrt(n))
        X['lsv60']=np.log(s60/s1440);X['lsv43200']=np.log(s1440/s43200)
        for n in (1440,10080):hi=c.rolling(n).max();lo=c.rolling(n).min();X[f'rng{n}']=((c-lo)/(hi-lo)).fillna(0.5)*2-1
        h=idx.hour+idx.minute/60;d=idx.dayofweek+h/24
        X['hs'],X['hc'],X['ds'],X['dc']=np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),np.sin(2*np.pi*d/7),np.cos(2*np.pi*d/7)
        dc_=c.resample('1D').last();ld=np.log(dc_);sd=ld.diff().rolling(20).std();Dd=pd.DataFrame(index=dc_.index)
        for n in (20,50,100):Dd[f'dsma{n}']=(ld-np.log(dc_.rolling(n).mean()))/(sd*np.sqrt(n))
        for n in (20,60,120):Dd[f'dret{n}']=(ld-ld.shift(n))/(sd*np.sqrt(n))
        Dd=Dd.shift(1)                                   # value for day D uses closes up to D-1 (completed days only)
        for k in CTXD:X[k]=Dd[k].reindex(idx.floor('D')).values
        X=X.replace([np.inf,-np.inf],np.nan);z=(r/s1440).clip(-8,8)
        os.makedirs(CACHE,exist_ok=True)
        np.savez(path,t=idx.as_unit('ns').asi8,r=r.values,lp=lp.values,s=s1440.values,z=z.values.astype('float32'),X=X[CTX+CTXD].values.astype('float32'))
    d=np.load(path);t=pd.to_datetime(d['t'],unit='ns')
    ok=np.isfinite(d['X'][:,:6]).all(1)&np.isfinite(d['z'])&np.isfinite(d['s'])
    return t,d['r'],d['lp'],d['s'],np.nan_to_num(d['z']),d['X'],ok

class Data:
    def __init__(s,sym,daily=False):
        s.t,s.r,s.lp,s.sig,s.z,X,ok=prep(sym);X=X if daily else X[:,:len(CTX)];i0,i1=s.span(TRAIN)
        mu,sd=np.nanmean(X[i0:i1],0),np.nanstd(X[i0:i1],0)+1e-9;s.X=np.clip(np.nan_to_num((X-mu)/sd),-10,10).astype('float32')
        s.rbar=s.r[i0:i1].mean();s.first=int(np.argmax(ok[WIN:]))+WIN
    def span(s,p):return int(np.searchsorted(s.t,pd.Timestamp(p[0]))),int(np.searchsorted(s.t,pd.Timestamp(p[1]),side='right'))

class Book:
    """Vectorised state of n independent position books."""
    def __init__(s,D,n):s.D=D;s.n=n
    def start(s,t0):
        s.t=np.array(t0,dtype=np.int64);s.pos=np.zeros(s.n);s.held=np.zeros(s.n);s.flat=np.zeros(s.n);s.entry=np.zeros(s.n)
    def obs(s):
        D=s.D;t=s.t;zw=D.z[t[:,None]+np.arange(-WIN+1,1)]
        pnl=np.where(s.pos!=0,s.pos*(D.lp[t]-s.entry)/(D.sig[t]*np.sqrt(np.maximum(s.held,1))),0.)
        st=np.c_[s.pos==-1,s.pos==0,s.pos==1,np.log1p(s.held)/5,np.clip(np.nan_to_num(pnl),-10,10)]
        return np.c_[zw,D.X[t],st].astype(np.float32)
    def act(s,a,cost,train):
        """a in {0,1,2}; returns per-book reward (training units) or real log P&L."""
        D=s.D;new=a.astype(float)-1;d=np.abs(new-s.pos);r1=D.r[s.t+1]
        if train:
            s.flat=np.where(new==0,s.flat+1,0);pen=np.where(new==0,K*(np.exp(np.minimum(s.flat/TAU,6))-1),0.)
            rew=new*(r1-D.rbar)-cost*d
        else:pen=0.;rew=new*r1-cost*d
        enter=(new!=s.pos)&(new!=0);s.entry=np.where(enter,D.lp[s.t],s.entry);s.held=np.where(new==0,0,np.where(enter,1,s.held+1))
        s.pos=new;s.t=s.t+1;return rew,pen

def make_vec(D,n,cost,penalty,seed):
    import gymnasium as gym
    from stable_baselines3.common.vec_env import VecEnv
    i0,i1=D.span(TRAIN);i0=max(i0,D.first)
    class V(VecEnv):
        def __init__(s):
            s.B=Book(D,n);s.rng=np.random.default_rng(seed);s.k=np.zeros(n,dtype=np.int64)
            super().__init__(n,gym.spaces.Box(-np.inf,np.inf,(WIN+D.X.shape[1]+5,),np.float32),gym.spaces.Discrete(3))
        def _starts(s,m):return s.rng.integers(i0,i1-EP-2,m)
        def reset(s):s.B.start(s._starts(n));s.k[:]=0;return s.B.obs()
        def step_async(s,a):s.a=np.asarray(a)
        def step_wait(s):
            rew,pen=s.B.act(s.a,cost,True);rew=(rew-(pen if penalty else 0))*SCALE;s.k+=1;done=s.k>=EP;o=s.B.obs();infos=[{} for _ in range(n)]
            if done.any():
                for i in np.where(done)[0]:infos[i]={'terminal_observation':o[i],'TimeLimit.truncated':True}
                st=s._starts(int(done.sum()));B=s.B;B.t[done]=st;B.pos[done]=0;B.held[done]=0;B.flat[done]=0;B.entry[done]=0;s.k[done]=0;o=B.obs()
            return o,rew.astype(np.float32),done,infos
        def close(s):pass
        def get_attr(s,name,indices=None):return [getattr(s,name,None)]*n
        def set_attr(s,name,value,indices=None):pass
        def env_method(s,name,*a,indices=None,**k):return [None]*n
        def env_is_wrapped(s,cls,indices=None):return [False]*n
        def seed(s,seed=None):return [None]*n
    return V()

def evaluate(model,D,period,cost):
    """Weekly chunks run as one batch; each starts flat and is closed (with cost) at its end. Returns per-minute P&L and positions."""
    import torch
    i0,i1=D.span(period);i0=max(i0,D.first);st=np.arange(i0,i1-1,EP);L=np.minimum(EP,i1-1-st);B=Book(D,len(st));B.start(st)
    P=np.zeros((len(st),EP));POS=np.zeros((len(st),EP))
    for k in range(int(L.max())):
        live=k<L
        with torch.no_grad():a=model.policy.get_distribution(torch.as_tensor(B.obs())).distribution.probs.argmax(1).numpy()
        a=np.where(live,a,1+B.pos);r,_=B.act(a,cost,False);P[:,k]=np.where(live,r,0);POS[:,k]=np.where(live,B.pos,0)
        B.t=np.minimum(B.t,len(D.r)-2)
    j=np.arange(len(st));P[j,L-1]-=np.abs(POS[j,L-1])*cost                               # close at the chunk end
    ts=np.concatenate([D.t[s+1:s+1+l] for s,l in zip(st,L)]);pnl=np.concatenate([P[j,:l] for j,l in enumerate(L)]);pos=np.concatenate([POS[j,:l] for j,l in enumerate(L)])
    return pd.DataFrame({'pnl':pnl,'pos':pos},index=ts)

def metrics(x,D,tp=None):
    d=x.pnl.resample('1D').sum();bh=pd.Series(D.r,index=D.t).loc[x.index].resample('1D').sum()
    sr=lambda q:float(q.mean()/q.std()*np.sqrt(365)) if q.std()>0 else 0.
    beta=float(np.cov(d,bh)[0,1]/bh.var()) if bh.var()>0 else 0.;al=d-beta*bh;eq=d.cumsum()
    p=x.pos.values;chg=np.r_[False,p[1:]!=p[:-1]];runs=np.diff(np.r_[0,np.where(chg)[0],len(p)]);lab=p[np.r_[0,np.where(chg)[0]]]
    hold=runs[lab!=0]
    return {'sharpe':round(sr(d),2),'alpha_sharpe':round(sr(al),2),'beta':round(beta,3),'cagr_pct':round((np.exp(d.mean()*365)-1)*100,1),
            'max_dd_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1),'trades_per_day':round(float(chg.sum()/len(d)),1),
            'time_in_market':round(float((p!=0).mean()),3),'median_hold_min':float(np.median(hold)) if len(hold) else 0.,
            **({} if tp is None else {'corr_trend_sign':round(float(x.pos.resample('1D').mean().corr(np.sign(tp).reindex(d.index))),3),
                                      'corr_trend_size':round(float(x.pos.resample('1D').mean().corr(tp.reindex(d.index))),3)}),
            'by_year_return_pct':{str(y):round((np.exp(q.sum())-1)*100,1) for y,q in d.groupby(d.index.year)}}

def trend_pos(sym):
    """trend.py position (BTC+ETH book, 20% vol) held during each UTC day, from minute closes."""
    import trend
    P=pd.DataFrame({c:data_m1.load(c).close.resample('1D').last() for c in ('BTCUSDT','ETHUSDT')}).dropna()
    return trend.target_positions(P).shift(1)[sym]

def run(job):
    import torch;torch.set_num_threads(1)
    from stable_baselines3 import PPO
    sym,var,cb,seed=job;tag=f'{sym}_{var}_{cb}bp_s{seed}';path=f'{RUNS}/{tag}.json'
    if os.path.exists(path):return tag,'cached'
    V=VARS[var];D=Data(sym,V['daily']);cost=cb*1e-4;env=make_vec(D,8,cost,V['pen'],seed);tp=trend_pos(sym)
    m=PPO('MlpPolicy',env,policy_kwargs=dict(net_arch=[32,32]),gamma=V['gamma'],gae_lambda=V['gae'],ent_coef=0.01,learning_rate=2e-4,n_steps=2048,batch_size=4096,seed=seed,device='cpu',verbose=0)
    best,bs,vals=-np.inf,None,[]
    for q in range(4):
        m.learn(QS,reset_num_timesteps=False);v=metrics(evaluate(m,D,VAL,cost),D,tp);vals.append({'steps':(q+1)*QS,**{k:v[k] for k in ('sharpe','alpha_sharpe','trades_per_day','time_in_market','corr_trend_sign')}})
        if v['sharpe']>best:best=v['sharpe'];bs={n:p.detach().clone() for n,p in m.policy.state_dict().items()}
    m.policy.load_state_dict(bs);x=evaluate(m,D,(VAL[1][:10],str(D.t[-2])),cost);x=x[x.index>='2024-01-01']
    torch.save(bs,f'{RUNS}/{tag}_policy.pt');res={'sym':sym,'variant':var,'cost_bp':cb,'seed':seed,'val':vals,'test':metrics(x,D,tp),'test_period':[str(x.index[0]),str(x.index[-1])]}
    pd.DataFrame({'pnl':x.pnl.resample('1D').sum(),'pos':x.pos.resample('1D').mean()}).to_parquet(f'{RUNS}/{tag}_daily.parquet');json.dump(res,open(path,'w'),indent=1);return tag,res['test']

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--jobs',type=int,default=4);a.add_argument('--costs',type=int,nargs='+',default=[5,2]);a=a.parse_args()
    os.makedirs(RUNS,exist_ok=True)
    for s in ('BTCUSDT','ETHUSDT'):prep(s)
    J=[(s,v,5,sd) for v in ('A','B','C') for s in ('BTCUSDT','ETHUSDT') for sd in range(3) if 5 in a.costs]
    J+=[(s,v,c,sd) for c in a.costs if c!=5 for v in ('A','B') for s in ('BTCUSDT','ETHUSDT') for sd in range(3)]
    from multiprocessing import Pool
    with Pool(a.jobs) as p:
        for tag,r in p.imap_unordered(run,J):print(tag,r,flush=True)
