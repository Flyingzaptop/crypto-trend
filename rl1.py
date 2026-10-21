"""RL1: unconstrained PPO/SAC on BTC and ETH (docs/ml/PROTOCOL_RL1.md).
usage: python rl1.py [--jobs 4] -> docs/ml/results/rl1_*.csv"""
import os,json,argparse,numpy as np,pandas as pd
import trend,ml_data as md

TRAIN_END,VAL,TEST='2022-12-31','2023','2024-01-01';WIN=30;EP=256;COST=10e-4;OUT='docs/ml/results'
BPD={'1d':1,'4h':6}

def frame(sym,bar,mirror=False):
    """Per-bar log return r, funding f, and the 5 causal features."""
    k=md.klines(sym,bar);p=k.close;f=trend.daily_funding(sym).reindex(p.index.floor('D')).fillna(0).values/BPD[bar]
    if mirror:p=1/p;f=-f
    n=BPD[bar];r=np.log(p).diff();sd=r.rolling(20*n).std()
    x=pd.DataFrame({'z':(r/sd).clip(-5,5),'lsd':np.log(sd)},index=p.index)
    for m in trend.SMAS:x[f'd{m}']=((np.log(p)-np.log(p.rolling(m*n).mean()))/(sd*np.sqrt(n))).clip(-10,10)   # distance in daily sigma
    x['f']=f*1000;x['r']=r;x['fund']=f
    return x.dropna()

def make_env(X,lo,hi,lev,train):
    import gymnasium as gym
    F=['z','lsd','d20','d50','d100','f']
    class Env(gym.Env):
        def __init__(s):
            s.X=X;tr=X[:TRAIN_END];s.mu=tr.lsd.mean();s.sg=tr.lsd.std()
            A=X[F].copy();A['lsd']=(A.lsd-s.mu)/s.sg;s.A=A.values.astype(np.float32);s.r=X.r.values;s.fu=X.fund.values
            idx=np.where((X.index>=pd.Timestamp(lo))&(X.index<=pd.Timestamp(hi)+pd.Timedelta(hours=23)))[0];s.i0,s.i1=max(idx[0],WIN),idx[-1]
            s.observation_space=gym.spaces.Box(-np.inf,np.inf,(WIN*len(F)+1,),np.float32);s.action_space=gym.spaces.Box(-1,1,(1,),np.float32)
        def obs(s):return np.r_[s.A[s.t-WIN+1:s.t+1].ravel(),s.pos/lev].astype(np.float32)
        def reset(s,seed=None,options=None):
            super().reset(seed=seed);s.pos=0.
            if train:s.t=int(s.np_random.integers(s.i0,max(s.i0+1,s.i1-EP)));s.end=min(s.t+EP,s.i1)
            else:s.t=s.i0;s.end=s.i1
            return s.obs(),{}
        def step(s,a):
            new=float(np.clip(a[0],-1,1))*lev;r=s.r[s.t+1];g=1+new*np.expm1(r)-new*s.fu[s.t+1]-COST*abs(new-s.pos)
            rew=float(np.log(max(g,1e-6)));s.pos=new;s.t+=1
            return s.obs(),rew,s.t>=s.end,g<=1e-6,{'pos':new,'ret':np.log(max(g,1e-6))}
    return Env()

def rollout(model,X,lo,hi,lev):
    e=make_env(X,lo,hi,lev,False);o,_=e.reset();pos,ret,ix=[],[],[]
    while True:
        a,_=model.predict(o,deterministic=True);t=e.t;o,rw,d,tr,i=e.step(a);pos.append(i['pos']);ret.append(i['ret']);ix.append(X.index[t+1])
        if d or tr:break
    return pd.DataFrame({'pos':pos,'ret':ret},index=ix)

def train_one(job):
    import torch;torch.set_num_threads(1)
    from stable_baselines3 import PPO,SAC
    sym,bar,algo,lev,seed,mirror=job;X=frame(sym,bar,mirror);steps=200_000 if algo=='PPO' else 50_000;every=10_000
    env=make_env(X,'2017-01-01',TRAIN_END,lev,True);cls={'PPO':PPO,'SAC':SAC}[algo]
    m=cls('MlpPolicy',env,seed=seed,verbose=0,policy_kwargs=dict(net_arch=[64,64]),device='cpu')
    best,bp=-np.inf,None
    for k in range(steps//every):
        m.learn(every,reset_num_timesteps=False);v=rollout(m,X,VAL+'-01-01',VAL+'-12-31',lev).ret.sum()
        if v>best:best,bp=v,{n:p.detach().clone() for n,p in m.policy.state_dict().items()}
    m.policy.load_state_dict(bp);full=rollout(m,X,'2018-06-01',X.index[-2].strftime('%Y-%m-%d'),lev)
    tag=f'{sym}_{bar}_{algo}_{lev}x_s{seed}'+('_mirror' if mirror else '')
    full.to_parquet(f'{OUT}/rl1_runs/{tag}.parquet');return tag,best

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--jobs',type=int,default=4);a=a.parse_args();os.makedirs(f'{OUT}/rl1_runs',exist_ok=True)
    J=[(s,b,'PPO',l,sd,False) for s in trend.COINS for b in ('1d','4h') for l in (1,3) for sd in range(5)]
    J+=[(s,b,'SAC',1,sd,False) for s in trend.COINS for b in ('1d','4h') for sd in range(5)]
    J+=[(s,b,'PPO',1,sd,True) for s in trend.COINS for b in ('1d','4h') for sd in range(5)]
    done=set(f[:-8] for f in os.listdir(f'{OUT}/rl1_runs'))
    J=[j for j in J if f'{j[0]}_{j[1]}_{j[2]}_{j[3]}x_s{j[4]}'+('_mirror' if j[5] else '') not in done]
    J.sort(key=lambda j:(j[2]=='SAC',j[5],j[1]=='4h'));print(len(J),'runs',flush=True)
    from multiprocessing import Pool
    with Pool(a.jobs) as p:
        for tag,v in p.imap_unordered(train_one,J):print(tag,'val',round(v,3),flush=True)
