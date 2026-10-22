"""RLM2: model-size sweep for the minute RL of RLM1 (docs/ml/PROTOCOL_RLM2.md). Train and validation metrics only - the sweep
never evaluates the test period (that is rlm2_test.py, for the selected point only).
usage: python rlm2.py --market BTCUSDT [--jobs 4] -> docs/ml/results/rlm2_runs/*.json"""
import os,json,argparse,numpy as np,pandas as pd
import rlm1,data_m1

RUNS='docs/ml/results/rlm2_runs'
MKT={'BTCUSDT':dict(cost=5,train=('2019-01-01','2022-12-31 23:59')),'ETHUSDT':dict(cost=5,train=('2019-01-01','2022-12-31 23:59')),
     'XAUUSD':dict(cost=1,train=('2017-01-01','2022-12-31 23:59'))}
VAL=('2023-01-01','2023-12-31 23:59')

def points():
    P=[(v,w,0.01,0.0) for v in ('A','C') for w in (8,16,32,64,128,256)]
    P+=[('C',w,0.05,0.0) for w in (32,64,128)]+[('C',w,0.01,1e-4) for w in (32,64,128)]
    return P

def tag(m,v,w,ent,wd,seed):return f'{m}_{v}_w{w}_ent{ent:g}_wd{wd:g}_s{seed}'

def trend_pos(m):
    """Trend rule position held during each day: trend.py BTC+ETH book for crypto; same rule on gold daily closes for XAUUSD."""
    if m!='XAUUSD':return rlm1.trend_pos(m)
    import trend
    g=pd.read_parquet(rlm1.BARS[m]);c=pd.Series(g.close.values,index=pd.to_datetime(g.timestamp).dt.tz_localize(None))
    return trend.target_positions(c.resample('1D').last().dropna().to_frame(m)).shift(1)[m].resample('1D').ffill()

def run(job):
    import torch;torch.set_num_threads(1)
    from stable_baselines3 import PPO
    m,v,w,ent,wd,seed=job;tg=tag(*job);path=f'{RUNS}/{tg}.json'
    if os.path.exists(path):return tg,'cached'
    V=rlm1.VARS[v];D=rlm1.Data(m,V['daily'],MKT[m]['train']);cost=MKT[m]['cost']*1e-4;env=rlm1.make_vec(D,8,cost,False,seed);tp=trend_pos(m)
    pk=dict(net_arch=[w,w])
    if wd>0:pk.update(optimizer_class=torch.optim.AdamW,optimizer_kwargs=dict(weight_decay=wd))
    M=PPO('MlpPolicy',env,policy_kwargs=pk,gamma=V['gamma'],gae_lambda=V['gae'],ent_coef=ent,learning_rate=2e-4,n_steps=2048,batch_size=4096,seed=seed,device='cpu',verbose=0)
    best,bs,vals=-np.inf,None,[]
    for q in range(4):
        M.learn(rlm1.QS,reset_num_timesteps=False);x=rlm1.metrics(rlm1.evaluate(M,D,VAL,cost),D,tp)
        vals.append({'steps':(q+1)*rlm1.QS,**{k:x[k] for k in ('sharpe','alpha_sharpe','trades_per_day','time_in_market','corr_trend_sign')}})
        if x['alpha_sharpe']>best:best=x['alpha_sharpe'];bs={n:p.detach().clone() for n,p in M.policy.state_dict().items()};vbest=x
    M.policy.load_state_dict(bs);tr=rlm1.metrics(rlm1.evaluate(M,D,MKT[m]['train'],cost),D,tp)
    torch.save(bs,f'{RUNS}/{tg}_policy.pt')
    res={'market':m,'variant':v,'width':w,'ent':ent,'wd':wd,'seed':seed,'val_curve':vals,'val':vbest,'train':tr}
    json.dump(res,open(path,'w'),indent=1);return tg,{k:vbest[k] for k in ('sharpe','alpha_sharpe','trades_per_day','time_in_market')}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--market',required=True);a.add_argument('--jobs',type=int,default=4)
    a.add_argument('--part',default='0/1',help="k[,k2]/n: run jobs whose index mod n is in {k,...} (split one market across machines)");a=a.parse_args()
    os.makedirs(RUNS,exist_ok=True);rlm1.prep(a.market)
    J=[(a.market,v,w,ent,wd,s) for v,w,ent,wd in points() for s in range(3)]
    J.sort(key=lambda j:-j[2])                                   # widest first: the long jobs do not end up alone at the tail
    ks,n=a.part.split('/');ks={int(x) for x in ks.split(',')};J=[j for i,j in enumerate(J) if i%int(n) in ks]   # e.g. '0,2/3'
    from multiprocessing import Pool
    with Pool(a.jobs) as p:
        for tg,r in p.imap_unordered(run,J):print(tg,r,flush=True)
