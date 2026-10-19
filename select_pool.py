"""Greedy pool expansion, as declared in docs/PROTOCOL_universe.md.
usage: python select_pool.py [--end 2023-12-31] [--out docs/results]
Needs data/universe.csv (universe.py) and the 1m cache (data_m1.py --universe)."""
import os,argparse,json,numpy as np,pandas as pd
import trend,data_m1

def load(symbols):
    P,F={},{}
    for s in symbols:
        try:P[s]=data_m1.daily_closes(s)
        except FileNotFoundError:print('no data:',s);continue
        F[s]=trend.daily_funding(s)
    return pd.DataFrame(P),pd.DataFrame(F)

def run(P,F,cols,cost=10.0):
    ret,N=trend.backtest(P[cols].dropna(how='all'),F.reindex(columns=cols),cost_bp=cost);return ret,N

def monthly(r):return np.exp(r.groupby(r.index.to_period('M')).sum())-1

def stats(r):
    eq=r.cumsum();return {'cagr_pct':round((np.exp(r.mean()*trend.ANN)-1)*100,1),'vol_pct':round(r.std()*np.sqrt(trend.ANN)*100,1),
        'sharpe':round(r.mean()/r.std()*np.sqrt(trend.ANN),2),'max_dd_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1)}

def compare(P,F,pool,x,min_months=12):
    """Monthly comparison pool+x vs pool over the full months x was in the book."""
    rb,_=run(P,F,pool);rc,N=run(P,F,pool+[x]);on=N[x].dropna()
    if on.empty:return None
    first=on.index[0].to_period('M')+1;last=P[x].last_valid_index().to_period('M')-1
    mb,mc=monthly(rb),monthly(rc);m=mb.index[(mb.index>=first)&(mb.index<=last)].intersection(mc.index)
    d=mc[m]-mb[m];lo,hi=first.start_time,last.end_time
    sh=lambda r:r[(r.index>=lo)&(r.index<=hi)].pipe(lambda q:q.mean()/q.std()*np.sqrt(trend.ANN))
    return {'symbol':x,'months':len(m),'from':str(first),'hit_pct':round((d>0).mean()*100,1) if len(m) else np.nan,
            'mean_delta_pct_yr':round(d.mean()*12*100,2) if len(m) else np.nan,
            'sharpe_pool':round(sh(rb),2),'sharpe_with':round(sh(rc),2),'enough':len(m)>=min_months}

def wait_load(P,F,x,end=None):
    """Add coin x to P/F, waiting for data_m1.py to finish it if it is still downloading."""
    import time
    while not os.path.exists(f'{data_m1.ROOT}/m1/spot_{x}.parquet'):time.sleep(20)
    p,f=load([x]);P[x]=p[x] if end is None else p[x][p.index<=end];F[x]=f[x];return P,F

def greedy(P,F,order,end=None):
    pool=[order[0]];log=[]
    for x in order[1:]:
        if x not in P:P,F=wait_load(P,F,x,end)
        c=compare(P,F,pool,x)
        if c is None:continue
        c['keep']=bool(c['enough'] and c['hit_pct']>50 and c['mean_delta_pct_yr']>0)
        if c['keep']:pool.append(x)
        c['pool_size']=len(pool);log.append(c)
        print(f"{x:12s} months {c['months']:3d}  hit {c['hit_pct']:5.1f}%  delta {c['mean_delta_pct_yr']:+6.2f}%/yr  "
              f"Sharpe {c['sharpe_pool']:.2f}->{c['sharpe_with']:.2f}  {'KEEP' if c['keep'] else '-'}",flush=True)
    return pool,pd.DataFrame(log)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--end',default=None,help='selection uses data up to this date only')
    a.add_argument('--out',default='docs/results');a=a.parse_args();os.makedirs(a.out,exist_ok=True)
    order=list(pd.read_csv('data/universe.csv').symbol);P,F=load(order[:1])
    Ps=P[P.index<=a.end] if a.end else P
    pool,log=greedy(Ps,F,order,a.end);P,F=load(order);tag=f'_to{a.end}' if a.end else '_full'
    log.to_csv(f'{a.out}/greedy{tag}.csv',index=False)
    res={'selection_end':a.end,'pool':pool}
    for name,cols in (('BTC',['BTCUSDT']),('BTC+ETH',['BTCUSDT','ETHUSDT']),('pool',pool)):
        for cost in (10,20):
            r,_=run(P,F,cols,cost)
            res[f'{name}_{cost}bp_full']=stats(r[r.index>='2018-01-01'])
            if a.end:res[f'{name}_{cost}bp_after']=stats(r[r.index>a.end])
    json.dump(res,open(f'{a.out}/summary{tag}.json','w'),indent=1);print(json.dumps(res,indent=1))
