"""Point-in-time top-K basket vs BTC+ETH, as declared in docs/PROTOCOL_top10.md.
usage: python basket.py [--k 10 5 20] [--out docs/results]   (needs python archive_d1.py)"""
import os,glob,json,argparse,numpy as np,pandas as pd
import trend,archive_d1
from select_pool import stats,monthly

STABLE={'USDC','FDUSD','USD1','RLUSD','EUR','TUSD','USDP','BUSD','DAI','XUSD','RWUSD','PYUSD','USDE','EURI','AEUR','BFUSD',
        'UST','USDS','PAX','USDSB','SUSD','GBP','AUD','BKRW','IDRT','BIDR','UAH','TRY','BRL','NGN','RUB','ZAR'}
GOLD={'PAXG','XAUT'}
LEV=('UP','DOWN','BULL','BEAR')

def universe():
    perps=set(open(f'{archive_d1.OUT}/_perps.txt').read().split())
    P,V={},{}
    for f in glob.glob(f'{archive_d1.OUT}/*USDT.parquet'):
        s=os.path.basename(f)[:-8];b=s[:-4]
        if b in STABLE|GOLD or b.endswith(LEV) or s not in perps:continue
        d=pd.read_parquet(f);P[s]=d.close;V[s]=d.quote_volume
    return pd.DataFrame(P).sort_index(),pd.DataFrame(V).sort_index()

def members(P,V,k,min_hist=130):
    """For each month: top-k by 30-day quote volume up to the end of the previous month, among coins with >= min_hist days."""
    hist=P.notna().cumsum();vol30=V.rolling(30,min_periods=20).mean();M=pd.DataFrame(False,index=P.index,columns=P.columns)
    for m in pd.period_range(P.index[0],P.index[-1],freq='M')[1:]:
        t=m.start_time-pd.Timedelta(days=1)
        if t not in vol30.index:continue
        ok=(hist.loc[t]>=min_hist)&P.loc[t].notna();top=vol30.loc[t][ok].nlargest(k).index
        M.loc[m.start_time:m.end_time,top]=True
    return M

def run(P,F,M,cost):
    held=M.any();cols=list(held[held].index)
    return trend.backtest(P[cols],F.reindex(columns=cols),cost_bp=cost,members=M[cols])

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--k',type=int,nargs='+',default=[10,5,20]);a.add_argument('--out',default='docs/results')
    a=a.parse_args();os.makedirs(a.out,exist_ok=True)
    P,V=universe();print(P.shape[1],'eligible pairs (incl. delisted)')
    Ms={k:members(P,V,k) for k in a.k};need=sorted(set().union(*[set(M.columns[M.any()]) for M in Ms.values()])|{'BTCUSDT','ETHUSDT'})
    F=pd.DataFrame({s:trend.daily_funding(s) for s in need})
    base=pd.DataFrame(False,index=P.index,columns=P.columns);base[['BTCUSDT','ETHUSDT']]=True
    res={};periods={'2018-01..':('2018-01-01',None),'2024-01..':('2024-01-01',None)}
    for cost in (10,20):
        rb,_=run(P,F,base,cost)
        for k,M in Ms.items():
            rk,N=run(P,F,M,cost);mb,mk=monthly(rb),monthly(rk)
            for name,(lo,hi) in periods.items():
                q=lambda r:r[r.index>=lo];m=mb.index[mb.index>=pd.Period(lo,'M')].intersection(mk.index);d=mk[m]-mb[m]
                res[f'top{k}_{cost}bp_{name}']={**stats(q(rk)),'hit_pct':round((d>0).mean()*100,1),'mean_delta_pct_yr':round(d.mean()*1200,2)}
                res[f'BTC+ETH_{cost}bp_{name}']=stats(q(rb))
            if cost==10:
                res[f'top{k}_yearly']={int(y):[round((np.exp(rk[rk.index.year==y].sum())-1)*100,1),round((np.exp(rb[rb.index.year==y].sum())-1)*100,1)]
                                       for y in sorted(set(rk.index.year)) if y>=2018}
                M[M.any(axis=1)].resample('MS').first().apply(lambda r:' '.join(c[:-4] for c in r.index[r]),axis=1).to_csv(f'{a.out}/top{k}_members.csv')
    json.dump(res,open(f'{a.out}/basket.json','w'),indent=1);print(json.dumps(res,indent=1))
