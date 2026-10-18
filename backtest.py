"""Backtest from public Binance data. usage: python backtest.py [--vol 0.20] [--cap 1] [--cost 10] [--start 2017-12-15] [--out results]
Prints yearly results and saves results/equity.png, results/daily.csv, results/summary.json."""
import argparse,os,json,numpy as np,pandas as pd
import trend

a=argparse.ArgumentParser();a.add_argument('--vol',type=float,default=0.20);a.add_argument('--cap',type=float,default=1.0)
a.add_argument('--cost',type=float,default=10.0);a.add_argument('--start',default='2017-12-15');a.add_argument('--out',default='results')
a=a.parse_args();os.makedirs(a.out,exist_ok=True)
P=pd.DataFrame({s:trend.daily_closes(s) for s in trend.COINS}).dropna()
F=pd.DataFrame({s:trend.daily_funding(s) for s in trend.COINS})
ret,N=trend.backtest(P,F,a.vol,a.cap,a.cost);ret=ret[ret.index>=a.start];N=N.reindex(ret.index)
bh=np.log(P).diff().mean(1).reindex(ret.index)
def stats(r):
    eq=r.cumsum();return {'cagr_pct':round((np.exp(r.mean()*trend.ANN)-1)*100,1),'vol_pct':round(r.std()*np.sqrt(trend.ANN)*100,1),
        'sharpe':round(r.mean()/r.std()*np.sqrt(trend.ANN),2),'max_drawdown_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1)}
rows=[]
for y in sorted(set(ret.index.year)):
    q=ret[ret.index.year==y];n=N[N.index.year==y];eq=q.cumsum()
    rows.append({'year':y,'strategy_pct':round((np.exp(q.sum())-1)*100,1),'buy_hold_pct':round((np.exp(bh[bh.index.year==y].sum())-1)*100,1),
                 'max_dd_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1),'avg_gross':round(float(n.abs().sum(1).mean()),2),'days':len(q)})
Y=pd.DataFrame(rows);S={'params':vars(a),'period':[str(ret.index[0].date()),str(ret.index[-1].date())],'strategy':stats(ret),'buy_hold':stats(bh),
   'growth_of_1':round(float(np.exp(ret.sum())),2),'yearly':rows}
print(Y.to_string(index=False));print(json.dumps({k:S[k] for k in ('period','strategy','buy_hold','growth_of_1')},indent=1))
json.dump(S,open(f'{a.out}/summary.json','w'),indent=1);pd.DataFrame({'ret':ret,'buy_hold':bh}).join(N.add_suffix('_pos')).to_csv(f'{a.out}/daily.csv')
try:
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4.5));ax.plot(np.exp(ret.cumsum()),label=f'trend BTC+ETH, {a.vol:.0%} vol target, cap {a.cap:g}x',color='#2a6fdb')
    ax.plot(np.exp(bh.cumsum()),label='buy & hold BTC+ETH 50/50',color='#999');ax.set_yscale('log');ax.grid(alpha=.3);ax.legend()
    ax.set_title('Growth of 1 (log scale), costs and funding included');fig.tight_layout();fig.savefig(f'{a.out}/equity.png',dpi=110)
except ImportError:pass
