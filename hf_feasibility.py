"""How much short-horizon edge exists in crypto before costs? usage: python hf_feasibility.py
For horizons 1m..4h on liquid coins (2022-2026): typical move, lag-1 autocorrelation, BTC->alt lead-lag,
and the gross next-bar spread (bp) of a market-neutral cross-sectional reversal (bottom-5 minus top-5 by last-bar residual)."""
import numpy as np,pandas as pd,data_m1
COINS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT','ADAUSDT','LINKUSDT','LTCUSDT','AVAXUSDT','DOTUSDT','TRXUSDT',
       'BCHUSDT','ETCUSDT','XLMUSDT','FILUSDT','NEARUSDT','ATOMUSDT','UNIUSDT','AAVEUSDT','ALGOUSDT','HBARUSDT','INJUSDT','SANDUSDT','CRVUSDT']
C=pd.DataFrame({s:data_m1.load(s).close['2022-01-01':] for s in COINS})
C.index=C.index.floor('min');C=C[~C.index.duplicated()].ffill(limit=3)
rows=[]
for h in ('1min','5min','15min','1h','4h'):
    P=C.resample(h).last();R=np.log(P).diff().dropna(how='all')
    mkt=R.mean(1);E=R.sub(mkt,axis=0)                                           # residual vs equal-weight market
    nxt=R.shift(-1);rank=E.rank(axis=1)
    lo=nxt.where(rank<=5).mean(1);hi=nxt.where(rank>=rank.max(1).values[:,None]-4).mean(1)
    spread=(lo-hi).dropna()                                                        # long losers, short winners
    lead=pd.concat([R.BTCUSDT,R.drop(columns='BTCUSDT').shift(-1).mean(1)],axis=1).dropna().corr().iat[0,1]
    ac=R.apply(lambda s:s.autocorr()).median()
    yr=spread.groupby(spread.index.year).mean()*1e4
    rows.append({'bar':h,'median_abs_move_bp':round(R.abs().median().median()*1e4,1),'autocorr_lag1':round(ac,3),
                 'btc_leads_alts_corr':round(lead,3),'reversal_gross_bp_per_bar':round(spread.mean()*1e4,2),
                 'reversal_hit_pct':round((spread>0).mean()*100,1),**{f'rev_{y}':round(v,2) for y,v in yr.items()}})
T=pd.DataFrame(rows);T.to_csv('docs/results/hf_feasibility.csv',index=False);print(T.T.to_string())
