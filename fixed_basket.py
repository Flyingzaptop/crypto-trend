"""Fixed all-time top-5 / top-10 baskets vs BTC+ETH (no selection). usage: python fixed_basket.py"""
import json,numpy as np,pandas as pd,select_pool as sp
T5=['BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','SOLUSDT'];T10=T5+['ADAUSDT','DOGEUSDT','TRXUSDT','LTCUSDT','LINKUSDT']
P,F=sp.load(T10);rows=[];rb,_=sp.run(P,F,T5[:2]);mb=sp.monthly(rb)
for name,c in (('BTC+ETH',T5[:2]),('top5',T5),('top10',T10)):
    for cost in (10,20):
        r,_=sp.run(P,F,c,cost)
        for lo in ('2018-01','2024-01'):
            row={'set':name,'cost_bp':cost,'from':lo,**sp.stats(r[r.index>=lo])}
            if name!='BTC+ETH':
                mk=sp.monthly(r);m=mb.index[(mb.index>=lo)&(mb.index<='2026-09')];d=mk[m]-mb[m]
                row.update(hit_pct=round((d>0).mean()*100,1),delta_pct_yr=round(d.mean()*1200,2))
            rows.append(row)
    r,_=sp.run(P,F,c);print(name,{y:round((np.exp(r[r.index.year==y].sum())-1)*100,1) for y in range(2018,2027)})
T=pd.DataFrame(rows);print(T.to_string(index=False));T.to_csv('docs/results/fixed_basket.csv',index=False)
