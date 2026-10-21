"""Phase 4 (channel D): trend vs mean-reversion switching by forecast vol regime (docs/vol/PROTOCOL_phase4.md)."""
import numpy as np,pandas as pd
import trend,strategies as st,tradfi as tf,phase1 as p1,phase3 as p3

OUT='docs/vol/results'

def run(P,F,ann,h,cap,cost,early,name,vix):
    R=np.log(P).diff();E,ok=p3.estimates(P,ann,h,20,vix);fc=E['har'].reindex(columns=P.columns);rv=np.sqrt((R**2).rolling(h).sum()*ann/h)
    T=sum(np.sign(P-P.rolling(n).mean()) for n in trend.SMAS)/3
    z=np.log(P).diff(5)/(R.rolling(20).std()*np.sqrt(5));M=-(z/2).clip(-1,1)
    up=fc>rv;lvl=fc>fc.rolling(252,min_periods=126).median()
    sig={'trend':T,'meanrev':M,'mix50':(T+M)/2,'S1 rise->T':T.where(up,M),"S1' rise->M":M.where(up,T),
         'S2 high->T':T.where(lvl,M),"S2' high->M":M.where(lvl,T)}
    st.COST=cost;rows=[]
    for k,S in sig.items():
        r=st.pnl(st.book(S.where(fc.notna()),R,ann,cap=cap),R,F)
        r=r[ok.reindex(r.index).fillna(False)];row={'portfolio':name,'strategy':k}
        for tag,(a,b) in (('early',early),('2024-26',('2024-01-01','2026-12-31'))):
            q=r[a:b];eq=q.cumsum();row.update({f'{tag}_sharpe':round(q.mean()/q.std()*np.sqrt(ann),2),f'{tag}_cagr_pct':round((np.exp(q.mean()*ann)-1)*100,1),
                                                f'{tag}_max_dd_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1)})
        rows.append(row)
    T_=pd.DataFrame(rows);base=T_[T_.strategy.isin(['trend','meanrev','mix50'])]
    sw=T_[T_.strategy.str.startswith('S')];best=sw.sort_values('early_sharpe',ascending=False).iloc[0]
    T_['selected']=T_.strategy==best.strategy
    T_['useful']=T_.selected&(best.early_sharpe-base.early_sharpe.max()>=0.2)&(best['2024-26_sharpe']-base['2024-26_sharpe'].max()>=0.2)&(best['2024-26_sharpe']>=0.5)
    return T_

if __name__=='__main__':
    vix=p1.cboe('VIX');out=[]
    D=pd.DataFrame({s:trend.daily_closes(s) for s in trend.COINS}).dropna();D=D[D.index<pd.Timestamp.now().floor('D')]
    F=pd.DataFrame({s:trend.daily_funding(s) for s in trend.COINS}).reindex(D.index).fillna(0)
    out.append(run(D,F,365,30,1.0,10e-4,('2021-01-01','2023-12-31'),'crypto BTC+ETH',vix));print(out[-1].to_string(index=False),flush=True)
    P=pd.DataFrame({t:tf.yahoo(t) for t in tf.TICKERS}).sort_index();P=P[P.index<pd.Timestamp.now().normalize()]
    out.append(run(P,P*0,252,21,3.0,5e-4,('2011-01-01','2023-12-31'),'ETF 19',vix));print(out[-1].to_string(index=False),flush=True)
    pd.concat(out).to_csv(f'{OUT}/phase4_switch.csv',index=False)
