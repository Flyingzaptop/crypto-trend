"""ML3: weekly cross-sectional ranking of the top-50 perps, LightGBM (docs/ml/PROTOCOL_ML3.md). usage: python ml3.py -> docs/ml/results/ml3*.csv"""
import numpy as np,pandas as pd,lightgbm as lgb
import trend,panel,ml_data as md
from ml1 import PARAMS

SEL=('2021-01-01','2023-12-31');VER='2024-01-01'

def weekly(C,Q,F,n=50):
    R=np.log(C).diff();sun=C.index[C.index.dayofweek==6];U=panel.universe(C,Q,F,n).loc[sun]
    lw=np.log(C.ffill(limit=6)).loc[sun];W={}           # last close within the week (delisted coins exit at their last close)
    W['mom1']=lw.diff(1);W['mom4']=lw.diff(4);W['mom12']=lw.diff(12)
    W['vol4']=R.rolling(28,min_periods=20).std().loc[sun];W['fund1']=F.fillna(0).rolling(7).sum().loc[sun];W['fund4']=F.fillna(0).rolling(28).sum().loc[sun]
    W['lvol']=np.log(Q.rolling(7).sum()+1).loc[sun]-np.log(Q.rolling(28).sum()/4+1).loc[sun]
    B=R['BTCUSDT'];W['beta12']=R.rolling(84,min_periods=60).cov(B).div(B.rolling(84,min_periods=60).var(),axis=0).loc[sun]
    W['size']=np.log(Q.rolling(30,min_periods=20).mean()+1).loc[sun]
    for k in list(W):W[f'{k}_rk']=W[k].where(U).rank(axis=1,pct=True)
    nxt=lw.shift(-1)-lw;tgt=nxt.where(U);tgt=tgt.sub(tgt.mean(1),axis=0)
    u=U.stack();idx=u[u].index
    rows=pd.DataFrame({k:v.stack(future_stack=True).reindex(idx) for k,v in W.items()});rows['y']=tgt.stack(future_stack=True).reindex(idx)
    return rows,U,R

def book(score,U,k=5,ls=True):
    """Weekly weights from a score frame (Sunday rows): long top-k (and short bottom-k)."""
    s=score.where(U);r=s.rank(axis=1,ascending=False);n=s.notna().sum(1)
    L=(r<=k).astype(float).mul(1/k);S=r.gt(n-k,axis=0).astype(float).mul(1/k) if ls else 0
    return L*0.5-S*0.5 if ls else L

def daily(Wk,C):
    return Wk.reindex(C.index).ffill().fillna(0)          # Sunday-close decision, held Mon..Sun

if __name__=='__main__':
    C,Q,F=panel.load();rows,U,R=weekly(C,Q,F);FEAT=[c for c in rows.columns if c!='y']
    rows=rows[rows.index.get_level_values(0)>='2020-01-01'];pred=pd.Series(np.nan,index=rows.index)
    for Y in range(2021,rows.index.get_level_values(0).max().year+1):
        t=rows.index.get_level_values(0);tr=(t+pd.Timedelta(days=7)<pd.Timestamp(f'{Y}-01-01'))&rows.y.notna();te=t.year==Y
        m=lgb.LGBMRegressor(**PARAMS).fit(rows.loc[tr,FEAT],rows.loc[tr,'y']);pred[te]=m.predict(rows.loc[te,FEAT])
    P=pred.unstack().reindex(U.index);ok=rows.dropna(subset=['y']).join(pred.rename('p'));ok=ok[ok.p.notna()]
    ic=ok.groupby(level=0).apply(lambda g:g.p.corr(g.y,method='spearman'))
    M4=rows.mom4.unstack().reindex(U.index)
    Rf=R.fillna(0);Ff=F.fillna(0);Pp=pd.DataFrame({s:md.klines(s,'1d').close for s in trend.COINS}).dropna()
    S={'ML L/S top5/bottom5':md.pnl(daily(book(P,U),C),Rf,Ff,simple=True),'ML long top5':md.pnl(daily(book(P,U,ls=False),C),Rf,Ff,simple=True),
       'mom4 L/S (no ML)':md.pnl(daily(book(M4,U),C),Rf,Ff,simple=True),'mom4 long top5 (no ML)':md.pnl(daily(book(M4,U,ls=False),C),Rf,Ff,simple=True),
       'EW top-50 long':md.pnl(daily(U.astype(float).div(U.sum(1),axis=0),C),Rf,Ff,simple=True),
       'trend BTC+ETH':trend.backtest(Pp,pd.DataFrame({s:trend.daily_funding(s) for s in trend.COINS}))[0]}
    out=[]
    for k,r in S.items():
        r=r['2021-01-04':];out.append({'series':k,**{f'sel_{a}':b for a,b in md.stats(r[:SEL[1]]).items()},**{f'ver_{a}':b for a,b in md.stats(r[VER:]).items()},
                                     **{str(y):b for y,b in md.by_year(r).items()}})
    T=pd.DataFrame(out);T.to_csv('docs/ml/results/ml3.csv',index=False);pd.DataFrame(S)['2021-01-04':].to_csv('docs/ml/results/ml3_daily_returns.csv')
    icy=ic.groupby(ic.index.year).mean().round(3);pd.DataFrame({'ic_mean':icy,'weeks':ic.groupby(ic.index.year).size()}).to_csv('docs/ml/results/ml3_ic.csv')
    to=daily(book(P,U),C).diff().abs().sum(1);print('avg coins in universe',round(U.sum(1)['2021':].mean(),1),'| L/S turnover per week',round(to['2021':].sum()/len(U['2021':]),2))
    pd.set_option('display.width',300);pd.set_option('display.max_columns',40);print(T.to_string(index=False));print('IC by year',icy.to_dict(),'| IC 2021-23',round(ic[:SEL[1]].mean(),3),'2024+',round(ic[VER:].mean(),3))
