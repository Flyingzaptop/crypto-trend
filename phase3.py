"""Phase 3 (channel C): volatility forecasts for position sizing (docs/vol/PROTOCOL_phase3.md). usage: python phase3.py"""
import numpy as np,pandas as pd
import lightgbm as lgb
import trend,strategies as st,tradfi as tf,tradfi2 as t2,phase1 as p1,vol_data as vd

OUT='docs/vol/results';PARAMS=p1.PARAMS
IVMAP={'SPY':'VIX','GLD':'GVZ','EEM':'VXEEM','EFA':'VXEFA','TLT':'VXTLT','USO':'OVX','SLV':'VXSLV','QQQ':'VXN','IWM':'RVX',
       'BTCUSDT':'DVOL_BTC','ETHUSDT':'DVOL_ETH'}

def estimates(P,ann,h,base_window,vix):
    """Annualized vol estimates per asset on each date: rv (rule's own), ewma, har, gbm, iv."""
    R=np.log(P).diff();out={}
    out['rv']=R.rolling(base_window).std()*np.sqrt(ann)
    out['ewma']=np.sqrt((R**2).ewm(alpha=0.06,adjust=False).mean()*ann)
    rows=[]
    for c in P:
        r=R[c];lr=lambda k:np.log(np.sqrt((r**2).rolling(k).sum()*ann/k))
        f=pd.DataFrame({'asset':c,'lrv1':np.log(r.abs()*np.sqrt(ann)+1e-4),'lrv5':lr(5),'lrv22':lr(h),'ret5':np.log(P[c]).diff(5),
                        'ret21':np.log(P[c]).diff(h),'lvix':np.log(vix.reindex(P.index).ffill()/100),
                        'y':np.log(np.sqrt((r**2).rolling(h).sum().shift(-h)*ann/h))})
        rows.append(f.dropna(subset=['lrv1','lrv5','lrv22']))
    X=pd.concat(rows);X['har']=np.nan;X['gbm']=np.nan;F=['lrv1','lrv5','lrv22','ret5','ret21','lvix']
    years=range(P.index.year.min()+3,2027)
    for c,x in X.groupby('asset'):
        Z=np.column_stack([np.ones(len(x)),x.lrv1,x.lrv5,x.lrv22]);y=x.y.values;first=x.index[0]
        for Y in years:
            cut=pd.Timestamp(f'{Y}-01-01')-pd.Timedelta(days=31);tr=(x.index<cut)&np.isfinite(y);te=(x.index.year==Y)
            if (cut-first).days<3*365 or te.sum()==0:continue
            b=np.linalg.lstsq(Z[tr],y[tr],rcond=None)[0];X.loc[(X.asset==c)&(X.index.year==Y),'har']=np.exp(Z[te]@b)
    for Y in years:
        cut=pd.Timestamp(f'{Y}-01-01')-pd.Timedelta(days=31);tr=(X.index<cut)&X.y.notna();te=X.index.year==Y
        if tr.sum()<1500 or te.sum()==0:continue
        m=lgb.LGBMRegressor(**PARAMS).fit(X.loc[tr,F],X.loc[tr,'y']);X.loc[te,'gbm']=np.exp(m.predict(X.loc[te,F]))
    X.loc[X.har.isna(),'gbm']=np.nan
    for k in ('har','gbm'):out[k]=X.reset_index().pivot_table(index='index' if 'index' in X.reset_index() else X.index.name,columns='asset',values=k).reindex(P.index)
    iv=pd.DataFrame({c:(p1.iv_series(IVMAP[c]).reindex(P.index).ffill(limit=3)/100 if c in IVMAP else np.nan) for c in P},index=P.index)
    out['iv']=iv
    ok=out['har'].notna()&out['gbm'].notna()                                    # common years with every estimate
    for k in ('ewma','har','gbm','iv'):out[k]=out[k].where(out[k].notna(),out['rv'])
    return out,ok.any(axis=1)

def summarize(r,ann):
    eq=r.cumsum();return {'sharpe':round(r.mean()/r.std()*np.sqrt(ann),2),'cagr_pct':round((np.exp(r.mean()*ann)-1)*100,1),
                          'max_dd_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1)}

def evaluate(name,returns,ann,ok,early):
    rows=[]
    for k,r in returns.items():
        r=r[ok.reindex(r.index).fillna(False)];row={'portfolio':name,'vol_est':k,'from':str(r.index[0].date())}
        for tag,(a,b) in (('early',early),('2024-26',('2024-01-01','2026-12-31'))):
            row.update({f'{tag}_{m}':v for m,v in summarize(r[a:b],ann).items()})
        rows.append(row)
    T=pd.DataFrame(rows).set_index('vol_est')
    for p in ('early','2024-26'):T[f'{p}_dSharpe']=(T[f'{p}_sharpe']-T.loc['rv',f'{p}_sharpe']).round(2);T[f'{p}_dDD']=(T[f'{p}_max_dd_pct']-T.loc['rv',f'{p}_max_dd_pct']).round(1)
    T['useful']=((T.early_dSharpe>=0.1)&(T['2024-26_dSharpe']>=0.1))|((T.early_dDD<=-3)&(T['2024-26_dDD']<=-3)&(T.early_dSharpe>=0)&(T['2024-26_dSharpe']>=0))
    T.loc['rv','useful']=False;return T.reset_index()

if __name__=='__main__':
    vix=p1.cboe('VIX');res=[]
    # 1 crypto trend BTC+ETH
    D=pd.DataFrame({s:trend.daily_closes(s) for s in trend.COINS}).dropna();D=D[D.index<pd.Timestamp.now().floor('D')]
    F=pd.DataFrame({s:trend.daily_funding(s) for s in trend.COINS}).reindex(D.index).fillna(0);R=np.log(D).diff()
    E,ok=estimates(D,365,30,20,vix);S=sum(np.sign(D-D.rolling(n).mean()) for n in trend.SMAS)/3;st.COST=10e-4
    rets={k:st.pnl(st.book(S,R,365,vol=v),R,F) for k,v in E.items()}
    res.append(evaluate('crypto trend BTC+ETH',rets,365,ok,('2021-01-01','2023-12-31')));print(res[-1].to_string(index=False),flush=True)
    # 2 multi-asset weekly ETF trend; 3 single assets
    P=pd.DataFrame({t:tf.yahoo(t) for t in tf.TICKERS}).sort_index();P=P[P.index<pd.Timestamp.now().normalize()];R=np.log(P).diff();Z=R*0
    E,ok=estimates(P,252,21,60,vix);st.COST=5e-4
    rets={k:st.pnl(t2.book_long(P,R,252,vol=v),R,Z) for k,v in E.items()}
    res.append(evaluate('ETF trend (19, weekly)',rets,252,ok,('2011-01-01','2023-12-31')));print(res[-1].to_string(index=False),flush=True)
    for a in ('SPY','GLD'):
        rets={k:st.pnl(t2.book_long(P[[a]],R[[a]],252,vol=v[[a]]),R[[a]],Z[[a]]) for k,v in E.items()}
        res.append(evaluate(f'{a} trend (weekly)',rets,252,ok,('2011-01-01','2023-12-31')));print(res[-1].to_string(index=False),flush=True)
    T=pd.concat(res);T.to_csv(f'{OUT}/phase3_sizing.csv',index=False)
