"""Addendum to docs/PROTOCOL_tradfi.md: A low-turnover weekly ETF trend, B hourly ETF trend, C leverage table for crypto trend.
usage: python tradfi2.py"""
import os,json,urllib.request,numpy as np,pandas as pd
import trend,strategies as st,tradfi as tf

def book_long(P,R,ann,smas=(50,100,200),vw=60,rw=60,cap=3.0,band=0.25):
    """Weekly (Friday) re-sizing with a no-trade band; same sizing idea as trend.target_positions."""
    S=sum(np.sign(P-P.rolling(n).mean()) for n in smas)/len(smas);N=st.book(S,R,ann,vw,rw,cap=cap)
    fri=N.index.dayofweek==4;W=N.where(pd.Series(fri,index=N.index),np.nan).ffill()
    out=W.copy();held=pd.Series(0.,index=N.columns)
    for i,t in enumerate(W.index):
        tgt=W.loc[t]
        chg=(tgt-held).abs()>band*held.abs().clip(lower=0.02)
        held=held.where(~chg|tgt.isna(),tgt).fillna(0);out.iloc[i]=held
    return out.where(N.notna()|W.notna())

def hourly(t):
    path=f'{tf.CACHE}/{t}_1h.csv'
    if os.path.exists(path):return pd.read_csv(path,index_col=0,parse_dates=True).iloc[:,0]
    req=urllib.request.Request(f'https://query1.finance.yahoo.com/v8/finance/chart/{t}?range=730d&interval=1h',headers={'User-Agent':'Mozilla/5.0'})
    r=json.loads(urllib.request.urlopen(req,timeout=60).read())['chart']['result'][0]
    s=pd.Series(r['indicators']['quote'][0]['close'],index=pd.to_datetime(r['timestamp'],unit='s'),name=t).dropna()
    s=s[~s.index.duplicated(keep='last')];s.to_csv(path);return s

def summary(r,ann):
    ys=r.groupby(r.index.year).apply(lambda q:tf.sharpe(q,ann))
    return {'cagr_pct':round((np.exp(r.mean()*ann)-1)*100,1),'vol_pct':round(r.std()*np.sqrt(ann)*100,1),'sharpe':round(tf.sharpe(r,ann),2),
            'max_dd_pct':round((1-np.exp(-(r.cumsum().cummax()-r.cumsum()).max()))*100,1),'pos_years_pct':round((ys>0).mean()*100),
            'slope':round(np.polyfit(ys.index,ys.values,1)[0],3) if len(ys)>2 else np.nan,**{str(y):round(v,2) for y,v in ys.items()}}

if __name__=='__main__':
    P=pd.DataFrame({t:tf.yahoo(t) for t in tf.TICKERS}).sort_index();P=P[P.index<pd.Timestamp.now().normalize()];R=np.log(P).diff()
    Z=R*0;st.COST=5e-4
    N=book_long(P,R,252);rA=st.pnl(N,R,Z)['2008':]
    print('A turnover/yr',round(N.diff().abs().sum(1)['2008':].mean()*252,1),'avg gross',round(N.abs().sum(1)['2008':].mean(),2))
    A=summary(rA,252)
    D=st.bars('1D');D=D[D.index<pd.Timestamp.now().floor('D')];F=pd.DataFrame({s:trend.daily_funding(s) for s in st.C}).reindex(D.index).fillna(0)
    crypto,_=trend.backtest(D,F);cr=tf.to_dates(crypto,P.index);j=pd.concat([rA,cr],axis=1).dropna();j=j[j.index>='2018-01-01'];cb=(j/j.std()).mean(1)
    A.update(corr=round(j.corr().iat[0,1],2),**{f'combo_{lo}+':round(tf.sharpe(cb[lo:],252),2) for lo in ('2018','2024')},
             **{f'crypto_{lo}+':round(tf.sharpe(j.iloc[:,1][lo:],252),2) for lo in ('2018','2024')})
    print('A',json.dumps(A))

    H=['SPY','QQQ','IWM','EFA','EEM','TLT','IEF','GLD','SLV','USO','UUP','FXE','FXY','VNQ']
    Ph=pd.DataFrame({t:hourly(t) for t in H}).sort_index().ffill(limit=2);Rh=np.log(Ph).diff();ann=252*7
    Sh=sum(np.sign(Ph-Ph.rolling(n).mean()) for n in trend.SMAS)/3;st.COST=2e-4;Nh=st.book(Sh,Rh,ann,cap=3.0)
    for c in (0,2):
        st.COST=c*1e-4;rh=st.pnl(Nh,Rh,Rh*0)
        print(f'B hourly cost {c}bp',json.dumps({k:v for k,v in summary(rh,ann).items()}),'turnover/yr',round(Nh.diff().abs().sum(1).mean()*ann,1))

    # C leverage table (crypto trend), actual and with Sharpe haircut to 0.7
    rows=[];st.COST=10e-4
    for v in (0.2,0.3,0.4,0.6,0.8,1.0):
        r,_=trend.backtest(D,F,v,v/0.2);r=r['2018':]
        for name,x in (('backtest',r),('sharpe0.7',r-r.mean()+0.7*r.std()/np.sqrt(365))):
            eq=x.cumsum();yr=x.groupby(x.index.year).sum()
            rows.append({'vol_target':v,'case':name,'cagr_pct':round((np.exp(x.mean()*365)-1)*100,1),
                         'max_dd_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1),'worst_year_pct':round((np.exp(yr.min())-1)*100,1),
                         'best_year_pct':round((np.exp(yr.max())-1)*100,1),'x_per_year_median':round(float(np.exp(yr.median())),2)})
    T=pd.DataFrame(rows);T.to_csv('docs/results/leverage.csv',index=False);print(T.to_string(index=False))
    json.dump(A,open('docs/results/tradfi_weekly.json','w'),indent=1)
