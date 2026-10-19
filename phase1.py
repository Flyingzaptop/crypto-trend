"""Phase 1: global map of the volatility premium and of its forecastability (docs/vol/PROTOCOL_phase1.md).
usage: python phase1.py -> docs/vol/results/phase1_*.csv"""
import os,json,urllib.request,io,numpy as np,pandas as pd
import lightgbm as lgb
import tradfi as tf,vol_data as vd,data_m1

OUT='docs/vol/results';CACHE=f'{data_m1.ROOT}/ivx'
MARKETS={'VIX':'^GSPC','VXN':'^NDX','RVX':'^RUT','VXD':'^DJI','VXEFA':'EFA','VXEEM':'EEM','VXEWZ':'EWZ','VXFXI':'FXI',
         'GVZ':'GLD','VXSLV':'SLV','VXGDX':'GDX','OVX':'USO','VXTLT':'TLT','VXAPL':'AAPL','VXAZN':'AMZN','VXGOG':'GOOG',
         'VXGS':'GS','VXIBM':'IBM','INDIAVIX':'^NSEI','DVOL_BTC':'BTCUSDT','DVOL_ETH':'ETHUSDT'}
PARAMS=dict(n_estimators=300,max_depth=3,learning_rate=0.03,min_child_samples=50,subsample=0.8,subsample_freq=1,verbose=-1)
FEAT=['lrv1','lrv5','lrv22','liv','div1','div5','div21','iv_rv','ret5','ret21','lvix']

def cboe(n):
    path=f'{CACHE}/{n}.csv'
    if not os.path.exists(path):
        req=urllib.request.Request(f'https://cdn.cboe.com/api/global/us_indices/daily_prices/{n}_History.csv',headers={'User-Agent':'Mozilla/5.0'})
        os.makedirs(CACHE,exist_ok=True);open(path,'wb').write(urllib.request.urlopen(req,timeout=60).read())
    d=pd.read_csv(path);d.index=pd.to_datetime(d.DATE);c='CLOSE' if 'CLOSE' in d else d.columns[-1]
    return d[c].astype(float).rename(n)

def iv_series(n):
    if n.startswith('DVOL'):return vd.dvol(n.split('_')[1])
    if n=='INDIAVIX':return tf.yahoo('^INDIAVIX')
    return cboe(n)

def underlying(t):
    if t.endswith('USDT'):return data_m1.daily_closes(t)
    return tf.yahoo(t)

def frame(n,vix):
    crypto=n.startswith('DVOL');A,H=(365,30) if crypto else (252,21)
    p=underlying(MARKETS[n]);iv=iv_series(n).reindex(p.index).ffill(limit=3)/100;r=np.log(p).diff()
    x=pd.DataFrame({'iv':iv,'r':r}).dropna(subset=['r'])
    rv=lambda k:np.sqrt((x.r**2).rolling(k).sum()*A/k)
    x['rv_fwd']=np.sqrt((x.r**2).rolling(H).sum().shift(-H)*A/H)
    x['lrv1']=np.log(x.r.abs()*np.sqrt(A)+1e-4);x['lrv5']=np.log(rv(5));x['lrv22']=np.log(rv(H))
    x['liv']=np.log(x.iv);x['div1']=x.liv.diff(1);x['div5']=x.liv.diff(5);x['div21']=x.liv.diff(H)
    x['iv_rv']=x.liv-x.lrv22;x['ret5']=np.log(p).reindex(x.index).diff(5);x['ret21']=np.log(p).reindex(x.index).diff(H)
    x['lvix']=np.log(vix.reindex(x.index).ffill()/100);x['mkt']=n;x['A'],x['H']=A,H
    return x.dropna(subset=['iv'])

def walk_forward(X):
    X['har']=np.nan;X['gbm']=np.nan
    years=range(X.index.year.min()+3,2027)
    for n,x in X.groupby('mkt'):
        Z=np.column_stack([np.ones(len(x)),x.lrv1,x.lrv5,x.lrv22]);y=np.log(x.rv_fwd).values
        for Y in years:
            tr=(x.index<pd.Timestamp(f'{Y}-01-01')-pd.Timedelta(days=31))&np.isfinite(y)&np.isfinite(Z).all(1)
            te=(x.index.year==Y)&np.isfinite(Z).all(1)
            if tr.sum()<3*200 or te.sum()==0:continue
            b=np.linalg.lstsq(Z[tr],y[tr],rcond=None)[0];X.loc[(X.mkt==n)&(X.index.year==Y)&np.isfinite(X[['lrv1','lrv5','lrv22']]).all(1).values,'har']=np.exp(Z[te]@b)
    X['tgt']=np.log(X.rv_fwd)-X.liv
    for Y in years:
        tr=(X.index<pd.Timestamp(f'{Y}-01-01')-pd.Timedelta(days=31))&X.tgt.notna()
        te=X.index.year==Y
        if tr.sum()<2000 or te.sum()==0:continue
        m=lgb.LGBMRegressor(**PARAMS).fit(X.loc[tr,FEAT],X.loc[tr,'tgt']);X.loc[te,'gbm']=X.loc[te,'iv']*np.exp(m.predict(X.loc[te,FEAT]))
    # a market's model forecast exists only once that market has 3 years of its own history (as for HAR)
    X.loc[X.har.isna(),'gbm']=np.nan
    return X

def periods(x):
    """Non-overlapping 30-day short variance swaps: entry every H rows."""
    H=int(x.H.iloc[0]);A=int(x.A.iloc[0]);q=x.iloc[::H].dropna(subset=['rv_fwd']).copy()
    q['pay']=(q.iv**2-q.rv_fwd**2)/(2*q.iv);q['ppy']=A/H;return q

def sharpe(v,ppy):return float(v.mean()/v.std()*np.sqrt(ppy)) if len(v)>3 and v.std()>0 else np.nan

if __name__=='__main__':
    vix=cboe('VIX');X=pd.concat([frame(n,vix) for n in MARKETS]);X=walk_forward(X)
    prem,swap,acc,money=[],[],[],[]
    for n,x in X.groupby('mkt',sort=False):
        v=((x.iv-x.rv_fwd)*100).dropna();yr=v.groupby(v.index.year).mean();yr=yr[yr.index<=2026]
        full=yr[[y for y in yr.index if (v.index.year==y).sum()>150]]
        prem.append({'mkt':n,'from':str(v.index[0].date()),'mean_vp':round(v.mean(),2),'median_vp':round(v.median(),2),'pos_pct':round((v>0).mean()*100),
                     'p5_vp':round(v.quantile(.05),1),'pos_years_pct':round((full>0).mean()*100),'slope_vp_per_yr':round(np.polyfit(full.index,full.values,1)[0],2),
                     'worst_dates':', '.join(str(d.date()) for d in v.nsmallest(60).index[::12][:5]),**{f'y{y}':round(m,1) for y,m in yr.items()}})
        q=periods(x);ppy=q.ppy.iloc[0]
        for lo,hi,tag in (('1990','2023','<2024'),('2024','2026','2024-26'),('1990','2026','all')):
            s=q.loc[lo:hi,'pay'];swap.append({'mkt':n,'period':tag,'n':len(s),'sharpe':round(sharpe(s,ppy),2),'pos_pct':round((s>0).mean()*100) if len(s) else np.nan,
                                                'worst_vp':round(s.min()*100,1) if len(s) else np.nan,'mean_vp':round(s.mean()*100,2) if len(s) else np.nan})
        for lo,hi,tag in (('2010','2023','2010-23'),('2024','2026','2024-26')):
            z=x.loc[lo:hi].dropna(subset=['rv_fwd','har','gbm'])
            for f in ('iv','har','gbm'):
                r=(z.rv_fwd/z[f])**2
                acc.append({'mkt':n,'period':tag,'forecast':f,'n':len(z),'mse_logvol':round(((np.log(z.rv_fwd)-np.log(z[f]))**2).mean(),4),'qlike':round((r-np.log(r)-1).mean(),4)})
            qq=q.loc[lo:hi].dropna(subset=['har','gbm'])
            row={'mkt':n,'period':tag,'n':len(qq),'always':round(sharpe(qq.pay,ppy),2)}
            for f in ('har','gbm'):
                sell=qq.iv>qq[f];row[f'filter_{f}']=round(sharpe(qq.pay.where(sell,0),ppy),2);row[f'longshort_{f}']=round(sharpe(qq.pay*np.where(sell,1,-1),ppy),2)
                row[f'sell_share_{f}']=round(sell.mean()*100)
            money.append(row)
    for name,rows in (('premium',prem),('varswap',swap),('forecast_acc',acc),('money',money)):pd.DataFrame(rows).to_csv(f'{OUT}/phase1_{name}.csv',index=False)
    P=pd.DataFrame(prem).set_index('mkt');S=pd.DataFrame(swap).pivot(index='mkt',columns='period',values='sharpe');M=pd.DataFrame(money).set_index(['mkt','period'])
    A=pd.DataFrame(acc).pivot_table(index=['mkt','period'],columns='forecast',values='mse_logvol')
    rows=[]
    for n in MARKETS:
        if n not in P.index:continue
        g=lambda per,c:M.loc[(n,per),c] if (n,per) in M.index else np.nan
        stable=(S.loc[n,'2024-26']>=0.5)&(P.loc[n,'pos_years_pct']>=70)&(P.loc[n,'slope_vp_per_yr']>=-0.3)
        useful=any((g('2010-23',f'filter_{f}')-g('2010-23','always')>=0.2) and (g('2024-26',f'filter_{f}')-g('2024-26','always')>=0.2) for f in ('har','gbm')) or \
               any((g('2010-23',f'longshort_{f}')>=0.5) and (g('2024-26',f'longshort_{f}')>=0.5) for f in ('har','gbm'))
        rows.append({'mkt':n,'underlying':MARKETS[n],'from':P.loc[n,'from'],'mean_premium_vp':P.loc[n,'mean_vp'],'pos_years_pct':P.loc[n,'pos_years_pct'],
                     'slope_vp_yr':P.loc[n,'slope_vp_per_yr'],'varswap_sharpe_<2024':S.loc[n,'<2024'],'varswap_sharpe_2024-26':S.loc[n,'2024-26'],
                     'mse_iv_24':A.loc[(n,'2024-26'),'iv'] if (n,'2024-26') in A.index else np.nan,'mse_har_24':A.loc[(n,'2024-26'),'har'] if (n,'2024-26') in A.index else np.nan,
                     'mse_gbm_24':A.loc[(n,'2024-26'),'gbm'] if (n,'2024-26') in A.index else np.nan,
                     'best_filter_gain_10-23':round(max(g('2010-23','filter_har'),g('2010-23','filter_gbm'))-g('2010-23','always'),2),
                     'best_filter_gain_24-26':round(max(g('2024-26','filter_har'),g('2024-26','filter_gbm'))-g('2024-26','always'),2),
                     'premium_stable':bool(stable),'forecast_useful':bool(useful)})
    R=pd.DataFrame(rows).sort_values('varswap_sharpe_2024-26',ascending=False);R.to_csv(f'{OUT}/phase1_rating.csv',index=False);print(R.to_string(index=False))
