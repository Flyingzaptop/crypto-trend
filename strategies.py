"""Five second-strategy candidates vs the daily trend, as declared in docs/PROTOCOL_strategies.md.
usage: python strategies.py   -> docs/results/strategies.csv (needs the 1m cache for BTC/ETH)"""
import io,os,zipfile,numpy as np,pandas as pd
import trend,data_m1,archive_d1
from select_pool import stats as _stats

C=list(trend.COINS);COST=10e-4

def stats(r,ann=365):
    s=_stats(r);s['sharpe']=round(r.mean()/r.std()*np.sqrt(ann),2) if r.std()>0 else np.nan;return s

def bars(rule):
    return pd.DataFrame({s:data_m1.load(s).close.resample(rule).last() for s in C}).dropna()

def perp_daily(sym):
    path=f'{data_m1.ROOT}/perp_d1_{sym}.parquet'
    if os.path.exists(path):return pd.read_parquet(path)[sym]
    parts=[]
    for k in archive_d1.ls(f'data/futures/um/monthly/klines/{sym}/1d/',keys=True):
        if k.endswith('.zip'):
            z=zipfile.ZipFile(io.BytesIO(trend._get(f'https://data.binance.vision/{k}')));d=pd.read_csv(z.open(z.namelist()[0]),header=None)
            parts.append(d.iloc[1:] if not str(d.iat[0,0]).isdigit() else d)
    d=pd.concat(parts);s=pd.Series(d[4].astype(float).values,index=data_m1._ts(d[0]),name=sym).sort_index()
    s=s[~s.index.duplicated()];s.to_frame().to_parquet(path);return s

def book(S,R,ann,vw=20,rw=60,target_vol=0.20,cap=1.0):
    """Same sizing as trend.target_positions, for any raw signal S in [-1,1]; windows in bars."""
    S=S*(0.5/(R.rolling(vw).std()*np.sqrt(ann))).clip(upper=2)
    pre=S/(S.shift(1)*R).rolling(rw,min_periods=rw//2).std();pre=pre.div(pre.notna().sum(1),axis=0)
    pr=(pre.shift(1)*R).sum(1,min_count=1);N=pre.mul((target_vol/np.sqrt(ann))/pr.rolling(rw,min_periods=rw//2).std(),axis=0)
    return N.div((N.abs().sum(1)/cap).clip(lower=1),axis=0)

def pnl(N,R,Fbar):
    N0=N.shift(1).fillna(0);r=(N0*R).sum(1)-(N0*Fbar.reindex(R.index).fillna(0)).sum(1)-N0.diff().abs().sum(1)*COST
    return r[N.shift(1).notna().any(axis=1)]

def daily(r):return r.groupby(r.index.floor('D')).sum()

def combine(a,b,ann=365):
    """50/50 risk mix: each leg divided by its own (full-period) vol, then rescaled to 20% vol."""
    j=pd.concat([a,b],axis=1).dropna();c=(j/j.std()).mean(1);return c*(0.2/np.sqrt(ann))/c.std()

if __name__=='__main__':
    D=bars('1D');D=D[D.index<pd.Timestamp.now().floor('D')];R=np.log(D).diff()
    F=pd.DataFrame({s:trend.daily_funding(s) for s in C}).reindex(D.index).fillna(0)
    out={}
    base,Nb=trend.backtest(D,F);out['0 trend (daily, current)']=base

    # 1 funding carry
    Pp=pd.DataFrame({s:perp_daily(s) for s in C}).reindex(D.index);Rp=np.log(Pp).diff()
    on=(F.rolling(7).mean()>0).astype(float).where(Pp.notna()).shift(1)       # decided with funding up to yesterday
    leg=(R-Rp+F)*on;cost=on.fillna(0).diff().abs()*2*COST
    out['1 funding carry (unlevered)']=((leg-cost)*0.5).sum(1,min_count=1).dropna().loc['2019-10-01':]

    # 2 short-term mean reversion: daily and 4h
    z=R/R.rolling(20).std();out['2a mean reversion daily']=pnl(book(-(z/2).clip(-1,1),R,365),R,F)
    H4=bars('4h');R4=np.log(H4).diff();F4=F.reindex(R4.index,method='ffill')/6;z4=R4/R4.rolling(120).std()
    out['2b mean reversion 4h']=daily(pnl(book(-(z4/2).clip(-1,1),R4,6*365,120,360),R4,F4))

    # 3 seasonality, fitted on 2018-2023, traded 2024+
    fit=R[(R.index>='2018-01-01')&(R.index<'2024-01-01')].mean(1).groupby(lambda t:t.dayofweek).mean().sort_values()
    sig=pd.Series(R.index.map(lambda t:1 if (t+pd.Timedelta('1D')).dayofweek in fit.index[-2:] else -1 if (t+pd.Timedelta('1D')).dayofweek in fit.index[:2] else 0),index=R.index)
    out['3a day-of-week (fit<2024)']=pnl(book(pd.DataFrame({s:sig for s in C}).astype(float),R,365),R,F)
    H1=bars('1h');R1=np.log(H1).diff();F1=F.reindex(R1.index,method='ffill')/24
    fh=R1[(R1.index>='2018-01-01')&(R1.index<'2024-01-01')].mean(1).groupby(lambda t:t.hour).mean().sort_values()
    nh=(R1.index+pd.Timedelta('1h')).hour;sh=pd.Series(np.where(np.isin(nh,fh.index[-6:]),1.,np.where(np.isin(nh,fh.index[:6]),-1.,0.)),index=R1.index)
    out['3b hour-of-day (fit<2024)']=daily(pnl(book(pd.DataFrame({s:sh for s in C}),R1,24*365,480,1440),R1,F1))

    # 4 trend on 4h bars
    S4=sum(np.sign(H4-H4.rolling(n).mean()) for n in trend.SMAS)/len(trend.SMAS)
    out['4 trend 4h']=daily(pnl(book(S4,R4,6*365,20,60),R4,F4))

    # 5 funding filter on the current trend
    F7=F.rolling(7).mean();thr=F7.rolling(365,min_periods=180).quantile(0.9);N=trend.target_positions(D)
    out['5 trend + funding filter']=pnl(N.where(~((N>0)&(F7>thr)),N*0.5),R,F)

    rows=[]
    for k,r in out.items():
        for lo in ('2018-01-01','2024-01-01'):
            q=r[r.index>=lo];row={'strategy':k,'from':lo[:4],**stats(q)}
            if not k.startswith('0'):
                j=pd.concat([q,base[base.index>=lo]],axis=1).dropna();row['corr_with_trend']=round(j.corr().iat[0,1],2)
                cb=combine(base,r);row['combo_sharpe']=stats(cb[cb.index>=lo])['sharpe'];row['trend_sharpe_same_days']=stats(base.reindex(cb[cb.index>=lo].index))['sharpe']
            rows.append(row)
    T=pd.DataFrame(rows);os.makedirs('docs/results',exist_ok=True);T.to_csv('docs/results/strategies.csv',index=False);print(T.to_string(index=False))
