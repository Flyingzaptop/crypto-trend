"""Weekly forward signal for ML3 (cross-sectional L/S on the top-50 Binance perps) plus the BTC+ETH trend.
Run once a week on MONDAY after 00:00 UTC: the decision uses the just-closed Sunday daily bar, positions are held Mon..Sun,
exactly as in the backtest (ml3.py / ml3_robust.py).
usage: python signal_ml3.py [--equity 10000] [--k 5] [--n 50] [--log forward_ml3.csv]

Data: data/ml/panel_*.parquet (build with `python panel.py`, monthly archives) extended to the last Sunday with
spot REST klines (data-api.binance.vision) and funding from fapi.binance.com; where fapi is blocked, daily funding is
estimated from the perp premium index (data.binance.vision daily premiumIndexKlines): 3 x (p + clamp(0.01% - p, +-0.05%)).
Model: LightGBM with the fixed ML3 parameters, retrained on every week whose target is already known."""
import os,io,json,zipfile,argparse,urllib.error,numpy as np,pandas as pd,lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor
import trend,panel,ml3,ml_data as md,archive_d1
from ml1 import PARAMS

API='https://data-api.binance.vision/api/v3/klines';FAPI='https://fapi.binance.com/fapi/v1'

def spot_tail(sym,start):
    """Daily close and quote volume from `start` (inclusive), completed days only; None if the pair no longer trades."""
    t=int(pd.Timestamp(start).tz_localize('UTC').timestamp()*1000)
    try:k=json.loads(trend._get(f'{API}?symbol={sym}&interval=1d&startTime={t}&limit=1000',tries=2))
    except urllib.error.HTTPError:return None
    if not k:return None
    d=pd.DataFrame({'close':[float(x[4]) for x in k],'qvol':[float(x[7]) for x in k]},index=pd.to_datetime([x[0] for x in k],unit='ms'))
    return d[d.index<pd.Timestamp.now(tz='UTC').floor('D').tz_localize(None)]

def funding_fapi(perp,start):
    t=int(pd.Timestamp(start).tz_localize('UTC').timestamp()*1000)
    r=json.loads(trend._get(f'{FAPI}/fundingRate?symbol={perp}&startTime={t}&limit=1000',tries=2))
    if not r:return pd.Series(dtype=float)
    s=pd.Series([float(x['fundingRate']) for x in r],index=pd.to_datetime([x['fundingTime'] for x in r],unit='ms'));return s.resample('1D').sum()

def funding_premium(perp,days):
    """Estimate from the daily premium-index close; empty days (perp not listed / delisted) are skipped."""
    out={}
    for d in days:
        ds=d.strftime('%Y-%m-%d')
        try:z=zipfile.ZipFile(io.BytesIO(trend._get(f'https://data.binance.vision/data/futures/um/daily/premiumIndexKlines/{perp}/1d/{perp}-1d-{ds}.zip',tries=2)))
        except Exception:continue
        x=pd.read_csv(z.open(z.namelist()[0]),header=None);x=x.iloc[1:] if not str(x.iat[0,0]).isdigit() else x;p=float(x.iat[0,4])
        out[d]=3*(p+min(max(1e-4-p,-5e-4),5e-4))
    return pd.Series(out,dtype=float)

def extend(C,Q,F,end):
    """Append days after the panel's last day up to `end` (inclusive)."""
    start=C.index[-1]+pd.Timedelta(days=1);days=pd.date_range(start,end,freq='D')
    if not len(days):return C,Q,F
    with ThreadPoolExecutor(16) as ex:tails=dict(zip(C.columns,ex.map(lambda s:spot_tail(s,start),C.columns)))
    C2=pd.DataFrame({s:t.close for s,t in tails.items() if t is not None}).reindex(days);Q2=pd.DataFrame({s:t.qvol for s,t in tails.items() if t is not None}).reindex(days)
    C=pd.concat([C,C2.reindex(columns=C.columns)]);Q=pd.concat([Q,Q2.reindex(columns=Q.columns)])
    # funding only matters for coins that can be in the universe: top-100 by recent turnover
    tv=Q.iloc[-30:].mean().sort_values(ascending=False);cand=[s for s in tv.index[:100] if C[s].iloc[-1:].notna().all()]
    perps=set(open(f'{archive_d1.OUT}/_perps.txt').read().split());F2=pd.DataFrame(index=days,columns=F.columns,dtype=float)
    try:funding_fapi('BTCUSDT',start);src='fapi'
    except Exception:src='premium-index estimate'
    def one(s):
        p=panel.perp_name(s,perps)
        if p is None:return s,pd.Series(dtype=float)
        try:return s,(funding_fapi(p,start) if src=='fapi' else funding_premium(p,days))
        except Exception:return s,pd.Series(dtype=float)
    with ThreadPoolExecutor(16) as ex:
        for s,f in ex.map(one,cand):F2[s]=f.reindex(days)
    print(f'extended {start.date()}..{end.date()}: {C2.notna().any().sum()} spot pairs, funding for {len(cand)} candidates via {src}')
    return C,Q,pd.concat([F,F2])

def ml3_weights(C,Q,F,n,k):
    rows,U,R=ml3.weekly(C,Q,F,n);FEAT=[c for c in rows.columns if c!='y'];rows=rows[rows.index.get_level_values(0)>='2020-01-01']
    last=U.index[-1];tr=rows.y.notna();m=lgb.LGBMRegressor(**PARAMS).fit(rows.loc[tr,FEAT],rows.loc[tr,'y'])
    te=rows.index.get_level_values(0)==last;p=pd.Series(m.predict(rows.loc[te,FEAT]),index=rows.index[te].get_level_values(1))
    W=ml3.book(p.to_frame().T.set_axis([last]),U.loc[[last]],k=k).iloc[0];return last,W[W!=0].sort_values(),p,int(tr.sum())

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--equity',type=float,default=None);a.add_argument('--k',type=int,default=5);a.add_argument('--n',type=int,default=50)
    a.add_argument('--log',default=None);a=a.parse_args()
    today=pd.Timestamp.now(tz='UTC').tz_localize(None).floor('D');sunday=today-pd.Timedelta(days=(today.dayofweek+1)%7 or 7)   # last fully closed Sunday bar
    C,Q,F=panel.load();C,Q,F=extend(C,Q,F,sunday);C,Q,F=C.loc[:sunday],Q.loc[:sunday],F.loc[:sunday]
    last,W,p,ntr=ml3_weights(C,Q,F,a.n,a.k)
    print(f'\nML3 L/S: decision at the close of Sunday {last.date()} (UTC), hold Mon..Sun. Universe top-{a.n}, {a.k} per leg, model trained on {ntr} coin-weeks.')
    for s,w in W.items():print(f'  {s:<14} {w:+.3f} x equity'+(f' = {w*a.equity:+,.0f} USD' if a.equity else '')+f'   pred {p[s]:+.4f}   close {C[s].iloc[-1]:.6g}')
    print(f'  gross {W.abs().sum():.2f}, net {W.sum():+.2f}')
    P=pd.DataFrame({s:trend.daily_closes(s,start=str((today-pd.Timedelta(days=500)).date())) for s in trend.COINS}).dropna().loc[:sunday]
    N=trend.target_positions(P).iloc[-1];print(f'\nTrend BTC+ETH (20% vol, cap 1x) from the same close:',', '.join(f'{s} {N[s]:+.3f}' for s in trend.COINS))
    if a.log:
        rows=[{'decision_date':last.date(),'book':'ml3','symbol':s,'weight':round(float(w),4),'pred':round(float(p[s]),5),'close':float(C[s].iloc[-1])} for s,w in W.items()]
        rows+=[{'decision_date':last.date(),'book':'trend','symbol':s,'weight':round(float(N[s]),4),'pred':np.nan,'close':float(P[s].iloc[-1])} for s in trend.COINS]
        J=pd.DataFrame(rows)
        if os.path.exists(a.log):
            old=pd.read_csv(a.log,parse_dates=['decision_date']);prev=old[old.decision_date==old.decision_date.max()]
            if len(prev) and prev.decision_date.iloc[0]<last:
                cl={**{s:C[s].iloc[-1] for s in prev.symbol if s in C},**{s:P[s].iloc[-1] for s in trend.COINS}}
                for b,g in prev.groupby('book'):
                    r=sum(w*(cl.get(s,c)/c-1) for s,w,c in zip(g.symbol,g.weight,g.close))
                    print(f'realized {b} week {prev.decision_date.iloc[0].date()} -> {last.date()}: {r:+.2%} (price only, before costs and funding)')
            if (old.decision_date==last).any():print('this decision date is already in',a.log,'- not appended');J=None
        if J is not None:J.to_csv(a.log,mode='a',header=not os.path.exists(a.log),index=False);print('appended to',a.log)
