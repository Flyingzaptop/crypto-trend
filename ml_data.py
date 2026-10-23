"""Shared data for the ML program (docs/ml/): Binance REST klines (any interval), Yahoo daily OHLC, cached in data/ml/."""
import os,json,urllib.request,numpy as np,pandas as pd
import trend,data_m1

CACHE=f'{data_m1.ROOT}/ml'

def klines(sym,interval='1d',start='2017-08-01'):
    """Spot klines from data-api.binance.vision: close, quote_volume; completed bars only."""
    path=f'{CACHE}/{sym}_{interval}.parquet'
    if os.path.exists(path):return pd.read_parquet(path)
    step={'1d':86400000,'4h':14400000,'1h':3600000}[interval];t=int(pd.Timestamp(start,tz='UTC').timestamp()*1000);rows=[]
    while True:
        k=json.loads(trend._get(f'https://data-api.binance.vision/api/v3/klines?symbol={sym}&interval={interval}&startTime={t}&limit=1000'))
        if not k:break
        rows+=k;t=k[-1][0]+step
        if len(k)<1000:break
    d=pd.DataFrame({'close':[float(x[4]) for x in rows],'quote_volume':[float(x[7]) for x in rows]},index=pd.to_datetime([x[0] for x in rows],unit='ms'))
    d=d[~d.index.duplicated()];d=d[d.index+pd.Timedelta(milliseconds=step)<=pd.Timestamp.now(tz='UTC').tz_localize(None)]
    os.makedirs(CACHE,exist_ok=True);d.to_parquet(path);return d

def yahoo_ohlc(t):
    """Daily open/close (split/dividend-adjusted via adjclose/close ratio), index = trading date."""
    path=f'{CACHE}/y_{t}.parquet'
    if os.path.exists(path):return pd.read_parquet(path)
    req=urllib.request.Request(f'https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1=0&period2=9999999999&interval=1d&events=div,split',headers={'User-Agent':'Mozilla/5.0'})
    r=json.loads(urllib.request.urlopen(req,timeout=60).read())['chart']['result'][0];q=r['indicators']['quote'][0]
    d=pd.DataFrame({'open':q['open'],'close':q['close']},index=pd.to_datetime(r['timestamp'],unit='s').normalize())
    adj=r['indicators'].get('adjclose',[{}])[0].get('adjclose')
    d['adj']=adj if adj is not None else d.close
    d=d.dropna();d=d[~d.index.duplicated(keep='last')];d=d[d.index<pd.Timestamp.now().normalize()]
    f=d.adj/d.close;d['open']=d.open*f;d['close']=d.adj;d=d[['open','close']]
    os.makedirs(CACHE,exist_ok=True);d.to_parquet(path);return d

def pnl(N,R,F,cost_bp=10.0,simple=False):
    """Same accounting as trend.backtest: N row t = position held during t+1.
    simple=True: book return = sum of w*(e^r - 1), returned as log(1 + that). Needed for high-vol books (alts), where a weighted
    sum of log returns understates longs and overstates shorts by ~sigma^2/2 per year."""
    N=N.shift(1);N0=N.fillna(0);F=F.reindex(R.index).fillna(0)
    g=(N0*(np.expm1(R) if simple else R)).sum(1)-(N0*F).sum(1)-N0.diff().abs().sum(1)*cost_bp*1e-4
    ret=np.log1p(g.clip(lower=-0.99)) if simple else g
    return ret[N.notna().any(axis=1)]

def stats(r,ann=365):
    eq=r.cumsum();return {'sharpe':round(r.mean()/r.std()*np.sqrt(ann),2) if r.std()>0 else np.nan,'cagr_pct':round((np.exp(r.mean()*ann)-1)*100,1),
                          'vol_pct':round(r.std()*np.sqrt(ann)*100,1),'max_dd_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1)}

def by_year(r,ann=365):return r.groupby(r.index.year).apply(lambda q:round(q.mean()/q.std()*np.sqrt(ann),2) if q.std()>0 else np.nan)
