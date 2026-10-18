"""Core of the BTC+ETH daily trend strategy: public Binance data loaders and the position rule.

The same `target_positions` function is used by the backtest and by the live signal, so what you trade is exactly
what was tested.

Rule (fixed, no fitting):
  per coin   signal = mean(sign(close - SMA_n)) for n in (20, 50, 100)          -> -1, -1/3, +1/3, +1
             sleeve = signal * min(2, 0.50 / vol20)                              (vol20 = 20-day realized vol, annualized)
  portfolio  each sleeve scaled to equal risk (trailing 60-day vol of its own returns), 1/n each (n = coins with a
             signal that day; coins enter after ~130 days of history), summed,
             then scaled to TARGET_VOL using the trailing 60-day vol of the combined book,
             gross exposure (|BTC| + |ETH|) capped at CAP x equity.
  timing     computed from completed daily closes (00:00 UTC), held for the next day.
"""
import io,json,time,zipfile,urllib.request,datetime as dt
import numpy as np,pandas as pd

COINS=('BTCUSDT','ETHUSDT')
SMAS=(20,50,100)
ANN=365

def _get(url,tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url,timeout=60) as r:return r.read()
        except Exception:
            if k==tries-1:raise
            time.sleep(2**k)

def daily_closes(symbol,start='2017-08-01'):
    """Spot daily closes from Binance public market data (data-api.binance.vision), completed days only."""
    t=int(pd.Timestamp(start,tz='UTC').timestamp()*1000);rows=[]
    while True:
        k=json.loads(_get(f'https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1d&startTime={t}&limit=1000'))
        if not k:break
        rows+=k;t=k[-1][0]+86400000
        if len(k)<1000:break
    s=pd.Series([float(x[4]) for x in rows],index=pd.to_datetime([x[0] for x in rows],unit='ms'),name=symbol)
    today=pd.Timestamp.now(tz='UTC').floor('D').tz_localize(None)
    return s[~s.index.duplicated()][s.index<today]

def daily_funding(symbol,start='2019-09',cache='data/funding'):
    """USD-M perpetual funding (sum per UTC day) from data.binance.vision monthly files; missing months -> 0.
    Completed months are cached in {cache}/{symbol}.csv."""
    import os;path=f'{cache}/{symbol}.csv';last=pd.Period(dt.date.today(),'M')-1
    if os.path.exists(path):
        old=pd.read_csv(path,index_col=0,parse_dates=True).iloc[:,0]
        if len(old)==0 or pd.Period(old.index[-1],'M')>=last:return old
    s=_funding(symbol,start)
    os.makedirs(cache,exist_ok=True);s.rename('funding').to_csv(path);return s

def _funding(symbol,start):
    parts=[];m=pd.Period(start,'M');last=pd.Period(dt.date.today(),'M')-1
    while m<=last:
        try:
            z=zipfile.ZipFile(io.BytesIO(_get(f'https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{m}.zip',tries=2)))
            parts.append(pd.read_csv(z.open(z.namelist()[0])))
        except Exception:pass
        m+=1
    if not parts:return pd.Series(dtype=float)
    f=pd.concat(parts);f.index=pd.to_datetime(f.calc_time,unit='ms')
    return f.last_funding_rate.astype(float).resample('1D').sum()

def target_positions(P,target_vol=0.20,cap=1.0):
    """P: DataFrame of daily closes (columns = coins). Returns DataFrame of target positions as a fraction of equity
    (+ long / - short), row t = position to hold during day t+1, decided with closes up to and including day t."""
    R=np.log(P).diff();W=(0.5/(R.rolling(20).std()*np.sqrt(ANN))).clip(upper=2)
    S=(sum(np.sign(P-P.rolling(n).mean()) for n in SMAS)/len(SMAS))*W
    sc=1/(S.shift(1)*R).rolling(60,min_periods=30).std()                       # equal-risk sleeves
    pre=S*sc;pre=pre.div(pre.notna().sum(1),axis=0)                            # 1/n of risk per coin with a signal
    pr=(pre.shift(1)*R).sum(1,min_count=1)
    ps=(target_vol/np.sqrt(ANN))/pr.rolling(60,min_periods=30).std()             # portfolio vol target
    N=pre.mul(ps,axis=0);g=N.abs().sum(1)
    return N.div((g/cap).clip(lower=1),axis=0)                                     # gross leverage cap

def backtest(P,F=None,target_vol=0.20,cap=1.0,cost_bp=10.0):
    """Daily log returns of the strategy: yesterday's targets x today's log returns, minus funding paid on perps and
    cost_bp per unit of position change."""
    N=target_positions(P,target_vol,cap).shift(1);R=np.log(P).diff()
    F=(F if F is not None else pd.DataFrame(0.,index=P.index,columns=P.columns)).reindex(P.index).fillna(0)
    N0=N.fillna(0)                                                             # coin not yet listed / no signal -> flat
    ret=(N0*R).sum(1)-(N0*F).sum(1)-N0.diff().abs().sum(1)*cost_bp*1e-4
    return ret[N.notna().any(axis=1)],N
