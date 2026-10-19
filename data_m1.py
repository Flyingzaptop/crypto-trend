"""Minute (1m) klines from Binance public archives (data.binance.vision) -> local parquet cache.

usage: python data_m1.py [BTCUSDT ETHUSDT ...] [--universe] [--market spot|um] [--jobs 6] [--drop-raw]

Monthly zips for completed months, daily zips for the current month. Raw zips are cached in data/raw/, the merged
series in data/m1/{market}_{symbol}.parquet. Re-running only downloads what is missing.
Archive timestamps switched from ms to us in 2025 (spot) - both are handled.
"""
import io,os,sys,shutil,zipfile,argparse,urllib.error,datetime as dt
import numpy as np,pandas as pd
import trend

ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base','taker_quote','ignore']
BASE={'spot':'https://data.binance.vision/data/spot','um':'https://data.binance.vision/data/futures/um'}

def _zip(url,path):
    """Cached download; returns DataFrame or None if the archive does not exist (404)."""
    if not os.path.exists(path):
        try:b=trend._get(url,tries=3)
        except urllib.error.HTTPError as e:
            if e.code==404:return None
            raise
        os.makedirs(os.path.dirname(path),exist_ok=True);open(path,'wb').write(b)
    z=zipfile.ZipFile(path);df=pd.read_csv(z.open(z.namelist()[0]),header=None)
    if not str(df.iat[0,0]).isdigit():df=df.iloc[1:]                       # futures files carry a header row
    df.columns=COLS[:df.shape[1]];return df

def _ts(x):
    x=x.astype('int64');return pd.to_datetime(np.where(x>1e14,x//1000,x),unit='ms')   # us -> ms

def download(symbol,market='spot',start='2017-08'):
    out=f'{ROOT}/m1/{market}_{symbol}.parquet';old=pd.read_parquet(out) if os.path.exists(out) else None
    today=dt.datetime.now(dt.timezone.utc).date();m=pd.Period(start,'M');cur=pd.Period(today,'M')
    if old is not None:m=max(m,pd.Period(old.index[-1],'M'))                 # resume from the last stored month
    parts=[];raw=f'{ROOT}/raw/{market}/{symbol}'
    while m<cur:
        f=f'{symbol}-1m-{m}.zip';d=_zip(f'{BASE[market]}/monthly/klines/{symbol}/1m/{f}',f'{raw}/{f}')
        if d is not None:parts.append(d);print(symbol,m,len(d),flush=True)
        m+=1
    day=max(cur.start_time.date(),old.index[-1].date() if old is not None else cur.start_time.date())
    while day<today:                                                         # current month: daily files (lag ~1 day)
        f=f'{symbol}-1m-{day}.zip';d=_zip(f'{BASE[market]}/daily/klines/{symbol}/1m/{f}',f'{raw}/daily/{f}')
        if d is not None:parts.append(d)
        day+=dt.timedelta(days=1)
    if not parts:return old
    df=pd.concat(parts);df.index=_ts(df.open_time);df.index.name='time'
    df=df[['open','high','low','close','volume','quote_volume','trades']].astype(float)
    if old is not None:df=pd.concat([old,df])
    df=df[~df.index.duplicated(keep='last')].sort_index()
    os.makedirs(os.path.dirname(out),exist_ok=True);df.to_parquet(out);return df

def load(symbol,market='spot'):
    return pd.read_parquet(f'{ROOT}/m1/{market}_{symbol}.parquet')

def daily_closes(symbol,market='spot'):
    """Daily close (UTC day, close = last minute's close) from the 1m cache, completed days only."""
    c=load(symbol,market).close.resample('1D').last().dropna()
    return c[c.index<pd.Timestamp.now(tz='UTC').floor('D').tz_localize(None)].rename(symbol)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('symbols',nargs='*',default=list(trend.COINS));a.add_argument('--market',default='spot')
    a.add_argument('--universe',action='store_true',help='all symbols from data/universe.csv (python universe.py)')
    a.add_argument('--jobs',type=int,default=1);a.add_argument('--drop-raw',action='store_true',help='delete cached zips after merging')
    a=a.parse_args();syms=list(pd.read_csv(f'{ROOT}/universe.csv').symbol) if a.universe else a.symbols
    def one(s):
        d=download(s,a.market)
        if a.drop_raw:shutil.rmtree(f'{ROOT}/raw/{a.market}/{s}',ignore_errors=True)
        return f'{s}: {len(d):,} minutes {d.index[0]} .. {d.index[-1]}'
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(a.jobs) as ex:
        for r in ex.map(one,syms):print(r,flush=True)
