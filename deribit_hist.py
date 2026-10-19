"""Deribit public history: hourly DVOL and option trades in sampled windows (docs/vol/PROTOCOL_phase0.md).
usage: python deribit_hist.py  -> data/vol/dvol_h_{CUR}.csv, data/vol/trades_{CUR}.parquet"""
import os,json,time,urllib.request,numpy as np,pandas as pd
from concurrent.futures import ThreadPoolExecutor
import vol_data as vd

H='https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time'

def get(u,tries=5):
    for k in range(tries):
        try:return json.loads(urllib.request.urlopen(u,timeout=60).read())['result']
        except Exception:
            if k==tries-1:raise
            time.sleep(2**k)

def dvol_hourly(cur):
    path=f'{vd.CACHE}/dvol_h_{cur}.csv'
    if os.path.exists(path):return pd.read_csv(path,index_col=0,parse_dates=True).iloc[:,0]
    rows,end=[],int(pd.Timestamp.now(tz='UTC').timestamp()*1000)
    while True:
        r=get(f'https://www.deribit.com/api/v2/public/get_volatility_index_data?currency={cur}&start_timestamp=1609459200000&end_timestamp={end}&resolution=3600')
        rows+=r['data']
        if not r.get('continuation') or not r['data']:break
        end=r['continuation']
    s=pd.Series({pd.to_datetime(x[0],unit='ms'):x[4] for x in rows},name=cur).sort_index();s=s[~s.index.duplicated()]
    s.to_csv(path);return s

def trades(cur,a,b):
    out,t=[],a
    while True:
        r=get(f'{H}?currency={cur}&kind=option&start_timestamp={t}&end_timestamp={b}&count=1000&sorting=asc&include_old=true')
        out+=r['trades']
        if not r.get('has_more') or not r['trades']:break
        t=r['trades'][-1]['timestamp']+1
    return out

def windows():
    rng=np.random.default_rng(0);w=[]
    for d in pd.date_range('2021-04-02','2026-10-16',freq='W-FRI'):w.append(('fri',d+pd.Timedelta('6h'),d+pd.Timedelta('10h')))
    for d in pd.date_range('2021-03-31','2026-10-14',freq='W-WED'):
        h=int(rng.integers(0,24));w.append(('wed',d+pd.Timedelta(hours=h),d+pd.Timedelta(hours=h+1)))
    return w

def sample(cur):
    path=f'{vd.CACHE}/trades_{cur}.parquet'
    if os.path.exists(path):return pd.read_parquet(path)
    ms=lambda t:int(t.timestamp()*1000)
    def one(w):
        tag,a,b=w;x=trades(cur,ms(a),ms(b))
        for z in x:z['window']=tag
        return x
    with ThreadPoolExecutor(8) as ex:rows=[z for x in ex.map(one,windows()) for z in x]
    df=pd.DataFrame(rows);df['time']=pd.to_datetime(df.timestamp,unit='ms')
    p=df.instrument_name.str.split('-',expand=True)
    df['expiry']=pd.to_datetime(p[1],format='%d%b%y')+pd.Timedelta('8h');df['strike']=p[2].astype(float);df['call']=p[3]=='C'
    df=df[['time','window','instrument_name','expiry','strike','call','price','mark_price','iv','index_price','direction','amount']]
    df.to_parquet(path);return df

if __name__=='__main__':
    for c in ('BTC','ETH'):
        s=dvol_hourly(c);print(c,'hourly DVOL',len(s),s.index[0],s.index[-1],flush=True)
        t=sample(c);print(c,'trades',len(t),t.time.min(),t.time.max(),t.window.value_counts().to_dict(),flush=True)
