"""Point-in-time altcoin panel for ML2/ML3: spot daily close/turnover (archive_d1, delisted included) + perp funding.
A coin counts as having a perp from its first funding day. usage: python panel.py -> data/ml/panel_{close,qvol,fund}.parquet"""
import os,io,re,zipfile,numpy as np,pandas as pd
import trend,archive_d1,ml_data as md
from concurrent.futures import ThreadPoolExecutor
from universe import STABLE,GOLD

LEV=re.compile(r'(UP|DOWN|BULL|BEAR)USDT$')

def perp_name(sym,perps):
    if sym in perps:return sym
    for p in ('1000','1000000','1M'):
        if p+sym in perps:return p+sym
    return None

def funding(perp):
    path=f'{md.CACHE}/fund/{perp}.parquet'
    if os.path.exists(path):return pd.read_parquet(path).iloc[:,0]
    parts=[]
    for k in archive_d1.ls(f'data/futures/um/monthly/fundingRate/{perp}/',keys=True):
        if not k.endswith('.zip'):continue
        z=zipfile.ZipFile(io.BytesIO(trend._get(f'https://data.binance.vision/{k}')));parts.append(pd.read_csv(z.open(z.namelist()[0])))
    s=pd.Series(dtype=float,name=perp)
    if parts:
        f=pd.concat(parts);f.index=pd.to_datetime(f.calc_time,unit='ms');s=f.last_funding_rate.astype(float).resample('1D').sum().rename(perp)
    os.makedirs(f'{md.CACHE}/fund',exist_ok=True);s.to_frame().to_parquet(path);return s

def load():
    p=f'{md.CACHE}/panel_close.parquet'
    if os.path.exists(p):return tuple(pd.read_parquet(f'{md.CACHE}/panel_{k}.parquet') for k in ('close','qvol','fund'))
    perps=set(open(f'{archive_d1.OUT}/_perps.txt').read().split())
    syms=[f[:-8] for f in os.listdir(archive_d1.OUT) if f.endswith('.parquet')]
    syms=[s for s in syms if s[:-4] not in STABLE|GOLD and not LEV.search(s) and perp_name(s,perps)]
    D={s:pd.read_parquet(f'{archive_d1.OUT}/{s}.parquet') for s in syms}
    C=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index();Q=pd.DataFrame({s:d.quote_volume for s,d in D.items()}).reindex(C.index)
    C=C[C.index>='2019-01-01'];Q=Q.reindex(C.index)
    cand=[s for s in C if C[s].notna().any()]          # all of them: ranking only among 'ever big' coins would leak the future
    print(len(syms),'spot pairs with a perp,',len(cand),'with data since 2019',flush=True)
    with ThreadPoolExecutor(16) as ex:fs=dict(zip(cand,ex.map(lambda s:funding(perp_name(s,perps)),cand)))
    F=pd.DataFrame({s:f for s,f in fs.items()}).reindex(C.index)
    C,Q=C[cand],Q[cand]
    for k,x in (('close',C),('qvol',Q),('fund',F)):x.to_parquet(f'{md.CACHE}/panel_{k}.parquet')
    return C,Q,F

def universe(C,Q,F,n,min_days=90,perp=True):
    """Bool frame: coin is in the top-n by 30-day average turnover, has a live perp (funding seen) and min_days of history."""
    live=F.notna().cumsum()>0 if perp else C.notna();age=C.notna().cumsum()>=min_days;ok=live&age&C.notna()
    tv=Q.rolling(30,min_periods=20).mean().where(ok);return tv.rank(axis=1,ascending=False)<=n

if __name__=='__main__':
    C,Q,F=load();U=universe(C,Q,F,50);print(C.shape,'coins per day in top-50 universe:',U.sum(1).resample('YE').mean().round(1).to_dict())
