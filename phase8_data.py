"""Phase 8 data: daily klines and funding of every Binance USD-M USDT perpetual (delisted included)."""
import os,io,zipfile,pandas as pd
from concurrent.futures import ThreadPoolExecutor
import archive_d1 as ad,trend,data_m1

OUT=f'{data_m1.ROOT}/perp_d1'

def klines(sym):
    path=f'{OUT}/{sym}.parquet'
    if os.path.exists(path):return pd.read_parquet(path)
    parts=[];keys=sorted(k for k in ad.ls(f'data/futures/um/monthly/klines/{sym}/1d/',keys=True) if k.endswith('.zip'))
    if not keys or keys[0].split('-1d-')[-1][:7]<'2020-12':                     # listed before 2021: not a new listing
        pd.DataFrame({'close':[],'quote_volume':[]}).to_parquet(path);return None
    for k in keys:
        if k.endswith('.zip'):
            z=zipfile.ZipFile(io.BytesIO(trend._get(f'https://data.binance.vision/{k}')));d=pd.read_csv(z.open(z.namelist()[0]),header=None)
            parts.append(d.iloc[1:] if not str(d.iat[0,0]).isdigit() else d)
    if not parts:return None
    d=pd.concat(parts);d.index=data_m1._ts(d[0]);d=d[[4,7]].astype(float);d.columns=['close','quote_volume']
    d=d[~d.index.duplicated()].sort_index();d.to_parquet(path);return d

if __name__=='__main__':
    os.makedirs(OUT,exist_ok=True)
    syms=[s for s in open(f'{ad.OUT}/_perps.txt').read().split() if s.endswith('USDT') and s.isascii()]
    def one(s):
        try:
            k=klines(s)
            if k is not None and k.index[0]>=pd.Timestamp('2020-12-01'):trend.daily_funding(s)
            return 1
        except Exception as e:print('fail',s,repr(e)[:80]);return 0
    with ThreadPoolExecutor(12) as ex:print('ok',sum(ex.map(one,syms)),'of',len(syms))
