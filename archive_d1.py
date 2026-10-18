"""Daily (1d) klines for ALL Binance spot USDT pairs ever listed, delisted included, from data.binance.vision.
usage: python archive_d1.py [--jobs 16]  -> data/d1/{symbol}.parquet (close, quote_volume), data/d1/_perps.txt
Completed months only (monthly archives)."""
import os,io,re,zipfile,argparse
import pandas as pd
import trend,data_m1

S3='https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
OUT=f'{data_m1.ROOT}/d1'

def ls(prefix,keys=False):
    """List an archive 'directory': sub-prefixes, or object keys with keys=True."""
    out,marker=[],''
    while True:
        x=trend._get(f'{S3}?delimiter=/&prefix={prefix}&marker={marker}').decode()
        out+=re.findall(r'<Key>([^<]*)</Key>' if keys else r'<Prefix>([^<]*/)</Prefix>',x)
        if '<IsTruncated>true' not in x:return [o for o in out if o!=prefix]
        marker=re.findall(r'<NextMarker>([^<]*)</NextMarker>',x)[0] if not keys else out[-1]

def one(sym):
    path=f'{OUT}/{sym}.parquet'
    if os.path.exists(path):return sym
    parts=[]
    for k in ls(f'data/spot/monthly/klines/{sym}/1d/',keys=True):
        if not k.endswith('.zip'):continue
        z=zipfile.ZipFile(io.BytesIO(trend._get(f'https://data.binance.vision/{k}')))
        d=pd.read_csv(z.open(z.namelist()[0]),header=None)
        if not str(d.iat[0,0]).isdigit():d=d.iloc[1:]
        parts.append(d)
    if not parts:return sym
    d=pd.concat(parts);d.index=data_m1._ts(d[0]);d.index.name='time'
    d=d[[4,7]].astype(float);d.columns=['close','quote_volume']
    d[~d.index.duplicated()].sort_index().to_parquet(path);return sym

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--jobs',type=int,default=16);a=a.parse_args();os.makedirs(OUT,exist_ok=True)
    syms=[p.split('/')[-2] for p in ls('data/spot/monthly/klines/')];syms=[s for s in syms if s.endswith('USDT') and s.isascii()]
    perps=[p.split('/')[-2] for p in ls('data/futures/um/monthly/klines/')]
    open(f'{OUT}/_perps.txt','w').write('\n'.join(perps));print(len(syms),'spot USDT pairs,',len(perps),'perps',flush=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(a.jobs) as ex:
        for i,s in enumerate(ex.map(one,syms)):
            if i%50==0:print(i,s,flush=True)
    print('done')
