"""Streaming aggregation of GDELT 2.0 15-minute packets into per-packet features (docs/ml/PROTOCOL_NEWS1.md).
Nothing raw is kept: download -> parse -> aggregate -> drop. Output per month: data/news/{events,gkg}_YYYYMM.parquet (resumable).
usage: python news_gdelt.py events [--jobs 12]      all packets since 2015-02-18
       python news_gdelt.py gkg [--every 4]         every n-th packet (hourly slices)"""
import os,io,sys,zipfile,argparse,urllib.request,numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor
import data_m1

OUT=f'{data_m1.ROOT}/news';MASTER='https://data.gdeltproject.org/gdeltv2/masterfilelist.txt'
REG={'me':{'IS','IR','IZ','SY','LE','JO','SA','YM','AE','QA','KU','BA','MU','EG','GZ','WE','TU'},'ru':{'RS','UP'},'cn':{'CH','TW'},'us':{'US'}}
THEMES={'gold':('ECON_GOLDPRICE','WB_2936_GOLD'),'infl':('ECON_INFLATION','WB_442_INFLATION'),
        'rates':('ECON_INTEREST_RATES','EPU_POLICY_FEDERAL_RESERVE','EPU_POLICY_CENTRAL_BANK','WB_1235_CENTRAL_BANKS'),
        'conflict':('ARMEDCONFLICT',),'sanct':('SANCTIONS',),'terror':('TERROR',),'oil':('ECON_OILPRICE',)}

def get(url,tries=4):
    import time
    for k in range(tries):
        try:return urllib.request.urlopen(url.replace('http://','https://'),timeout=60).read()
        except Exception as e:
            if getattr(e,'code',None)==404:return None
            if k==tries-1:return None
            time.sleep(2**k)

def unzip(b):z=zipfile.ZipFile(io.BytesIO(b));return z.read(z.namelist()[0])

def events(url):
    b=get(url);t=pd.Timestamp(url.rsplit('/',1)[1][:14]);r={'t':t}
    if b is None:return r
    d=pd.read_csv(io.BytesIO(unzip(b)),sep='\t',header=None,usecols=[29,30,31,34,53],dtype={29:'float',30:'float',31:'float',34:'float',53:'str'},on_bad_lines='skip')
    d.columns=['q','g','m','tone','cc'];d=d.dropna(subset=['q']);r['n']=len(d)
    if not len(d):return r
    c=d[d.q>=3];r.update(conf_n=len(c),conf_m=c.m.sum(),conf_g=c.g.mean(),conf_gmin=c.g.min(),
                         tone=np.average(d.tone,weights=d.m.clip(lower=1)),neg=(d.tone<0).mean())
    for k,s in REG.items():
        x=c[c.cc.isin(s)];r.update({f'conf_{k}_n':len(x),f'conf_{k}_m':x.m.sum(),f'conf_{k}_g':x.g.mean() if len(x) else np.nan})
    return r

def gkg(url):
    b=get(url);t=pd.Timestamp(url.rsplit('/',1)[1][:14]);r={'t':t}
    if b is None:return r
    th,tn=[],[]
    for l in unzip(b).decode('utf-8','ignore').split('\n'):
        p=l.split('\t')
        if len(p)<16:continue
        try:tone=float(p[15].split(',',1)[0])
        except ValueError:continue
        th.append(set(p[7].split(';')));tn.append(tone)
    tn=np.array(tn);r['n']=len(tn)
    if not len(tn):return r
    r.update(tone=tn.mean(),neg=(tn<-2).mean())
    for k,names in THEMES.items():
        m=np.array([bool(s.intersection(names)) for s in th]);r[f'{k}_share']=m.mean();r[f'{k}_tone']=tn[m].mean() if m.any() else np.nan
    return r

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('kind',choices=['events','gkg']);a.add_argument('--jobs',type=int,default=12);a.add_argument('--every',type=int,default=1,choices=[1,4])
    a=a.parse_args();os.makedirs(OUT,exist_ok=True)
    tag='export.CSV.zip' if a.kind=='events' else 'gkg.csv.zip'
    U=[l.split()[2] for l in get(MASTER).decode().splitlines() if len(l.split())>2 and l.split()[2].endswith(tag)]
    U=[u for u in U if u.rsplit('/',1)[1][:4]>='2015' and u.rsplit('/',1)[1][:6]<pd.Timestamp.now().strftime('%Y%m')]   # completed months only
    if a.every==4:U=[u for u in U if u.rsplit('/',1)[1][10:12]=='00']                 # hourly slices: the packet at minute 00
    by={};[by.setdefault(u.rsplit('/',1)[1][:6],[]).append(u) for u in U];fn=events if a.kind=='events' else gkg
    print(a.kind,len(U),'packets in',len(by),'months',flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        for ym,us in sorted(by.items()):
            path=f'{OUT}/{a.kind}_{ym}.parquet'
            if os.path.exists(path):continue
            D=pd.DataFrame(list(ex.map(fn,us,chunksize=8))).sort_values('t');D.to_parquet(path);print(ym,len(D),'packets',int(D.n.isna().sum()),'missing',flush=True)
