"""Phase 6 data: closed Polymarket price markets (gamma API) and YES price history (CLOB). docs/vol/PROTOCOL_phase6.md"""
import os,json,time,urllib.request,random,numpy as np,pandas as pd
from concurrent.futures import ThreadPoolExecutor
import data_m1

ROOT=f'{data_m1.ROOT}/poly';UA={'User-Agent':'Mozilla/5.0'};TAGS=['bitcoin','ethereum','sp-500','gold','cpi','fed-rates']

def get(u,tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=60) as r:return json.loads(r.read())
        except Exception:
            if k==tries-1:raise
            time.sleep(2**k)

def events(tag):
    path=f'{ROOT}/events_{tag}.json'
    if os.path.exists(path):return json.load(open(path))
    def win(d):
        e=d+pd.Timedelta(days=3);res=[]
        for off in range(0,2000,100):
            x=get(f'https://gamma-api.polymarket.com/events?tag_slug={tag}&closed=true&limit=100&offset={off}'
                  f'&end_date_min={d:%Y-%m-%dT%H:%M:%SZ}&end_date_max={e:%Y-%m-%dT%H:%M:%SZ}')
            res+=x
            if len(x)<100:break
        return res
    out={}
    with ThreadPoolExecutor(8) as ex:
        for x in ex.map(win,pd.date_range('2024-01-01','2026-10-20',freq='3D')):
            for ev in x:out[ev['id']]=ev
    print(tag,len(out),'events',flush=True)
    os.makedirs(ROOT,exist_ok=True);json.dump(list(out.values()),open(path,'w'));return list(out.values())

def markets():
    path=f'{ROOT}/markets.parquet'
    if os.path.exists(path):return pd.read_parquet(path)
    rows=[]
    for tag in TAGS:
        for ev in events(tag):
            for m in ev.get('markets',[]):
                try:op=json.loads(m.get('outcomePrices') or '[]');tok=json.loads(m.get('clobTokenIds') or '[]');oc=json.loads(m.get('outcomes') or '[]')
                except Exception:continue
                if len(op)!=2 or len(tok)!=2 or not m.get('endDate'):continue
                if {float(op[0]),float(op[1])}!={0.0,1.0}:continue                         # unresolved / voided
                rows.append({'tag':tag,'event':ev.get('title'),'event_start':ev.get('startDate'),'question':m.get('question'),'market_id':m.get('id'),
                             'start':m.get('startDate') or ev.get('startDate'),'end':m.get('endDate'),'yes_token':tok[0],'outcome0':oc[0] if oc else '',
                             'yes_won':float(op[0])==1.0,'volume':float(m.get('volume') or 0),'description':(m.get('description') or '')[:600]})
    M=pd.DataFrame(rows).drop_duplicates('market_id');M.to_parquet(path);return M

def history(tok,end):
    path=f'{ROOT}/h/{tok[:40]}.json'
    if os.path.exists(path):return json.load(open(path))
    e=int(pd.Timestamp(end).timestamp());h=get(f'https://clob.polymarket.com/prices-history?market={tok}&startTs={e-8*86400}&endTs={e}&fidelity=60')['history']
    os.makedirs(os.path.dirname(path),exist_ok=True);json.dump(h,open(path,'w'));return h

if __name__=='__main__':
    M=markets();print(len(M),'resolved binary markets',M.tag.value_counts().to_dict(),flush=True)
    hourly=M.question.str.contains('Up or Down',case=False)&M.event.str.contains(r'[AP]M ET',regex=True)
    random.seed(0);keep=~hourly|pd.Series([random.random()<0.125 for _ in range(len(M))],index=M.index)
    S=M[keep];print(len(S),'markets to fetch (hourly up/down sampled 1/8)',flush=True)
    def one(r):
        try:history(r.yes_token,r.end);return 1
        except Exception:return 0
    with ThreadPoolExecutor(8) as ex:print('histories',sum(ex.map(one,S.itertuples())))
