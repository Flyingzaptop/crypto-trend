"""Candidate coin list for the pool expansion (docs/PROTOCOL_universe.md).
usage: python universe.py [--n 100]  -> data/universe.csv (rank, symbol, avg 30d quote volume, first day)"""
import os,json,argparse,urllib.error,datetime as dt
import pandas as pd
import trend

STABLE={'USDC','FDUSD','USD1','RLUSD','EUR','TUSD','USDP','BUSD','DAI','XUSD','RWUSD','PYUSD','USDE','EURI','AEUR','BFUSD','UST','USDS'}
GOLD={'PAXG','XAUT'}
API='https://data-api.binance.vision/api/v3'

def has_perp(sym):
    m=pd.Period(dt.date.today(),'M')-1
    try:trend._get(f'https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip',tries=2);return True
    except urllib.error.HTTPError:return False

def first_day(sym):
    k=json.loads(trend._get(f'{API}/klines?symbol={sym}&interval=1d&startTime=0&limit=1'));return pd.to_datetime(k[0][0],unit='ms')

def build(n=100):
    info=json.loads(trend._get(f'{API}/exchangeInfo?permissions=SPOT'))['symbols']
    syms=[s['symbol'] for s in info if s['quoteAsset']=='USDT' and s['status']=='TRADING' and s['baseAsset'].isascii()
          and s['baseAsset'] not in STABLE|GOLD]
    tick={t['symbol']:float(t['quoteVolume']) for t in json.loads(trend._get(f'{API}/ticker/24hr'))}
    syms=sorted(syms,key=lambda s:-tick.get(s,0))[:300]                   # pre-filter, then rank by 30-day average
    rows=[]
    for s in syms:
        k=json.loads(trend._get(f'{API}/klines?symbol={s}&interval=1d&limit=31'))[:-1]
        rows.append({'symbol':s,'vol30_musd':sum(float(x[7]) for x in k)/max(len(k),1)/1e6})
    df=pd.DataFrame(rows).sort_values('vol30_musd',ascending=False);out=[]
    for r in df.itertuples():
        if len(out)==n:break
        fd=first_day(r.symbol)
        if (pd.Timestamp.now()-fd).days<365:continue                        # >= 1 year of history
        if not has_perp(r.symbol):print('no perp:',r.symbol);continue      # tokenized stocks etc. have no USD-M perp
        out.append({'rank':len(out)+1,'symbol':r.symbol,'vol30_musd':round(r.vol30_musd,1),'first_day':fd.date()});print(out[-1],flush=True)
    return pd.DataFrame(out)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--n',type=int,default=100);a=a.parse_args()
    u=build(a.n);os.makedirs('data',exist_ok=True);u.to_csv('data/universe.csv',index=False);print(len(u),'symbols -> data/universe.csv')
