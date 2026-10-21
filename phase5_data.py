"""Phase 5 data: earnings events from SEC EDGAR (8-K item 2.02) and daily prices from Yahoo (docs/vol/PROTOCOL_phase5.md)."""
import os,sys,json,time,urllib.request,numpy as np,pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'scripts'))
import earnings_collector as ec,data_m1

ROOT=f'{data_m1.ROOT}/eq';SEC_UA={'User-Agent':'crypto-trend-research admin@example.org'}
SECTOR_ETF={'Information Technology':'XLK','Financials':'XLF','Health Care':'XLV','Consumer Discretionary':'XLY','Consumer Staples':'XLP',
            'Energy':'XLE','Industrials':'XLI','Materials':'XLB','Utilities':'XLU','Real Estate':'XLRE','Communication Services':'XLC'}

def sec(url):
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=SEC_UA),timeout=60) as r:time.sleep(0.12);return json.loads(r.read())
        except Exception:
            if k==4:raise
            time.sleep(2**k)

def universe():
    path=f'{ROOT}/universe.csv'
    if os.path.exists(path):return pd.read_csv(path)
    html=ec.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',raw=True).decode('utf-8','ignore')
    import re
    tab=re.search(r'<table[^>]*id="constituents".*?</table>',html,re.S).group(0);rows=[]
    for row in re.findall(r'<tr[^>]*>(.*?)</tr>',tab,re.S):
        cells=[re.sub(r'<[^>]+>','',c).strip() for c in re.findall(r'<td[^>]*>(.*?)</td>',row,re.S)]
        if len(cells)>3:rows.append({'symbol':cells[0].replace('.','-'),'sector':cells[2]})
    sp=pd.DataFrame(rows);ndx=ec.wiki_symbols('Nasdaq-100')
    u=pd.concat([sp,pd.DataFrame({'symbol':sorted(ndx-set(sp.symbol)),'sector':'none'})]).drop_duplicates('symbol')
    tick=sec('https://www.sec.gov/files/company_tickers.json');cik={v['ticker'].replace('.','-'):v['cik_str'] for v in tick.values()}
    u['cik']=u.symbol.map(cik);os.makedirs(ROOT,exist_ok=True);u.to_csv(path,index=False);return u

def filings(cik):
    path=f'{ROOT}/sec_{cik}.parquet'
    if os.path.exists(path):return pd.read_parquet(path)
    d=sec(f'https://data.sec.gov/submissions/CIK{int(cik):010d}.json');parts=[pd.DataFrame(d['filings']['recent'])]
    for f in d['filings'].get('files',[]):
        if f['filingTo']>='2009-06-01':parts.append(pd.DataFrame(sec(f"https://data.sec.gov/submissions/{f['name']}")))
    x=pd.concat(parts);x=x[(x.form=='8-K')&x['items'].fillna('').str.contains('2.02')][['filingDate','acceptanceDateTime','items']]
    x['sic']=d.get('sic');x.to_parquet(path);return x

def prices(t):
    path=f'{ROOT}/px_{t}.csv'
    if os.path.exists(path):return pd.read_csv(path,index_col=0,parse_dates=True)
    r=ec.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1=1230768000&period2=9999999999&interval=1d&events=div,split')['chart']['result'][0]
    q=r['indicators']['quote'][0];adj=r['indicators']['adjclose'][0]['adjclose']
    d=pd.DataFrame({'open':q['open'],'close':q['close'],'adj':adj,'volume':q['volume']},index=pd.to_datetime(r['timestamp'],unit='s').normalize())
    d=d.dropna();d=d[~d.index.duplicated(keep='last')];d['adj_open']=d.open*d.adj/d.close;d.to_csv(path);return d

if __name__=='__main__':
    u=universe();print(len(u),'companies',u.cik.notna().sum(),'with CIK',flush=True)
    ok=0
    for i,r in u[u.cik.notna()].iterrows():
        try:filings(r.cik);ok+=1
        except Exception as e:print('sec fail',r.symbol,e)
    print('sec ok',ok,flush=True)
    syms=list(u.symbol)+['SPY']+list(SECTOR_ETF.values())
    def one(t):
        try:prices(t);return 1
        except Exception as e:print('px fail',t,repr(e)[:80]);return 0
    with ThreadPoolExecutor(4) as ex:print('prices ok',sum(ex.map(one,syms)))
