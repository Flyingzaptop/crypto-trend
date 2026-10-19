#!/usr/bin/env python3
"""Daily snapshot of the options-implied earnings move for S&P 500 + Nasdaq-100 companies (standard library only).

usage:
  python earnings_collector.py                 # snapshot: events in the next 1..10 trading days -> data/earnings/snapshots_YYYYMMDD.csv
  python earnings_collector.py --outcomes      # realized moves for past events -> data/earnings/outcomes.csv
  options: --out DIR (default data/earnings), --days N (default 10), --max N (limit tickers, for testing)

Run once a day after the US close. Sources: Nasdaq earnings calendar API (dates, before/after market),
Yahoo Finance option chains and daily prices (cookie + crumb, no key). Errors on one ticker are logged and skipped."""
import argparse,csv,datetime as dt,http.cookiejar,json,os,re,sys,time,urllib.request,urllib.error

UA={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36','Accept':'application/json,text/html,*/*'}
JAR=http.cookiejar.CookieJar();OPENER=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(JAR));CRUMB=None
SNAP_COLS=['snapshot_utc','symbol','event_date','event_time','reaction_date','expiry','expiry_role','days_to_expiry','spot','strike_offset',
           'strike','call_bid','call_ask','call_last','call_iv','put_bid','put_ask','put_last','put_iv','straddle_mid','straddle_bid','straddle_ask',
           'implied_move_pct','eps_forecast','market_cap']
OUT_COLS=['symbol','event_date','event_time','reaction_date','close_before','open_reaction','close_reaction','move_open_pct','move_close_pct','recorded_utc']

def log(*a):print(dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S'),*a,file=sys.stderr,flush=True)

def get(url,tries=5,raw=False):
    for k in range(tries):
        try:
            with OPENER.open(urllib.request.Request(url,headers=UA),timeout=30) as r:
                b=r.read();time.sleep(0.4);return b if raw else json.loads(b)
        except urllib.error.HTTPError as e:
            if e.code in (429,500,502,503,504) and k<tries-1:time.sleep(5*2**k);continue
            raise
        except (urllib.error.URLError,TimeoutError,ConnectionError):
            if k<tries-1:time.sleep(3*2**k);continue
            raise

def crumb():
    global CRUMB
    if CRUMB is None:
        try:OPENER.open(urllib.request.Request('https://fc.yahoo.com',headers=UA),timeout=30)
        except Exception:pass                                                   # sets the cookie even on 404
        CRUMB=get('https://query1.finance.yahoo.com/v1/test/getcrumb',raw=True).decode().strip()
    return CRUMB

def trading_days(start,n):
    d,out=start,[]
    while len(out)<n:
        d+=dt.timedelta(days=1)
        if d.weekday()<5:out.append(d)
    return out

def next_trading_day(d):return trading_days(d,1)[0]

# ---------- universe ----------
def wiki_symbols(page):
    html=get(f'https://en.wikipedia.org/wiki/{page}',raw=True).decode('utf-8','ignore')
    m=re.search(r'<table[^>]*id="constituents".*?</table>',html,re.S)
    if not m:return set()
    syms=set()
    for row in re.findall(r'<tr[^>]*>(.*?)</tr>',m.group(0),re.S):
        cells=re.findall(r'<td[^>]*>(.*?)</td>',row,re.S)
        for c in cells[:2]:
            t=re.sub(r'<[^>]+>','',c).strip()
            if re.fullmatch(r'[A-Z]{1,5}([.-][A-Z])?',t):syms.add(t.replace('.','-'));break
    return syms

def universe(out):
    path=os.path.join(out,'universe.csv');syms=set()
    try:syms=wiki_symbols('List_of_S%26P_500_companies')|wiki_symbols('Nasdaq-100')
    except Exception as e:log('universe fetch failed:',e)
    if len(syms)>400:
        with open(path,'w',newline='') as f:f.write('symbol\n'+'\n'.join(sorted(syms))+'\n')
        return syms
    if os.path.exists(path):
        with open(path) as f:return {l.strip() for l in f if l.strip() and l.strip()!='symbol'}
    log('no universe list; falling back to market cap >= $10B');return None

# ---------- calendar ----------
def calendar(days):
    today=dt.date.today();ev=[]
    for d in [today]+trading_days(today,days):
        try:rows=get(f'https://api.nasdaq.com/api/calendar/earnings?date={d}')['data']['rows'] or []
        except Exception as e:log('calendar',d,e);continue
        for r in rows:
            t={'time-pre-market':'BMO','time-after-hours':'AMC'}.get(r.get('time'),'UNK')
            cap=re.sub(r'[^0-9]','',r.get('marketCap') or '') or '0'
            if d==today and t=='BMO':continue                                    # already reacted before this after-close snapshot
            ev.append({'symbol':r['symbol'].replace('.','-').replace('/','-'),'event_date':d,'event_time':t,'eps_forecast':r.get('epsForecast',''),'market_cap':int(cap)})
    return ev

def reaction_date(e):
    return e['event_date'] if e['event_time']=='BMO' else next_trading_day(e['event_date'])  # UNK treated as after close

# ---------- options ----------
def chain(sym,expiry=None):
    u=f'https://query1.finance.yahoo.com/v7/finance/options/{sym}?crumb={crumb()}'+(f'&date={expiry}' if expiry else '')
    return get(u)['optionChain']['result'][0]

def mid(o):
    b,a=o.get('bid') or 0,o.get('ask') or 0
    return (b+a)/2 if b>0 and a>0 else None

def snapshot_rows(e,now):
    sym=e['symbol'];base=chain(sym);spot=base['quote'].get('regularMarketPrice');exps=sorted(base.get('expirationDates',[]))
    if not spot or not exps:return []
    rd=reaction_date(e);rd_ts=int(dt.datetime(rd.year,rd.month,rd.day,tzinfo=dt.timezone.utc).timestamp())
    post=[x for x in exps if x>=rd_ts];pre=[x for x in exps if x<rd_ts and x>now.timestamp()]
    picks=([('post',post[0])] if post else [])+([('pre',pre[-1])] if pre else [])
    rows=[]
    for role,x in picks:
        c=chain(sym,x);calls={o['strike']:o for o in c['options'][0].get('calls',[])};puts={o['strike']:o for o in c['options'][0].get('puts',[])}
        ks=sorted(set(calls)&set(puts))
        if not ks:continue
        i=min(range(len(ks)),key=lambda j:abs(ks[j]-spot));exp_d=dt.datetime.fromtimestamp(x,dt.timezone.utc).date()
        for off in (-1,0,1):
            if not 0<=i+off<len(ks):continue
            k=ks[i+off];C,P=calls[k],puts[k];cm,pm=mid(C),mid(P)
            sm=cm+pm if cm is not None and pm is not None else None
            rows.append({'snapshot_utc':now.strftime('%Y-%m-%d %H:%M'),'symbol':sym,'event_date':e['event_date'],'event_time':e['event_time'],'reaction_date':rd,
                         'expiry':exp_d,'expiry_role':role,'days_to_expiry':(exp_d-now.date()).days,'spot':spot,'strike_offset':off,'strike':k,
                         'call_bid':C.get('bid'),'call_ask':C.get('ask'),'call_last':C.get('lastPrice'),'call_iv':C.get('impliedVolatility'),
                         'put_bid':P.get('bid'),'put_ask':P.get('ask'),'put_last':P.get('lastPrice'),'put_iv':P.get('impliedVolatility'),
                         'straddle_mid':sm,'straddle_bid':(C.get('bid') or 0)+(P.get('bid') or 0),'straddle_ask':(C.get('ask') or 0)+(P.get('ask') or 0),
                         'implied_move_pct':round(sm/spot*100,3) if sm and off==0 else None,'eps_forecast':e['eps_forecast'],'market_cap':e['market_cap']})
    return rows

def append(path,cols,rows):
    new=not os.path.exists(path)
    with open(path,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols)
        if new:w.writeheader()
        w.writerows(rows)

def run_snapshot(a):
    os.makedirs(a.out,exist_ok=True);now=dt.datetime.now(dt.timezone.utc);uni=universe(a.out);ev=calendar(a.days)
    ev=[e for e in ev if (e['symbol'] in uni if uni else e['market_cap']>=10_000_000_000)]
    seen=set();ev=[e for e in ev if not (e['symbol'] in seen or seen.add(e['symbol']))]
    if a.max:ev=ev[:a.max]
    log(f'{len(ev)} events in universe');path=os.path.join(a.out,f'snapshots_{now:%Y%m%d}.csv');n=0
    for e in ev:
        try:
            rows=snapshot_rows(e,now);append(path,SNAP_COLS,rows);n+=len(rows)
        except Exception as ex:log('skip',e['symbol'],repr(ex)[:120])
    log(f'wrote {n} rows -> {path}')

# ---------- outcomes ----------
def daily(sym,d0,d1):
    p1=int(dt.datetime(d0.year,d0.month,d0.day,tzinfo=dt.timezone.utc).timestamp());p2=int(dt.datetime(d1.year,d1.month,d1.day,tzinfo=dt.timezone.utc).timestamp())+86400
    r=get(f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d')['chart']['result'][0];q=r['indicators']['quote'][0]
    return [(dt.datetime.fromtimestamp(t,dt.timezone.utc).date(),o,c) for t,o,c in zip(r['timestamp'],q['open'],q['close']) if o and c]

def run_outcomes(a):
    path=os.path.join(a.out,'outcomes.csv');done=set()
    if os.path.exists(path):
        with open(path) as f:done={(r['symbol'],r['event_date']) for r in csv.DictReader(f)}
    ev={}
    for fn in sorted(os.listdir(a.out)):
        if fn.startswith('snapshots_') and fn.endswith('.csv'):
            with open(os.path.join(a.out,fn)) as f:
                for r in csv.DictReader(f):ev[(r['symbol'],r['event_date'])]=r
    today=dt.date.today();rows=[]
    for (sym,ed),r in ev.items():
        rd=dt.date.fromisoformat(r['reaction_date'])
        if (sym,ed) in done or rd>=today:continue
        try:
            bars=daily(sym,rd-dt.timedelta(days=10),rd);before=[b for b in bars if b[0]<rd];on=[b for b in bars if b[0]==rd]
            if not before or not on:continue
            cb,(_,o,c)=before[-1][2],on[0]
            rows.append({'symbol':sym,'event_date':ed,'event_time':r['event_time'],'reaction_date':rd,'close_before':cb,'open_reaction':o,'close_reaction':c,
                         'move_open_pct':round((o/cb-1)*100,3),'move_close_pct':round((c/cb-1)*100,3),'recorded_utc':dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M')})
        except Exception as ex:log('skip',sym,repr(ex)[:120])
    append(path,OUT_COLS,rows);log(f'wrote {len(rows)} outcomes -> {path}')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--outcomes',action='store_true');ap.add_argument('--out',default='data/earnings')
    ap.add_argument('--days',type=int,default=10);ap.add_argument('--max',type=int,default=0);a=ap.parse_args()
    run_outcomes(a) if a.outcomes else run_snapshot(a)
