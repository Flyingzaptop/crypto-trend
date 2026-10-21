#!/usr/bin/env python3
"""Implied vs realized earnings moves from earnings_collector.py data (standard library only).

usage: python earnings_forward_analysis.py [--data data/earnings] [--out data/earnings/forward_summary.csv]

For every event with an outcome, uses the last snapshot taken before the reaction day:
  * straddle_move  = ATM straddle mid / spot of the 'post' expiry (includes normal vol until that expiry);
  * event_sigma    = event-only standard deviation of the log move:
      with a 'pre' expiry:  var_e = iv_post^2 * T_post - iv_pre^2 * T_pre
      without it:           var_e = iv_post^2 * T_post - sigma_norm^2 * (T_post - 1/252), sigma_norm from 60 days of prices
    (iv = mean of ATM call and put IV; T in years, trading-day based: days_to_expiry * 252/365 / 252);
  * event_move     = event_sigma * sqrt(2/pi)  (expected |move| under a normal distribution);
and compares them with the realized |close-to-close move| on the reaction day.
Prints the aggregate: mean/median ratio realized/implied, share of events where realized > implied, and the P&L in % of
spot of selling the event-only straddle (event_move - |realized|) — before costs and spreads (straddle bid/ask spread is reported)."""
import argparse,csv,glob,json,math,os,statistics as st,sys,time,urllib.request,datetime as dt

UA={'User-Agent':'Mozilla/5.0'}

def sigma_norm(sym,before):
    p2=int(dt.datetime(before.year,before.month,before.day,tzinfo=dt.timezone.utc).timestamp());p1=p2-120*86400
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d',headers=UA),timeout=30) as r:
                c=[x for x in json.loads(r.read())['chart']['result'][0]['indicators']['quote'][0]['close'] if x]
            time.sleep(0.3);lr=[math.log(b/a) for a,b in zip(c[:-1],c[1:])][-60:]
            return st.pstdev(lr)*math.sqrt(252) if len(lr)>20 else None
        except Exception:time.sleep(2**k)
    return None

def f(x):
    try:return float(x)
    except (TypeError,ValueError):return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/earnings');ap.add_argument('--out',default=None);a=ap.parse_args()
    out=a.out or os.path.join(a.data,'forward_summary.csv')
    outcomes={}
    with open(os.path.join(a.data,'outcomes.csv')) as fh:
        for r in csv.DictReader(fh):outcomes[(r['symbol'],r['event_date'])]=r
    snaps={}
    for fn in sorted(glob.glob(os.path.join(a.data,'snapshots_*.csv'))):
        with open(fn) as fh:
            for r in csv.DictReader(fh):
                key=(r['symbol'],r['event_date'])
                if key not in outcomes or r['strike_offset']!='0' or r['snapshot_utc'][:10]>=r['reaction_date']:continue
                snaps.setdefault(key,{}).setdefault(r['snapshot_utc'],{})[r['expiry_role']]=r
    rows=[]
    for key,by_time in snaps.items():
        s=by_time[max(by_time)];post=s.get('post');pre=s.get('pre');o=outcomes[key]
        if not post:continue
        spot=f(post['spot']);mid=f(post['straddle_mid']);ivp=[f(post['call_iv']),f(post['put_iv'])];ivp=[x for x in ivp if x]
        if not spot or not mid or not ivp:continue
        Tp=max(int(post['days_to_expiry']),1)/365;iv_post=sum(ivp)/len(ivp);var=iv_post**2*Tp;method='post-only'
        if pre:
            ivq=[x for x in (f(pre['call_iv']),f(pre['put_iv'])) if x];Tq=max(int(pre['days_to_expiry']),0)/365
            if ivq and Tq<Tp:var-=(sum(ivq)/len(ivq))**2*Tq;method='pre/post'
        if method=='post-only':
            sn=sigma_norm(key[0],dt.date.fromisoformat(o['reaction_date']))
            if sn is None:continue
            var-=sn**2*max(Tp-1/252,0)
        es=math.sqrt(var) if var>0 else 0.0;em=es*math.sqrt(2/math.pi);real=abs(math.log(1+float(o['move_close_pct'])/100))
        rows.append({'symbol':key[0],'event_date':key[1],'method':method,'straddle_move_pct':round(mid/spot*100,3),'event_move_pct':round(em*100,3),
                     'realized_abs_pct':round(real*100,3),'ratio_real_to_event':round(real/em,3) if em>0 else '',
                     'sell_event_pnl_pct':round((em-real)*100,3),'straddle_spread_pct_of_mid':round((f(post['straddle_ask'])-f(post['straddle_bid']))/mid*100,2) if f(post['straddle_ask']) and f(post['straddle_bid']) else ''})
    if not rows:print('no matched events yet');return
    with open(out,'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    r=[x['ratio_real_to_event'] for x in rows if x['ratio_real_to_event']!=''];pnl=[x['sell_event_pnl_pct'] for x in rows]
    sp=[x['straddle_spread_pct_of_mid'] for x in rows if x['straddle_spread_pct_of_mid']!='']
    print(f"events {len(rows)} (pre/post {sum(x['method']=='pre/post' for x in rows)})")
    print(f"realized/event-implied: mean {st.mean(r):.2f}, median {st.median(r):.2f}, share realized > implied {sum(x>1 for x in r)/len(r):.1%}")
    print(f"selling the event move, % of spot per event: mean {st.mean(pnl):.3f}, median {st.median(pnl):.3f}, "
          f"t {st.mean(pnl)/(st.pstdev(pnl)/math.sqrt(len(pnl))) if len(pnl)>1 and st.pstdev(pnl)>0 else float('nan'):.2f}")
    if sp:print(f"straddle bid/ask spread, % of mid: median {st.median(sp):.1f}")
    print('->',out)

if __name__=='__main__':main()
