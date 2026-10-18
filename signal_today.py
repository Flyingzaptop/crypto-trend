"""Today's target positions. Run once a day after 00:00 UTC.
usage: python signal_today.py [--equity 10000] [--vol 0.20] [--cap 1] [--log forward.csv]
Prints the target position per coin as a fraction of equity and in USD (+ long / - short perpetual), and optionally
appends a row to a CSV journal for forward tracking."""
import argparse,os,pandas as pd
import trend

a=argparse.ArgumentParser();a.add_argument('--equity',type=float,default=None);a.add_argument('--vol',type=float,default=0.20)
a.add_argument('--cap',type=float,default=1.0);a.add_argument('--log',default=None);a=a.parse_args()
P=pd.DataFrame({s:trend.daily_closes(s,start=str((pd.Timestamp.now()-pd.Timedelta(days=500)).date())) for s in trend.COINS}).dropna()
N=trend.target_positions(P,a.vol,a.cap);last=N.iloc[-1];d=P.index[-1].date()
print(f'Signal from the close of {d} (UTC). Hold until the next daily close. Target vol {a.vol:.0%}, gross cap {a.cap:g}x.')
for s in trend.COINS:
    side='LONG' if last[s]>0 else 'SHORT' if last[s]<0 else 'FLAT'
    usd=f'  = {last[s]*a.equity:+,.0f} USD' if a.equity else ''
    print(f'  {s}: {last[s]:+.3f} x equity ({side}){usd}   close {P[s].iloc[-1]:,.2f}')
print(f'  gross exposure {last.abs().sum():.3f} x equity')
if a.log:
    row={'date':d,**{f'{s}_pos':round(float(last[s]),4) for s in trend.COINS},**{f'{s}_close':float(P[s].iloc[-1]) for s in trend.COINS}}
    pd.DataFrame([row]).to_csv(a.log,mode='a',header=not os.path.exists(a.log),index=False);print('appended to',a.log)
