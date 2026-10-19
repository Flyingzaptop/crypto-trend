"""Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) for the positive findings of the vol research program."""
import glob,numpy as np,pandas as pd
from scipy.stats import norm,skew,kurtosis
import phase1 as p1,phase1b as p1b,trend

def dsr(r,sr_trials,n_trials):
    """r: per-period returns of the chosen strategy; sr_trials: per-period Sharpes of all trials in its family."""
    r=pd.Series(r).dropna();T=len(r);sr=r.mean()/r.std();g3=skew(r);g4=kurtosis(r,fisher=False)
    v=np.var(sr_trials,ddof=1) if len(sr_trials)>1 else (1-g3*sr+(g4-1)/4*sr*sr)/(T-1)
    e=0.5772156649;sr0=np.sqrt(v)*((1-e)*norm.ppf(1-1/n_trials)+e*norm.ppf(1-1/(n_trials*np.e)))
    z=(sr-sr0)*np.sqrt(T-1)/np.sqrt(1-g3*sr+(g4-1)/4*sr*sr);return {'T':T,'SR_period':round(sr,3),'SR0_period':round(sr0,3),'N':n_trials,'DSR_prob':round(norm.cdf(z),3)}

def count_trials():
    n=0;rows=[]
    for f,col in (('docs/vol/results/vol_backtest*.csv',None),('docs/vol/results/phase0_rerun.csv',None),('docs/vol/results/phase1_money.csv',None),
                  ('docs/vol/results/phase1b_cross_section.csv',None),('docs/vol/results/phase1b_sizing_by_market.csv',None),('docs/vol/results/phase2_*.csv',None),
                  ('docs/vol/results/phase3*_*.csv',None),('docs/vol/results/phase4_switch.csv',None),('docs/vol/results/phase6_rules.csv',None),
                  ('docs/vol/results/phase7_straddle.csv',None),('docs/vol/results/phase8_portfolio.csv',None)):
        for p in glob.glob(f):
            k=len(pd.read_csv(p));rows.append((p.split('/')[-1],k));n+=k
    for p in glob.glob('docs/results/*.csv'):k=len(pd.read_csv(p));rows.append((p.split('/')[-1],k));n+=k
    return n,rows

if __name__=='__main__':
    N,rows=count_trials();print('rows across result tables (upper bound on variants tried):',N)
    out=[]
    # 1 phase 7: hedged weekly short straddle, always sell, maker hedge, 2024-26
    R=pd.read_csv('docs/vol/results/phase7_trades.csv',parse_dates=['entry'])
    S=[];
    for (v,bp),g in R.groupby(['variant','hedge_bp']):
        w=g.groupby('entry').pnl_pct.mean();w=w[w.index>='2024-01-01'];S.append(w.mean()/w.std())
    rr=pd.read_csv('docs/vol/results/phase0_rerun.csv');S+=list(rr['2024-26_sharpe']/np.sqrt(52))
    w=R[(R.variant=='always_sell')&(R.hedge_bp==2)].groupby('entry').pnl_pct.mean();w=w[w.index>='2024-01-01']
    for n_ in (20,N):out.append({'finding':'Deribit: продажа недельного ATM-стрэддла с хеджем (мейкер), 2024-26','N_scope':'фаза 7' if n_==20 else 'вся программа',**dsr(w,S,n_)})
    # 2 short variance swap on VIX / VXTLT, 2024-26 monthly
    X=p1b.load();MS=pd.read_csv('docs/vol/results/phase1_money.csv');sr_m=list(MS[MS.period=='2024-26'].always/np.sqrt(12))
    for mkt in ('VIX','VXTLT','VXEWZ'):
        q=p1b.periods(X[X.mkt==mkt],0).loc['2024':'2026'].pay
        for n_ in (105,N):out.append({'finding':f'короткий variance swap {mkt}, 2024-26 (без издержек)','N_scope':'фаза 1' if n_==105 else 'вся программа',**dsr(q,sr_m,n_)})
    # 3 base crypto trend, 2018-2026 daily (variants tried in the original research ~ 20)
    D=pd.DataFrame({s:trend.daily_closes(s) for s in trend.COINS}).dropna();r,_=trend.backtest(D,pd.DataFrame({s:trend.daily_funding(s) for s in trend.COINS}))
    r=r['2018-01-01':]
    out.append({'finding':'тренд BTC+ETH (текущая стратегия), 2018-26','N_scope':'исходное исследование (~20 вариантов)',**dsr(r,[],20)})
    T=pd.DataFrame(out);T.to_csv('docs/vol/results/deflated_sharpe.csv',index=False);pd.set_option('display.width',250);print(T.to_string(index=False))
    pd.DataFrame(rows,columns=['table','rows']).to_csv('docs/vol/results/trial_counts.csv',index=False)
