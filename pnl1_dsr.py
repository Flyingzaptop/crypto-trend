"""Deflated Sharpe for the PNL1 bar-level baseline: the test series of each selected market x bar (3-seed ensemble), trials = validation
Sharpes of all points (per-day), N = number of trained runs. usage: python pnl1_dsr.py -> docs/ml/results/pnl1_dsr.csv"""
import glob,json,numpy as np,pandas as pd
from deflated import dsr

if __name__=='__main__':
    P=pd.read_csv('docs/ml/results/pnl1_points.csv');runs=len(glob.glob('docs/ml/results/pnl1_runs/*_s[0-9].json'));D=pd.read_csv('docs/ml/results/pnl1_ensemble_daily.csv',index_col=0,parse_dates=True)
    rows=[]
    for k in D:
        m,b=k.split('_');ann=365 if m!='XAUUSD' else 252;r=D[k].dropna();trials=(P[(P.market==m)&(P.bar==b)].val/np.sqrt(ann)).values
        for n_ in (len(trials),runs):rows.append({'series':k,'N_trials':n_,**dsr(r,trials,max(n_,2))})
    T=pd.DataFrame(rows);T.to_csv('docs/ml/results/pnl1_dsr.csv',index=False);print(T.to_string(index=False))
