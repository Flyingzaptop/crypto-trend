"""Deflated Sharpe (Bailey & Lopez de Prado) for the ML program's only protocol pass (ML3 L/S), 2024-26.
Trial Sharpes = every variant the program reported on 2024-26 (all phases), converted to per-day. usage: python ml_deflated.py"""
import numpy as np,pandas as pd
from deflated import dsr

R='docs/ml/results'
def trials():
    s=[]
    d=pd.read_csv(f'{R}/ml1_variants.csv');s+=list(d[(d.tag=='real')&d.series.str.startswith('h')].ver_sharpe/np.sqrt(365))
    import os
    if os.path.exists(f'{R}/rl1_configs.csv'):d=pd.read_csv(f'{R}/rl1_configs.csv');d=d[~d.config.str.contains('BASE|MIRROR')];s+=list(d.test_sharpe_mean/np.sqrt(365))
    d=pd.read_csv(f'{R}/x1.csv');d=d[d.strategy.str.startswith('(')];s+=list(d.ver_sharpe/np.sqrt(365))
    d=pd.read_csv(f'{R}/ml2.csv');d=d[d.series.str.match(r'^(lgbm|mlp) ')];s+=list(d.ver_sharpe/np.sqrt(365))
    d=pd.read_csv(f'{R}/ml3.csv');s+=list(d[d.series.str.startswith(('ML','mom4'))].ver_sharpe/np.sqrt(365))
    d=pd.read_csv(f'{R}/ml3_sensitivity.csv');s+=list(d.sharpe_2024_26/np.sqrt(365))
    d=pd.read_csv(f'{R}/ml3_checks.csv');d=d[d.case.str.startswith('factor')];s+=list(d.sharpe_2024_26/np.sqrt(365))
    return np.array(s)

if __name__=='__main__':
    S=trials();r=pd.read_csv(f'{R}/ml3_daily_returns.csv',index_col=0,parse_dates=True)['ML L/S top5/bottom5']['2024-01-01':]
    out=[]
    for n in (len(S),2*len(S),200):out.append({'series':'ML3 L/S, daily, 2024-26','N_trials':n,**dsr(r,S,n)})
    T=pd.DataFrame(out);T.to_csv(f'{R}/ml_deflated.csv',index=False);print('trials with a 2024-26 Sharpe:',len(S),'| per-day SR sd',round(S.std(),4));print(T.to_string(index=False))
