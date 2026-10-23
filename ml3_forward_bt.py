"""Backtest of the frozen forward version of ML3 (signal_ml3.py): top-30, REG LightGBM, shorts ~ 1/vol4 capped at 0.15.
One fixed variant, not a search. Weekly buy-and-hold accounting, 10 bp. usage: python ml3_forward_bt.py -> docs/ml/results/ml3_forward_version.csv"""
import pandas as pd
import panel,ml3,ml3_check as mc,ml_data as md
import ml3_robust as rb,signal_ml3 as sg

if __name__=='__main__':
    C,Q,F=panel.load();out=[]
    for n in (30,50):
        rows,U,_=ml3.weekly(C,Q,F,n);rb.PARAMS=sg.REG                                         # same walk-forward as ml3_robust, REG params
        P=rb.predict(rows,range(2021,2027)).reindex(U.index);vol=rows.vol4.unstack().reindex(U.index)
        r=mc.weekly_bh(sg.book_fwd(P,U,vol),C,F,U)
        out.append({'version':f'forward top-{n}'+(' (main)' if n==30 else ' (comparison)'),'sharpe_2021_23':mc.st(r,'2021-01-04','2023-12-31'),'sharpe_2024_26':mc.st(r,'2024-01-01','2099'),
                    'max_dd_2021_23_pct':md.stats(r['2021-01-04':'2023-12-31'],52)['max_dd_pct'],'max_dd_2024_26_pct':md.stats(r['2024':],52)['max_dd_pct'],
                    'cagr_2024_26_pct':md.stats(r['2024':],52)['cagr_pct'],**{str(y):mc.st(r,f'{y}-01-01',f'{y}-12-31') for y in range(2021,2027)}})
        print(out[-1],flush=True)
    pd.DataFrame(out).to_csv('docs/ml/results/ml3_forward_version.csv',index=False)
