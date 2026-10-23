"""ML3 sensitivity (diagnostic, not for selection): universe size x names per leg, weekly buy-and-hold accounting, 10 bp.
usage: python ml3_sens.py -> docs/ml/results/ml3_sensitivity.csv"""
import pandas as pd
import panel,ml3,ml3_check as mc

if __name__=='__main__':
    C,Q,F=panel.load();out=[]
    for n in (30,50,100):
        rows,U,R=ml3.weekly(C,Q,F,n);P,_=mc.predict(rows);P=P.reindex(U.index)
        for k in (3,5,10):
            if 2*k>n//2:continue
            r=mc.weekly_bh(ml3.book(P,U,k=k),C,F,U);out.append({'universe':n,'per_leg':k,**{f'sharpe_{y}':mc.st(r,f'{y}-01-01',f'{y}-12-31') for y in range(2021,2027)},
                                                              'sharpe_2021_23':mc.st(r,'2021-01-04','2023-12-31'),'sharpe_2024_26':mc.st(r,'2024-01-01','2099')});print(out[-1],flush=True)
    pd.DataFrame(out).to_csv('docs/ml/results/ml3_sensitivity.csv',index=False)
