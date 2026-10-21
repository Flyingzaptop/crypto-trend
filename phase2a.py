"""Phase 2A addendum: alpha of CBOE option-selling indices vs S&P 500 under 0 / 0.3% / 1% annual costs."""
import numpy as np,pandas as pd
import phase1 as p1,phase2 as p2,tradfi as tf

if __name__=='__main__':
    spx=tf.yahoo('^SP500TR').pct_change();rf=p2.rf_daily(spx.index);rows=[]
    for n in ('PUT','BXM','BXMD','CNDR','CLL'):
        lv=p1.cboe(n).dropna();g=lv.index.to_series().diff().dt.days;big=g[g>30]
        if len(big):lv=lv[lv.index>=big.index[-1]]
        r0=lv.pct_change().dropna()
        for c in (0.0,0.003,0.01):
            r=r0-c/252
            for seg,(a,b) in (('2008-23',('2008','2023')),('2024-26',('2024','2026')),('all',('1988','2026'))):
                q=r[a:b];al,t,beta=p2.alpha(q,spx,rf)
                rows.append({'index':n,'from':str(r.index[0].date()),'cost_pct_yr':c*100,'seg':seg,'alpha_pct_yr':al,'alpha_t':t,'beta':beta,'sharpe':p2.stats(q,rf)['sharpe']})
    T=pd.DataFrame(rows);T.to_csv('docs/vol/results/phase2a_cost_sensitivity.csv',index=False)
    print(T.pivot_table(index=['index','seg'],columns='cost_pct_yr',values=['alpha_pct_yr','alpha_t']).round(2).to_string())
