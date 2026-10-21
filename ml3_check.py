"""Robustness checks of ML3 (not for selection): weekly buy-and-hold accounting, 20 bp costs, mirror, placebo, single factors, importance.
usage: python ml3_check.py -> docs/ml/results/ml3_checks.csv"""
import numpy as np,pandas as pd,lightgbm as lgb
import panel,ml_data as md,ml3
from ml1 import PARAMS

def predict(rows):
    FEAT=[c for c in rows.columns if c!='y'];rows=rows[rows.index.get_level_values(0)>='2020-01-01'];pred=pd.Series(np.nan,index=rows.index);imp=[]
    for Y in range(2021,rows.index.get_level_values(0).max().year+1):
        t=rows.index.get_level_values(0);tr=(t+pd.Timedelta(days=7)<pd.Timestamp(f'{Y}-01-01'))&rows.y.notna();te=t.year==Y
        m=lgb.LGBMRegressor(**PARAMS,importance_type='gain').fit(rows.loc[tr,FEAT],rows.loc[tr,'y']);pred[te]=m.predict(rows.loc[te,FEAT])
        imp.append(pd.Series(m.feature_importances_,index=FEAT)/m.feature_importances_.sum())
    return pred.unstack(),pd.concat(imp,axis=1).mean(1).sort_values(ascending=False)

def weekly_bh(Wk,C,F,U,cost_bp=10):
    """Weights set on Sunday close, held (drifting) for the week: book return = sum w*(e^r_week - 1) - funding - cost."""
    lw=np.log(C.ffill(limit=6)).loc[U.index];rw=np.expm1((lw.shift(-1)-lw).fillna(0));fw=F.fillna(0).rolling(7).sum().shift(-7).reindex(U.index).fillna(0)
    Wk=Wk.reindex(U.index).fillna(0);g=(Wk*rw).sum(1)-(Wk*fw).sum(1)-Wk.diff().abs().sum(1).fillna(Wk.abs().sum(1))*cost_bp*1e-4
    r=np.log1p(g.clip(lower=-0.99)).shift(1).dropna()             # stamp the week's return on the Sunday it ends
    return r

def st(r,lo,hi):x=r[lo:hi];return round(x.mean()/x.std()*np.sqrt(52),2)

if __name__=='__main__':
    C,Q,F=panel.load();out=[]
    for tag,(c,f) in {'real':(C,F),'mirror':(1/C,-F)}.items():
        rows,U,R=ml3.weekly(c,Q,f);P,imp=predict(rows);P=P.reindex(U.index)
        if tag=='real':imp.round(3).to_csv('docs/ml/results/ml3_importance.csv');print(imp.round(3).head(10).to_dict())
        cases={'ML L/S':ml3.book(P,U),'ML long top5':ml3.book(P,U,ls=False)}
        if tag=='real':
            rng=np.random.default_rng(0)
            for i in range(5):cases[f'placebo random L/S #{i}']=ml3.book(pd.DataFrame(rng.normal(size=P.shape),index=P.index,columns=P.columns),U)
            for k in ('mom1','mom4','mom12','vol4','fund1','fund4','lvol','beta12','size'):
                W=rows[k].unstack().reindex(U.index);cases[f'factor {k}: long high / short low']=ml3.book(W,U)
        for k,W in cases.items():
            for cb in ((10,20) if k.startswith('ML') else (10,)):
                r=weekly_bh(W,c,f,U,cb);out.append({'data':tag,'case':k,'cost_bp':cb,'sharpe_2021_23':st(r,'2021-01-04','2023-12-31'),'sharpe_2024_26':st(r,'2024-01-01','2099'),
                                                     'cagr_2024_26_pct':round((np.exp(r['2024':].mean()*52)-1)*100,1),'max_dd_2024_26_pct':md.stats(r['2024':],52)['max_dd_pct']})
                print(out[-1],flush=True)
    pd.DataFrame(out).to_csv('docs/ml/results/ml3_checks.csv',index=False)
