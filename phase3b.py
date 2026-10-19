"""Phase 3 clean: single-layer vol targeting with RV20 / EWMA / HAR / IV (docs/vol/PROTOCOL_phase3.md addendum)."""
import numpy as np,pandas as pd
import trend,tradfi as tf,phase1 as p1,phase3 as p3

OUT='docs/vol/results';TARGET=0.15;CAP=2.0

def metrics(r,w,ann):
    eq=r.cumsum();m=r.groupby(r.index.to_period('M'))
    mv=m.std()*np.sqrt(ann)/TARGET;mv=mv[m.count()>=10]
    return {'sharpe':round(r.mean()/r.std()*np.sqrt(ann),2),'cagr_pct':round((np.exp(r.mean()*ann)-1)*100,1),
            'max_dd_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1),'worst_month_pct':round((np.exp(m.sum().min())-1)*100,1),
            'rv_over_target_mean':round(mv.mean(),2),'rv_over_target_sd':round(mv.std(),3),'avg_lev':round(w.abs().mean(),2)}

def run(P,ann,h,cost,early,vix):
    R=np.log(P).diff();E,ok=p3.estimates(P,ann,h,20,vix);S=sum(np.sign(P-P.rolling(n).mean()) for n in trend.SMAS)/3;rows=[]
    for a in P:
        on=E['har'][a].notna() if False else None
        for strat in ('buyhold_scaled','trend'):
            for k in ('rv','ewma','har','iv'):
                v=E[k][a].reindex(P.index)
                w=(TARGET/v).clip(upper=CAP)*(S[a] if strat=='trend' else 1.0);w=w.shift(1)
                r=(w*R[a]-w.diff().abs()*cost).dropna();valid=ok.reindex(r.index).fillna(False)&E['har'][a].reindex(r.index).notna()
                for seg,(x,y) in (('early',early),('2024-26',('2024-01-01','2026-12-31'))):
                    q=r[valid][x:y];rows.append({'asset':a,'strategy':strat,'vol_est':k,'seg':seg,**metrics(q,w.reindex(q.index),ann)})
        r=R[a].dropna()
        for seg,(x,y) in (('early',early),('2024-26',('2024-01-01','2026-12-31'))):
            q=r[x:y];rows.append({'asset':a,'strategy':'buyhold_unscaled','vol_est':'-','seg':seg,**metrics(q,pd.Series(1.0,index=q.index),ann)})
    return pd.DataFrame(rows)

if __name__=='__main__':
    vix=p1.cboe('VIX')
    P=pd.DataFrame({t:tf.yahoo(t) for t in ('SPY','EFA','TLT','GLD','USO')}).sort_index();P=P[P.index<pd.Timestamp.now().normalize()]
    A=run(P,252,21,5e-4,('2011-01-01','2023-12-31'),vix)
    D=pd.DataFrame({s:trend.daily_closes(s) for s in trend.COINS}).dropna();D=D[D.index<pd.Timestamp.now().floor('D')]
    B=run(D,365,30,10e-4,('2021-01-01','2023-12-31'),vix)
    T=pd.concat([A,B]);T.to_csv(f'{OUT}/phase3b_clean.csv',index=False)
    # verdict per asset/strategy/estimator vs RV20
    V=[]
    for (a,s),g in T[T.vol_est!='-'].groupby(['asset','strategy']):
        g=g.set_index(['vol_est','seg'])
        for k in ('ewma','har','iv'):
            ok=True;d={}
            for seg in ('early','2024-26'):
                b,x=g.loc[('rv',seg)],g.loc[(k,seg)]
                disp=x.rv_over_target_sd<=0.9*b.rv_over_target_sd
                tail=(x.worst_month_pct>=0.9*b.worst_month_pct) or (x.max_dd_pct<=0.9*b.max_dd_pct)   # worst_month negative: less negative = better
                sh=x.sharpe>=b.sharpe-0.1;ok&=bool(disp and tail and sh)
                d[f'{seg}_disp']=round(x.rv_over_target_sd/b.rv_over_target_sd,2);d[f'{seg}_dd']=round(x.max_dd_pct/b.max_dd_pct,2) if b.max_dd_pct else np.nan
                d[f'{seg}_worstm']=round(x.worst_month_pct/b.worst_month_pct,2) if b.worst_month_pct else np.nan;d[f'{seg}_dSharpe']=round(x.sharpe-b.sharpe,2)
            V.append({'asset':a,'strategy':s,'vol_est':k,'better_than_rv20':ok,**d})
    V=pd.DataFrame(V);V.to_csv(f'{OUT}/phase3b_verdict.csv',index=False)
    pd.set_option('display.width',250);print(V.to_string(index=False))
    print(V.groupby(['strategy','vol_est']).better_than_rv20.sum())
