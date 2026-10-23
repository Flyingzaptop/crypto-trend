"""ML3 robustness checks requested by the coordinator (not for selection; the protocol variant stays the main one).
Main accounting here = weekly buy-and-hold (ml3_check.weekly_bh); daily constant-weight accounting (ml3.py) shown for comparison.
usage: python ml3_robust.py -> docs/ml/results/ml3_robust_*.csv"""
import json,numpy as np,pandas as pd,lightgbm as lgb
import trend,panel,ml_data as md,ml3,ml3_check as mc
from ml1 import PARAMS
from deflated import dsr

R_='docs/ml/results'

def predict(rows,years,train_from='2020-01-01',fixed_until=None):
    """Yearly walk-forward (purge 1 week); fixed_until: one model trained on targets ending before it, never retrained."""
    FEAT=[c for c in rows.columns if c!='y'];rows=rows[rows.index.get_level_values(0)>=train_from];t=rows.index.get_level_values(0);pred=pd.Series(np.nan,index=rows.index)
    if fixed_until:
        tr=(t+pd.Timedelta(days=7)<pd.Timestamp(fixed_until))&rows.y.notna();te=t>=pd.Timestamp(fixed_until)
        pred[te]=lgb.LGBMRegressor(**PARAMS).fit(rows.loc[tr,FEAT],rows.loc[tr,'y']).predict(rows.loc[te,FEAT]);return pred.unstack()
    for Y in years:
        tr=(t+pd.Timedelta(days=7)<pd.Timestamp(f'{Y}-01-01'))&rows.y.notna();te=t.year==Y
        if tr.sum()<200 or te.sum()==0:continue
        pred[te]=lgb.LGBMRegressor(**PARAMS).fit(rows.loc[tr,FEAT],rows.loc[tr,'y']).predict(rows.loc[te,FEAT])
    return pred.unstack()

def sh(r,lo,hi,a=52):x=r[lo:hi];return round(x.mean()/x.std()*np.sqrt(a),2) if len(x)>4 and x.std()>0 else np.nan
def yrs(r,a=52):return {str(y):sh(r,f'{y}-01-01',f'{y}-12-31',a) for y in sorted(set(r.index.year))}

if __name__=='__main__':
    C,Q,F=panel.load();out={}
    rows,U,R=ml3.weekly(C,Q,F);P=predict(rows,range(2021,2027)).reindex(U.index);W=ml3.book(P,U)
    base=mc.weekly_bh(W,C,F,U);daily=md.pnl(ml3.daily(W,C),R.fillna(0),F.fillna(0),simple=True)

    # 1a another epoch: spot-ranked top-50 (no perp requirement), train 2019, test 2020
    rs,Us,_=ml3.weekly(C,Q,F,perp=False);Ps=predict(rs,[2020,2021],train_from='2019-01-01').reindex(Us.index);rsp=mc.weekly_bh(ml3.book(Ps,Us),C,F,Us)
    # 1b one model trained on <=2022, never retrained, applied 2023-26
    Pf=predict(rows,None,fixed_until='2023-01-01').reindex(U.index);rf=mc.weekly_bh(ml3.book(Pf,U),C,F,U)
    out['1_epochs']=[{'case':'protocol (yearly retrain), weekly accounting',**yrs(base)},{'case':'protocol, daily constant-weight accounting',**yrs(daily,365)},
                     {'case':'spot-ranked top-50, trained 2019 only -> 2020','2020':sh(rsp,'2020-01-01','2020-12-31'),'coins_2020':round(Us.loc['2020'].sum(1).mean(),1)},
                     {'case':'single model trained <=2022, no retrain',**{k:v for k,v in yrs(rf).items() if k>='2023'},'2023-26':sh(rf,'2023-01-01','2099')}]

    # 2 sensitivity: k x N x rebalance day
    S=[]
    for dow,dn in ((6,'Sun'),(2,'Wed')):
        for n in (30,50,100):
            rw,Uw,_=ml3.weekly(C,Q,F,n,dow);Pw=predict(rw,range(2021,2027)).reindex(Uw.index)
            for k in (3,5,10):
                r=mc.weekly_bh(ml3.book(Pw,Uw,k=k),C,F,Uw);S.append({'day':dn,'universe':n,'per_leg':k,'2021-23':sh(r,'2021-01-04','2023-12-31'),'2024-26':sh(r,'2024-01-01','2099'),**yrs(r)})
                print(S[-1],flush=True)
    out['2_sensitivity']=S

    # 3 executability: perp delisted before spot; costs; short-squeeze weeks
    lastf=F.apply(lambda s:s.last_valid_index());dead=pd.DataFrame({c:U.index>lastf[c] if pd.notna(lastf[c]) else True for c in U.columns},index=U.index)
    Wx=W.where(~dead,0.);n_dead=int(((W!=0)&dead).sum().sum())
    E=[{'case':f'all positions, {cb} bp','cost_bp':cb,'2021-23':sh(mc.weekly_bh(W,C,F,U,cb),'2021-01-04','2023-12-31'),'2024-26':sh(mc.weekly_bh(W,C,F,U,cb),'2024-01-01','2099')} for cb in (10,20,40)]
    rx=mc.weekly_bh(Wx,C,F,U);E.append({'case':f'without positions in coins whose perp was already delisted ({n_dead} coin-weeks)','cost_bp':10,'2021-23':sh(rx,'2021-01-04','2023-12-31'),'2024-26':sh(rx,'2024-01-01','2099')})
    lw=np.log(C.ffill(limit=6)).loc[U.index];rw_=np.expm1((lw.shift(-1)-lw).fillna(0));fw=F.fillna(0).rolling(7).sum().shift(-7).reindex(U.index).fillna(0)
    contrib=W*rw_-W*fw;sq=(W<0)&(fw/7<-0.001)                                            # short held through a week of funding < -0.1%/day
    tot=contrib.sum().sum();E.append({'case':'short positions in squeeze weeks (avg funding < -0.1%/day)','coin_weeks':int(sq.sum().sum()),'pnl_sum':round(float(contrib[sq].sum().sum()),3),
                                      'share_of_total_pnl':round(float(contrib[sq].sum().sum()/tot),3),'total_pnl_sum':round(float(tot),3)})
    out['3_executability']=E

    # 4 vol target 20% on the trailing 12-week vol of the book (leverage cap 3)
    lev=(0.20/np.sqrt(52)/base.rolling(12,min_periods=8).std().shift(1)).clip(upper=3).fillna(1);rv=np.log1p(np.expm1(base)*lev)
    m=rv.groupby([rv.index.year,rv.index.month]).sum();Y=[]
    for y in sorted(set(rv.index.year)):
        q=rv.loc[str(y)];eq=q.cumsum();Y.append({'year':y,'return_pct':round((np.exp(q.sum())-1)*100,1),'max_dd_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1),
                                             'worst_month_pct':round((np.exp(m.loc[y].min())-1)*100,1),'avg_leverage':round(float(lev.loc[str(y)].mean()),2)})
    for lo,hi in (('2021-01-04','2023-12-31'),('2024-01-01','2099')):x=rv[lo:hi];Y.append({'year':f'{lo[:4]}-{hi[:4] if hi<"2099" else "26"}',**md.stats(x,52)})
    out['4_vol_target']=Y

    # 5 portfolio with the BTC+ETH trend (weekly, same Sunday-to-Sunday weeks), 50/50 risk with weights from 2021-23 vol
    Pp=pd.DataFrame({s:md.klines(s,'1d').close for s in trend.COINS}).dropna();tr,_=trend.backtest(Pp,pd.DataFrame({s:trend.daily_funding(s) for s in trend.COINS}))
    tw=tr.groupby(U.index[U.index.searchsorted(tr.index).clip(max=len(U.index)-1)]).sum().reindex(base.index)
    j=pd.concat([base.rename('ml3'),tw.rename('trend')],axis=1).dropna()['2021-01-04':];pre=j[:'2023-12-31'];w=1/pre.std();cb=(j*w).sum(1)*(pre.trend.std()/(pre*w).sum(1).std())
    out['5_with_trend']=[{'series':k,'corr_2021_23':round(pre.corr().iat[0,1],2),'corr_2024_26':round(j['2024':].corr().iat[0,1],2),
                          **{f'{a}_{lo[:4]}':b for lo,hi in (('2021-01-04','2023-12-31'),('2024-01-01','2099')) for a,b in md.stats(x[lo:hi],52).items() if a in ('sharpe','cagr_pct','max_dd_pct')}}
                         for k,x in (('trend',j.trend),('ML3 L/S',j.ml3),('trend + ML3 50/50 risk',cb))]

    # 6 deflated Sharpe of the weekly series 2024-26
    import ml_deflated as mdf
    allS=mdf.trials()*np.sqrt(365)/np.sqrt(52);ml3S=np.r_[[s['2024-26'] for s in S],[e['2024-26'] for e in E if '2024-26' in e]]/np.sqrt(52)
    allS=np.r_[allS,ml3S];x=base['2024-01-01':]
    out['6_deflated']=[{'scope':'whole ML program + checks','N':len(allS),**dsr(x,allS,len(allS))},{'scope':'ML3 only (protocol + checks)','N':len(ml3S)+4,**dsr(x,ml3S,len(ml3S)+4)}]
    for k,v in out.items():pd.DataFrame(v).to_csv(f'{R_}/ml3_robust_{k}.csv',index=False);print('\n##',k);print(pd.DataFrame(v).to_string(index=False))
