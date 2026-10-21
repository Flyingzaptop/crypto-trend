"""Drawdowns of the BTC+ETH trend (rule from README, unchanged) by year and regime; trend + carry; bootstrap at Sharpe 0.7.
usage: python drawdowns.py -> docs/DRAWDOWNS.md tables in docs/results/drawdowns_*.csv"""
import numpy as np,pandas as pd
import trend,carry as cy

OUT='docs/results';REG={'20%/1x':(0.20,1.0),'30%/1.5x':(0.30,1.5),'40%/2x':(0.40,2.0)};RNG=np.random.default_rng(0)

def year_rows(name,r):
    rows=[]
    for y,q in r.groupby(r.index.year):
        eq=np.exp(q.cumsum());peak=np.maximum.accumulate(np.r_[1.0,eq.values])[1:]
        dd=1-eq.values/peak;under=dd>0;run=best=0
        for u in under:run=run+1 if u else 0;best=max(best,run)
        m=q.groupby(q.index.month).sum()
        rows.append({'series':name,'year':y,'days':len(q),'return_pct':round((eq.iloc[-1]-1)*100,1),'max_dd_in_year_pct':round(dd.max()*100,1),
                     'worst_vs_jan1_pct':round((min(eq.min(),1.0)-1)*100,1),'worst_month_pct':round((np.exp(m.min())-1)*100,1),'longest_underwater_days':int(best)})
    return rows

def episodes(name,r,thr=0.10):
    eq=np.exp(r.cumsum());peak=eq.cummax();dd=1-eq/peak;out=[];i=0;idx=eq.index
    while i<len(eq):
        if dd.iloc[i]>0:
            j=i
            while j<len(eq) and dd.iloc[j]>0:j+=1
            seg=dd.iloc[i:j];trough=seg.idxmax()
            if seg.max()>=thr:
                pk=idx[i-1] if i>0 else idx[0];rec=idx[j] if j<len(eq) else None
                out.append({'series':name,'peak':pk.date(),'trough':trough.date(),'depth_pct':round(seg.max()*100,1),'recovered':rec.date() if rec is not None else 'not yet',
                            'peak_to_trough_days':(trough-pk).days,'total_days':((rec or idx[-1])-pk).days})
            i=j
        else:i+=1
    return out

def boot_mdd(r,years,n=2000,b=30):
    x=r.values;L=int(365*years);k=int(np.ceil(L/b));res=[]
    for _ in range(n):
        s=np.concatenate([x[i:i+b] for i in RNG.integers(0,len(x)-b,k)])[:L];eq=np.cumsum(s);res.append(1-np.exp(-(np.maximum.accumulate(np.r_[0,eq])[1:]-eq).max()))
    return np.median(res)*100,np.percentile(res,95)*100

if __name__=='__main__':
    D=pd.DataFrame({s:trend.daily_closes(s) for s in trend.COINS}).dropna();F=pd.DataFrame({s:trend.daily_funding(s) for s in trend.COINS})
    S={}
    for k,(v,c) in REG.items():r,_=trend.backtest(D,F,v,c);S[f'trend {k}']=r['2018-01-01':]
    S['buy&hold BTC+ETH 50/50']=np.log(D).diff().mean(1)['2018-01-01':]
    import strategies as st
    Fc=F.reindex(D.index).fillna(0);Pp=pd.DataFrame({s:st.perp_daily(s) for s in trend.COINS}).reindex(D.index)   # spot = REST daily closes
    car,_=cy.carry(D,Fc,Pp,pd.DataFrame(1.,index=D.index,columns=D.columns))
    base=S['trend 20%/1x'];cr=car.reindex(base.index).fillna(0)
    for w in (0.5,0.8):S[f'trend 20%/1x + {int(w*100)}% carry']=(base+w*cr)['2020-01-02':]
    Y=pd.DataFrame([row for n,r in S.items() for row in year_rows(n,r)]);Y.to_csv(f'{OUT}/drawdowns_by_year.csv',index=False)
    E=pd.DataFrame([row for n,r in S.items() for row in episodes(n,r)]);E.to_csv(f'{OUT}/drawdowns_episodes.csv',index=False)
    r=S['trend 20%/1x'];M=(np.exp(r.groupby([r.index.year,r.index.month]).sum())-1).mul(100).round(1).unstack();M.columns=[f'{m:02d}' for m in M.columns]
    M['year']=(np.exp(r.groupby(r.index.year).sum())-1).mul(100).round(1);M.to_csv(f'{OUT}/drawdowns_monthly_20.csv')
    B=[]
    for k in REG:
        x=S[f'trend {k}'];hon=x-x.mean()+0.7*x.std()/np.sqrt(365)
        for nm,s in (('backtest',x),('Sharpe 0.7',hon)):
            for yrs in (1,3,5):m,p=boot_mdd(s,yrs);B.append({'regime':k,'case':nm,'years':yrs,'mdd_median_pct':round(m,1),'mdd_p95_pct':round(p,1)})
    B=pd.DataFrame(B);B.to_csv(f'{OUT}/drawdowns_bootstrap.csv',index=False)
    pd.set_option('display.width',250);print('data to',D.index[-1].date(),'| carry data to',car.index[-1].date())
    print(Y.to_string(index=False));print(E.to_string(index=False));print(M.to_string());print(B.to_string(index=False))
