"""Phase 1b: is the volatility forecast useful for selling the premium? (docs/vol/PROTOCOL_phase1b.md)"""
import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
import phase1 as p1

OUT='docs/vol/results';SEG={'2010-23':('2010-01-01','2023-12-31'),'2024-26':('2024-01-01','2026-12-31')}
US=[m for m in p1.MARKETS if m not in ('INDIAVIX','DVOL_BTC','DVOL_ETH')]
CRISES=[('2008-09-01','2009-03-31'),('2020-02-01','2020-04-30'),('2022-01-01','2022-12-31'),('2024-07-01','2024-08-31'),('2025-03-01','2025-04-30')]
RNG=np.random.default_rng(0)

def load():
    path=f'{p1.CACHE}/phase1_panel.parquet'
    if os.path.exists(path):return pd.read_parquet(path)
    X=p1.walk_forward(pd.concat([p1.frame(n,p1.cboe('VIX')) for n in p1.MARKETS]));X.to_parquet(path);return X

def periods(x,o):
    H=int(x.H.iloc[0]);q=x.iloc[o::H].dropna(subset=['rv_fwd']).copy();q['pay']=(q.iv**2-q.rv_fwd**2)/(2*q.iv);return q

def sizing(pp):
    s=pp.expanding(12).std().shift(1);return (pp/s).clip(0,2)

def tails(r,ref_worst):
    eq=r.cumsum();return {'worst':r.min(),'mdd':(eq.cummax()-eq).max(),'worst5':r.reindex(ref_worst).mean()}

def sharpe(r,ppy):return r.mean()/r.std()*np.sqrt(ppy) if len(r)>3 and r.std()>0 else np.nan

def boot(r,ppy,n=2000,b=3):
    r=r.dropna().values
    if len(r)<6:return (np.nan,np.nan)
    k=int(np.ceil(len(r)/b));s=[]
    for _ in range(n):
        idx=np.concatenate([np.arange(i,i+b) for i in RNG.integers(0,len(r)-b+1,k)])[:len(r)];x=r[idx];s.append(x.mean()/x.std()*np.sqrt(ppy) if x.std()>0 else np.nan)
    return tuple(np.round(np.nanpercentile(s,[5,95]),2))

if __name__=='__main__':
    X=load();ic_rows,q_rows,size_rows,pool={},[],[],{f:[] for f in ('har','gbm')}
    ic_rows=[]
    for n,x in X.groupby('mkt',sort=False):
        H=int(x.H.iloc[0]);ppy=x.A.iloc[0]/H
        for f in ('har','gbm'):
            for seg,(a,b) in SEG.items():
                ics,res=[],[]
                for o in range(H):
                    q=periods(x,o).dropna(subset=[f]);q['pp']=np.log(q.iv)-np.log(q[f]);q['w']=sizing(q.pp);s=q.loc[a:b].dropna(subset=['w'])
                    if len(s)<8:continue
                    ics.append(spearmanr(s.pp,s.pay).statistic);pool[f].append(s.assign(seg=seg,mkt=n)[['pp','pay','seg','mkt']])
                    c=s.pay;w=s.pay*s.w;w=w*(c.std()/w.std()) if w.std()>0 else w;worst=c.nsmallest(5).index
                    tc,tw=tails(c,worst),tails(w,worst)
                    res.append({'sh_const':sharpe(c,ppy),'sh_sized':sharpe(w,ppy),**{f'{k}_const':v for k,v in tc.items()},**{f'{k}_sized':v for k,v in tw.items()}})
                if not ics:continue
                R=pd.DataFrame(res).mean()
                ic_rows.append({'mkt':n,'forecast':f,'seg':seg,'ic':round(np.nanmean(ics),3),'offsets':len(ics)})
                size_rows.append({'mkt':n,'forecast':f,'seg':seg,**{k:round(v,4) for k,v in R.items()}})
    IC=pd.DataFrame(ic_rows);SZ=pd.DataFrame(size_rows)
    for f in ('har','gbm'):
        P=pd.concat(pool[f])
        for seg in SEG:
            z=P[P.seg==seg];ic=spearmanr(z.pp,z.pay).statistic
            z=z.assign(qn=z.groupby('mkt').pp.transform(lambda v:pd.qcut(v.rank(method='first'),5,labels=False)))
            qm=(z.groupby('qn').pay.mean()*100).round(2).tolist()
            q_rows.append({'forecast':f,'seg':seg,'pooled_ic':round(ic,3),'q1..q5_mean_pay_vp':qm,'monotonic':bool(np.all(np.diff(qm)>0))})
    Q=pd.DataFrame(q_rows)
    # (b) pooled over markets: mean of per-market metrics
    agg=SZ.groupby(['forecast','seg'])[[c for c in SZ if c not in ('mkt','forecast','seg')]].mean().round(4).reset_index()

    # (c) cross-section on US-calendar markets selected with pre-2024 data
    sel=[]
    for n in US:
        x=X[X.mkt==n];q=periods(x,0);pre=q[:'2023-12-31'];yr=pre.pay.groupby(pre.index.year).mean()
        if len(yr)<3:continue
        if sharpe(pre.pay,12)>=0.3 and (yr>0).mean()>=0.7 and np.polyfit(yr.index,yr.values*100,1)[0]>=-0.3:sel.append(n)
    cs_rows=[];corr_rows=[]
    for f in ('har','gbm'):
        per={}
        for o in range(21):
            frames={}
            for n in sel:
                q=periods(X[X.mkt==n],o).dropna(subset=[f]);q['pp']=np.log(q.iv)-np.log(q[f]);q['z']=q.pp/q.pp.expanding(12).std().shift(1)
                q['r']=q.pay/q.pay.expanding(12).std().shift(1);frames[n]=q[['r','z','pay']]
            R=pd.concat({n:v.r for n,v in frames.items()},axis=1);Z=pd.concat({n:v.z for n,v in frames.items()},axis=1);PAY=pd.concat({n:v.pay for n,v in frames.items()},axis=1)
            ok=R.notna()&Z.notna();R,Z=R.where(ok),Z.where(ok)
            eq=R.mean(1)
            wz=Z.clip(lower=0);wz=wz.div(wz.sum(1),axis=0);pz=(R*wz).sum(1,min_count=1)
            top=Z.rank(1,ascending=False)<=ok.sum(1).div(2).values[:,None];pt=R.where(top).mean(1)
            per[o]={'equal':eq,'by_forecast':pz,'top_half':pt}
            if f=='har':
                cr=PAY.where(ok);inc=pd.Series(False,index=cr.index)
                for a,b in CRISES:inc|=(cr.index>=pd.Timestamp(a)-pd.Timedelta(days=30))&(cr.index<=b)
                def avgc(D):c=D.corr().values;return np.nanmean(c[np.triu_indices_from(c,1)])
                corr_rows.append({'offset':o,'all':avgc(cr),'crises':avgc(cr[inc]),'normal':avgc(cr[~inc])})
        for v in ('equal','by_forecast','top_half'):
            for seg,(a,b) in SEG.items():
                m=[];bs=[]
                for o in range(21):
                    r=per[o][v].loc[a:b].dropna();base=per[o]['equal'].loc[a:b].dropna()
                    rr=r*(base.std()/r.std()) if r.std()>0 else r;worst=base.nsmallest(5).index
                    m.append({'sharpe':sharpe(r,12),**tails(rr,worst)})
                    if o==0 and seg=='2024-26':bs=boot(r,12)
                M=pd.DataFrame(m).mean()
                cs_rows.append({'forecast':f,'weights':v,'seg':seg,'markets':len(sel),**{k:round(val,4) for k,val in M.items()},'boot90_sharpe':bs if seg=='2024-26' else ''})
    CS=pd.DataFrame(cs_rows);CR=pd.DataFrame(corr_rows).mean().round(3)
    # (d) bootstrap for constant selling per market (offset 0, 2024-26)
    bt=[]
    for n,x in X.groupby('mkt',sort=False):
        H=int(x.H.iloc[0]);q=periods(x,0).loc['2024':'2026'];bt.append({'mkt':n,'sharpe_2024_26':round(sharpe(q.pay,x.A.iloc[0]/H),2),'boot90':boot(q.pay,x.A.iloc[0]/H)})
    BT=pd.DataFrame(bt)
    for name,T in (('ic',IC),('quintiles',Q),('sizing_by_market',SZ),('sizing_pooled',agg),('cross_section',CS),('bootstrap',BT)):T.to_csv(f'{OUT}/phase1b_{name}.csv',index=False)
    CR.to_frame('avg_pairwise_corr').to_csv(f'{OUT}/phase1b_crisis_corr.csv')
    pd.set_option('display.width',250)
    print('selected markets:',sel);print(Q.to_string(index=False));print(agg.to_string(index=False));print(CS.to_string(index=False));print(CR);print(BT.to_string(index=False))
