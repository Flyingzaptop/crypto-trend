"""NEWS1: GDELT news aggregates vs XAUUSD (docs/ml/PROTOCOL_NEWS1.md).
usage: python news1.py [--part corr|events|model|all] -> docs/ml/results/news1_*.csv, docs/ml/img/news1_*.png"""
import glob,json,argparse,numpy as np,pandas as pd
import data_m1

R='docs/ml/results';OUT=f'{data_m1.ROOT}/news';GOLD=f'{data_m1.ROOT}/ml/gold/xauusd_m1.parquet'
H={'15m':15,'1h':60,'4h':240,'1d':1440};LAG=pd.Timedelta(minutes=15);TRAIN_END='2022-12-31 23:59';VAL=('2023-01-01','2023-12-31 23:59');TEST0='2024-01-01'
EVF=['n','conf_n','conf_m','conf_g','conf_gmin','tone','neg']+[f'conf_{r}_{k}' for r in ('me','ru','cn','us') for k in ('n','m','g')]

def features():
    """One row per Events packet; availability a = T + 15 min. GKG (hourly slices) joined as of its own availability (as-of, backward)."""
    E=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{OUT}/events_*.parquet'))]).drop_duplicates('t').sort_values('t').set_index('t')
    E=E[EVF].add_prefix('ev_');E.index=E.index+LAG
    G=sorted(glob.glob(f'{OUT}/gkg_*.parquet'))
    if G:
        K=pd.concat([pd.read_parquet(p) for p in G]).drop_duplicates('t').sort_values('t').set_index('t');K=K.drop(columns=[c for c in K if c=='t'],errors='ignore').add_prefix('gk_');K.index=K.index+LAG
        E=pd.merge_asof(E.reset_index().rename(columns={'index':'a','t':'a'}),K.reset_index().rename(columns={'t':'a'}),on='a',direction='backward',tolerance=pd.Timedelta('2h')).set_index('a')
    E.index.name='a'
    for c in [c for c in E if c.endswith(('_n','_m'))]:E[c]=np.log1p(E[c])
    return E

def zscore(F,win='30D'):
    """z = (x - mean of the PAST 30 days) / std of the past 30 days (current packet excluded)."""
    m=F.rolling(win,min_periods=500).mean().shift(1);s=F.rolling(win,min_periods=500).std().shift(1)
    return ((F-m)/s).clip(-10,10).add_prefix('z_')

def gold():
    g=pd.read_parquet(GOLD);c=pd.Series(g.close.values,index=pd.to_datetime(g.timestamp).dt.tz_localize(None)).sort_index();return c[~c.index.duplicated()]

def targets(a,c):
    """Entry = first bar in [a, a+5m]; exit = last bar <= a+h, which must be >= a+h-30m (no closure inside the horizon)."""
    t=c.index.values;v=np.log(c.values);ai=a.values;i=np.searchsorted(t,ai);ok=(i<len(t))&(t[np.minimum(i,len(t)-1)]<=ai+np.timedelta64(5,'m'))
    out=pd.DataFrame(index=a);out['entry_i']=np.where(ok,i,-1)
    for k,m in H.items():
        e=ai+np.timedelta64(m,'m');j=np.searchsorted(t,e,side='right')-1;good=ok&(j>=0)&(t[np.maximum(j,0)]>=e-np.timedelta64(30,'m'))&(j>i)
        out[f'r_{k}']=np.where(good,v[np.maximum(j,0)]-v[np.minimum(i,len(t)-1)],np.nan);out[f'x_{k}']=np.where(good,j,-1)
    for k in H:out[f'abs_{k}']=out[f'r_{k}'].abs()
    return out

def corr_by_year(Z,T,tag='real'):
    rows=[];D=Z.join(T)
    for f in [c for c in Z]:
        for y_ in [c for c in T if c.startswith(('r_','abs_'))]:
            g=D[[f,y_]].dropna();rho=g.groupby(g.index.year).apply(lambda q:q[f].corr(q[y_],method='spearman') if len(q)>200 else np.nan)
            rows.append({'shift':tag,'feature':f,'target':y_,**{str(k):round(v,4) for k,v in rho.items()},'mean':round(rho.mean(),4),
                         'same_sign_years':int(max((rho>0).sum(),(rho<0).sum())),'years':int(rho.notna().sum())})
    return pd.DataFrame(rows)

def event_study(Z,c,feats,rng):
    t=c.index.values;v=np.log(c.values);offs=np.arange(-4,193)*15                       # -1h .. +2d, minutes
    def path(ts):
        P=np.full((len(ts),len(offs)),np.nan);a=ts.values;i0=np.searchsorted(t,a);base_ok=(i0<len(t))&(t[np.minimum(i0,len(t)-1)]<=a+np.timedelta64(5,'m'))
        b=v[np.minimum(i0,len(t)-1)]
        for k,o in enumerate(offs):
            e=a+np.timedelta64(int(o),'m');j=np.searchsorted(t,e,side='right')-1;good=base_ok&(j>=0)&(t[np.maximum(j,0)]>=e-np.timedelta64(30,'m'))
            P[:,k]=np.where(good,v[np.maximum(j,0)]-b,np.nan)
        return P
    how=lambda ix:(ix.dayofweek*24+ix.hour);rows=[];curves={}
    for f,sign in feats:
        s=sign*Z[f].dropna();thr=s[:TRAIN_END].quantile(0.99)
        for per,lo,hi in (('2015-22','2015-01-01',TRAIN_END),('2023-26','2023-01-01','2099')):
            x=s[lo:hi];hit=x[x>=thr].index;ev=[]
            for h in hit:
                if not ev or h-ev[-1]>pd.Timedelta('1D'):ev.append(h)                    # decluster: one event per day
            ev=pd.DatetimeIndex(ev);pool=x.index;hw=how(pool);ctrl=[]
            for hv,nv in pd.Series(how(ev)).value_counts().items():
                cand=pool[hw==hv];ctrl+=list(rng.choice(cand,min(len(cand),int(np.ceil(2000*nv/len(ev)))),replace=False))
            Pe=path(ev);Pc=path(pd.DatetimeIndex(ctrl));me,mc=np.nanmean(Pe,0),np.nanmean(Pc,0)
            B=np.array([np.nanmean(Pe[rng.integers(0,len(Pe),len(Pe))],0)-mc for _ in range(1000)])
            curves[(f,per)]=(offs,me-mc,np.nanpercentile(B,2.5,0),np.nanpercentile(B,97.5,0))
            for lab,o in (('+1h',60),('+4h',240),('+1d',1440),('+2d',2880)):
                k=int(np.where(offs==o)[0][0]);rows.append({'feature':f,'sign':sign,'period':per,'events':len(ev),'h':lab,'excess_bp':round((me[k]-mc[k])*1e4,2),
                                                         'ci_lo_bp':round(np.nanpercentile(B[:,k],2.5)*1e4,2),'ci_hi_bp':round(np.nanpercentile(B[:,k],97.5)*1e4,2),
                                                         'abs_ratio':round(np.nanmean(np.abs(Pe[:,k]))/np.nanmean(np.abs(Pc[:,k])),3)})
    return pd.DataFrame(rows),curves

def book(c,T,sig,h,cost=1e-4):
    """Minute position book: +-1 from entry bar to exit bar of each signal, summed and clipped to [-1, 1]; daily P&L."""
    n=len(c);d=np.zeros(n+1);e=T.loc[sig.index,'entry_i'].values;x=T.loc[sig.index,f'x_{h}'].values;s=np.sign(sig.values)
    ok=(e>=0)&(x>e);np.add.at(d,e[ok],s[ok]);np.add.at(d,x[ok],-s[ok]);pos=np.clip(np.cumsum(d)[:n],-1,1)   # pos[i] held from bar i to i+1
    r=np.diff(np.log(c.values));pnl=pos[:-1]*r-cost*np.abs(np.diff(np.r_[0,pos]))[:-1]
    return pd.Series(pnl,index=c.index[1:]).resample('1D').sum().loc[lambda q:q!=0]

def stats(d,bh):
    sh=lambda q:float(q.mean()/q.std()*np.sqrt(252)) if len(q)>20 and q.std()>0 else np.nan
    j=pd.concat([d,bh],axis=1).fillna(0);j=j[j.iloc[:,1]!=0];beta=np.cov(j.iloc[:,0],j.iloc[:,1])[0,1]/j.iloc[:,1].var() if len(j)>20 else 0
    al=j.iloc[:,0]-beta*j.iloc[:,1];eq=d.cumsum()
    return {'sharpe':round(sh(d),2),'alpha_sharpe':round(sh(al),2),'beta':round(beta,3),'ann_ret_pct':round(d.sum()/max(len(bh.loc[d.index[0]:d.index[-1]]),1)*252*100,2) if len(d) else 0,
            'max_dd_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1) if len(d) else 0,'days':len(d)}

def models(Z,T,c,tag='real',only=None):
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    X=Z.copy();how=X.index.dayofweek*24+X.index.hour;X['how_s'],X['how_c']=np.sin(2*np.pi*how/168),np.cos(2*np.pi*how/168);X=X.fillna(0)
    bh=pd.Series(np.diff(np.log(c.values)),index=c.index[1:]).resample('1D').sum().loc[lambda q:q!=0];rows=[]
    for h in ('1h','4h','1d'):
        y=T[f'r_{h}'];ok=y.notna()&(T.entry_i>=0);tr=ok&(X.index<=TRAIN_END);yy=(y>0).astype(int)
        for mname in ('lgbm','logreg'):
            if only and (mname,h)!=only:continue
            if mname=='lgbm':m=lgb.LGBMClassifier(max_depth=3,num_leaves=15,n_estimators=200,learning_rate=0.05,min_child_samples=500,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,random_state=0)
            else:
                from sklearn.pipeline import make_pipeline;from sklearn.preprocessing import StandardScaler
                m=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=500))
            m.fit(X[tr],yy[tr]);p=pd.Series(m.predict_proba(X[ok])[:,1],index=X.index[ok]);thr=np.percentile(np.abs(p[X.index[tr]]-0.5),95)
            sig=(p-0.5)[np.abs(p-0.5)>=thr]
            from sklearn.metrics import roc_auc_score
            for per,lo,hi in (('train','2015-01-01',TRAIN_END),('val',*VAL),('test',TEST0,'2099')):
                s=sig[lo:hi];d=book(c,T,s,h) if len(s) else pd.Series(dtype=float);yv=yy[ok][lo:hi];pv=p[lo:hi]
                rows.append({'shift':tag,'model':mname,'h':h,'period':per,'signals':len(s),'auc':round(roc_auc_score(yv,pv),4) if yv.nunique()>1 else np.nan,
                             'hit':round(float((np.sign(s)==np.sign(T.loc[s.index,f'r_{h}'])).mean()),3) if len(s) else np.nan,**stats(d,bh[lo:hi])})
    return pd.DataFrame(rows)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--part',default='all');a=a.parse_args()
    F=features();Z=zscore(F);c=gold();Z=Z[(Z.index>=c.index[0])&(Z.index<=c.index[-1])];T=targets(Z.index,c);rng=np.random.default_rng(0)
    print('packets',len(Z),Z.index[0],Z.index[-1],'| with entry',int((T.entry_i>=0).sum()),'| gkg cols',sum(c_.startswith('z_gk_') for c_ in Z),flush=True)
    SH={'+7d':'7D','-7d':'-7D','+30d':'30D','-30d':'-30D'}
    def shifted(s):Zs=Z.copy();Zs.index=Zs.index+pd.Timedelta(s);return Zs.reindex(Z.index,method='nearest',tolerance=pd.Timedelta('20min'))
    if a.part in ('corr','all'):
        C=corr_by_year(Z,T);C.to_csv(f'{R}/news1_corr.csv',index=False)
        P=pd.concat([corr_by_year(shifted(s),T,k) for k,s in SH.items()]);P.to_csv(f'{R}/news1_corr_placebo.csv',index=False)
        for k,g in pd.concat([C,P]).groupby('shift'):print(k,'median |mean rho|',round(g['mean'].abs().median(),4),'max |mean rho|',round(g['mean'].abs().max(),4),'pairs >=8/11 same sign',int((g.same_sign_years>=8).sum()),'of',len(g),flush=True)
    if a.part in ('events','all'):
        feats=[('z_ev_conf_n',1),('z_ev_conf_me_n',1),('z_ev_conf_ru_n',1),('z_ev_tone',-1)]+[(f,s) for f,s in (('z_gk_conflict_share',1),('z_gk_tone',-1)) if f in Z]
        E,curves=event_study(Z,c,feats,rng);E.to_csv(f'{R}/news1_events.csv',index=False);print(E.to_string(index=False),flush=True)
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(len(feats),2,figsize=(10,2.6*len(feats)),squeeze=False)
        for i,(f,_) in enumerate(feats):
            for j,per in enumerate(('2015-22','2023-26')):
                o,m,lo,hi=curves[(f,per)];x=o/60;ax[i,j].plot(x,m*1e4);ax[i,j].fill_between(x,lo*1e4,hi*1e4,alpha=0.3);ax[i,j].axhline(0,color='k',lw=0.5);ax[i,j].axvline(0,color='k',lw=0.5)
                ax[i,j].set_title(f'{f} top 1%: {per} (gold, excess vs same-hour control)',fontsize=8);ax[i,j].set_ylabel('bp');ax[i,j].set_xlabel('hours')
        fig.tight_layout();fig.savefig('docs/ml/img/news1_event_study.png',dpi=100)
    if a.part in ('model','all'):
        M=models(Z,T,c);V=M[M.period=='val'].sort_values('sharpe',ascending=False);best=(V.iloc[0].model,V.iloc[0].h)
        print('selected',best,flush=True);Pm=pd.concat([models(shifted(s),T,c,k,only=best) for k,s in SH.items()])
        A=pd.concat([M,Pm]);A['selected']=(A.model==best[0])&(A.h==best[1]);A.to_csv(f'{R}/news1_models.csv',index=False)
        pd.set_option('display.width',250);print(A.to_string(index=False))
