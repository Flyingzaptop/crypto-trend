"""Volatility selling study, steps 1-4 of docs/PROTOCOL_vol.md. usage: python vol.py -> docs/results/vol_*.csv"""
import json,numpy as np,pandas as pd
from scipy.stats import norm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import trend,vol_data as vd

OUT='docs/results';YEARS=range(2021,2027);EXITS=True   # EXITS=False: hold to expiry (diagnostic only)

# ---------- step 1: premium ----------
def premium(p):
    rows=[]
    for cur,x in p.groupby('coin'):
        x=x.dropna(subset=['dvol'])
        for h in (7,30):
            v=(x.dvol-x[f'rv_fwd{h}']).dropna()*100
            for y,q in list(v.groupby(v.index.year))+[('all',v)]:
                rows.append({'coin':cur,'h':h,'year':y,'mean_vp':round(q.mean(),1),'median_vp':round(q.median(),1),
                             'pos_pct':round((q>0).mean()*100),'p5_vp':round(q.quantile(.05),1),'n':len(q)})
    return pd.DataFrame(rows)

EPISODES={'2021-05 crash':('2021-05-01','2021-05-31'),'2022-05/06 LUNA':('2022-05-01','2022-06-30'),'2022-11 FTX':('2022-11-01','2022-11-30'),
          '2023-03 SVB':('2023-03-01','2023-03-31'),'2024-08 carry unwind':('2024-07-25','2024-08-15'),'2025-10 crash':('2025-10-01','2025-10-31')}
def episodes(p):
    rows=[]
    for cur,x in p.groupby('coin'):
        v=(x.dvol-x.rv_fwd7)*100
        for k,(a,b) in EPISODES.items():
            q=v[a:b].dropna()
            if len(q):rows.append({'coin':cur,'episode':k,'worst_vrp7_vp':round(q.min(),1),'date':str(q.idxmin().date()),'mean_vrp7_vp':round(q.mean(),1)})
    return pd.DataFrame(rows)

# ---------- step 2: HAR ----------
def har(p):
    """Walk-forward HAR forecasts of log RV_fwd(h), per coin: for year Y train on rows dated < Y-01-01 - h days."""
    out=pd.DataFrame(index=p.index)
    res=[]
    for h in (7,30):
        fc=pd.Series(np.nan,index=range(len(p)))
        for cur in p.coin.unique():
            m=(p.coin==cur).values;x=p[m]
            X=np.column_stack([np.ones(m.sum()),x.f_lrv1,x.f_lrv7,x.f_lrv30]);y=np.log(x[f'rv_fwd{h}']).values
            pred=np.full(m.sum(),np.nan)
            for Y in YEARS:
                tr=(x.index<pd.Timestamp(f'{Y}-01-01')-pd.Timedelta(days=h))&(x.index>='2018-01-01')&np.isfinite(y)&np.isfinite(X).all(1)
                te=(x.index.year==Y)&np.isfinite(X).all(1)
                b=np.linalg.lstsq(X[tr],y[tr],rcond=None)[0];pred[te]=X[te]@b
            fc[np.where(m)[0]]=pred
        p[f'har{h}']=np.exp(fc.values)
    return p

def accuracy(p,cols,lo='2022-01-01',hi='2026-12-31'):
    rows=[]
    for cur in p.coin.unique():
        for h in (7,30):
            x=p[(p.coin==cur)&(p.index>=lo)&(p.index<=hi)].dropna(subset=[f'rv_fwd{h}','dvol',f'har{h}'])
            t=x[f'rv_fwd{h}']
            for name,f in cols(h):
                if f not in x or x[f].isna().all():continue
                q=x.dropna(subset=[f]);tt,ff=q[f'rv_fwd{h}'],q[f]
                r=(tt/ff)**2
                rows.append({'coin':cur,'h':h,'forecast':name,'mse_logvol':round(((np.log(tt)-np.log(ff))**2).mean(),4),
                             'qlike':round((r-np.log(r)-1).mean(),4),'bias_logvol':round((np.log(ff)-np.log(tt)).mean(),3),'n':len(q)})
    return pd.DataFrame(rows)

# ---------- step 4: model ----------
PARAMS=dict(n_estimators=300,max_depth=3,learning_rate=0.03,min_child_samples=50,subsample=0.8,subsample_freq=1,verbose=-1)
def model(p):
    ok=p.dvol.notna()
    p['y7']=np.log(p.rv_fwd7)-np.log(p.dvol);p['y30']=np.log(p.rv_fwd30)-np.log(p.dvol)
    p['tail']=(p.ret_fwd7.abs()>2*p.dvol*np.sqrt(7/365)).astype(float).where(p.ret_fwd7.notna())
    for c in ('m7','m30','ptail'):p[c]=np.nan
    imp={}
    for Y in range(2023,2027):
        te=ok&(p.index.year==Y)
        for h,col,tgt in ((7,'m7','y7'),(30,'m30','y30'),(7,'ptail','tail')):
            tr=ok&(p.index<pd.Timestamp(f'{Y}-01-01')-pd.Timedelta(days=h))&p[tgt].notna()
            if tgt=='tail':
                mdl=lgb.LGBMClassifier(**PARAMS).fit(p.loc[tr,vd.FEATURES],p.loc[tr,tgt]);p.loc[te,col]=mdl.predict_proba(p.loc[te,vd.FEATURES])[:,1]
            else:
                mdl=lgb.LGBMRegressor(**PARAMS).fit(p.loc[tr,vd.FEATURES],p.loc[tr,tgt]);p.loc[te,col]=mdl.predict(p.loc[te,vd.FEATURES])
                if Y==2026:imp[tgt]=dict(sorted(zip(vd.FEATURES,mdl.booster_.feature_importance('gain').round()),key=lambda z:-z[1])[:6])
    p['mod7']=p.dvol*np.exp(p.m7);p['mod30']=p.dvol*np.exp(p.m30)
    q=p.dropna(subset=['ptail','tail']);auc=roc_auc_score(q['tail'],q['ptail'])
    return p,{'tail_auc_2023_26':round(auc,3),'tail_base_rate':round(q['tail'].mean(),3),'importance_2026':imp}

# ---------- step 3: synthetic backtest ----------
def bs(S,K,s,T,call):
    T=np.maximum(T,1e-9);d1=(np.log(S/K)+0.5*s*s*T)/(s*np.sqrt(T));d2=d1-s*np.sqrt(T)
    return np.where(call,S*norm.cdf(d1)-K*norm.cdf(d2),K*norm.cdf(-d2)-S*norm.cdf(-d1))

def legs(S0,s,T,kind,k,w):
    """[(strike, is_call, sign)], sign +1 = sold."""
    u=s*np.sqrt(T)
    if kind=='straddle':return [(S0,False,1),(S0,True,1)]
    return [(S0*np.exp(-k*u),False,1),(S0*np.exp(k*u),True,1),(S0*np.exp(-(k+w)*u),False,-1),(S0*np.exp((k+w)*u),True,-1)]

SIGMA=lambda S,K,s,T:s                     # pricing vol per leg (phase 0 replaces with a smile)
SPREAD=lambda S,K,s,T,price:0.05*price     # half-spread paid per leg per trade (phase 0 replaces with measured)
def px(S,K,s,T,call):return bs(S,K,SIGMA(S,K,s,T),T,call)
def cost(S,K,s,T,price):return np.minimum(0.0003*S,0.125*price)+SPREAD(S,K,s,T,price)

def fee(S,price,expiry=False):
    f=np.minimum((0.00015 if expiry else 0.0003)*S,0.125*price)
    return f if expiry else f+0.05*price

def run_coin(x,h,kind,k,w,allow,risk=0.05):
    """Daily P&L as a fraction of equity for one coin. x: rows of one coin with close, dvol; allow: bool Series (entry filter)."""
    x=x.dropna(subset=['dvol','close']);S,V=x.close.values,x.dvol.values;idx=x.index
    entry=(idx.dayofweek==6)
    if h==28:entry=entry&(((idx-pd.Timestamp('2021-01-03')).days//7)%4==0)
    al=allow.reindex(idx).fillna(False).values
    pnl=np.zeros(len(idx));pos=None;trades=[]
    for i in range(len(idx)):
        if pos:
            T=max(pos['exp']-i,0)/365;val=sum(sg*px(S[i],K,V[i],T,c) for K,c,sg in pos['legs'])
            pnl[i]+=(pos['val']-val)*pos['n'];pos['val']=val
            done=pos['exp']==i;early=EXITS and (not done) and (pos['credit']-val>=0.5*pos['credit'] or val-pos['credit']>=2*pos['credit'])
            if done:
                pnl[i]-=pos['n']*sum(fee(S[i],max(S[i]-K,0) if c else max(K-S[i],0),True) for K,c,sg in pos['legs'])
            elif early:
                pnl[i]-=pos['n']*sum(cost(S[i],K,V[i],T,px(S[i],K,V[i],T,c)) for K,c,sg in pos['legs'])
            if done or early:trades.append((idx[pos['i0']],idx[i],pos['credit_frac'],early));pos=None
        if pos is None and entry[i] and al[i] and i+h<len(idx):
            T=h/365;L=legs(S[i],V[i],T,kind,k,w);pr=[px(S[i],K,V[i],T,c) for K,c,sg in L]
            credit=sum(sg*q for (K,c,sg),q in zip(L,pr))
            if kind=='straddle':
                Sx=S[i]*np.exp(3*V[i]*np.sqrt(T));maxloss=(Sx-S[i])-credit
            else:
                maxloss=max(L[0][0]-L[2][0],L[3][0]-L[1][0])-credit
            if credit<=0 or maxloss<=0:continue
            n=risk/maxloss                                                     # per unit of equity
            pnl[i]-=n*sum(cost(S[i],K,V[i],T,q) for (K,c,sg),q in zip(L,pr))
            pos={'legs':L,'val':credit,'credit':credit,'n':n,'exp':i+h,'i0':i,'credit_frac':credit/S[i]}
    return pd.Series(pnl,index=idx),trades

def run(p,h,kind,k,w,filt):
    parts=[]
    for cur,x in p.groupby('coin'):
        hh=30 if h==28 else 7
        allow={'always':lambda:pd.Series(True,index=x.index),'har':lambda:x.dvol>x[f'har{hh}'],
               'model':lambda:x[f'm{hh}']<0,'model+tail':lambda:(x[f'm{hh}']<0)&(x.ptail<0.25)}[filt]()
        r,_=run_coin(x,h,kind,k,w,allow.fillna(False));parts.append(r)
    r=pd.concat(parts,axis=1).fillna(0).sum(1)                               # P&L fractions of a shared equity
    return np.log1p(r.clip(lower=-0.99))

def stats(r):
    if r.std()==0:return {'cagr_pct':0,'sharpe':np.nan,'max_dd_pct':0,'worst_week_pct':0,'worst_month_pct':0}
    eq=r.cumsum();w=r.groupby(r.index.to_period('W')).sum();m=r.groupby(r.index.to_period('M')).sum()
    return {'cagr_pct':round((np.exp(r.mean()*365)-1)*100,1),'sharpe':round(r.mean()/r.std()*np.sqrt(365),2),
            'max_dd_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1),
            'worst_week_pct':round((np.exp(w.min())-1)*100,1),'worst_month_pct':round((np.exp(m.min())-1)*100,1)}

if __name__=='__main__':
    p=vd.panel();p=har(p);p,minfo=model(p)
    premium(p).to_csv(f'{OUT}/vol_premium.csv',index=False);episodes(p).to_csv(f'{OUT}/vol_episodes.csv',index=False)
    acc=accuracy(p,lambda h:[('DVOL','dvol'),('HAR',f'har{h}')]);acc.to_csv(f'{OUT}/vol_forecast_acc.csv',index=False)
    acc2=accuracy(p,lambda h:[('DVOL','dvol'),('HAR',f'har{h}'),('model',f'mod{h}')],lo='2023-01-01');acc2.to_csv(f'{OUT}/vol_forecast_acc_2023.csv',index=False)
    D=pd.DataFrame({s:trend.daily_closes(s) for s in trend.COINS}).dropna();F=pd.DataFrame({s:trend.daily_funding(s) for s in trend.COINS})
    tr,_=trend.backtest(D,F)
    rows=[];IS=('2021-04-01','2023-12-31');OOS=('2024-01-01','2026-12-31')
    grid=[(k,w) for k in (0.5,1.0) for w in (0.5,1.0)]
    best={}
    for h in (7,28):
        for kind,g in (('condor',grid),('straddle',[(0,0)])):
            for k,w in g:
                for filt in ('always','har','model','model+tail'):
                    r=run(p,h,kind,k,w,filt);row={'h':h,'kind':kind,'k':k,'w':w,'filter':filt}
                    for name,(a,b) in (('IS',IS),('OOS',OOS),('2023-26',('2023-01-01','2026-12-31'))):
                        for kk,v in stats(r[a:b]).items():row[f'{name}_{kk}']=v
                    yr=r.groupby(r.index.year).sum();row.update({f'y{y}_pct':round((np.exp(v)-1)*100,1) for y,v in yr.items()})
                    j=pd.concat([r,tr],axis=1).dropna();row['corr_trend']=round(j.corr().iat[0,1],2);rows.append(row)
                    print(h,kind,k,w,filt,row['IS_sharpe'],row['OOS_sharpe'],row['OOS_max_dd_pct'],flush=True)
    T=pd.DataFrame(rows);T.to_csv(f'{OUT}/vol_backtest.csv',index=False)
    sel={}
    for h in (7,28):
        c=T[(T.h==h)&(T.kind=='condor')&(T['filter']=='always')].sort_values('IS_sharpe',ascending=False).iloc[0];sel[h]=(c.k,c.w)
    json.dump({'selected_kw':{str(h):v for h,v in sel.items()},**minfo},open(f'{OUT}/vol_meta.json','w'),indent=1,default=float)
    print(json.dumps({'selected':{str(h):v for h,v in sel.items()},**minfo},default=float,indent=1))
