"""ML1: meta-labeling of the BTC+ETH trend (docs/ml/PROTOCOL_ML1.md). usage: python ml1.py -> docs/ml/results/ml1_*.csv"""
import json,numpy as np,pandas as pd,lightgbm as lgb
import trend,vol_data,ml_data as md

C=list(trend.COINS);SEL=('2020-01-01','2023-12-31');VER='2024-01-01';HS=(5,20);THR=(0.45,0.50,0.55)
PARAMS=dict(n_estimators=200,learning_rate=0.03,num_leaves=7,max_depth=3,min_child_samples=50,subsample=0.8,subsample_freq=1,
            colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)

def load():
    K={s:md.klines(s,'1d') for s in C};P=pd.DataFrame({s:K[s].close for s in C}).dropna()
    V=pd.DataFrame({s:K[s].quote_volume for s in C}).reindex(P.index)
    F=pd.DataFrame({s:trend.daily_funding(s) for s in C}).reindex(P.index).fillna(0)
    DV=pd.DataFrame({s:vol_data.dvol(s[:3]) for s in C}).reindex(P.index)/100
    X=pd.DataFrame({k:md.yahoo_ohlc(t).close for k,t in (('spx','^GSPC'),('dxy','DX-Y.NYB'),('gold','GC=F'))}).sort_index()
    X=np.log(X).diff(20).reindex(P.index.union(X.index)).ffill().reindex(P.index)   # last close <= t (US close is before 00:00 UTC)
    return P,V,F,DV,X

def features(P,V,F,DV,X,N):
    R=np.log(P).diff();corr=R[C[0]].rolling(30).corr(R[C[1]]);rows=[]
    for s in C:
        lp=np.log(P[s]);r=R[s];sd=r.rolling(20).std();x=pd.DataFrame(index=P.index)
        for h in (1,5,20,60,120):x[f'ret{h}']=lp.diff(h)
        for h in (5,20,60):x[f'rv{h}']=r.rolling(h).std()*np.sqrt(365)
        x['rv20_60']=x.rv20/x.rv60;x['dvol']=DV[s];x['dvol_rv']=np.log(DV[s])-np.log(x.rv20)
        for h in (1,7,30):x[f'fund{h}']=F[s].rolling(h).sum()
        x['lvol']=np.log(V[s])-np.log(V[s].rolling(30).mean());x['corr30']=corr
        for k in X:x[k]=X[k]
        for n in trend.SMAS:x[f'dsma{n}']=(lp-np.log(P[s].rolling(n).mean()))/sd
        sig=sum(np.sign(P[s]-P[s].rolling(n).mean()) for n in trend.SMAS)/3;x['sig']=sig;x['side']=np.sign(N[s]);x['eth']=float(s==C[1])
        for h in HS:x[f'y{h}']=(np.sign(N[s])*r.rolling(h).sum().shift(-h)>0).astype(float).where(r.shift(-h).notna())
        x['coin']=s;rows.append(x[N[s].notna()&(N[s]!=0)])
    return pd.concat(rows)

FEAT=None
def walk_forward(Z,h):
    """Out-of-sample probability per row; model for year Y trained on rows with t+h < Y-01-01."""
    p=pd.Series(np.nan,index=pd.MultiIndex.from_arrays([Z.index,Z.coin]))
    for Y in range(2020,Z.index.max().year+1):
        lim=pd.Timestamp(f'{Y}-01-01');tr=Z[(Z.index+pd.Timedelta(days=h)<lim)&Z[f'y{h}'].notna()];te=Z[Z.index.year==Y]
        m=lgb.LGBMClassifier(**PARAMS).fit(tr[FEAT],tr[f'y{h}']);p.loc[list(zip(te.index,te.coin))]=m.predict_proba(te[FEAT])[:,1]
    return p.unstack('coin')

def run(P,V,F,DV,X,tag):
    N=trend.target_positions(P);R=np.log(P).diff();Z=features(P,V,F,DV,X,N)
    global FEAT;FEAT=[c for c in Z.columns if c not in ('coin',*[f'y{h}' for h in HS])]
    out={'buy&hold 50/50':R.mean(1).dropna(),'trend':md.pnl(N,R,F)};auc={}
    for h in HS:
        pr=walk_forward(Z,h).reindex(N.index)[C]
        y=Z[[f'y{h}','coin']].set_index('coin',append=True)[f'y{h}'];pp=pr.stack().reindex(y.index);ok=y.notna()&pp.notna()
        from sklearn.metrics import roc_auc_score
        for lo,hi in ((SEL[0],SEL[1]),(VER,'2099')):
            k=ok&(y.index.get_level_values(0)>=lo)&(y.index.get_level_values(0)<=hi);auc[f'h{h} {lo[:4]}+']=round(roc_auc_score(y[k],pp[k]),3)
        G=N.where(pr.notna());g=N.abs().sum(1)
        S=(G*(2*pr).clip(0,2));S=S.div((S.abs().sum(1)/1.0).clip(lower=1),axis=0);out[f'h{h} scale']=md.pnl(S,R,F)
        for t in THR:out[f'h{h} gate {t}']=md.pnl(G*(pr>=t),R,F)
    first=pd.Timestamp('2020-01-02');out={k:v[v.index>=first] for k,v in out.items()}
    rows=[]
    for k,r in out.items():
        s=r[SEL[0]:SEL[1]];v=r[VER:];row={'tag':tag,'series':k,'sel_sharpe':md.stats(s)['sharpe'],**{f'ver_{a}':b for a,b in md.stats(v).items()}}
        lev=out['trend'][SEL[0]:SEL[1]].std()/s.std();row.update({f'vm_ver_{a}':b for a,b in md.stats(v*lev).items() if a!='sharpe'})
        row.update({str(y):b for y,b in md.by_year(r).items()});rows.append(row)
    return pd.DataFrame(rows),auc,out

if __name__=='__main__':
    P,V,F,DV,X=load();print('data',P.index[0].date(),P.index[-1].date())
    T,auc,out=run(P,V,F,DV,X,'real');Tm,aucm,_=run(1/P,V,-F,DV,X,'mirror')
    ml=[k for k in T.series if k.startswith('h')];best=T[T.series.isin(ml)].sort_values('sel_sharpe').series.iloc[-1]
    A=pd.concat([T,Tm]);A['selected']=A.series==best;A.to_csv('docs/ml/results/ml1_variants.csv',index=False)
    pd.DataFrame({k:v for k,v in out.items()}).to_csv('docs/ml/results/ml1_daily_returns.csv')
    json.dump({'auc_real':auc,'auc_mirror':aucm,'selected':best,'n_variants':len(ml)},open('docs/ml/results/ml1_summary.json','w'),indent=1)
    pd.set_option('display.width',300);pd.set_option('display.max_columns',40);print(A.to_string(index=False));print('AUC',auc,'\nAUC mirror',aucm,'\nselected',best)
