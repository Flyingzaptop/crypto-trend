"""ML2: rare CUSUM-event entries with triple-barrier labels, LightGBM vs torch MLP (docs/ml/PROTOCOL_ML2.md).
usage: python ml2.py -> docs/ml/results/ml2_*.csv"""
import json,numpy as np,pandas as pd,lightgbm as lgb
import trend,panel,ml_data as md
from ml1 import PARAMS

HS=(5,20);WIN=60;SEL=('2020-01-01','2023-12-31');VER='2024-01-01';ALWAYS=['BTCUSDT','ETHUSDT']

def frames(C,Q,F):
    R=np.log(C).diff();sd=R.rolling(20).std();lc=np.log(C);B=R['BTCUSDT']
    X={f'ret{h}':lc.diff(h) for h in (1,5,20,60)}
    X.update({f'rv{h}':R.rolling(h).std()*np.sqrt(365) for h in (5,20,60)});X['rv20_60']=X['rv20']/X['rv60']
    X.update({f'dsma{n}':(lc-np.log(C.rolling(n).mean()))/sd for n in trend.SMAS})
    X.update({f'fund{h}':F.fillna(0).rolling(h).sum() for h in (1,7,30)})
    X['lvol']=np.log(Q+1)-np.log(Q.rolling(30).mean()+1)
    cov=R.rolling(60).cov(B);X['beta']=cov.div(B.rolling(60).var(),axis=0);X['corr']=R.rolling(60).corr(B)
    for h in (5,20):X[f'btc{h}']=pd.DataFrame({c:lc['BTCUSDT'].diff(h) for c in C})
    X['tvrank']=Q.rolling(30,min_periods=20).mean().rank(axis=1,ascending=False)
    return R,sd,X

def cusum(r,h):
    """Symmetric CUSUM events: +1 / -1 on the day the filter fires (threshold h per day)."""
    out=np.zeros(len(r));sp=sn=0.
    for i in range(len(r)):
        if np.isnan(r[i]) or np.isnan(h[i]):sp=sn=0.;continue
        sp=max(0.,sp+r[i]);sn=min(0.,sn+r[i])
        if sp>h[i]:out[i]=1;sp=sn=0.
        elif sn<-h[i]:out[i]=-1;sp=sn=0.
    return out

def events(C,R,sd,U,H):
    rows=[];Rv=R.values;S=sd.values;idx=C.index
    for j,c in enumerate(C.columns):
        e=cusum(Rv[:,j],S[:,j]);last=np.where(~np.isnan(C.values[:,j]))[0]
        if not len(last):continue
        last=last[-1]
        for i in np.where((e!=0)&U[c].values)[0]:
            if i+1>last:continue
            b=S[i,j]*np.sqrt(H);path=np.nancumsum(Rv[i+1:min(i+H,last)+1,j]);hit=np.where((path>=b)|(path<=-b))[0]
            k=hit[0] if len(hit) else len(path)-1;y=float(path[k]>0)
            rows.append({'t':idx[i],'coin':c,'i':i,'j':j,'ev':e[i],'y':y,'exit':i+1+k,'full':len(path)==H or len(hit)>0,'sd':S[i,j]})
    return pd.DataFrame(rows)

def mlp_fit_predict(Xtr,ytr,Xte,seed=0):
    import torch;torch.manual_seed(seed);torch.set_num_threads(1)
    mu,sg=np.nanmean(Xtr,0),np.nanstd(Xtr,0)+1e-9;f=lambda A:torch.tensor(np.nan_to_num((A-mu)/sg).clip(-10,10),dtype=torch.float32)
    net=torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1],64),torch.nn.ReLU(),torch.nn.Dropout(0.2),torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Dropout(0.2),torch.nn.Linear(32,1))
    opt=torch.optim.Adam(net.parameters(),1e-3);A=f(Xtr);y=torch.tensor(ytr,dtype=torch.float32)[:,None];lf=torch.nn.BCEWithLogitsLoss()
    g=torch.Generator().manual_seed(seed)
    for _ in range(30):
        net.train()
        for b in torch.randperm(len(A),generator=g).split(256):opt.zero_grad();l=lf(net(A[b]),y[b]);l.backward();opt.step()
    net.eval()
    with torch.no_grad():return torch.sigmoid(net(A)).numpy()[:,0],torch.sigmoid(net(f(Xte))).numpy()[:,0]

def design(E,X,R,sd,Q):
    snap=np.column_stack([X[k].values[E.i,E.j] for k in X]+[E.ev.values,E.coin.isin(['BTCUSDT']).values,E.coin.isin(['ETHUSDT']).values]).astype(float)
    z=(R/sd).clip(-5,5).values;lq=(np.log(Q+1)-np.log(Q.rolling(30).mean()+1)).values
    win=np.stack([np.r_[z[i-WIN+1:i+1,j],lq[i-WIN+1:i+1,j]] if i>=WIN else np.full(2*WIN,np.nan) for i,j in zip(E.i,E.j)])
    return snap,np.c_[snap,win]

def trade(E,p,thr,H,C,R,F):
    N=np.zeros(C.shape)
    for (_,e),pp in zip(E.iterrows(),p):
        if abs(pp-0.5)<thr:continue
        w=min(0.5,0.02/(e.sd*np.sqrt(H)))*np.sign(pp-0.5);N[e.i:e.exit,e.j]+=w            # row t = held during t+1 ... exit day
    N=pd.DataFrame(N,index=C.index,columns=C.columns);g=N.abs().sum(1);N=N.div((g/2).clip(lower=1),axis=0)
    return md.pnl(N,R.fillna(0),F.fillna(0),simple=True),N

def run(C,Q,F,variants,tag):
    R,sd,X=frames(C,Q,F);U=panel.universe(C,Q,F,10);U[ALWAYS]=C[ALWAYS].notna()&(C[ALWAYS].notna().cumsum()>=100)
    out={};info={}
    for H in sorted({h for _,h in variants}):
        E=events(C,R,sd,U,H);E=E[E.t>='2019-04-01'].reset_index(drop=True);snap,full=design(E,X,R,sd,Q)
        for model in sorted({m for m,h in variants if h==H}):
            A=snap if model=='lgbm' else full;p=np.full(len(E),np.nan);thr=pd.Series(dtype=float)
            for Y in range(2020,E.t.max().year+1):
                tr=(E.t+pd.Timedelta(days=H)<pd.Timestamp(f'{Y}-01-01')).values;te=(E.t.dt.year==Y).values
                if model=='lgbm':m=lgb.LGBMClassifier(**PARAMS).fit(A[tr],E.y[tr]);ptr=m.predict_proba(A[tr])[:,1];p[te]=m.predict_proba(A[te])[:,1]
                else:ptr,p[te]=mlp_fit_predict(A[tr],E.y[tr].values,A[te])
                thr[Y]=np.percentile(np.abs(ptr-0.5),90)
            ok=~np.isnan(p);E2=E[ok];p2=p[ok];th=E2.t.dt.year.map(thr).values
            keep=np.abs(p2-0.5)>=th;r,N=trade(E2[keep],p2[keep],0,H,C,R,F)
            from sklearn.metrics import roc_auc_score
            k=f'{model} H{H}';out[k]=r['2020-01-01':]
            info[k]={'events':int(len(E2)),'trades':int(keep.sum()),'trades_per_year':round(keep.sum()/((E2.t.max()-E2.t.min()).days/365),1),
                     'auc_sel':round(roc_auc_score(E2.y[E2.t<VER],p2[(E2.t<VER).values]),3),'auc_ver':round(roc_auc_score(E2.y[E2.t>=VER],p2[(E2.t>=VER).values]),3),
                     'hit_rate_traded':round(float(((p2[keep]>0.5)==(E2.y[keep]==1)).mean()),3),'avg_gross':round(float(N.abs().sum(1)['2020':].mean()),2)}
            print(tag,k,info[k],md.stats(out[k][SEL[0]:SEL[1]]),md.stats(out[k][VER:]),flush=True)
    return out,info,U,R,F

if __name__=='__main__':
    C,Q,F=panel.load();V=[(m,h) for m in ('lgbm','mlp') for h in HS]
    out,info,U,R,F2=run(C,Q,F,V,'real')
    best=max(out,key=lambda k:md.stats(out[k][SEL[0]:SEL[1]])['sharpe']);m,h=best.split();h=int(h[1:])
    mir,minfo,_,_,_=run(1/C,Q,-F,[(m,h)],'mirror')
    P=pd.DataFrame({s:md.klines(s,'1d').close for s in trend.COINS}).dropna();Ft=pd.DataFrame({s:trend.daily_funding(s) for s in trend.COINS})
    base={'trend':trend.backtest(P,Ft)[0],'buy&hold BTC+ETH':np.log(P).diff().mean(1).dropna(),
          'EW universe long':np.log(C).diff().where(U.shift(1).fillna(False).astype(bool)).mean(1).fillna(0)}
    rows=[]
    def add(name,r,**kw):
        r=r['2020-01-01':];rows.append({'series':name,**kw,**{f'sel_{a}':b for a,b in md.stats(r[SEL[0]:SEL[1]]).items()},
                                        **{f'ver_{a}':b for a,b in md.stats(r[VER:]).items()},**{str(y):b for y,b in md.by_year(r).items()}})
    for k,r in base.items():add(k,r)
    for k,r in out.items():add(k,r,selected=k==best,**info[k])
    for k,r in mir.items():add(f'MIRROR {k}',r,**minfo[k])
    t=base['trend'];j=pd.concat([t,out[best]],axis=1).dropna();pre=j[:'2023-12-31'];w=1/pre.std();cb=(j*w).sum(1)*(t[:'2023-12-31'].std()/(pre*w).sum(1).std())
    add(f'trend + {best} (50/50 risk)',cb,corr_pre24=round(pre.corr().iat[0,1],2))
    T=pd.DataFrame(rows);T.to_csv('docs/ml/results/ml2.csv',index=False);pd.DataFrame(out).to_csv('docs/ml/results/ml2_daily_returns.csv')
    pd.set_option('display.width',300);pd.set_option('display.max_columns',60);print(T.to_string(index=False));print('selected',best)
