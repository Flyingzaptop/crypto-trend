"""Phase 2: the volatility premium on real prices (docs/vol/PROTOCOL_phase2.md). usage: python phase2.py"""
import os,urllib.request,urllib.error,numpy as np,pandas as pd
import tradfi as tf,phase1 as p1,data_m1

OUT='docs/vol/results';FUT=f'{data_m1.ROOT}/vxfut'
CBOE_IDX=['PUT','WPUT','PUTR','BXM','BXMW','BXMD','BXY','BXR','CNDR','BFLY','CLL','CLLZ','PPUT','CMBO','RXM','VXTH','VPD','VPN','SPRO']
WEEKLY={'WPUT','BXMW'}
PERIODS={'<2008':('1986','2007'),'2008-23':('2008','2023'),'2024-26':('2024','2026'),'all':('1986','2026')}
EPIS={'2008-09..2009-02':('2008-09-01','2009-02-28'),'2018-02':('2018-02-01','2018-02-28'),'2020-02..03':('2020-02-15','2020-03-31'),'2022':('2022-01-01','2022-12-31')}

def rf_daily(idx):
    return (tf.yahoo('^IRX').reindex(idx).ffill()/100/252).fillna(0)

def stats(r,rf):
    """r: daily simple returns of a total-return series; Sharpe over T-bill."""
    r=r.dropna();x=r-rf.reindex(r.index).fillna(0);lr=np.log1p(r);eq=lr.cumsum();m=lr.groupby(r.index.to_period('M')).sum()
    return {'cagr_pct':round((np.exp(lr.mean()*252)-1)*100,1),'vol_pct':round(r.std()*np.sqrt(252)*100,1),
            'sharpe':round(x.mean()/x.std()*np.sqrt(252),2) if x.std()>0 else np.nan,
            'max_dd_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1),'worst_month_pct':round((np.exp(m.min())-1)*100,1)}

def alpha(r,b,rf):
    """Monthly regression of excess returns on S&P 500 excess returns: annual alpha (%), t, beta."""
    j=pd.concat([r,b,rf],axis=1).dropna();m=np.log1p(j).groupby(j.index.to_period('M')).sum();y=m.iloc[:,0]-m.iloc[:,2];x=m.iloc[:,1]-m.iloc[:,2]
    X=np.column_stack([np.ones(len(x)),x]);c,res,*_=np.linalg.lstsq(X,y,rcond=None);e=y-X@c;s2=(e**2).sum()/(len(y)-2)
    se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)));return round(c[0]*1200,2),round(c[0]/se[0],2),round(c[1],2)

def part_a():
    spx=tf.yahoo('^SP500TR').pct_change();rf=rf_daily(spx.index);rows=[]
    for n in CBOE_IDX+['SPXTR']:
        if n=='SPXTR':s=spx
        else:
            lv=p1.cboe(n).dropna();g=lv.index.to_series().diff().dt.days;big=g[g>30]
            if len(big):lv=lv[lv.index>=big.index[-1]]                         # sparse early history (PUT before 2007)
            s=lv.pct_change()
        haircut=0 if n=='SPXTR' else (0.03 if n in WEEKLY else 0.01)/252
        r=(s-haircut).dropna();r=r[r.index>=spx.dropna().index[0]]
        row={'index':n,'from':str(r.index[0].date())}
        for k,(a,b) in PERIODS.items():
            q=r[a:b]
            if len(q)>200:row.update({f'{k}_{m}':v for m,v in stats(q,rf).items()})
        row['alpha_pct_yr'],row['alpha_t'],row['beta']=alpha(r,spx,rf) if n!='SPXTR' else (0.0,np.nan,1.0)
        for k,(a,b) in EPIS.items():
            q=r[a:b];row[f'ep_{k}']=round((np.prod(1+q)-1)*100,1) if len(q) else np.nan
        rows.append(row)
    T=pd.DataFrame(rows);T.to_csv(f'{OUT}/phase2_cboe_indices.csv',index=False);return T

def spx_har():
    vix=p1.cboe('VIX');x=p1.frame('VIX',vix);Z=np.column_stack([np.ones(len(x)),x.lrv1,x.lrv5,x.lrv22]);y=np.log(x.rv_fwd).values;h=pd.Series(np.nan,index=x.index)
    for Y in range(1993,2027):
        tr=(x.index<pd.Timestamp(f'{Y}-01-01')-pd.Timedelta(days=31))&np.isfinite(y)&np.isfinite(Z).all(1);te=(x.index.year==Y)&np.isfinite(Z).all(1)
        b=np.linalg.lstsq(Z[tr],y[tr],rcond=None)[0];h[te]=np.exp(Z[te]@b)
    return h,x.iv

def short_vol(ret,signal,rf,borrow,cost=2e-4):
    """Short a vol product: daily simple returns ret, signal in {0,1} decided at t-1 close; 20% ex-ante vol, cap 1."""
    w=(0.20/(ret.rolling(60).std()*np.sqrt(252))).clip(upper=1.0)*signal;w=w.shift(1).fillna(0)
    return -w*ret+rf.reindex(ret.index).fillna(0)-w*borrow/252-w.diff().abs().fillna(0)*cost

def vx_expiries():
    out=[]
    for y in range(2013,2027):
        for mth in range(1,13):
            nm=pd.Timestamp(y,mth,1)+pd.offsets.MonthBegin(1);f3=nm+pd.offsets.WeekOfMonth(week=2,weekday=4) if nm.weekday()!=4 or True else nm
            fr=pd.date_range(nm,nm+pd.Timedelta(days=27),freq='W-FRI')[2];out.append(fr-pd.Timedelta(days=30))
    return out

def vx_contracts():
    os.makedirs(FUT,exist_ok=True);parts=[]
    for e in vx_expiries():
        path=f'{FUT}/VX_{e.date()}.csv'
        if not os.path.exists(path):
            for d in (e,e-pd.Timedelta(days=1)):                                # holiday-shifted expiries
                try:
                    req=urllib.request.Request(f'https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{d.date()}.csv',headers={'User-Agent':'Mozilla/5.0'})
                    open(path,'wb').write(urllib.request.urlopen(req,timeout=60).read());e=d;break
                except urllib.error.HTTPError:continue
        if os.path.exists(path):
            d=pd.read_csv(path);d['date']=pd.to_datetime(d['Trade Date']);d['exp']=e;parts.append(d[['date','exp','Settle']])
    return pd.concat(parts)

def vx_cm30(V):
    """Daily return of a constant 30-day-maturity long VIX futures position (two nearest contracts)."""
    V=V[V.Settle>0];P=V.pivot_table(index='date',columns='exp',values='Settle');R=P.pct_change(fill_method=None);idx=P.index;out=[]
    for i in range(1,len(idx)):
        t=idx[i-1];ex=[e for e in P.columns if e>t and pd.notna(P.loc[t,e])];ex.sort()
        if len(ex)<2:out.append(np.nan);continue
        d1,d2=(ex[0]-t).days,(ex[1]-t).days;w1=min(max((d2-30)/(d2-d1),0),1)
        out.append(w1*R.loc[idx[i],ex[0]]+(1-w1)*R.loc[idx[i],ex[1]])
    return pd.Series(out,index=idx[1:]).dropna()

def part_b():
    h,iv=spx_har();VV=pd.concat([p1.cboe('VIX'),p1.cboe('VIX3M')],axis=1).dropna()/100;vix,v3=VV.VIX,VV.VIX3M;rows=[];series={}
    sig={'always':lambda i:pd.Series(1.0,index=i),'contango':lambda i:(vix<v3).reindex(i).astype(float),
         'premium':lambda i:(iv>h).reindex(i).astype(float),'both':lambda i:((vix<v3).reindex(i).fillna(False)&(iv>h).reindex(i).fillna(False)).astype(float)}
    prods={'VIXY':tf.yahoo('VIXY').pct_change().dropna()}
    try:prods['VX_CM30']=vx_cm30(vx_contracts())
    except Exception as e:print('VX futures unavailable:',e)
    for p,ret in prods.items():
        rf=rf_daily(ret.index)
        for f,fn in sig.items():
            s=fn(ret.index).fillna(0)
            for borrow in ((0.02,0.06) if p=='VIXY' else (0.0,)):
                r=short_vol(ret,s,rf,borrow);series[(p,f,borrow)]=r;row={'product':p,'filter':f,'borrow':borrow,'in_market_pct':round(s.mean()*100)}
                for k,(a,b) in (('2011-23',('2011','2023')),('2024-26',('2024','2026')),('all',('2011','2026'))):
                    q=r[a:b]
                    if len(q)>200:row.update({f'{k}_{m}':v for m,v in stats(q,rf).items()})
                for k,(a,b) in EPIS.items():
                    q=r[a:b];row[f'ep_{k}']=round((np.prod(1+q)-1)*100,1) if len(q) else np.nan
                rows.append(row)
    sv=tf.yahoo('SVXY').pct_change().dropna();rf=rf_daily(sv.index)
    rows.append({'product':'SVXY buy&hold','filter':'-','borrow':0,**{f'{k}_{m}':v for k,(a,b) in (('2011-23',('2011','2023')),('2024-26',('2024','2026')),('all',('2011','2026'))) for m,v in stats(sv[a:b],rf).items()}})
    T=pd.DataFrame(rows);T.to_csv(f'{OUT}/phase2_short_vix.csv',index=False);return T

PAIRS={'TQQQ/SQQQ':('TQQQ','SQQQ','^NDX'),'UPRO/SPXU':('UPRO','SPXU','^GSPC'),'SPXL/SPXS':('SPXL','SPXS','^GSPC'),'TNA/TZA':('TNA','TZA','^RUT'),'SOXL/SOXS':('SOXL','SOXS','^SOX')}
def part_c():
    rows=[]
    for name,(a,b,u) in PAIRS.items():
        P=pd.DataFrame({'a':tf.yahoo(a),'b':tf.yahoo(b)}).dropna();R=P.pct_change().dropna();rf=rf_daily(R.index)
        up=tf.yahoo(u);lr=np.log(up).diff();rv=np.sqrt((lr**2).rolling(21).sum()*12);lrv=lambda k:np.log(np.sqrt((lr**2).rolling(k).sum()*252/k))
        Z=pd.DataFrame({'c':1.0,'l1':np.log(lr.abs()*np.sqrt(252)+1e-4),'l5':lrv(5),'l22':lrv(21),'y':np.log(rv.shift(-21))}).dropna(subset=['l1','l5','l22'])
        fc=pd.Series(np.nan,index=Z.index)
        for Y in range(Z.index.year.min()+3,2027):
            tr=(Z.index<pd.Timestamp(f'{Y}-01-01')-pd.Timedelta(days=31))&Z.y.notna();te=Z.index.year==Y
            if tr.sum()<500:continue
            c=np.linalg.lstsq(Z.loc[tr,['c','l1','l5','l22']],Z.loc[tr,'y'],rcond=None)[0];fc[te]=np.exp(Z.loc[te,['c','l1','l5','l22']]@c)
        hi=(fc>fc.rolling(252,min_periods=126).median()).astype(float).reindex(R.index).fillna(0)
        for filt,s in (('always',pd.Series(1.0,index=R.index)),('high_vol_forecast',hi)):
            for borrow in (0.03,0.10):
                # short 50% in each leg, reset monthly; between resets each leg's weight drifts with its own price
                out=[];wa=wb=0.0;m=None
                for t,(ra,rb) in R.iterrows():
                    if t.to_period('M')!=m:
                        m=t.to_period('M');tgt=s.shift(1).fillna(0).get(t,0)*0.5;cost=(abs(tgt-wa)+abs(tgt-wb))*2e-4;wa=wb=tgt
                    else:cost=0
                    pnl=-wa*ra-wb*rb-(wa+wb)*borrow/252-cost+rf[t];out.append(pnl)
                    wa*=1+ra;wb*=1+rb
                r=pd.Series(out,index=R.index);row={'pair':name,'filter':filt,'borrow':borrow,'from':str(R.index[0].date())}
                for k,(a_,b_) in (('<2024',('2008','2023')),('2024-26',('2024','2026')),('all',('2008','2026'))):
                    q=r[a_:b_]
                    if len(q)>200:row.update({f'{k}_{m_}':v for m_,v in stats(q,rf).items()})
                rows.append(row)
    T=pd.DataFrame(rows);T.to_csv(f'{OUT}/phase2_letf_decay.csv',index=False);return T

if __name__=='__main__':
    pd.set_option('display.width',250)
    A=part_a();print(A[['index','from','2008-23_sharpe','2024-26_sharpe','all_cagr_pct','all_sharpe','all_max_dd_pct','alpha_pct_yr','alpha_t','beta']].to_string(index=False),flush=True)
    C=part_c();print(C[['pair','filter','borrow','<2024_sharpe','2024-26_sharpe','all_cagr_pct','all_max_dd_pct']].to_string(index=False),flush=True)
    B=part_b();print(B[['product','filter','borrow','2011-23_sharpe','2024-26_sharpe','all_cagr_pct','all_max_dd_pct','ep_2018-02','ep_2020-02..03']].to_string(index=False))
