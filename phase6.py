"""Phase 6: Polymarket price markets vs a log-normal volatility model (docs/vol/PROTOCOL_phase6.md)."""
import re,numpy as np,pandas as pd
from scipy.stats import norm
import phase6_data as d6,deribit_hist as dh,vol,vol_data as vd,data_m1

OUT='docs/vol/results';HOR={'1h':pd.Timedelta('1h'),'24h':pd.Timedelta('24h'),'7d':pd.Timedelta('7D')};MAXAGE={'1h':pd.Timedelta('6h'),'24h':pd.Timedelta('24h'),'7d':pd.Timedelta('24h')}
SPLIT=pd.Timestamp('2025-07-01',tz='UTC');FEE=0.07

def kind(q):
    q=q.lower()
    if 'up or down' in q:return 'updown'
    if re.search(r'\babove\b',q):return 'above'
    if re.search(r'\b(reach|hit|dip|drop|fall)\b',q):return 'touch'
    return 'other'

def strike(q):
    m=re.search(r'\$\s?([\d,]+(?:\.\d+)?)\s?([kK])?',q)
    if not m:return np.nan
    v=float(m.group(1).replace(',',''));return v*1000 if m.group(2) else v

def asset(r):
    t=(r.question+' '+r.event).lower()
    return 'BTC' if 'bitcoin' in t or 'btc' in t else 'ETH' if 'ethereum' in t or 'eth ' in t else None

def snap_price(h,ts,maxage):
    if not h:return np.nan
    t=np.array([x['t'] for x in h]);p=np.array([x['p'] for x in h]);i=np.searchsorted(t,ts.timestamp(),side='right')-1
    return p[i] if i>=0 and ts.timestamp()-t[i]<=maxage.total_seconds() else np.nan

def build():
    M=pd.read_parquet(f'{d6.ROOT}/sample.parquet');M['kind']=M.question.map(kind);M['K']=M.question.map(strike);M['asset']=M.apply(asset,axis=1)
    M['end']=pd.to_datetime(M.end,utc=True);M['start']=pd.to_datetime(M.start,utc=True,errors='coerce')
    m1={a:data_m1.load(f'{a}USDT')[['close','high','low']] for a in ('BTC','ETH')}
    for a in m1:m1[a].index=m1[a].index.tz_localize('UTC').floor('min');m1[a]=m1[a][~m1[a].index.duplicated()]
    dv={a:(dh.dvol_hourly(a)/100).tz_localize('UTC') for a in ('BTC','ETH')}
    p=vd.panel();p=vol.har(p);har={a:p[p.coin==a].har7.tz_localize('UTC') for a in ('BTC','ETH')}
    rows=[]
    for r in M.itertuples():
        path=f'{d6.ROOT}/h/{r.yes_token[:40]}.json'
        try:h=__import__('json').load(open(path))
        except Exception:continue
        for hn,dt_ in HOR.items():
            ts=r.end-dt_;px=snap_price(h,ts,MAXAGE[hn])
            if not np.isfinite(px):continue
            row={'market_id':r.market_id,'tag':r.tag,'asset':r.asset,'kind':r.kind,'hor':hn,'end':r.end,'ts':ts,'price':px,'y':float(r.yes_won),'K':r.K,'question':r.question}
            if r.asset in m1 and r.kind in ('above','updown','touch') and ts>m1[r.asset].index[0] and ts<m1[r.asset].index[-1]:
                c=m1[r.asset];S=c.close.asof(ts-pd.Timedelta('1min'));tau=(r.end-ts).total_seconds()/(365*86400)
                for nm,sv in (('har',har[r.asset].asof((ts-pd.Timedelta('1D')).normalize())),('dvol',dv[r.asset].asof(ts))):
                    if not np.isfinite(sv) or tau<=0:continue
                    s=sv*np.sqrt(tau)
                    if r.kind=='above' and np.isfinite(r.K):q=norm.cdf((np.log(S/r.K)-s*s/2)/s)
                    elif r.kind=='updown':
                        win=pd.Timedelta('15min') if re.search(r'\d:\d\d\s?[AP]M\s?-\s?\d',r.event) else pd.Timedelta('1h') if re.search(r'[AP]M ET',r.event) else pd.Timedelta('1D');st=r.end-win
                        if ts<st:q=norm.cdf(-s/2)
                        else:
                            o=c.close.asof(st-pd.Timedelta('1min'));q=norm.cdf((np.log(S/o)-s*s/2)/s)
                        if str(r.question).lower().find('down')>=0 and False:q=1-q
                    elif r.kind=='touch' and np.isfinite(r.K):
                        st=r.start if pd.notna(r.start) else r.end-pd.Timedelta('30D');seg=c.loc[max(st,c.index[0]):ts]
                        up=r.K>S
                        if (up and seg.high.max()>=r.K) or ((not up) and seg.low.min()<=r.K):q=1.0
                        else:b=abs(np.log(r.K/S));q=min(1.0,2*norm.cdf(-b/s))
                    else:continue
                    row[f'm_{nm}']=float(np.clip(q,1e-4,1-1e-4))
            rows.append(row)
    return pd.DataFrame(rows)

def brier(p,y):return float(((p-y)**2).mean())
def logloss(p,y):p=np.clip(p,1e-4,1-1e-4);return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())

def trade(df,side,cost):
    """side: Series of +1 buy YES / -1 buy NO / 0; pays price+cost+taker fee, returns per $ staked."""
    p=np.where(side>0,df.price,1-df.price);win=np.where(side>0,df.y,1-df.y);c=p+cost+FEE*p*(1-p)
    r=(win-c)/c;return pd.Series(r,index=df.index)[side!=0]

def tstat(r):return r.mean()/r.std()*np.sqrt(len(r)) if len(r)>2 and r.std()>0 else np.nan

if __name__=='__main__':
    pd.set_option('display.width',250);D=build();D.to_parquet(f'{d6.ROOT}/panel.parquet')
    print(len(D),'snapshots',D.groupby(['kind','hor']).size().unstack().to_string())
    # (a) accuracy, crypto price markets with model
    A=[]
    for (k,hn),g in D[D.kind.isin(['above','updown','touch'])].dropna(subset=['m_har','m_dvol']).groupby(['kind','hor']):
        A.append({'kind':k,'hor':hn,'n':len(g),'brier_market':round(brier(g.price,g.y),4),'brier_har':round(brier(g.m_har,g.y),4),'brier_dvol':round(brier(g.m_dvol,g.y),4),
                  'logloss_market':round(logloss(g.price,g.y),4),'logloss_har':round(logloss(g.m_har,g.y),4),'logloss_dvol':round(logloss(g.m_dvol,g.y),4)})
    A=pd.DataFrame(A);A.to_csv(f'{OUT}/phase6_accuracy.csv',index=False);print(A.to_string(index=False))
    # calibration (all tags), market price buckets
    D['bucket']=pd.cut(D.price,np.linspace(0,1,11),include_lowest=True)
    C=D.groupby(['hor','bucket'],observed=True).agg(n=('y','size'),mean_price=('price','mean'),freq_yes=('y','mean')).round(3).reset_index();C.bucket=C.bucket.astype(str)
    C.to_csv(f'{OUT}/phase6_calibration.csv',index=False);print(C.to_string(index=False))
    tails=[];
    for lo,hi,nm in ((0,0.02,'<2c'),(0.02,0.05,'2-5c'),(0.05,0.10,'5-10c'),(0.90,0.95,'90-95c'),(0.95,0.98,'95-98c'),(0.98,1.0,'>98c')):
        g=D[(D.price>=lo)&(D.price<hi)];tails.append({'bucket':nm,'n':len(g),'mean_price':round(g.price.mean(),4),'freq_yes':round(g.y.mean(),4)})
    T=pd.DataFrame(tails);T.to_csv(f'{OUT}/phase6_tails.csv',index=False);print(T.to_string(index=False))
    # (b) rules: choose parameters on first half, verdict on second
    first=D.end<SPLIT;rows=[]
    for cost in (0.01,0.02):
        for hn in HOR:
            g=D[D.hor==hn]
            for X in (0.02,0.05,0.10):
                side=np.where(g.price<X,-1,0)
                for half,mask in (('first',first[g.index]),('second',~first[g.index])):
                    r=trade(g[mask],side[mask.values],cost);rows.append({'rule':'R1 tails NO','param':X,'hor':hn,'cost':cost,'half':half,'n':len(r),'mean_ret':round(r.mean(),4),'t':round(tstat(r),2)})
            gm=g.dropna(subset=['m_har','m_dvol'])
            for nm in ('har','dvol'):
                for N in (0.05,0.10,0.20):
                    d=gm[f'm_{nm}']-gm.price;side=np.where(d>N,1,np.where(-d>N,-1,0))
                    for half,mask in (('first',first[gm.index]),('second',~first[gm.index])):
                        r=trade(gm[mask],side[mask.values],cost);rows.append({'rule':f'R2 model {nm}','param':N,'hor':hn,'cost':cost,'half':half,'n':len(r),'mean_ret':round(r.mean(),4) if len(r) else np.nan,'t':round(tstat(r),2)})
    R=pd.DataFrame(rows);R.to_csv(f'{OUT}/phase6_rules.csv',index=False)
    sel=[]
    for (rule,cost),g in R.groupby(['rule','cost']):
        f=g[(g.half=='first')&(g.n>=100)].sort_values('mean_ret',ascending=False)
        if f.empty:continue
        b=f.iloc[0];s=g[(g.half=='second')&(g.param==b.param)&(g.hor==b.hor)].iloc[0]
        sel.append({'rule':rule,'cost':cost,'param':b.param,'hor':b.hor,'first_n':b.n,'first_mean_ret':b.mean_ret,'first_t':b.t,'second_n':s.n,'second_mean_ret':s.mean_ret,'second_t':s.t,
                    'works':bool(s.n>=100 and s.mean_ret>0 and s.t>2)})
    S=pd.DataFrame(sel);S.to_csv(f'{OUT}/phase6_selected.csv',index=False);print(S.to_string(index=False))
