"""Phase 0: real Deribit option prices vs Black-Scholes(DVOL) (docs/vol/PROTOCOL_phase0.md).
usage: python phase0.py -> docs/vol/results/phase0_*.csv, docs/vol/results/phase0_smile.json"""
import json,numpy as np,pandas as pd
from scipy.stats import norm
import deribit_hist as dh,vol,vol_data as vd

OUT='docs/vol/results';XB=np.arange(-3.25,3.26,0.5);TB=[0,3.5,10.5,30.5,60.5];TL=['1-3d','4-10d','11-30d','31-60d']

def bs_coin(x,k,s,T,call):
    """Option price in units of the underlying: F=1, strike k=K/F."""
    d1=(-np.log(k)+0.5*s*s*T)/(s*np.sqrt(T));d2=d1-s*np.sqrt(T)
    return np.where(call,norm.cdf(d1)-k*norm.cdf(d2),k*norm.cdf(-d2)-norm.cdf(-d1))

def implied(p,k,T,call):
    lo,hi=np.full(len(p),0.01),np.full(len(p),6.0)
    for _ in range(60):
        m=(lo+hi)/2;up=bs_coin(None,k,m,T,call)>p;hi=np.where(up,m,hi);lo=np.where(up,lo,m)
    return (lo+hi)/2

def load(cur):
    t=dh.sample(cur).copy();d=dh.dvol_hourly(cur)/100
    t['T']=(t.expiry-t.time).dt.total_seconds()/86400/365;t['Td']=t['T']*365
    t=t[(t.Td>=1)&(t.Td<=60)&(t.price>0)&(t.mark_price>0)].sort_values('time')
    dv=d.rename('dvol').to_frame();dv.index=dv.index.astype('datetime64[ns]');dv.index.name='time';t['time']=t.time.astype('datetime64[ns]')
    t=pd.merge_asof(t,dv.reset_index().sort_values('time'),on='time',direction='backward')
    t['k']=t.strike/t.index_price;t['x']=np.log(t.k)/(t.dvol*np.sqrt(t['T']))
    t['otm']=np.where(t.call,t.k>=1,t.k<1)
    t['mark_iv']=implied(t.mark_price.values,t.k.values,t['T'].values,t.call.values)
    t['ratio']=t.mark_iv/t.dvol
    t['sgn']=np.where(t.direction=='buy',1,-1)
    t['half_rel']=(t.price-t.mark_price)*t.sgn/t.mark_price;t['half_coin']=(t.price-t.mark_price)*t.sgn
    t['xb']=pd.cut(t.x,XB);t['tb']=pd.cut(t.Td,TB,labels=TL);t['coin']=cur
    # sanity: does Deribit's own trade iv agree with the price? (drop broken rows)
    t=t[(t.mark_iv>0.02)&(t.mark_iv<5.9)&t.dvol.notna()]
    return t

def design(x,tb,reg):
    cols=[]
    for b in TL:
        m=(tb==b).astype(float);cols+=[m,m*x,m*x**2,m*x**3]
    cols+=[(reg==1).astype(float),(reg==2).astype(float)]
    return np.column_stack(cols)

if __name__=='__main__':
    T=pd.concat([load(c) for c in ('BTC','ETH')]);o=T[T.otm]
    print(len(T),'trades',len(o),'OTM',T.groupby(['coin','window']).size().to_dict(),flush=True)
    # 1 smile
    sm=o.groupby(['coin','tb','xb'],observed=True).ratio.agg(['median','count']).round(3).reset_index();sm.xb=sm.xb.astype(str)
    sm.to_csv(f'{OUT}/phase0_smile_table.csv',index=False)
    # time-of-day check near the money
    atm=o[o.x.abs()<0.5].groupby(['coin','window','tb'],observed=True).ratio.median().round(3).unstack('window');atm.to_csv(f'{OUT}/phase0_tod.csv')
    # 2 spread
    sp=T.groupby(['coin','tb','xb'],observed=True)[['half_rel','half_coin']].median().round(5).reset_index();sp.xb=sp.xb.astype(str)
    sp.to_csv(f'{OUT}/phase0_spread_table.csv',index=False)
    # 3 correction fit (pre-2024 OTM), check 2024+
    q=o[np.isfinite(o.x)&(o.x.abs()<=4)].copy();cut=q[q.time<'2024-01-01'].dvol.quantile([1/3,2/3]).values
    q['reg']=np.digitize(q.dvol,cut);X=design(q.x.values,q.tb.astype(str).values,q.reg.values);y=np.log(q.ratio.values)
    tr=(q.time<'2024-01-01').values;beta=np.linalg.lstsq(X[tr],y[tr],rcond=None)[0]
    te=~tr;mse_fit=float(((X[te]@beta-y[te])**2).mean());mse_flat=float((y[te]**2).mean())
    hr=T[T.time<'2024-01-01'].groupby(['tb','xb'],observed=True).half_rel.median()
    meta={'n_trades':len(T),'n_otm':len(o),'dvol_regime_cuts':cut.round(4).tolist(),'beta':beta.round(5).tolist(),
          'mse_2024_26_fitted':round(mse_fit,4),'mse_2024_26_flat':round(mse_flat,4),
          'median_ratio_atm':round(float(o[o.x.abs()<0.25].ratio.median()),3),
          'half_spread_rel_median':{f'{a}|{b}':round(v,4) for (a,b),v in hr.items()}}
    json.dump(meta,open(f'{OUT}/phase0_smile.json','w'),indent=1);print(json.dumps({k:v for k,v in meta.items() if k!='half_spread_rel_median'},indent=1))

    # 4 rerun the earlier backtest with corrected prices
    def tbin(Td):return np.digitize(Td,TB[1:-1])
    def sigma(S,K,s,T_):
        if T_<=0:return s
        x=np.log(K/S)/(s*np.sqrt(T_));x=np.clip(x,-4,4);b=TL[min(tbin(T_*365),3)];r=int(np.digitize(s,cut))
        return s*float(np.exp(design(np.array([x]),np.array([b]),np.array([r]))@beta)[0])
    xb_mid=np.array([iv.mid for iv in hr.index.get_level_values('xb').unique()])
    def spread(S,K,s,T_,price):
        if T_<=0:return 0.0
        x=np.clip(np.log(K/S)/(s*np.sqrt(T_)),-3,3);b=TL[min(tbin(T_*365),3)]
        h=hr.xs(b,level='tb');m=np.array([iv.mid for iv in h.index]);rel=float(h.iloc[np.abs(m-x).argmin()])
        return max(max(rel,0)*price,0.0001*S)
    p=vd.panel();p=vol.har(p)
    cases=[(7,'condor',0.5,1.0),(28,'condor',1.0,1.0),(7,'straddle',0,0),(28,'straddle',0,0)]
    rows=[]
    for label,sg,spf in (('synthetic (old)',lambda S,K,s,T_:s,lambda S,K,s,T_,pr:0.05*pr),('corrected',sigma,spread)):
        vol.SIGMA,vol.SPREAD=sg,spf
        for exits in (True,False):
            vol.EXITS=exits
            for h,kind,k,w in cases:
                r=vol.run(p,h,kind,k,w,'always');row={'prices':label,'exits':'50%/2x' if exits else 'hold','h':h,'kind':kind,'k':k,'w':w}
                for n,(a,b) in (('2021-26',('2021-04-01','2026-12-31')),('2024-26',('2024-01-01','2026-12-31'))):
                    s_=vol.stats(r[a:b]);row.update({f'{n}_sharpe':s_['sharpe'],f'{n}_cagr':s_['cagr_pct'],f'{n}_dd':s_['max_dd_pct']})
                rows.append(row);print(row,flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/phase0_rerun.csv',index=False)
