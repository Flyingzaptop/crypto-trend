"""Phase 7(b): weekly ATM straddle on Deribit with daily perp delta hedge, real entry prices (docs/vol/PROTOCOL_phase7.md)."""
import numpy as np,pandas as pd
from scipy.stats import norm
import deribit_hist as dh,phase0 as p0,vol,vol_data as vd,trend,data_m1

OUT='docs/vol/results'

def entries(cur,half):
    t=dh.sample(cur);t=t[t.window=='fri'].copy();t['day']=t.time.dt.normalize()
    t=t[(t.time.dt.hour>=8)&(t.time.dt.hour<10)];rows=[]
    for d,g in t.groupby('day'):
        exp=d+pd.Timedelta(days=7,hours=8);g=g[g.expiry==exp]
        if g.empty:continue
        idx=g.index_price.median();c=g[g.call].groupby('strike').mark_price.median();p=g[~g.call].groupby('strike').mark_price.median()
        ks=c.index.intersection(p.index)
        if len(ks)==0:continue
        k=ks[np.abs(ks-idx).argmin()]
        if abs(k/idx-1)>0.03:continue
        T=7/365-0/365;kr=k/idx
        iv=float(np.mean([p0.implied(np.array([c[k]]),np.array([kr]),np.array([T]),np.array([True]))[0],p0.implied(np.array([p[k]]),np.array([kr]),np.array([T]),np.array([False]))[0]]))
        rows.append({'entry':d+pd.Timedelta(hours=8),'expiry':exp,'K':k,'index':idx,'call':c[k]*idx,'put':p[k]*idx,'iv':iv,'half':half})
    return pd.DataFrame(rows)

def bs_delta(S,K,s,T):
    d1=(np.log(S/K)+0.5*s*s*T)/(s*np.sqrt(T));return norm.cdf(d1)-1+norm.cdf(d1)        # call delta + put delta

def run(cur,E,har,fund,hedge_bp):
    m=data_m1.load(f'{cur}USDT').close;m.index=m.index.floor('min');m=m[~m.index.duplicated()];mt=m.index.values;mv=m.values
    out=[]
    for r in E.itertuples():
        fc=har.asof(r.entry.normalize()-pd.Timedelta('1D'))
        sign=1 if np.log(r.iv)-np.log(fc)>0 else -1                              # +1 = sell straddle
        days=pd.date_range(r.entry,r.expiry,freq='1D');ix=np.searchsorted(mt,days.values,side='right')-1;S=np.where(ix>=0,mv[np.clip(ix,0,None)],np.nan)
        if np.isnan(S).any():continue
        prem=r.call+r.put;half=r.half*prem                                     # half-spread in $ on the whole straddle
        fee_in=2*min(0.0003*r.index,0.125*prem/2)
        payoff=abs(S[-1]-r.K);fee_out=min(0.00015*S[-1],0.125*payoff) if payoff>0 else 0
        # option P&L per 1 unit for a seller; buyer is the negative minus the same costs
        opt_sell=prem-payoff;cost_opt=half+fee_in+fee_out
        hedge=0.0;h_cost=0.0;pos=0.0;fsum=0.0
        for i in range(len(days)-1):
            T=(r.expiry-days[i]).total_seconds()/(365*86400)
            dlt=bs_delta(S[i],r.K,r.iv,T)                                       # straddle delta (long)
            tgt=dlt                                                               # seller is short delta -> hedge long dlt
            h_cost+=abs(tgt-pos)*S[i]*hedge_bp*1e-4;pos=tgt
            f_=pos*S[i]*fund.get(days[i].normalize(),0.0);fsum+=f_;hedge+=pos*(S[i+1]-S[i])-f_     # long perp pays funding
        h_cost+=abs(pos)*S[-1]*hedge_bp*1e-4
        for variant,sg in (('always_sell',1),('har_sign',sign)):
            pnl=sg*(opt_sell+hedge)-cost_opt-h_cost                               # hedge flips with the option side
            out.append({'coin':cur,'entry':r.entry,'variant':variant,'side':sg,'iv':r.iv,'har':fc,'pnl_pct':pnl/r.index*100,'funding_paid_pct':sg*fsum/r.index*100,'hedge_cost_pct':h_cost/r.index*100,'option_cost_pct':cost_opt/r.index*100})
    return pd.DataFrame(out)

def stats(w):
    x=w/100;eq=x.cumsum()
    return {'n':len(w),'mean_pct_wk':round(w.mean(),3),'sharpe':round(x.mean()/x.std()*np.sqrt(52),2) if x.std()>0 else np.nan,
            'worst_week_pct':round(w.min(),2),'max_dd_pct':round((eq.cummax()-eq).max()*100,1)}

if __name__=='__main__':
    pd.set_option('display.width',250)
    sp=pd.read_csv(f'{OUT}/phase0_spread_table.csv');p=vd.panel();p=vol.har(p);res=[]
    for cur in ('BTC','ETH'):
        half=float(sp[(sp.coin==cur)&(sp.tb=='4-10d')&(sp.xb=='(-0.25, 0.25]')].half_rel.iloc[0])
        E=entries(cur,half);har=p[p.coin==cur].har7;fund=trend.daily_funding(f'{cur}USDT').fillna(0)
        print(cur,len(E),'weekly entries, ATM half-spread',round(half*100,2),'% of price',flush=True)
        for bp in (5,2):res.append(run(cur,E,har,fund,bp).assign(hedge_bp=bp))
    R=pd.concat(res);R.to_csv(f'{OUT}/phase7_trades.csv',index=False)
    rows=[]
    for (v,bp),g in R.groupby(['variant','hedge_bp']):
        w=g.groupby('entry').pnl_pct.mean()                                      # BTC and ETH equally weighted
        for seg,(a,b) in (('<2024',('2021','2023')),('2024-26',('2024','2026'))):
            rows.append({'variant':v,'hedge_bp':bp,'seg':seg,**stats(w[a:b])})
        yr=w.groupby(w.index.year).sum().round(2).to_dict();rows[-1]['by_year_pct']=yr
    T=pd.DataFrame(rows);T.to_csv(f'{OUT}/phase7_straddle.csv',index=False);print(T.to_string(index=False))
    D=pd.DataFrame({s:trend.daily_closes(s) for s in trend.COINS}).dropna();tr,_=trend.backtest(D,pd.DataFrame({s:trend.daily_funding(s) for s in trend.COINS}))
    w=R[(R.variant=='always_sell')&(R.hedge_bp==5)].groupby('entry').pnl_pct.mean();w.index=w.index.normalize()
    print('corr with trend (weekly):',round(pd.concat([w,tr.groupby(pd.Grouper(freq='W-FRI',label='left')).sum().reindex(w.index)],axis=1).corr().iat[0,1],2))
