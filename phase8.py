"""Phase 8: short new Binance perpetual listings (docs/vol/PROTOCOL_phase8.md)."""
import os,re,glob,numpy as np,pandas as pd
import trend,phase8_data as d8

OUT='docs/vol/results'
EXCL=re.compile(r'^(BTC|ETH|USDC|FDUSD|TUSD|BUSD|USDP|DAI|PAXG|XAUT|XAU|XAG|EUR|GBP)')

def load():
    P,F={},{}
    for f in glob.glob(f'{d8.OUT}/*.parquet'):
        s=os.path.basename(f)[:-8];b=s[:-4]
        if EXCL.match(b) or s.endswith(('UPUSDT','DOWNUSDT')):continue
        d=pd.read_parquet(f)
        if d.empty or d.index[0]<pd.Timestamp('2021-01-01') or d.index[0]>pd.Timestamp('2026-08-31'):continue
        base=re.sub(r'^(1000000|100000|10000|1000|1M)','',b)
        P[s]=d.close;fp=f'data/funding/{s}.csv'
        F[s]=pd.read_csv(fp,index_col=0,parse_dates=True).iloc[:,0] if os.path.exists(fp) else pd.Series(dtype=float)
    P=pd.DataFrame(P).sort_index();F=pd.DataFrame(F).reindex(P.index).fillna(0)
    # drop multiplier duplicates of an asset that already traded as a perp (e.g. 1000XYZ after XYZ)
    first=P.apply(lambda s:s.first_valid_index());keep=[]
    for s in P:
        base=re.sub(r'^(1000000|100000|10000|1000|1M)','',s[:-4])+'USDT'
        if base!=s and base in first and first[base]<first[s]:continue
        keep.append(s)
    return P[keep],F[keep],first[keep]

def backtest(P,F,first,k,H,hedge,btc,btc_f,cost=5e-4,cap=0.2):
    R=P.pct_change(fill_method=None);on=pd.DataFrame(False,index=P.index,columns=P.columns)
    for s,d in first.items():
        a=d+pd.Timedelta(days=k);on.loc[a:a+pd.Timedelta(days=H-1),s]=True
    on&=P.notna()
    n=on.sum(1);w=on.div(n.where(n>0),axis=0).clip(upper=cap).fillna(0)          # short weights, gross <= 1
    W=w.shift(1).fillna(0);pnl=-(W*R.fillna(0)).sum(1)+(W*F).sum(1)-W.diff().abs().sum(1)*cost        # simple returns: a short gains at most 100%
    if hedge:
        g=W.sum(1);pnl+=g*btc.pct_change().reindex(P.index).fillna(0)-g*btc_f.reindex(P.index).fillna(0)-g.diff().abs().fillna(0)*cost
    pnl=np.log1p(pnl.clip(lower=-0.99));return pnl[pnl.index>=pd.Timestamp('2021-01-15')],on

def stats(r):
    eq=r.cumsum();m=r.groupby(r.index.to_period('M')).sum()
    return {'cagr_pct':round((np.exp(r.mean()*365)-1)*100,1),'sharpe':round(r.mean()/r.std()*np.sqrt(365),2) if r.std()>0 else np.nan,
            'max_dd_pct':round((1-np.exp(-(eq.cummax()-eq).max()))*100,1),'worst_month_pct':round((np.exp(m.min())-1)*100,1)}

if __name__=='__main__':
    pd.set_option('display.width',250)
    P,F,first=load();print(len(P.columns),'new perps 2021-2026; per year',first.dt.year.value_counts().sort_index().to_dict(),flush=True)
    btc=pd.read_parquet(f'{d8.OUT}/BTCUSDT.parquet').close;btc_f=trend.daily_funding('BTCUSDT')
    # single-short outcomes by listing cohort
    rows=[]
    for s,d in first.items():
        p=P[s].dropna()
        for k in (1,7):
            for H in (30,90,180):
                a=p[p.index>=d+pd.Timedelta(days=k)]
                if len(a)<2:continue
                end=a[a.index<=a.index[0]+pd.Timedelta(days=H)]
                ret=end.iloc[-1]/a.iloc[0]-1;fund=F[s].reindex(end.index).sum()
                rows.append({'symbol':s,'listed':d,'k':k,'H':H,'short_ret_pct':(-ret+fund)*100,'up_gt_100':ret>1,'complete':(end.index[-1]-a.index[0]).days>=H-3})
    O=pd.DataFrame(rows);O.to_csv(f'{OUT}/phase8_single_shorts.csv',index=False)
    C=O[O.complete].assign(cohort=np.where(O[O.complete].listed<'2024-01-01','<2024','2024-26')).groupby(['cohort','k','H']).agg(
        n=('short_ret_pct','size'),mean_pct=('short_ret_pct','mean'),median_pct=('short_ret_pct','median'),win_pct=('short_ret_pct',lambda v:(v>0).mean()*100),
        squeeze_gt100_pct=('up_gt_100',lambda v:v.mean()*100)).round(1).reset_index();print(C.to_string(index=False))
    C.to_csv(f'{OUT}/phase8_cohorts.csv',index=False)
    res=[];series={}
    for k in (1,7):
        for H in (30,90,180):
            for hedge in (False,True):
                pre=first[first<'2024-01-01'];post=first[first>='2024-01-01']
                r1,_=backtest(P[pre.index],F[pre.index],pre,k,H,hedge,btc,btc_f);r2,_=backtest(P[post.index],F[post.index],post,k,H,hedge,btc,btc_f)
                r2=r2['2024-01-01':];series[(k,H,hedge)]=r2
                row={'k':k,'H':H,'hedge':hedge,**{f'pre_{a}':b for a,b in stats(r1[:'2025-06-30']).items()},**{f'post_{a}':b for a,b in stats(r2).items()},
                     'post_h1_pct':round((np.exp(r2[:'2025-04-30'].sum())-1)*100,1),'post_h2_pct':round((np.exp(r2['2025-05-01':].sum())-1)*100,1),
                     **{f'post_y{y}':round((np.exp(v)-1)*100,1) for y,v in r2.groupby(r2.index.year).sum().items()}}
                res.append(row)
    T=pd.DataFrame(res);best=T.sort_values('pre_sharpe',ascending=False).iloc[0]
    T['selected']=(T.k==best.k)&(T.H==best.H)&(T.hedge==best.hedge)
    T['works']=T.selected&(T.post_sharpe>=0.7)&(T.post_max_dd_pct<=35)&(T.post_h1_pct>0)&(T.post_h2_pct>0)
    T.to_csv(f'{OUT}/phase8_portfolio.csv',index=False);print(T.to_string(index=False))
