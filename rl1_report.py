"""RL1 report from the saved runs (docs/ml/results/rl1_runs/*.parquet). usage: python rl1_report.py -> docs/ml/results/rl1_*.csv"""
import os,numpy as np,pandas as pd
import trend,ml_data as md,rl1

D=f'{rl1.OUT}/rl1_runs';ANN={'1d':365,'4h':2190}

def sh(r,a):return round(r.mean()/r.std()*np.sqrt(a),2) if len(r)>1 and r.std()>0 else 0.0

if __name__=='__main__':
    runs=[f[:-8] for f in sorted(os.listdir(D))];rows=[];ens={}
    P=pd.DataFrame({s:md.klines(s,'1d').close for s in trend.COINS}).dropna();F=pd.DataFrame({s:trend.daily_funding(s) for s in trend.COINS})
    base={}
    for s in trend.COINS:
        r,N=trend.backtest(P[[s]],F[[s]]);base[s]={'trend':r,'pos':N[s],'bh':np.log(P[s]).diff().dropna()}
    for tag in runs:
        sym,bar,algo,lev,seed,*m=tag.split('_');x=pd.read_parquet(f'{D}/{tag}.parquet');a=ANN[bar];t=x[rl1.TEST:];v=x.loc[rl1.VAL]
        pd_=x.pos.resample('1D').mean();c=pd_.corr(base[sym]['pos'].shift(1).reindex(pd_.index)) if not m else np.nan
        st=md.stats(t.ret,a)
        rows.append({'config':f'{sym} {bar} {algo} {lev}'+(' MIRROR' if m else ''),'seed':seed,'val_2023_logret':round(v.ret.sum(),3),
                     'test_sharpe':st['sharpe'],'test_cagr_pct':st['cagr_pct'],'test_max_dd_pct':st['max_dd_pct'],'test_vol_pct':st['vol_pct'],
                     'mean_pos':round(t.pos.mean(),2),'mean_abs_pos':round(t.pos.abs().mean(),2),'turnover_per_year':round(t.pos.diff().abs().sum()/(len(t)/a),1),
                     'corr_with_trend_pos':round(c,2),'train_sharpe_2018_22':sh(x[:rl1.TRAIN_END].ret,a)})
        ens.setdefault(rows[-1]['config'],[]).append(x)
    R=pd.DataFrame(rows);R.to_csv(f'{rl1.OUT}/rl1_seeds.csv',index=False)
    S=R.groupby('config').agg(val=('val_2023_logret','mean'),test_sharpe_mean=('test_sharpe','mean'),test_sharpe_sd=('test_sharpe','std'),
                              seeds_pos=('test_sharpe',lambda q:int((q>0).sum())),test_cagr_mean=('test_cagr_pct','mean'),test_dd_mean=('test_max_dd_pct','mean'),
                              mean_pos=('mean_pos','mean'),turnover=('turnover_per_year','mean'),corr_trend=('corr_with_trend_pos','mean'),train_sharpe=('train_sharpe_2018_22','mean')).round(2)
    for k,xs in ens.items():       # ensemble: average position of the seeds, re-priced with costs
        sym,bar=k.split()[:2];X=rl1.frame(sym,bar,'MIRROR' in k);p=pd.concat([x.pos for x in xs],axis=1).mean(1);r=X.r.reindex(p.index);f=X.fund.reindex(p.index)
        g=1+p.shift(1).fillna(0)*np.expm1(r)-p.shift(1).fillna(0)*f-rl1.COST*p.diff().abs().fillna(0);er=np.log(g.clip(lower=1e-6))
        S.loc[k,'ensemble_test_sharpe']=md.stats(er[rl1.TEST:],ANN[bar])['sharpe'];S.loc[k,'ensemble_test_dd']=md.stats(er[rl1.TEST:],ANN[bar])['max_dd_pct']
    for s in trend.COINS:
        for nm in ('trend','bh'):r=base[s][nm][rl1.TEST:];S.loc[f'{s} BASE {nm}',['test_sharpe_mean','test_cagr_mean','test_dd_mean']]=[md.stats(r)['sharpe'],md.stats(r)['cagr_pct'],md.stats(r)['max_dd_pct']]
    sel={s:S[S.index.str.startswith(s)&~S.index.str.contains('MIRROR|BASE')].val.idxmax() for s in trend.COINS};S['selected']=S.index.isin(sel.values())
    S.to_csv(f'{rl1.OUT}/rl1_configs.csv');pd.set_option('display.width',300);pd.set_option('display.max_columns',30);print(S.to_string());print('selected',sel)
