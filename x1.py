"""X1: slow patterns on ETFs and the combo with the crypto trend (docs/ml/PROTOCOL_X1.md). usage: python x1.py -> docs/ml/results/x1_*.csv"""
import json,numpy as np,pandas as pd
import trend,tradfi as tf,ml_data as md

T=['SPY','QQQ','IWM','EFA','EEM','TLT','IEF','GLD','SLV','USO','DBC','UUP','FXE','FXY','VNQ'];ANN=252;START='2008-01-01';VER='2024-01-01'

def row(name,r,**kw):
    s,v=r[:'2023-12-31'],r[VER:];out={'strategy':name,**kw}
    out.update({f'pre24_{k}':x for k,x in md.stats(s,ANN).items()});out.update({f'ver_{k}':x for k,x in md.stats(v,ANN).items()})
    out.update({str(y):x for y,x in md.by_year(r,ANN).items()});return out

def tsmom(C):
    R=np.log(C).diff();me=C.groupby([C.index.year,C.index.month]).tail(1).index          # month-end trading days
    M=np.log(C.loc[me]);sig=np.sign(M.shift(1)-M.shift(12))                              # 12-1: P[m-1]/P[m-12]
    sd=R.rolling(60).std().loc[me]*np.sqrt(ANN);W=sig*0.40/sd;W=W.div(W.notna().sum(1),axis=0)
    pv=(W.reindex(C.index).ffill().shift(1)*R).sum(1).rolling(60).std().loc[me]*np.sqrt(ANN)     # trailing book vol at rebalance
    W=W.mul(0.10/pv,axis=0);N=W.reindex(C.index).ffill()                               # decided at month-end close, held from next day
    return md.pnl(N,R,R*0,cost_bp=5.0)

def overnight(O,C,t,c):
    on=np.log(O[t]/C[t].shift(1));return (on-2*c*1e-4).dropna()

def tom(C,t):
    c=C[t].dropna();R=np.log(c).diff();d=pd.Series(c.index,index=c.index)
    k=d.groupby([d.index.year,d.index.month]).cumcount();last=d.groupby([d.index.year,d.index.month]).transform('max')==d
    inwin=pd.Series(last.values|(k.values<3),index=c.index)                              # last day of month + first 3 days
    N=inwin.astype(float).shift(-1).fillna(0)                                            # N row t = held during t+1
    return md.pnl(N.to_frame(t),R.to_frame(t),R.to_frame(t)*0,cost_bp=5.0)

if __name__=='__main__':
    Y={t:md.yahoo_ohlc(t) for t in T};C=pd.DataFrame({t:Y[t].close for t in T}).sort_index();O=pd.DataFrame({t:Y[t].open for t in T}).reindex(C.index)
    C=C[C.index>='2005-01-01'];O=O.reindex(C.index);rows=[];S={}
    r=tsmom(C)[START:];S['tsmom']=r;rows.append(row('(a) TSMOM 12-1, 15 ETF',r))
    for t in ('SPY','QQQ','IWM','GLD'):
        R=np.log(C[t]).diff()[START:];rows.append(row(f'buy&hold {t}',R.dropna()))
        rows.append(row(f'(b) intraday open->close {t}',(np.log(C[t]/O[t]))[START:].dropna(),cost_bp_side=0))
        for c in (0,1,2,5):
            r=overnight(O,C,t,c)[START:];rows.append(row(f'(b) overnight {t}',r,cost_bp_side=c))
            if c==2:S[f'overnight {t}']=r
    for t in ('SPY','QQQ','IWM'):r=tom(C,t)[START:];S[f'tom {t}']=r;rows.append(row(f'(c) turn of month {t}',r))
    # (d) combo with crypto trend on ETF dates
    P=pd.DataFrame({s:md.klines(s,'1d').close for s in trend.COINS}).dropna();F=pd.DataFrame({s:trend.daily_funding(s) for s in trend.COINS})
    cr,_=trend.backtest(P,F);cr=tf.to_dates(cr,C.index);cr=cr[cr.index>='2018-01-01'];rows.append(row('crypto trend BTC+ETH (ETF dates)',cr))
    cands={'tsmom':S['tsmom'],'overnight SPY':S['overnight SPY'],'tom SPY':S['tom SPY']};combo={}
    for k,x in cands.items():
        j=pd.concat([cr,x.reindex(cr.index).fillna(0)],axis=1);pre=j[:'2023-12-31'];w=1/pre.std();c=(j*w).sum(1)*(cr[:'2023-12-31'].std()/((pre*w).sum(1).std()))
        combo[k]=c;rows.append(row(f'(d) crypto trend + {k} (50/50 risk)',c,corr_pre24=round(pre.corr().iat[0,1],2)))
    best=max(combo,key=lambda k:md.stats(combo[k][:'2023-12-31'],ANN)['sharpe'])
    D=pd.DataFrame(rows);D['selected_combo']=D.strategy==f'(d) crypto trend + {best} (50/50 risk)';D.to_csv('docs/ml/results/x1.csv',index=False)
    pd.set_option('display.width',300);pd.set_option('display.max_columns',50);print(D.drop(columns=[c for c in D.columns if c.isdigit()]).to_string(index=False));print('best combo',best)
