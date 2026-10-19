"""Same trend rule on 19 traditional-market ETFs (docs/PROTOCOL_tradfi.md). usage: python tradfi.py
Prices: Yahoo Finance adjusted closes, cached in data/tradfi/{ticker}.csv."""
import os,json,urllib.request,numpy as np,pandas as pd
import trend,strategies as st,stable

TICKERS=['SPY','QQQ','IWM','EFA','EEM','EWJ','TLT','IEF','LQD','TIP','GLD','SLV','USO','DBA','DBC','UUP','FXE','FXY','VNQ']
ANN=252;CACHE=f'{st.data_m1.ROOT}/tradfi'

def yahoo(t):
    path=f'{CACHE}/{t}.csv'
    if os.path.exists(path):return pd.read_csv(path,index_col=0,parse_dates=True).iloc[:,0]
    req=urllib.request.Request(f'https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1=0&period2=9999999999&interval=1d&events=div,split',headers={'User-Agent':'Mozilla/5.0'})
    r=json.loads(urllib.request.urlopen(req,timeout=60).read())['chart']['result'][0]
    s=pd.Series(r['indicators']['adjclose'][0]['adjclose'],index=pd.to_datetime(r['timestamp'],unit='s').normalize(),name=t).dropna()
    s=s[~s.index.duplicated(keep='last')];os.makedirs(CACHE,exist_ok=True);s.to_csv(path);return s

def sharpe(r,ann):return r.mean()/r.std()*np.sqrt(ann)

def to_dates(r,idx):
    """Sum crypto daily log returns onto the next ETF trading date (weekends roll into Monday)."""
    pos=idx.searchsorted(r.index);ok=pos<len(idx);return r[ok].groupby(idx[pos[ok]]).sum()

if __name__=='__main__':
    P=pd.DataFrame({t:yahoo(t) for t in TICKERS}).sort_index();P=P[P.index<pd.Timestamp.now().normalize()]
    print('first dates:',P.apply(lambda s:s.first_valid_index().date()).to_dict())
    R=np.log(P).diff();S=sum(np.sign(P-P.rolling(n).mean()) for n in trend.SMAS)/3
    Z=pd.DataFrame(0.,index=P.index,columns=P.columns)
    D=st.bars('1D');D=D[D.index<pd.Timestamp.now().floor('D')];F=pd.DataFrame({s:trend.daily_funding(s) for s in st.C}).reindex(D.index).fillna(0)
    crypto,_=trend.backtest(D,F);cr=to_dates(crypto,P.index)
    rows=[];res={}
    for cap in (1,2):
        for cost in (5,10):
            st.COST=cost*1e-4;r=st.pnl(st.book(S,R,ANN,cap=cap),R,Z);r=r[r.index>='2008-01-01'];res[(cap,cost)]=r
            ys=r.groupby(r.index.year).apply(lambda q:sharpe(q,ANN))
            j=pd.concat([r,cr],axis=1).dropna();j=j[j.index>='2018-01-01'];cb=(j/j.std()).mean(1)
            row={'cap':cap,'cost_bp':cost,'cagr_pct':round((np.exp(r.mean()*ANN)-1)*100,1),'vol_pct':round(r.std()*np.sqrt(ANN)*100,1),
                 'sharpe':round(sharpe(r,ANN),2),'max_dd_pct':round((1-np.exp(-(r.cumsum().cummax()-r.cumsum()).max()))*100,1),
                 'pos_years_pct':round((ys>0).mean()*100),'slope_per_yr':round(np.polyfit(ys.index,ys.values,1)[0],3),
                 'corr_crypto':round(j.corr().iat[0,1],2)}
            for lo in ('2018','2024'):
                q=lambda x:x[x.index>=lo];row[f'crypto_{lo}+']=round(sharpe(q(j.iloc[:,1]),ANN),2);row[f'combo_{lo}+']=round(sharpe(q(cb),ANN),2)
            row.update({str(y):round(v,2) for y,v in ys.items()});rows.append(row)
    T=pd.DataFrame(rows);T.to_csv('docs/results/tradfi.csv',index=False);print(T.T.to_string())
    r=res[(1,5)];j=pd.concat([r,cr],axis=1).dropna();j=j[j.index>='2018-01-01'];cb=(j/j.std()).mean(1)*(0.2/np.sqrt(ANN))/((j/j.std()).mean(1)).std()
    for lo in ('2018','2024'):
        q=lambda x:x[x.index>=lo];c=q(cb);k=q(j.iloc[:,1]);k=k*(0.2/np.sqrt(ANN))/k.std()
        dd=lambda x:round((1-np.exp(-(x.cumsum().cummax()-x.cumsum()).max()))*100,1)
        print(lo,'crypto alone (20% vol): CAGR',round((np.exp(k.mean()*ANN)-1)*100,1),'DD',dd(k),'| combo: CAGR',round((np.exp(c.mean()*ANN)-1)*100,1),'DD',dd(c))
