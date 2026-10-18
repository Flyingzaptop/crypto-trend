"""Funding carry (long spot + short perp, BTC/ETH 50/50) with different on/off rules. usage: python carry.py"""
import numpy as np,pandas as pd
import trend,strategies as st

def load():
    D=st.bars('1D');D=D[D.index<pd.Timestamp.now().floor('D')]
    F=pd.DataFrame({s:trend.daily_funding(s) for s in st.C}).reindex(D.index).fillna(0)
    Pp=pd.DataFrame({s:st.perp_daily(s) for s in st.C}).reindex(D.index);return D,F,Pp

def hysteresis(f30,f7):
    on=pd.DataFrame(0.,index=f30.index,columns=f30.columns)
    for c in f30:
        x=0.
        for i,(a,b) in enumerate(zip(f30[c],f7[c])):
            if x==0 and a>0:x=1.
            elif x==1 and a<0 and b<0:x=0.
            on.iat[i,on.columns.get_loc(c)]=x
    return on

def carry(D,F,Pp,on):
    R=np.log(D).diff();Rp=np.log(Pp).diff();on=on.where(Pp.notna()).shift(1)       # decided with data up to yesterday
    r=((R-Rp+F)*on-on.fillna(0).diff().abs()*2*st.COST)*0.5
    return r.sum(1,min_count=1).dropna().loc['2020-01-02':],int((on.fillna(0).diff().abs()>0).sum().sum())

if __name__=='__main__':
    D,F,Pp=load();f7,f30=F.rolling(7).mean(),F.rolling(30).mean()
    rules={'0 7d>0 (original)':(f7>0).astype(float),'a always on':pd.DataFrame(1.,index=F.index,columns=F.columns),
           'b 30d>0':(f30>0).astype(float),'c hysteresis 30d/7d':hysteresis(f30,f7)}
    rows=[]
    for k,on in rules.items():
        r,n=carry(D,F,Pp,on);row={'rule':k,'switches':n}
        for y in range(2020,2027):row[str(y)]=round((np.exp(r[r.index.year==y].sum())-1)*100,1)
        for lo,hi in (('2020','2023'),('2024','2026')):
            q=r.loc[lo:hi];row[f'sharpe_{lo}-{hi[2:]}']=round(q.mean()/q.std()*np.sqrt(365),2);row[f'cagr_{lo}-{hi[2:]}']=round((np.exp(q.mean()*365)-1)*100,1)
        row['max_dd']=round((1-np.exp(-(r.cumsum().cummax()-r.cumsum()).max()))*100,2);rows.append(row)
    T=pd.DataFrame(rows);T.to_csv('docs/results/carry.csv',index=False);print(T.T.to_string())
