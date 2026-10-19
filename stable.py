"""Candidates with a time-stable edge (docs/PROTOCOL_strategies.md, addendum 2). usage: python stable.py"""
import numpy as np,pandas as pd
import trend,strategies as st,archive_d1

def yearly_sharpe(r):return r.groupby(r.index.year).apply(lambda q:q.mean()/q.std()*np.sqrt(365))

def report(name,r,base):
    r=r[r.index>='2018-01-01'];ys=yearly_sharpe(r);ys=ys[ys.index<=2026];cb=st.combine(base,r)
    j=pd.concat([r,base],axis=1).dropna();q=lambda x:x[x.index>='2024-01-01']
    return {'strategy':name,**{str(y):round(v,2) for y,v in ys.items()},'pos_years_pct':round((ys>0).mean()*100),
            'slope_per_yr':round(np.polyfit(ys.index,ys.values,1)[0],3),'sharpe_all':st.stats(r)['sharpe'],
            'corr':round(j.corr().iat[0,1],2),'combo_24-26':st.stats(q(cb))['sharpe'],'trend_24-26':st.stats(q(base.reindex(cb.index)))['sharpe']}

if __name__=='__main__':
    D=st.bars('1D');D=D[D.index<pd.Timestamp.now().floor('D')];R=np.log(D).diff()
    F=pd.DataFrame({s:trend.daily_funding(s) for s in st.C}).reindex(D.index).fillna(0)
    base,Nb=trend.backtest(D,F);rows=[report('0 trend BTC+ETH (current)',base,base)]

    # 1 ETH/BTC ratio trend: position x in the ratio = +x ETH, -x BTC
    q=(D.ETHUSDT/D.BTCUSDT).to_frame('ETHBTC');Rq=np.log(q).diff()
    S=sum(np.sign(q-q.rolling(n).mean()) for n in trend.SMAS)/3;N=st.book(S,Rq,365)
    Fq=(F.ETHUSDT-F.BTCUSDT).to_frame('ETHBTC');N0=N.shift(1)
    r1=(N0*Rq).sum(1)-(N0*Fq).sum(1)-N0.diff().abs().sum(1)*2*st.COST;rows.append(report('1 ETH/BTC trend',r1[N0.notna().any(axis=1)],base))

    # 2 Donchian 50-day breakout, state held between signals
    hi,lo=D.rolling(50).max().shift(1),D.rolling(50).min().shift(1)
    S2=pd.DataFrame(np.where(D>hi,1.,np.where(D<lo,-1.,np.nan)),index=D.index,columns=D.columns).ffill()
    rows.append(report('2 Donchian 50d BTC+ETH',st.pnl(st.book(S2,R,365),R,F),base))

    # 3 gold (PAXG spot; no perp funding before 2025 -> 0)
    g=pd.read_parquet(f'{archive_d1.OUT}/PAXGUSDT.parquet').close.rename('PAXGUSDT').to_frame();g=g[g.index>='2019-10-01']
    rg,_=trend.backtest(g);rows.append(report('3 trend gold (PAXG)',rg,base))

    # 4 long-only trend
    N4=trend.target_positions(D).clip(lower=0);rows.append(report('4 trend long-only BTC+ETH',st.pnl(N4,R,F),base))

    T=pd.DataFrame(rows);T.to_csv('docs/results/stable.csv',index=False);print(T.T.to_string())
