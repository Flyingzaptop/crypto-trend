"""Offline checks of the position rule (no network): causality, cap, sign logic. run: python -m pytest -q"""
import os,sys,numpy as np,pandas as pd
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trend

def fake(n=600,seed=0):
    rng=np.random.default_rng(seed);idx=pd.date_range('2020-01-01',periods=n)
    return pd.DataFrame({c:100*np.exp(np.cumsum(rng.normal(0.001,0.03,n))) for c in trend.COINS},index=idx)

def test_no_lookahead():
    P=fake();N=trend.target_positions(P);Q=P.copy();Q.iloc[400:]*=np.linspace(1,3,len(Q)-400)[:,None]   # change the future
    M=trend.target_positions(Q);pd.testing.assert_frame_equal(N.iloc[:400],M.iloc[:400])

def test_cap():
    for cap in (0.5,1,2):assert (trend.target_positions(fake(),0.8,cap).abs().sum(1).dropna()<=cap+1e-9).all()

def test_uptrend_is_long():
    idx=pd.date_range('2020-01-01',periods=400);P=pd.DataFrame({c:np.exp(np.linspace(0,2,400))*(1+0.01*np.sin(np.arange(400))) for c in trend.COINS},index=idx)
    assert (trend.target_positions(P).iloc[-50:]>0).all().all()

def test_backtest_uses_yesterdays_position():
    P=fake();ret,N=trend.backtest(P);R=np.log(P).diff();T=trend.target_positions(P)
    d=ret.index[100];assert abs(ret[d]-((T.shift(1).loc[d]*R.loc[d]).sum()-T.shift(1).diff().abs().loc[d].sum()*1e-3))<1e-12
