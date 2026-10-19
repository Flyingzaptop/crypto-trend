"""Data for the volatility study (docs/PROTOCOL_vol.md): DVOL from Deribit, 5-minute realized variance, features.
Every row t is a decision at the close of UTC day t (00:00 of day t+1)."""
import os,json,urllib.request,numpy as np,pandas as pd
import trend,data_m1

COINS={'BTC':'BTCUSDT','ETH':'ETHUSDT'};CACHE=f'{data_m1.ROOT}/vol'

def dvol(cur):
    """Daily DVOL close (vol points), index = UTC day of the candle."""
    path=f'{CACHE}/dvol_{cur}.csv'
    if os.path.exists(path):return pd.read_csv(path,index_col=0,parse_dates=True).iloc[:,0]
    rows,end=[],int(pd.Timestamp.now(tz='UTC').timestamp()*1000)
    while True:
        u=f'https://www.deribit.com/api/v2/public/get_volatility_index_data?currency={cur}&start_timestamp=1609459200000&end_timestamp={end}&resolution=1D'
        r=json.loads(urllib.request.urlopen(u,timeout=60).read())['result'];rows+=r['data']
        if not r.get('continuation'):break
        end=r['continuation']
    s=pd.Series({pd.to_datetime(x[0],unit='ms'):x[4] for x in rows},name=cur).sort_index()
    s=s[s.index<pd.Timestamp.now().floor('D')];os.makedirs(CACHE,exist_ok=True);s.to_csv(path);return s

def daily_bars(sym):
    """Per UTC day: 5-minute realized variance, bipower variation, max |5m return|, close, quote volume."""
    path=f'{CACHE}/rv_{sym}.parquet'
    if os.path.exists(path):return pd.read_parquet(path)
    m=data_m1.load(sym);m.index=m.index.floor('min');m=m[~m.index.duplicated()]
    c=m.close.resample('5min').last().ffill();r=np.log(c).diff().dropna()
    g=r.groupby(r.index.floor('D'))
    d=pd.DataFrame({'rv':g.apply(lambda x:(x**2).sum()),
                    'bv':g.apply(lambda x:np.pi/2*(x.abs()*x.abs().shift(1)).sum()),
                    'jmax':g.apply(lambda x:x.abs().max()),
                    'close':m.close.resample('1D').last(),'qvol':m.quote_volume.resample('1D').sum()}).dropna()
    d=d[d.index<pd.Timestamp.now().floor('D')];os.makedirs(CACHE,exist_ok=True);d.to_parquet(path);return d

def rv(var,h):
    """Annualized vol from daily variances over the trailing h days (inclusive of t)."""
    return np.sqrt(var.rolling(h).sum()*365/h)

def panel():
    out=[]
    for cur,sym in COINS.items():
        d=daily_bars(sym);v=d.rv;D=dvol(cur).reindex(d.index)/100;f=trend.daily_funding(sym).reindex(d.index).fillna(0)
        x=pd.DataFrame(index=d.index)
        x['coin']=cur;x['close']=d.close;x['dvol']=D
        for h in (1,7,30):x[f'rv{h}']=rv(v,h)
        for h in (7,30):x[f'rv_fwd{h}']=rv(v,h).shift(-h)
        x['ret_fwd7']=np.log(d.close).shift(-7)-np.log(d.close)
        x['f_lrv1'],x['f_lrv7'],x['f_lrv30']=np.log(x.rv1),np.log(x.rv7),np.log(x.rv30)
        x['f_jmax']=d.jmax;x['f_bvratio']=d.bv.rolling(7).sum()/v.rolling(7).sum()
        x['f_ldvol']=np.log(D);x['f_ddvol1']=np.log(D).diff(1);x['f_ddvol7']=np.log(D).diff(7);x['f_iv_rv30']=np.log(D)-np.log(x.rv30)
        x['f_fund7']=f.rolling(7).sum();x['f_lvol']=np.log(d.qvol)-np.log(d.qvol.rolling(30).mean())
        for h in (1,7,30):x[f'f_ret{h}']=np.log(d.close).diff(h)
        x['f_eth']=float(cur=='ETH');out.append(x)
    return pd.concat(out)

FEATURES=['f_lrv1','f_lrv7','f_lrv30','f_jmax','f_bvratio','f_ldvol','f_ddvol1','f_ddvol7','f_iv_rv30','f_fund7','f_lvol','f_ret1','f_ret7','f_ret30','f_eth']
