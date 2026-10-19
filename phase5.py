"""Phase 5 history: earnings-day moves, their predictability, and post-earnings drift (docs/vol/PROTOCOL_phase5.md)."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from zoneinfo import ZoneInfo
import phase5_data as d5,phase1 as p1

OUT='docs/vol/results';NY=ZoneInfo('America/New_York')

def events():
    u=d5.universe();spy=d5.prices('SPY');vix=p1.cboe('VIX');sec={k:d5.prices(v).adj for k,v in d5.SECTOR_ETF.items()};rows=[]
    for _,c in u.iterrows():
        try:f=d5.filings(c.cik);px=d5.prices(c.symbol)
        except Exception:continue
        if f.empty or len(px)<300:continue
        idx=px.index;lr=np.log(px.adj).diff();t=pd.to_datetime(f.acceptanceDateTime,utc=True).dt.tz_convert(NY)
        f=f.assign(t=t.values).sort_values('t');last=None
        for ts in f.t:
            if ts.year<2010:continue
            if last is not None and (ts-last).days<10:continue
            last=ts;day=pd.Timestamp(ts.date())
            after=ts.hour+ts.minute/60>=16
            k=idx.searchsorted(day,side='left')                                       # first trading day >= filing date
            if after and k+1<len(idx) and k>=61:
                # filed after the close: the release may have been that morning or after the close -> pick the day with the
                # larger volume spike (volume is not the predicted quantity, unlike |move|)
                med=px.volume.iloc[k-61:k-1].median()
                if idx[k]>day or px.volume.iloc[k+1]/med>px.volume.iloc[k]/med:k=k+1 if idx[k]==day else k
            if k<62 or k+61>len(idx):continue
            d=idx[k]
            mv=lr.iloc[k];om=np.log(px.adj_open.iloc[k]/px.adj.iloc[k-1]);sig=lr.iloc[k-61:k-1].std()
            sp=np.log(spy.adj.reindex(idx)).diff()
            fut={h:(np.log(px.adj.iloc[k+h]/px.adj.iloc[k])-np.log(spy.adj.reindex(idx).iloc[k+h]/spy.adj.reindex(idx).iloc[k])) for h in (5,20,60)}
            se=sec.get(c.sector)
            futs={h:(np.log(px.adj.iloc[k+h]/px.adj.iloc[k])-np.log(se.reindex(idx).iloc[k+h]/se.reindex(idx).iloc[k])) if se is not None else np.nan for h in (5,20,60)}
            rows.append({'symbol':c.symbol,'sector':c.sector,'accept_et':ts.strftime('%Y-%m-%d %H:%M'),'timing':('AMC' if px.index[k]>day else 'AMC-sameday') if after else ('BMO' if ts.hour+ts.minute/60<9.5 else 'DURING'),
                         'react':d,'move':mv,'open_move':om,'sigma60':sig,'z':abs(mv)/sig,'mkt':sp.iloc[k],'vix':vix.asof(idx[k-1])/100,
                         'dollar_vol60':float((px.close*px.volume).iloc[k-61:k-1].mean()),**{f'car{h}':v for h,v in fut.items()},**{f'sar{h}':v for h,v in futs.items()}})
    E=pd.DataFrame(rows).sort_values('react').reset_index(drop=True);E['abs']=E.move.abs()
    g=E.groupby('symbol').abs
    E['past4']=g.transform(lambda s:s.shift(1).rolling(4,min_periods=4).mean());E['past8']=g.transform(lambda s:s.shift(1).rolling(8,min_periods=8).mean())
    E['year']=E.react.dt.year;return E

def predict(E):
    X=E.dropna(subset=['past4','past8','sigma60','vix']).copy();X['ldv']=np.log(X.dollar_vol60)
    S=pd.get_dummies(X.sector,prefix='s',dtype=float);cols=['past4','past8','sigma60','vix','ldv']
    Z=pd.concat([X[cols],S],axis=1);Z.insert(0,'c',1.0);X['lin']=np.nan
    for Y in range(2013,2027):
        tr=(X.year<Y).values;te=(X.year==Y).values
        if te.sum()==0:continue
        b=np.linalg.lstsq(Z.values[tr],X['abs'].values[tr],rcond=None)[0];X.loc[te,'lin']=Z.values[te]@b
    rows=[]
    for Y,g in X[X.year>=2013].groupby('year'):
        for f in ('past4','past8','sigma60','vix','lin'):
            r2=1-((g['abs']-g[f])**2).sum()/((g['abs']-g['abs'].mean())**2).sum() if f=='lin' else np.nan
            rows.append({'year':Y,'forecast':f,'ic':round(spearmanr(g[f],g['abs']).statistic,3),'r2':round(r2,3) if f=='lin' else np.nan,'n':len(g)})
    return X,pd.DataFrame(rows)

def pead(E):
    E=E.copy();E['s']=E.move-E.mkt;E['q']=E.groupby(E.react.dt.to_period('Q')).s.transform(lambda v:pd.qcut(v.rank(method='first'),5,labels=False)+1)
    rows=[]
    for seg,m in (('2010-23',E.year<=2023),('2024-26',E.year>=2024)):
        for base in ('car','sar'):
            for h in (5,20,60):
                z=E[m].dropna(subset=[f'{base}{h}'])
                qm=z.groupby('q')[f'{base}{h}'].mean()*100
                per=z.groupby(z.react.dt.to_period('Q')).apply(lambda g:g.loc[g.q==5,f'{base}{h}'].mean()-g.loc[g.q==1,f'{base}{h}'].mean()).dropna()
                ls=per.mean()*100;t=per.mean()/per.std()*np.sqrt(len(per)) if len(per)>2 else np.nan
                rows.append({'seg':seg,'vs':'SPY' if base=='car' else 'sector','h':h,'n_events':len(z),**{f'Q{int(k)}_pct':round(v,2) for k,v in qm.items()},
                             'Q5-Q1_pct':round(ls,2),'t_quarters':round(t,2),'Q5-Q1_net02_pct':round(ls-0.2,2),'Q5-Q1_net04_pct':round(ls-0.4,2)})
    return pd.DataFrame(rows)

if __name__=='__main__':
    pd.set_option('display.width',250)
    E=events();E.to_csv(f'{OUT}/phase5_events.csv.gz',index=False)
    print(len(E),'events',E.symbol.nunique(),'companies',E.timing.value_counts().to_dict())
    A=E.groupby('year').agg(n=('abs','size'),median_abs_pct=('abs',lambda v:round(v.median()*100,2)),mean_abs_pct=('abs',lambda v:round(v.mean()*100,2)),
                            median_z=('z',lambda v:round(v.median(),2)),share_z_gt3=('z',lambda v:round((v>3).mean()*100,1)),
                            open_share=('open_move',lambda v:np.nan)).reset_index()
    A['open_share_of_move']=E.assign(r=(E.open_move.abs()/E['abs']).clip(0,3)).groupby('year').r.median().round(2).values
    A=A.drop(columns='open_share');A.to_csv(f'{OUT}/phase5_moves_by_year.csv',index=False);print(A.to_string(index=False))
    X,IC=predict(E);IC.to_csv(f'{OUT}/phase5_predictability.csv',index=False)
    s=IC.assign(seg=np.where(IC.year<=2023,'2013-23','2024-26')).groupby(['seg','forecast'])[['ic','r2']].mean().round(3);print(s.to_string())
    P=pead(E);P.to_csv(f'{OUT}/phase5_pead.csv',index=False);print(P.to_string(index=False))
