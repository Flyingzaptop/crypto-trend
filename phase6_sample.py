"""Phase 6 sample per coordinator: up to 400 resolved markets per category, evenly over quarters 2024-2026."""
import os,numpy as np,pandas as pd
from concurrent.futures import ThreadPoolExecutor
import phase6 as p6,phase6_data as d6

if __name__=='__main__':
    M=pd.read_parquet(f'{d6.ROOT}/markets.parquet').drop_duplicates('market_id');M['kind']=M.question.map(p6.kind)
    M['asset']=M.apply(p6.asset,axis=1);M['end']=pd.to_datetime(M.end,utc=True,format='ISO8601')
    crypto=M.tag.isin(['bitcoin','ethereum']);daily_ud=(M.kind=='updown')&~M.event.str.contains(r'[AP]M',regex=True)
    cats={'BTC above':crypto&(M.asset=='BTC')&(M.kind=='above'),'ETH above':crypto&(M.asset=='ETH')&(M.kind=='above'),
          'BTC daily up/down':(M.asset=='BTC')&daily_ud,'ETH daily up/down':(M.asset=='ETH')&daily_ud,
          'BTC/ETH touch':crypto&M.asset.isin(['BTC','ETH'])&(M.kind=='touch'),
          'S&P 500':M.tag=='sp-500','gold':M.tag=='gold','macro (CPI, Fed)':M.tag.isin(['cpi','fed-rates'])}
    have=lambda t:os.path.exists(f'{d6.ROOT}/h/{t[:40]}.json');parts=[]
    for c,m in cats.items():
        g=M[m].copy();g['cat']=c;g['q']=g.end.dt.to_period('Q');g['have']=g.yes_token.map(have);per=int(np.ceil(400/max(g.q.nunique(),1)));sel=[]
        for q,x in g.groupby('q'):
            h=x[x.have];n=x[~x.have];sel.append(pd.concat([h.head(per),n.sample(max(0,min(per-len(h),len(n))),random_state=0)]))
        s=pd.concat(sel).head(400);parts.append(s);print(c,len(g),'->',len(s),'already downloaded',int(s.have.sum()),flush=True)
    S=pd.concat(parts).drop_duplicates('market_id').drop(columns=['q','have']);S.to_parquet(f'{d6.ROOT}/sample.parquet')
    def one(r):
        try:d6.history(r.yes_token,r.end.isoformat());return 1
        except Exception:return 0
    with ThreadPoolExecutor(8) as ex:print(len(S),'markets; histories ok',sum(ex.map(one,S.itertuples())))
