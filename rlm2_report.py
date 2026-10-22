"""RLM2 report: per-point means over seeds, selection by mean validation alpha-Sharpe (per market), heatmaps depth x width,
params -> val curve, and the ONE test evaluation of the selected point (docs/ml/PROTOCOL_RLM2.md).
usage: python rlm2_report.py [--test] -> docs/ml/results/rlm2_points.csv, rlm2_selected.csv, rlm2_test.csv, docs/ml/img/rlm2_*.png"""
import glob,json,argparse,numpy as np,pandas as pd

R='docs/ml/results';RUNS=f'{R}/rlm2_runs'
TEST={'BTCUSDT':('2024-01-01','2099-01-01'),'ETHUSDT':('2024-01-01','2099-01-01'),'XAUUSD':('2024-01-01','2026-09-30 23:59')}

def load():
    rows=[]
    for p in glob.glob(f'{RUNS}/*.json'):
        j=json.load(open(p))
        if 'depth' not in j:continue                                   # GPU-engine runs only
        rows.append({k:j[k] for k in ('market','variant','space','width','depth','ent','wd','seed','params')}|
                    {f'val_{k}':j['val'][k] for k in ('sharpe','alpha_sharpe','trades_per_day','time_in_market','corr_trend_sign')}|
                    {f'train_{k}':j['train'][k] for k in ('sharpe','alpha_sharpe','trades_per_day','time_in_market','corr_trend_sign')}|{'file':p})
    return pd.DataFrame(rows)

def plots(P):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    for m,g in P.groupby('market'):
        sp=sorted(g.space.unique());fig,ax=plt.subplots(2,len(sp),figsize=(4.6*len(sp),7),squeeze=False)
        for c,s in enumerate(sp):
            for r,col in enumerate(('val_alpha_sharpe','train_alpha_sharpe')):
                H=g[g.space==s].pivot_table(index='depth',columns='width',values=col);a=ax[r,c];im=a.imshow(H.values,cmap='RdBu',vmin=-max(1,np.nanmax(abs(H.values))),vmax=max(1,np.nanmax(abs(H.values))))
                a.set_xticks(range(len(H.columns)),H.columns);a.set_yticks(range(len(H.index)),H.index);a.set_xlabel('width');a.set_ylabel('depth')
                a.set_title(f'{m} {s}: {"val 2023" if r==0 else "train"} alpha-Sharpe',fontsize=9)
                for (i,j),v in np.ndenumerate(H.values):a.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8)
        fig.tight_layout();fig.savefig(f'docs/ml/img/rlm2_heatmap_{m}.png',dpi=110);plt.close(fig)
    fig,ax=plt.subplots(1,P.market.nunique(),figsize=(5*P.market.nunique(),3.6),squeeze=False)
    for a,(m,g) in zip(ax[0],P.groupby('market')):
        for s,q in g.groupby('space'):
            q=q.sort_values('params');a.plot(q.params,q.val_alpha_sharpe,'o',label=f'{s} val');a.plot(q.params,q.train_alpha_sharpe,'x',label=f'{s} train')
        a.set_xscale('log');a.axhline(0,color='grey',lw=0.5);a.set_xlabel('parameters (actor+critic)');a.set_ylabel('alpha-Sharpe');a.set_title(m,fontsize=9);a.legend(fontsize=7)
    fig.tight_layout();fig.savefig('docs/ml/img/rlm2_params_vs_val.png',dpi=110);plt.close(fig)

def test(S):
    import torch,rlm1,rl_gpu
    out=[]
    for _,s in S.iterrows():
        m=s.market;V=rlm1.VARS[s.variant];D=rlm1.Data(m,V['daily'],rl_gpu.MKT[m]['train']);T=rl_gpu.TData(D,torch.device('cpu'));cost=rl_gpu.MKT[m]['cost']*1e-4
        tp=rlm1.trend_pos(m) if m!='XAUUSD' else __import__('rlm2').trend_pos(m)
        for f in s.files:
            ag=rl_gpu.Agent(T.dim,int(s.width),int(s.depth),s.space);ag.load_state_dict(torch.load(f.replace('.json','_policy.pt'),map_location='cpu'))
            x=rl_gpu.evaluate(ag,T,TEST[m],cost);x=x[x.index>=TEST[m][0]];r=rlm1.metrics(x,D,tp)
            out.append({'market':m,'point':s.point,'seed':json.load(open(f))['seed'],**{k:v for k,v in r.items() if k!='by_year_return_pct'},**{f'ret_{y}':v for y,v in r['by_year_return_pct'].items()}})
            print(out[-1],flush=True)
    return pd.DataFrame(out)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--test',action='store_true');a=a.parse_args();X=load()
    key=['market','variant','space','depth','width','ent','wd']
    P=X.groupby(key).agg(seeds=('seed','count'),params=('params','first'),**{c:(c,'mean') for c in X.columns if c.startswith(('val_','train_'))}).round(3).reset_index()
    P['point']=P.apply(lambda r:f"{r.variant} {r.space} d{r.depth} w{r.width}",axis=1);P.to_csv(f'{R}/rlm2_points.csv',index=False)
    S=P.loc[P.groupby('market').val_alpha_sharpe.idxmax()].copy()
    S['files']=S.apply(lambda r:list(X[(X[key]==r[key]).all(1)].file),axis=1);S.drop(columns='files').to_csv(f'{R}/rlm2_selected.csv',index=False)
    pd.set_option('display.width',250);pd.set_option('display.max_columns',30);print(P.drop(columns=['ent','wd']).to_string(index=False));print('\nselected:\n',S[['market','point','val_alpha_sharpe','train_alpha_sharpe']].to_string(index=False))
    import os;os.makedirs('docs/ml/img',exist_ok=True);plots(P)
    if a.test:test(S).to_csv(f'{R}/rlm2_test.csv',index=False)
