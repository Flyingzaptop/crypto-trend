"""RLM2 report: per-point means over seeds, selection by mean validation alpha-Sharpe (per market), heatmaps depth x width,
params -> val curve, and the ONE test evaluation of the selected point (docs/ml/PROTOCOL_RLM2.md).
usage: python rlm2_report.py [--test] -> docs/ml/results/rlm2_points.csv, rlm2_selected.csv, rlm2_test.csv, docs/ml/img/rlm2_*.png"""
import glob,json,argparse,numpy as np,pandas as pd

R='docs/ml/results';RUNS=f'{R}/rlm2_runs'
TEST={'BTCUSDT':('2024-01-01','2099-01-01'),'ETHUSDT':('2024-01-01','2099-01-01'),'XAUUSD':('2024-01-01','2026-09-30 23:59')}

M=('sharpe','alpha_sharpe','trades_per_day','time_in_market','corr_trend_sign')
def load():
    """One row per run: stochastic (5-repeat) train/val metrics from <tag>_stoch.json (addendum 6), argmax metrics for reference.
    group: 'sweep' (abs, argmax-selected checkpoint), 'delta_se' (delta rerun, stochastic checkpoint),
    'delta_argmax' (first delta batch, 25% checkpoint - excluded from selection), 'diag' (lr 5e-5 point - excluded)."""
    rows=[]
    for p in glob.glob(f'{RUNS}/*.json'):
        if p.endswith('_stoch.json'):continue
        j=json.load(open(p))
        if 'depth' not in j:continue                                   # GPU-engine runs only
        sp=p.replace('.json','_stoch.json');st=json.load(open(sp)) if __import__('os').path.exists(sp) else None
        g='diag' if 'lr5e' in p else 'delta_se' if p.endswith('_se.json') else 'delta_argmax' if j['space']=='delta' else 'sweep'
        r={k:j[k] for k in ('market','variant','space','width','depth','ent','wd','seed','params')}|{'group':g,'file':p}
        r|={f'argmax_val_{k}':j['val'][k] for k in M}|{f'argmax_train_{k}':j['train'][k] for k in M}
        if st:
            r|={f'val_{k}':st['val'][k] for k in M}|{f'train_{k}':st['train'][k] for k in M}
            pr=st.get('val_probs',{});r|={f'p_flat_{i}':v for i,v in enumerate(pr.get('flat',[]))}|{f'p_inpos_{i}':v for i,v in enumerate(pr.get('in_pos',[]))}
        rows.append(r)
    return pd.DataFrame(rows)

def plots(P):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    for m,g in P.groupby('market'):
        sp=sorted(g.space.unique());fig,ax=plt.subplots(2,len(sp),figsize=(4.6*len(sp),7),squeeze=False)
        for c,s in enumerate(sp):
            for r,col in enumerate(('sel_val_alpha','sel_train_alpha')):
                H=g[g.space==s].pivot_table(index='depth',columns='width',values=col);a=ax[r,c];im=a.imshow(H.values,cmap='RdBu',vmin=-max(1,np.nanmax(abs(H.values))),vmax=max(1,np.nanmax(abs(H.values))))
                a.set_xticks(range(len(H.columns)),H.columns);a.set_yticks(range(len(H.index)),H.index);a.set_xlabel('width');a.set_ylabel('depth')
                a.set_title(f'{m} {s} ({"argmax" if s=="abs" else "stochastic"}): {"val 2023" if r==0 else "train"} alpha-Sharpe',fontsize=9)
                for (i,j),v in np.ndenumerate(H.values):a.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8)
        fig.tight_layout();fig.savefig(f'docs/ml/img/rlm2_heatmap_{m}.png',dpi=110);plt.close(fig)
    fig,ax=plt.subplots(1,P.market.nunique(),figsize=(5*P.market.nunique(),3.6),squeeze=False)
    for a,(m,g) in zip(ax[0],P.groupby('market')):
        for s,q in g.groupby('space'):
            q=q.sort_values('params');a.plot(q.params,q.sel_val_alpha,'o',label=f'{s} val');a.plot(q.params,q.sel_train_alpha,'x',label=f'{s} train')
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
            for rep in (range(5) if s.space=='delta' else [0]):     # addendum 7: delta stochastic x5, abs argmax
                x=rl_gpu.evaluate(ag,T,TEST[m],cost,s.space=='delta',rep);x=x[x.index>=TEST[m][0]];r=rlm1.metrics(x,D,tp)
                out.append({'market':m,'point':s.point,'seed':json.load(open(f))['seed'],'rep':rep,**{k:v for k,v in r.items() if k!='by_year_return_pct'},**{f'ret_{y}':v for y,v in r['by_year_return_pct'].items()}})
            print(out[-1],flush=True)
    return pd.DataFrame(out)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--test',action='store_true');a=a.parse_args();X=load()
    key=['market','group','variant','space','depth','width','ent','wd']
    P=X.groupby(key).agg(seeds=('seed','count'),params=('params','first'),**{c:(c,'mean') for c in X.columns if c.startswith(('val_','train_','argmax_','p_'))}).round(3).reset_index()
    P['point']=P.apply(lambda r:f"{r.variant} {r.space} d{r.depth} w{r.width}"+(' lr5e-5' if r.group=='diag' else ''),axis=1);P.to_csv(f'{R}/rlm2_points.csv',index=False)
    P['sel_val_alpha']=np.where(P.space=='abs',P.argmax_val_alpha_sharpe,P.val_alpha_sharpe)        # addendum 7: abs argmax, delta stochastic
    P['sel_train_alpha']=np.where(P.space=='abs',P.argmax_train_alpha_sharpe,P.train_alpha_sharpe)
    E=P[P.group.isin(['sweep','delta_se'])];S=E.loc[E.groupby('market').sel_val_alpha.idxmax()].copy()
    S['files']=S.apply(lambda r:list(X[(X[key]==r[key]).all(1)].file),axis=1);S.drop(columns='files').to_csv(f'{R}/rlm2_selected.csv',index=False)
    pd.set_option('display.width',250);pd.set_option('display.max_columns',30);print(P[['market','group','space','depth','width','params','sel_val_alpha','sel_train_alpha','val_alpha_sharpe','argmax_val_alpha_sharpe','val_trades_per_day','argmax_val_trades_per_day']].to_string(index=False));print('\nselected:\n',S[['market','point','sel_val_alpha','sel_train_alpha']].to_string(index=False))
    import os;os.makedirs('docs/ml/img',exist_ok=True);plots(P[P.group.isin(['sweep','delta_se'])])
    if a.test:test(S).to_csv(f'{R}/rlm2_test.csv',index=False)
