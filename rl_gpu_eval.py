"""Stochastic re-evaluation of saved RLM2 policies (PROTOCOL_RLM2.md, addendum 6): 5 sampled repeats on train and validation 2023,
mean metrics, and mean action probabilities on validation when flat / in a position. Writes <tag>_stoch.json next to each run.
usage: python rl_gpu_eval.py [--workers 8] [--worker i]"""
import os,sys,glob,json,argparse,subprocess,numpy as np,torch
import rlm1,rl_gpu

KEYS=('sharpe','alpha_sharpe','trades_per_day','time_in_market','corr_trend_sign','corr_trend_size','beta','cagr_pct','max_dd_pct')

def one(path,dev):
    j=json.load(open(path));m=j['market'];V=rlm1.VARS[j['variant']];D=rlm1.Data(m,V['daily'],rl_gpu.MKT[m]['train']);T=rl_gpu.TData(D,dev)
    cost=rl_gpu.MKT[m]['cost']*1e-4;tp=rlm1.trend_pos(m) if m!='XAUUSD' else __import__('rlm2').trend_pos(m)
    ag=rl_gpu.Agent(T.dim,j['width'],j['depth'],j.get('space','abs')).to(dev);ag.load_state_dict(torch.load(path.replace('.json','_policy.pt'),map_location=dev))
    out={'run':os.path.basename(path)[:-5]}
    for nm,per in (('val',rl_gpu.VAL),('train',rl_gpu.MKT[m]['train'])):
        R=[];P={} 
        for rep in range(5):
            x=rl_gpu.evaluate(ag,T,per,cost,True,rep,P if (nm=='val' and rep==0) else None);R.append(rlm1.metrics(x,D,tp))
        out[nm]={k:float(np.nanmean([r.get(k,np.nan) for r in R])) for k in KEYS};out[nm+'_reps_alpha']=[r['alpha_sharpe'] for r in R]
        if P:out['val_probs']={s:(P[s+'_sum']/max(P[s+'_n'],1)).cpu().numpy().round(4).tolist() for s in ('flat','in_pos')}|{s+'_share':P[s+'_n']/(P['flat_n']+P['in_pos_n']) for s in ('flat','in_pos')}
    json.dump(out,open(path.replace('.json','_stoch.json'),'w'),indent=1,default=float);return out

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=8);a.add_argument('--worker',type=int,default=None);a.add_argument('--pattern',default='*')
    a=a.parse_args()
    F=sorted(p for p in glob.glob(f'{rl_gpu.RUNS}/{a.pattern}.json') if not p.endswith('_stoch.json'))
    if a.worker is None:
        ps=[subprocess.Popen([sys.executable,'-W','ignore',__file__,'--workers',str(a.workers),'--worker',str(i),'--pattern',a.pattern],stdout=open(f'/tmp/eval_w{i}.log' if not torch.cuda.is_available() else f'eval_w{i}.log','w'),stderr=subprocess.STDOUT) for i in range(a.workers)]
        [p.wait() for p in ps];print('eval done')
    else:
        dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for p in F[a.worker::a.workers]:
            if not os.path.exists(p.replace('.json','_stoch.json')):r=one(p,dev);print(r['run'],r['val']['alpha_sharpe'],flush=True)
