"""After the M1 sweep (on the VM): select per market by mean validation Sharpe over seeds (addendum 3 grid), then test (once),
placebo (1 seed) and walk-forward (1 seed) for the selected points, run in parallel. Writes POST_FINISHED.
usage: python pnl_m1_post.py --workers 6"""
import glob,json,sys,argparse,subprocess,pandas as pd
RUNS='docs/ml/results/pnl_m1_runs';WF={'BTCUSDT':range(2021,2027),'ETHUSDT':range(2021,2027),'XAUUSD':range(2019,2027)}
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=6);a=a.parse_args()
    R=pd.DataFrame([json.load(open(p)) for p in glob.glob(f'{RUNS}/*_s[0-9].json')]);R=R[R.model.isin(['gru64','cnn'])&R.seed.isin([0,1])]
    R['val_sharpe']=R.val.apply(lambda v:v['sharpe']);P=R.groupby(['market','cost_bp','loss','model']).agg(n=('seed','count'),val=('val_sharpe','mean')).reset_index()
    P=P[P.n==2];S=P.loc[P.groupby(['market','cost_bp']).val.idxmax()];S.to_csv(f'{RUNS}/SELECTED.csv',index=False);print(S,flush=True)
    cmds=[]
    for _,s in S.iterrows():
        b=[sys.executable,'-W','ignore','pnl_m1.py'];arg=[s.market,f'{s.cost_bp:g}',s.loss,s.model]
        cmds.append(b+['test',*arg,'0,1']);cmds.append(b+['placebo',*arg,'0'])
        cmds+=[b+['wf',*arg,'0',str(y)] for y in WF[s.market]]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(a.workers) as ex:list(ex.map(lambda c:subprocess.run(c,stdout=open('post.log','a'),stderr=subprocess.STDOUT),cmds))
    open('POST_FINISHED','w').write('ok')
