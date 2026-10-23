"""PNL1-M1 sweep driver on one GPU: K worker processes share jobs.txt (one 'MARKET COST LOSS MODEL SEED' per line); done jobs are skipped.
usage: python pnl_m1_gpu.py jobs.txt --workers 6"""
import os,sys,argparse,subprocess
RUNS='docs/ml/results/pnl_m1_runs'
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('jobs');a.add_argument('--workers',type=int,default=6);a.add_argument('--worker',type=int,default=None);a=a.parse_args()
    J=[l.split() for l in open(a.jobs) if l.strip()]
    if a.worker is None:
        ps=[subprocess.Popen([sys.executable,'-W','ignore',__file__,a.jobs,'--workers',str(a.workers),'--worker',str(i)],stdout=open(f'pnl_w{i}.log','w'),stderr=subprocess.STDOUT) for i in range(a.workers)]
        [p.wait() for p in ps];print('all done')
    else:
        for m,cb,loss,kind,seed in J[a.worker::a.workers]:
            if os.path.exists(f'{RUNS}/{m}_{float(cb):g}bp_{loss}_{kind}_s{seed}.json'):continue
            subprocess.run([sys.executable,'-W','ignore','pnl_m1.py','run',m,cb,loss,kind,seed])
