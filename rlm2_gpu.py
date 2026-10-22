"""RLM2 sweep driver on one GPU: K worker processes, each runs its share of `rl_gpu.py run ...` jobs one after another.
usage: python rlm2_gpu.py --markets BTCUSDT XAUUSD --workers 8 --steps 26214400"""
import os,sys,argparse,subprocess,itertools
RUNS='docs/ml/results/rlm2_runs'

def jobs(markets):
    J=[]
    for m in markets:
        for sp,d,w,s in itertools.product(('delta','abs'),(1,2,4),(16,64,256),(0,1)):J.append((m,'C',w,d,0.01,0.0,s,sp))
    return J

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--markets',nargs='+',required=True);a.add_argument('--workers',type=int,default=8);a.add_argument('--steps',type=int,required=True)
    a.add_argument('--worker',type=int,default=None);a=a.parse_args();J=jobs(a.markets)
    if a.worker is None:
        ps=[subprocess.Popen([sys.executable,'-W','ignore',__file__,'--markets',*a.markets,'--workers',str(a.workers),'--steps',str(a.steps),'--worker',str(i)],
                             stdout=open(f'rlm2_w{i}.log','w'),stderr=subprocess.STDOUT) for i in range(a.workers)]
        [p.wait() for p in ps];print('all done')
    else:
        for m,v,w,d,ent,wd,s,sp in J[a.worker::a.workers]:
            tg=f'{m}_{v}_{sp}_w{w}_d{d}_ent{ent:g}_wd{wd:g}_s{s}'
            if os.path.exists(f'{RUNS}/{tg}.json'):continue
            subprocess.run([sys.executable,'-W','ignore','rl_gpu.py','run',m,v,str(w),str(d),str(ent),str(wd),str(s),sp,'--steps',str(a.steps)])
