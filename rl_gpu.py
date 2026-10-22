"""GPU-native PPO for the minute RL of RLM1/RLM2 (CleanRL style, PyTorch). Same environment rules as rlm1.Book, vectorised
over N environments on tensors; PPO rollout, GAE and updates all on the device.
usage:
  python rl_gpu.py check   [--device cpu]          equivalence vs the numpy Book (BTC and gold, 10k steps)
  python rl_gpu.py bench   [--device cuda]         steps/s on a short run
  python rl_gpu.py run MARKET VARIANT WIDTH DEPTH ENT WD SEED [abs|delta] [--device cuda] -> docs/ml/results/rlm2_runs/<tag>.json (train/val only)"""
import os,sys,json,time,argparse,numpy as np,pandas as pd,torch
import rlm1

WIN,EP,K,TAU,SCALE=rlm1.WIN,rlm1.EP,rlm1.K,rlm1.TAU,rlm1.SCALE
MKT={'BTCUSDT':dict(cost=5,train=('2019-01-01','2022-12-31 23:59')),'ETHUSDT':dict(cost=5,train=('2019-01-01','2022-12-31 23:59')),
     'XAUUSD':dict(cost=1,train=('2017-01-01','2022-12-31 23:59'))}
VAL=('2023-01-01','2023-12-31 23:59');RUNS='docs/ml/results/rlm2_runs'
# PPO settings for every point of the sweep (fixed before the sweep; see docs/ml/STATUS.md)
CFG=dict(steps=6_000_000,n_env=1024,T=256,epochs=10,minibatch=16384,lr=2e-4,clip=0.2,vf_coef=0.5,max_grad=0.5)

class TData:
    """rlm1.Data moved to tensors."""
    def __init__(s,D,dev):
        s.D=D;s.dev=dev;f=lambda a,dt:torch.as_tensor(np.ascontiguousarray(a),dtype=dt,device=dev)
        s.z=f(D.z,torch.float32);s.X=f(D.X,torch.float32);s.r=f(D.r,torch.float64);s.lp=f(D.lp,torch.float64);s.sig=f(D.sig,torch.float64)
        s.rbar=float(D.rbar);s.ar=torch.arange(-WIN+1,1,device=dev);s.n=len(D.r);s.dim=WIN+D.X.shape[1]+5

class TBook:
    """Tensor version of rlm1.Book: same state, same rules."""
    def __init__(s,T,n):s.T=T;s.n=n;dev=T.dev;s.z64=lambda:torch.zeros(n,dtype=torch.float64,device=dev)
    def start(s,t0):
        s.t=torch.as_tensor(t0,dtype=torch.long,device=s.T.dev).clone();s.pos=s.z64();s.held=s.z64();s.flat=s.z64();s.entry=s.z64()
    def obs(s):
        T=s.T;t=s.t;zw=T.z[t[:,None]+T.ar]
        pnl=torch.where(s.pos!=0,s.pos*(T.lp[t]-s.entry)/(T.sig[t]*torch.sqrt(torch.clamp(s.held,min=1))),torch.zeros_like(s.pos))
        pnl=torch.nan_to_num(pnl).clamp(-10,10)
        st=torch.stack([(s.pos==-1).double(),(s.pos==0).double(),(s.pos==1).double(),torch.log1p(s.held)/5,pnl],1)
        return torch.cat([zw,T.X[t],st.float()],1)
    def act(s,a,cost,train):
        T=s.T;new=a.double()-1;d=(new-s.pos).abs();r1=T.r[s.t+1]
        if train:
            s.flat=torch.where(new==0,s.flat+1,torch.zeros_like(s.flat))
            pen=torch.where(new==0,K*(torch.exp(torch.clamp(s.flat/TAU,max=6))-1),torch.zeros_like(s.flat));rew=new*(r1-T.rbar)-cost*d
        else:pen=torch.zeros_like(new);rew=new*r1-cost*d
        enter=(new!=s.pos)&(new!=0);s.entry=torch.where(enter,T.lp[s.t],s.entry)
        s.held=torch.where(new==0,torch.zeros_like(s.held),torch.where(enter,torch.ones_like(s.held),s.held+1))
        s.pos=new;s.t=s.t+1;return rew,pen

def mlp(i,o,w,depth,gain):
    L=[];d=i
    for _ in range(depth):l=torch.nn.Linear(d,w);torch.nn.init.orthogonal_(l.weight,np.sqrt(2));torch.nn.init.zeros_(l.bias);L+=[l,torch.nn.Tanh()];d=w
    l=torch.nn.Linear(d,o);torch.nn.init.orthogonal_(l.weight,gain);torch.nn.init.zeros_(l.bias);return torch.nn.Sequential(*L,l)

NACT={'abs':3,'delta':4}
def to_abs(a,pos,space):
    """Action -> index of the new absolute position {0:-1, 1:0, 2:+1}. 'abs': the action is the position itself.
    'delta': 0 hold, 1 go long, 2 go short, 3 close."""
    if space=='abs':return a
    keep=(pos+1).long();return torch.where(a==0,keep,torch.where(a==1,2,torch.where(a==2,0,torch.ones_like(a))))

class Agent(torch.nn.Module):
    """Separate actor and critic MLPs with tanh, orthogonal init (as SB3 MlpPolicy with net_arch=[w]*depth).
    'delta': the policy's output bias makes P(hold) ~ 0.97 at the start."""
    def __init__(s,dim,w,depth,space='abs'):
        super().__init__();s.space=space;s.pi=mlp(dim,NACT[space],w,depth,0.01);s.v=mlp(dim,1,w,depth,1.0)
        if space=='delta':
            with torch.no_grad():s.pi[-1].bias[0]=float(np.log(0.97/0.01))
    def nparams(s):return sum(p.numel() for p in s.parameters())

def evaluate(ag,T,period,cost,stoch=False,seed=0):
    """As rlm1.evaluate: weekly chunks in one batch, each starts flat and is closed with cost at its end.
    stoch=True samples actions from the policy (fixed seed) instead of argmax - needed for 'delta', where argmax is almost always "hold"."""
    g=torch.Generator(device=T.dev).manual_seed(seed)
    D=T.D;i0,i1=D.span(period);i0=max(i0,D.first);st=np.arange(i0,i1-1,EP);L=torch.as_tensor(np.minimum(EP,i1-1-st),device=T.dev)
    B=TBook(T,len(st));B.start(st);P=torch.zeros(len(st),EP,dtype=torch.float64,device=T.dev);POS=torch.zeros_like(P)
    with torch.no_grad():
        for k in range(int(L.max())):
            live=k<L;lg=ag.pi(B.obs());a=torch.multinomial(torch.softmax(lg.float(),1),1,generator=g)[:,0] if stoch else lg.argmax(1);a=to_abs(a,B.pos,ag.space);a=torch.where(live,a,(1+B.pos).long());r,_=B.act(a,cost,False)
            P[:,k]=torch.where(live,r,torch.zeros_like(r));POS[:,k]=torch.where(live,B.pos,torch.zeros_like(r));B.t=B.t.clamp(max=T.n-2)
    j=torch.arange(len(st),device=T.dev);P[j,L-1]-=POS[j,L-1].abs()*cost
    P,POS,L=P.cpu().numpy(),POS.cpu().numpy(),L.cpu().numpy()
    ts=np.concatenate([D.t[s+1:s+1+l] for s,l in zip(st,L)]);pnl=np.concatenate([P[j,:l] for j,l in enumerate(L)]);pos=np.concatenate([POS[j,:l] for j,l in enumerate(L)])
    return pd.DataFrame({'pnl':pnl,'pos':pos},index=ts)

def train(market,variant,w,depth,ent,wd,seed,dev,cfg=CFG,on_quarter=None,log=None,space='abs',stoch_eval=False):
    """PPO on device. Returns the agent with the best checkpoint by validation alpha-Sharpe and the per-quarter validation metrics."""
    torch.manual_seed(seed);rng=torch.Generator(device=dev).manual_seed(seed);V=rlm1.VARS[variant];gamma,lam=V['gamma'],V['gae']
    D=rlm1.Data(market,V['daily'],MKT[market]['train']);T=TData(D,dev);cost=MKT[market]['cost']*1e-4;pen_on=V['pen']
    i0,i1=D.span(D.train);i0=max(i0,D.first);N,Tn=cfg['n_env'],cfg['T']
    ag=Agent(T.dim,w,depth,space).to(dev);opt=(torch.optim.AdamW(ag.parameters(),lr=cfg['lr'],eps=1e-5,weight_decay=wd) if wd>0 else torch.optim.Adam(ag.parameters(),lr=cfg['lr'],eps=1e-5))
    rs=lambda m:torch.randint(i0,i1-EP-2,(m,),generator=rng,device=dev)
    B=TBook(T,N);B.start(rs(N));k=torch.zeros(N,dtype=torch.long,device=dev);o=B.obs()
    obs=torch.zeros(Tn,N,T.dim,device=dev);act=torch.zeros(Tn,N,dtype=torch.long,device=dev);lpb=torch.zeros(Tn,N,device=dev)
    rew=torch.zeros(Tn,N,device=dev);don=torch.zeros(Tn,N,device=dev);val=torch.zeros(Tn,N,device=dev)
    iters=max(1,cfg['steps']//(N*Tn));quarters={int(round(iters*q/4)) for q in (1,2,3,4)};tp=rlm1.trend_pos(market) if market!='XAUUSD' else None
    if market=='XAUUSD':import rlm2;tp=rlm2.trend_pos(market)
    best,bs,vals,t0=-np.inf,None,[],time.time()
    for it in range(1,iters+1):
        with torch.no_grad():
            for s in range(Tn):
                obs[s]=o;lg=ag.pi(o);dist=torch.distributions.Categorical(logits=lg);a=dist.sample();act[s]=a;lpb[s]=dist.log_prob(a);val[s]=ag.v(o)[:,0]
                r,pen=B.act(to_abs(a,B.pos,space),cost,True);r=(r-(pen if pen_on else 0))*SCALE;k+=1;dn=k>=EP;o=B.obs()
                if dn.any():                                            # time-limit truncation: bootstrap with V(terminal obs), as SB3
                    r=r+gamma*torch.where(dn,ag.v(o)[:,0].double(),torch.zeros_like(r))
                    m=int(dn.sum());B.t[dn]=rs(m);B.pos[dn]=0;B.held[dn]=0;B.flat[dn]=0;B.entry[dn]=0;k[dn]=0;o=B.obs()
                rew[s]=r.float();don[s]=dn.float()
            nv=ag.v(o)[:,0];adv=torch.zeros_like(rew);g=torch.zeros(N,device=dev)
            for s in reversed(range(Tn)):
                nxt=nv if s==Tn-1 else val[s+1];nd=1-don[s];delta=rew[s]+gamma*nxt*nd-val[s];g=delta+gamma*lam*nd*g;adv[s]=g
            ret=adv+val
        bo,ba,bl,badv,bret=obs.reshape(-1,T.dim),act.reshape(-1),lpb.reshape(-1),adv.reshape(-1),ret.reshape(-1);n=bo.shape[0]
        for _ in range(cfg['epochs']):
            perm=torch.randperm(n,device=dev,generator=rng)
            for mb in perm.split(cfg['minibatch']):
                lg=ag.pi(bo[mb]);dist=torch.distributions.Categorical(logits=lg);nl=dist.log_prob(ba[mb]);ratio=(nl-bl[mb]).exp()
                A=badv[mb];A=(A-A.mean())/(A.std()+1e-8)
                pl=torch.max(-A*ratio,-A*ratio.clamp(1-cfg['clip'],1+cfg['clip'])).mean();vl=((ag.v(bo[mb])[:,0]-bret[mb])**2).mean()
                loss=pl+cfg['vf_coef']*vl-ent*dist.entropy().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(ag.parameters(),cfg['max_grad']);opt.step()
        if it in quarters:
            x=rlm1.metrics(evaluate(ag,T,VAL,cost,stoch_eval,seed),D,tp);vals.append({'steps':it*N*Tn,**{q:x[q] for q in ('sharpe','alpha_sharpe','trades_per_day','time_in_market','corr_trend_sign')}})
            if log:print(f'{log} it {it}/{iters} {time.time()-t0:.0f}s val {vals[-1]}',flush=True)
            if x['alpha_sharpe']>best:best=x['alpha_sharpe'];bs={kk:vv.detach().clone() for kk,vv in ag.state_dict().items()};vb=x
    ag.load_state_dict(bs);return ag,T,D,cost,tp,vals,vb,time.time()-t0

def check(dev):
    """Same random action sequence through numpy Book and TBook: obs, rewards, penalties, positions, held, pnl."""
    out={}
    for m in ('BTCUSDT','XAUUSD'):
        D=rlm1.Data(m,True,MKT[m]['train']);T=TData(D,dev);n=8;i0,i1=D.span(D.train);i0=max(i0,D.first)
        t0=np.random.default_rng(0).integers(i0,i1-20000,n);A=np.random.default_rng(1).integers(0,3,(10000,n))
        A[:2000]=np.random.default_rng(2).choice([0,1,1,1,2],(2000,n))               # long flat stretches too (penalty path)
        nb=rlm1.Book(D,n);nb.start(t0);tb=TBook(T,n);tb.start(t0);err={'obs':0.,'rew':0.,'pen':0.,'pos':0.,'held':0.,'entry':0.}
        for k in range(10000):
            err['obs']=max(err['obs'],float(np.abs(nb.obs()-tb.obs().cpu().numpy()).max()))
            r1,p1=nb.act(A[k],5e-4,True);r2,p2=tb.act(torch.as_tensor(A[k],device=dev),5e-4,True)
            err['rew']=max(err['rew'],float(np.abs(r1-r2.cpu().numpy()).max()));err['pen']=max(err['pen'],float(np.abs(np.asarray(p1)-p2.cpu().numpy()).max()))
            for q in ('pos','held','entry'):err[q]=max(err[q],float(np.abs(getattr(nb,q)-getattr(tb,q).cpu().numpy()).max()))
        out[m]={q:float(f'{v:.2e}') for q,v in err.items()};out[m]['pass']=all(v<1e-5 for v in err.values())
        # 'delta' zero strategy: always "hold" from flat == always flat (abs action 1) in the numpy Book
        nb=rlm1.Book(D,n);nb.start(t0);tb=TBook(T,n);tb.start(t0);e=0.
        for k in range(10000):
            e=max(e,float(np.abs(nb.obs()-tb.obs().cpu().numpy()).max()));r1,p1=nb.act(np.ones(n,dtype=int),5e-4,True)
            r2,p2=tb.act(to_abs(torch.zeros(n,dtype=torch.long,device=dev),tb.pos,'delta'),5e-4,True)
            e=max(e,float(np.abs(r1-r2.cpu().numpy()).max()),float(np.abs(np.asarray(p1)-p2.cpu().numpy()).max()),float(np.abs(nb.pos-tb.pos.cpu().numpy()).max()))
        out[m]['delta_always_hold_max_err']=float(f'{e:.2e}');out[m]['pass']=out[m]['pass'] and e<1e-5
    return out

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('cmd');a.add_argument('args',nargs='*');a.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a.add_argument('--steps',type=int,default=None);a.add_argument('--stoch_eval',action='store_true');a.add_argument('--n_env',type=int,default=None);a=a.parse_args();dev=torch.device(a.device)
    cfg=dict(CFG,**{k:v for k,v in (('steps',a.steps),('n_env',a.n_env)) if v})
    if a.cmd=='check':r=check(dev);print(json.dumps(r,indent=1));json.dump(r,open('docs/ml/results/rl_gpu_equivalence.json','w'),indent=1)
    elif a.cmd in ('bench','run'):
        m,v,w,d,ent,wd,seed,sp=(a.args+['BTCUSDT','A','32','2','0.01','0','0','abs'][len(a.args):])
        w,d,ent,wd,seed=int(w),int(d),float(ent),float(wd),int(seed);tg=f'{m}_{v}_{sp}_w{w}_d{d}_ent{ent:g}_wd{wd:g}_s{seed}'+('_se' if a.stoch_eval else '')
        if a.cmd=='bench':cfg['steps']=a.steps or cfg['n_env']*cfg['T']*8
        ag,T,D,cost,tp,vals,vb,sec=train(m,v,w,d,ent,wd,seed,dev,cfg,log=tg,space=sp,stoch_eval=a.stoch_eval)
        sps=cfg['steps']/sec;print(f'{tg}: {cfg["steps"]} steps in {sec:.0f}s = {sps:,.0f} steps/s -> {6e6/sps/60:.1f} min per 6M (incl. 4 val evals)')
        if a.cmd=='run':
            tr=rlm1.metrics(evaluate(ag,T,MKT[m]['train'],cost,a.stoch_eval,seed),D,tp);os.makedirs(RUNS,exist_ok=True);torch.save(ag.state_dict(),f'{RUNS}/{tg}_policy.pt')
            json.dump({'market':m,'variant':v,'space':sp,'eval':'stochastic' if a.stoch_eval else 'argmax','width':w,'depth':d,'ent':ent,'wd':wd,'seed':seed,'params':ag.nparams(),'cfg':cfg,'sec':round(sec),
                       'val_curve':vals,'val':vb,'train':tr},open(f'{RUNS}/{tg}.json','w'),indent=1,default=float)
