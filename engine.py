"""Neurogenesis engine: Izhikevich spiking network, 6 functional regions, reward-modulated STDP,
episodic/semantic memory with replay, neuromodulators, nociception-inspired threat pathway."""
import numpy as np, json
from collections import deque
REG = {'SENS':60,'CTX':120,'HIPP':40,'AMY':30,'BG':40,'PFC':30}
# (pre, post, prob, weight) -- weights are current-based; inhibitory presynaptic cells are negated
PATHS = [('SENS','CTX',.12,14),('CTX','CTX',.1,.25),('CTX','HIPP',.2,3),('HIPP','CTX',.1,.5),('CTX','AMY',.2,3),
         ('AMY','CTX',.1,.5),('CTX','PFC',.2,3),('PFC','PFC',.15,2),('PFC','BG',.2,.3),('AMY','BG',.1,.5),('BG','BG',.5,2.5)]
GP=3  # gain of plastic CTX->BG pathway
TASK = {0:{'approach':(1.0,0.0),'avoid':(0.0,0.0)}, 1:{'approach':(0.0,0.8),'avoid':(0.3,0.0)}}  # (reward, harm)
class Brain:
    def __init__(s, seed=0): s.seed=seed; s.reset()
    def reset(s):
        r=s.rng=np.random.default_rng(s.seed); s.sl={}; i=0
        for k,n in REG.items(): s.sl[k]=slice(i,i+n); i+=n
        N=s.N=i; s.t=0; s.reg=np.concatenate([[k]*n for k,n in REG.items()])
        s.inh=np.zeros(N,bool)
        for k in ('CTX','HIPP','AMY','BG','PFC'): s.inh[s.sl[k].start:s.sl[k].start+REG[k]//5]=True
        ri=r.random(N); s.a=np.where(s.inh,.02+.08*ri,.02); s.b=np.where(s.inh,.25-.05*ri,.2)
        s.c=np.where(s.inh,-65.,-65+15*ri**2); s.d=np.where(s.inh,2.,8-6*ri**2)
        s.v=np.full(N,-65.); s.u=s.b*s.v; s.g=np.zeros(N); s.spk=np.zeros(N,bool)
        W=np.zeros((N,N))
        for p,q,pr,w in PATHS:
            m=r.random((REG[q],REG[p]))<pr; W[s.sl[q],s.sl[p]]+=m*w*(.5+r.random((REG[q],REG[p])))
        W[:,s.inh]*=-2.0  # inhibitory columns
        bg=s.sl['BG']; ie=np.flatnonzero(~s.inh[bg])+bg.start; W[np.ix_(ie,ie)]=0  # no exc->exc inside BG: competition via inhibitory cells
        s.W=W; ex=[j for j in range(REG['BG']) if not s.inh[bg.start+j]]
        s.grp=[np.array(ex[:len(ex)//2]),np.array(ex[len(ex)//2:])]  # BG action groups: approach, avoid
        s.ctx_ex=np.where(~s.inh[s.sl['CTX']])[0]
        s.Wp=r.random((REG['BG'],len(s.ctx_ex)))*.2  # plastic CTX->BG
        s.Wp[[i for i in range(REG['BG']) if s.inh[bg.start+i]]]*=0
        s.xpre=np.zeros(len(s.ctx_ex)); s.xpost=np.zeros(REG['BG']); s.elig=np.zeros_like(s.Wp)
        s.V=np.zeros((2,2)); s.DA=.0; s.HT=.5; s.NE=.2; s.ACh=.5
        s.emo=dict(reward=0,threat=0,uncertainty=1,frustration=0,curiosity=1,stress=0,arousal=.2,valence=0)
        s.mem=[]; s.sem={}; s.hist=[]; s.raster=deque(maxlen=200); s.rate=np.zeros(N)
    def step(s, I_ext):
        gain=1+s.ACh*.5; I=s.W@s.g+I_ext*gain+(1.2+3*s.NE)*s.rng.standard_normal(s.N)
        I[s.sl['BG']]+=GP*s.Wp@s.g[s.sl['CTX']][s.ctx_ex]
        for _ in range(2):
            s.v+=.5*(.04*s.v**2+5*s.v+140-s.u+I); s.u+=.5*s.a*(s.b*s.v-s.u)
        s.spk=s.v>=30; s.v[s.spk]=s.c[s.spk]; s.u[s.spk]+=s.d[s.spk]; s.v=np.clip(s.v,-100,30)
        s.g=s.g*.8+s.spk; s.rate=.98*s.rate+.02*s.spk*1000; s.t+=1; s.raster.append(np.flatnonzero(s.spk))
        # STDP traces (CTX exc -> BG), eligibility accumulates; dopamine decides sign/size later
        bg=s.spk[s.sl['BG']]; pre=s.spk[s.sl['CTX']][s.ctx_ex]
        s.xpre*=.9; s.xpost*=.9
        s.elig+=.02*np.outer(bg,s.xpre)-.02*np.outer(s.xpost,pre)
        s.xpre+=pre; s.xpost+=bg
        return s.spk
    def present(s, stim, pain=0., ms=100, learn=True):
        I=np.zeros(s.N)
        if stim is not None: I[s.sl['SENS'].start+30*stim:s.sl['SENS'].start+30*stim+30]=12
        if pain>0: I[s.sl['AMY']]+=pain*15
        s.elig[:]=0; cnt=np.zeros(s.N); 
        for _ in range(ms): cnt+=s.step(I)
        return cnt
    def choose(s, cnt):
        bg=s.sl['BG'].start; sc=[cnt[bg+g].sum() for g in s.grp]
        eps=.1*(1+s.emo['curiosity'])*s.emo['uncertainty']  # curiosity-driven exploration
        if s.rng.random()<min(.3,eps) or sc[0]==sc[1]: return int(s.rng.random()<.5)
        return int(sc[1]>sc[0])
    def learn(s, rpe, lr=.5, act=None):
        s.DA=rpe; E=s.elig
        if act is not None:  # credit assignment: only the selected action's BG group is eligible
            m=np.zeros(len(s.Wp),bool); m[s.grp[act]]=True; E=E*m[:,None]
        s.Wp=np.clip(s.Wp+lr*rpe*E,0,3)*(s.Wp>0)
        nz=s.Wp.sum(1,keepdims=True); tgt=.1*s.Wp.shape[1]  # soft homeostatic scaling toward target total input
        s.Wp*=np.where(nz>0,np.clip((tgt/np.maximum(nz,1e-9))**.1,.9,1.1),1)
    def trial(s, stim, learn=True):
        cnt=s.present(stim,learn=learn); act=s.choose(cnt); rew,harm=TASK[stim][('approach','avoid')[act]]
        out=rew-harm; rpe=out-s.V[stim,act]
        E0=s.elig.copy()
        if harm>0: s.present(None,pain=harm,ms=40)
        s.elig=E0  # keep the decision-phase eligibility trace   # nociception-like input to amygdala
        threat=float(s.rate[s.sl['AMY']].mean()/50)
        e=s.emo; e['reward']=rew; e['threat']=min(1,max(harm,threat)); e['valence']=out
        e['uncertainty']=min(1,.9*e['uncertainty']+.1*abs(rpe)); e['frustration']=.8*e['frustration']+.2*(out<=0)
        e['curiosity']=.5+.5*e['uncertainty']; e['stress']=.9*e['stress']+.1*e['threat']; e['arousal']=min(1,.5*e['threat']+.5*e['uncertainty']+.1)
        s.NE=.2+.6*e['arousal']; s.ACh=.3+.7*e['uncertainty']; s.HT=.5+.3*(1-e['stress'])
        if learn: s.V[stim,act]+=.3*rpe; s.learn(rpe,act=act)
        s.encode(stim,act,out,harm,rpe,cnt)
        s.hist.append(dict(stim=stim,act=act,out=out,harm=harm,rpe=rpe,correct=int(out==max(TASK[stim]['approach'][0]-TASK[stim]['approach'][1],TASK[stim]['avoid'][0]-TASK[stim]['avoid'][1]))))
        s.forget(); return s.hist[-1]
    # ---- memory ----
    def encode(s, stim, act, out, harm, rpe, cnt):
        sal=min(1,abs(out)+harm+abs(rpe)*.5)*(1+s.emo['arousal'])  # emotional significance boosts encoding
        h=s.sl['HIPP']; idx=np.argsort(cnt[h])[-6:].tolist()       # sparse hippocampal index
        s.mem.append(dict(t=s.t,stim=stim,act=act,out=out,harm=harm,strength=.3+sal,index=idx,cue=cnt[s.sl['SENS']].tolist()))
    def forget(s):
        for m in s.mem: m['strength']*=.995
        s.mem=[m for m in s.mem if m['strength']>.02][-300:]
    def retrieve(s, stim, k=3):
        cue=np.zeros(60); cue[30*stim:30*stim+30]=1; best=[]
        for m in s.mem:
            c=np.array(m['cue']); sim=c@cue/(np.linalg.norm(c)*np.linalg.norm(cue)+1e-9); best.append((sim*m['strength'],m))
        best.sort(key=lambda x:-x[0]); out=[m for _,m in best[:k]]
        for m in out: m['strength']*=1.1   # reconsolidation: retrieval strengthens
        return out
    def sleep(s, n=30):
        """Replay strong episodes: re-present stimulus, apply stored outcome as dopamine, build semantic memory."""
        if not s.mem: return 0
        w=np.array([m['strength'] for m in s.mem]); w/=w.sum()
        for i in s.rng.choice(len(s.mem),n,p=w):
            m=s.mem[i]; cnt=s.present(m['stim'],ms=60); k=f"{m['stim']}:{m['act']}"
            sv=s.sem.get(k,0.); s.sem[k]=sv+.1*(m['out']-sv); s.learn(.5*(m['out']-s.V[m['stim'],m['act']]),lr=.02,act=m['act'])
            m['strength']*=1.05
        return n
    def accuracy(s, n=20): h=s.hist[-n:]; return float(np.mean([x['correct'] for x in h])) if h else 0.
    def save(s, path):
        np.savez(path+'.npz',W=s.W,Wp=s.Wp,V=s.V); json.dump(dict(mem=s.mem,sem=s.sem,hist=s.hist[-500:],t=s.t),open(path+'.json','w'))
    def load(s, path):
        z=np.load(path+'.npz'); s.W,s.Wp,s.V=z['W'],z['Wp'],z['V']; d=json.load(open(path+'.json')); s.mem,s.sem,s.hist,s.t=d['mem'],d['sem'],d['hist'],d['t']
    def snapshot(s):
        R={k:float(s.rate[s.sl[k]].mean()) for k in REG}
        return dict(t=s.t,rates=R,emo=s.emo,mod=dict(DA=float(s.DA),HT=s.HT,NE=s.NE,ACh=s.ACh),acc=s.accuracy(),
            hist=[x['correct'] for x in s.hist[-200:]],mem=[{k:m[k] for k in('t','stim','act','out','harm','strength')} for m in s.mem[-8:]],
            sem=s.sem,V=s.V.tolist(),raster=[[i,r.tolist()] for i,r in enumerate(s.raster)],N=s.N,
            bounds={k:[sl.start,sl.stop] for k,sl in s.sl.items()},wp_mean=float(s.Wp.mean()))
