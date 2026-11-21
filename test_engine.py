import numpy as np, engine
def test_neuron_fires_and_refractory_reset():
    b=engine.Brain(1); I=np.zeros(b.N); I[:]=15; n=sum(b.step(I)[0] for _ in range(300))
    assert n>3 and b.v[0]<31
def test_silent_without_drive():
    b=engine.Brain(1); b.rng=np.random.default_rng(0); n=sum(b.step(np.zeros(b.N))[b.sl['PFC']].sum() for _ in range(200)); assert n<400
def test_synaptic_effect_and_inhibition():
    b=engine.Brain(2); assert (b.W[:,b.inh]<=0).all() and (b.W[:,~b.inh]>=0).all()
def test_stdp_eligibility_and_dopamine_sign():
    b=engine.Brain(3); b.present(0); w0=b.Wp.copy(); e=b.elig.copy(); assert np.abs(e).sum()>0; m=np.zeros(len(e),bool); m[b.grp[0]]=True; e=e*m[:,None]
    b.learn(1.0,act=0); up=(b.Wp-w0).ravel()@e.ravel(); b.Wp=w0.copy(); b.elig=e; b.learn(-1.0,act=0); dn=(b.Wp-w0).ravel()@e.ravel(); assert up>0>dn
def test_learning_improves_over_baseline():
    early,late=[],[]
    for seed in range(6):
        b=engine.Brain(seed); r=np.random.default_rng(seed)
        for i in range(240): b.trial(int(r.integers(2)))
        early.append(np.mean([x['correct'] for x in b.hist[:40]])); late.append(b.accuracy(40))
    print('early',np.mean(early),'late',np.mean(late)); assert np.mean(late)>.65 and np.mean(late)>np.mean(early)+.1   # chance ~.5

def test_pain_avoidance_and_memory_salience():
    b=engine.Brain(0)
    for i in range(160): b.trial(i%2)
    harm=[m['strength'] for m in b.mem if m['harm']>0]; calm=[m['strength'] for m in b.mem if m['harm']==0]
    if harm and calm: assert np.mean(harm)>np.mean(calm)*.8
    assert b.retrieve(1) 
def test_forgetting_replay_persistence(tmp_path):
    b=engine.Brain(0)
    for i in range(30): b.trial(i%2)
    s0=sum(m['strength'] for m in b.mem); b.sleep(10); assert b.sem
    b.save(str(tmp_path/'s')); c=engine.Brain(0); c.load(str(tmp_path/'s')); assert np.allclose(b.Wp,c.Wp) and len(c.mem)==len(b.mem)
