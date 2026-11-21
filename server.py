"""Local server: runs the engine in a background thread and exposes it over HTTP. python server.py -> http://localhost:8000"""
import json, threading, time, os, numpy as np
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from engine import Brain
B=Brain(0); L=threading.Lock(); S=dict(run=False,delay=.3); rng=np.random.default_rng(1); SAVE=os.path.join(os.path.dirname(__file__),'state')
def loop():
    while True:
        if S['run']:
            with L: B.trial(int(rng.integers(2)))
        time.sleep(S['delay'])
threading.Thread(target=loop,daemon=True).start()
class H(BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def send(s,obj,code=200,ct='application/json'):
        d=obj if isinstance(obj,bytes) else json.dumps(obj).encode(); s.send_response(code); s.send_header('Content-Type',ct); s.end_headers(); s.wfile.write(d)
    def do_GET(s):
        if s.path.startswith('/api/state'):
            with L: d=B.snapshot(); d['run']=S['run']; s.send(d)
        elif s.path.startswith('/api/neuron/'):
            i=int(s.path.split('/')[-1])
            with L: s.send(dict(id=i,region=str(B.reg[i]),v=float(B.v[i]),rate=float(B.rate[i]),inhibitory=bool(B.inh[i]),a=float(B.a[i]),d=float(B.d[i]),n_in=int((B.W[i]!=0).sum())))
        else: s.send(open(os.path.join(os.path.dirname(__file__),'index.html'),'rb').read(),ct='text/html')
    def do_POST(s):
        q=json.loads(s.rfile.read(int(s.headers.get('Content-Length',0))) or b'{}'); p=s.path; global B
        with L:
            if p=='/api/start': S['run']=True
            elif p=='/api/pause': S['run']=False
            elif p=='/api/speed': S['delay']=max(.01,float(q['delay']))
            elif p=='/api/reset': B.reset(); S['run']=False
            elif p=='/api/trial': return s.send(B.trial(int(q['stim'])))
            elif p=='/api/sleep': B.sleep(int(q.get('n',30)))
            elif p=='/api/retrieve': return s.send([{k:m[k] for k in('t','stim','act','out','harm','strength')} for m in B.retrieve(int(q['stim']))])
            elif p=='/api/experiment':
                e=Brain(int(q.get('seed',5))); r=np.random.default_rng(5)
                for _ in range(int(q.get('n',160))): e.trial(int(r.integers(2)))
                h=[x['correct'] for x in e.hist]; return s.send(dict(early=float(np.mean(h[:40])),late=float(np.mean(h[-40:])),chance=.5,curve=h))
            elif p=='/api/save': B.save(SAVE)
            elif p=='/api/load': B.load(SAVE)
        s.send(dict(ok=True))
if __name__=='__main__':
    port = int(os.environ.get('PORT', '8000'))
    print(f'Listening on 0.0.0.0:{port}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', port), H).serve_forever()
