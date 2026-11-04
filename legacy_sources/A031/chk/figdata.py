import numpy as np, json
from check import F_of, pq
from check2 import shock_positions
from check4 import phi_merged
rng = np.random.default_rng(7)
out = {'eta': [], 'inc_med': [], 'inc_p95': [], 'mer_med': [], 'mer_p95': []}
for eta in (1.5, 2.0, 3.0, 4.0, 5.0, 6.0):
    inc=[]; mer=[]
    for dt in (0.004, 0.01, 0.03):
        for trial in range(120):
            X = rng.uniform(0,0.1,(3,2))
            if abs(np.linalg.det(np.c_[X[1]-X[0],X[2]-X[0]]))<2e-3: continue
            ts=np.arange(0,0.25,dt); P=[shock_positions(X,t,eta=eta) for t in ts]
            steps=[pq(F_of(P[k],P[k+1])) for k in range(len(ts)-1)]
            inc.append(abs(sum(np.angle(p) for p,q in steps))); mer.append(abs(phi_merged(steps,0.01,2)))
    out['eta'].append(eta); out['inc_med'].append(float(np.median(inc))); out['inc_p95'].append(float(np.quantile(inc,.95)))
    out['mer_med'].append(float(np.median(mer))); out['mer_p95'].append(float(np.quantile(mer,.95)))
# noise telescoping: fixed material triangle vs re-sampled triangle each step
Om=1.0; h=0.05; nn=[10,20,40,80,160]; fixed=[]; resamp=[]
for n in nn:
    T=2.0; ts=np.linspace(0,T,n+1); X=np.array([[0,0],[h,0],[0.5*h,0.87*h]]); ef=[]; er=[]
    Rm=lambda t: np.array([[np.cos(Om*t),-np.sin(Om*t)],[np.sin(Om*t),np.cos(Om*t)]])
    for trial in range(300):
        noise=[rng.normal(0,0.05*h,(3,2)) for t in ts]
        P=[(Rm(t)@X.T).T+e for t,e in zip(ts,noise)]
        ef.append(sum(np.angle(pq(F_of(P[k],P[k+1]))[0]) for k in range(n))-Om*T)
        # re-sampled: independent noise realisation of the start frame each step (mimics re-triangulation / different stars)
        er.append(sum(np.angle(pq(F_of((Rm(ts[k])@X.T).T+rng.normal(0,0.05*h,(3,2)),P[k+1]))[0]) for k in range(n))-Om*T)
    fixed.append(float(np.std(ef))); resamp.append(float(np.std(er)))
out['n']=nn; out['fixed']=fixed; out['resamp']=resamp
# shock signature: full-crossing blocks and smooth steps
sig={'lnJ':[], 'mu':[], 'kind':[]}
for eta in np.linspace(1.2,6,25):
    X = np.array([[0.02,0.02],[0.08,0.03],[0.04,0.09]]); P0=shock_positions(X,0.0,eta=eta); P1=shock_positions(X,0.25,eta=eta)
    p,q=pq(F_of(P0,P1)); sig['lnJ'].append(float(np.log(abs(p)**2-abs(q)**2))); sig['mu'].append(float(abs(q/p))); sig['kind'].append('shock')
for _ in range(150):
    L=rng.normal(size=(2,2)); S=0.5*(L+L.T); W=0.5*(L-L.T); dt=0.05
    F=np.eye(2)+dt*(S*rng.uniform(0,3)+W*rng.uniform(0,6))
    p,q=pq(F); sig['lnJ'].append(float(np.log(abs(p)**2-abs(q)**2))); sig['mu'].append(float(abs(q/p))); sig['kind'].append('smooth')
for _ in range(60):   # partial-crossing single steps
    eta=rng.uniform(1.5,6); X=rng.uniform(0,0.1,(3,2)); t0=rng.uniform(0.0,0.2)
    if abs(np.linalg.det(np.c_[X[1]-X[0],X[2]-X[0]]))<2e-3: continue
    p,q=pq(F_of(shock_positions(X,t0,eta=eta),shock_positions(X,t0+0.02,eta=eta)))
    if abs(q/p)>1e-6: sig['lnJ'].append(float(np.log(abs(p)**2-abs(q)**2))); sig['mu'].append(float(abs(q/p))); sig['kind'].append('partial')
out['sig']=sig
json.dump(out, open('figdata.json','w'))
print({k: (v if k!='sig' else len(v['mu'])) for k,v in out.items()})
