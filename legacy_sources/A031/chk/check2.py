import numpy as np
from check import F_of, pq
rng = np.random.default_rng(1)
# (a) planar shock (normal along x) sweeping one material triangle, piston model
def shock_positions(X, t, s=1.0, eta=4.0, x0=-0.05):
    xs = x0 + s*t; up = s*(1-1/eta)
    tc = (X[:,0]-x0)/s                       # crossing time of each particle (pre-shock at rest)
    disp = np.where(t > tc, up*(t-tc), 0.0)
    return X + np.c_[disp, np.zeros(len(X))]
for eta in (1.5, 2.5, 4.0, 6.0):
  for dt in (0.004, 0.01, 0.03):
    errs=[]; bounds=[]; blocks=[]
    for trial in range(300):
        X = rng.uniform(0,0.1,(3,2)); 
        if abs(np.linalg.det(np.c_[X[1]-X[0],X[2]-X[0]]))<2e-3: continue
        ts = np.arange(0, 0.25, dt); P = [shock_positions(X,t,eta=eta) for t in ts]
        inc=0; bnd=0; mus=[]
        for k in range(len(ts)-1):
            p,q = pq(F_of(P[k],P[k+1])); inc += np.angle(p); mus.append(abs(q/p))
        for k in range(len(mus)-1): bnd += np.arcsin(min(1,mus[k]*mus[k+1]))
        pb,_ = pq(F_of(P[0],P[-1])); blocks.append(np.angle(pb)); errs.append(abs(inc)); bounds.append(bnd)
    errs=np.array(errs); bounds=np.array(bounds)
    print(f'eta {eta} dt {dt}: |sum arg p| median {np.median(errs):.3f} p95 {np.quantile(errs,.95):.3f} | bound viol frac {(errs>bounds+1e-12).mean():.3f} | block |arg| max {np.abs(blocks).max():.1e}  (mu_shock={(eta-1)/(eta+1):.2f})')
# (b) noise: rigid rotation, fixed material triangle, sum of incremental polar angles vs truth
Om=1.0; h=0.05
for n in (10, 40, 160):
    T=2.0; ts=np.linspace(0,T,n+1); X=np.array([[0,0],[h,0],[0.5*h,0.87*h]])
    errs=[]
    for trial in range(400):
        P=[(np.array([[np.cos(Om*t),-np.sin(Om*t)],[np.sin(Om*t),np.cos(Om*t)]])@X.T).T + rng.normal(0,0.05*h,(3,2)) for t in ts]
        inc=sum(np.angle(pq(F_of(P[k],P[k+1]))[0]) for k in range(n)); errs.append(inc-Om*T)
    print(f'noise 0.05h, n={n}: std of sum arg p - truth = {np.std(errs):.4f}')
