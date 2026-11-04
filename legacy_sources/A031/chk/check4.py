import numpy as np
from check import F_of, pq
from check2 import shock_positions
rng = np.random.default_rng(3)
def compose(p1,q1,p2,q2): return p2*p1+q2*np.conj(q1), p2*q1+q2*np.conj(p1)
def phi_merged(steps, mu_star, hold=2):
    phi=0; P,Q=1+0j,0j; inblk=False; quiet=0
    for p,q in steps:
        hot = abs(q/p)>mu_star
        if hot or (inblk and quiet<hold):
            P,Q=compose(P,Q,p,q); inblk=True; quiet = 0 if hot else quiet+1
        else:
            if inblk: phi+=np.angle(P); P,Q=1+0j,0j; inblk=False; quiet=0
            phi+=np.angle(p)
    if inblk: phi+=np.angle(P)
    return phi
for mu_star in (0.05, 0.01, 0.002):
  for hold in (0, 2, 4):
    res=[]
    for eta in (2.5, 6.0):
      for dt in (0.004, 0.01):
        for trial in range(150):
            X = rng.uniform(0,0.1,(3,2))
            if abs(np.linalg.det(np.c_[X[1]-X[0],X[2]-X[0]]))<2e-3: continue
            ts=np.arange(0,0.25,dt); Pp=[shock_positions(X,t,eta=eta) for t in ts]
            steps=[pq(F_of(Pp[k],Pp[k+1])) for k in range(len(ts)-1)]
            res.append(abs(phi_merged(steps,mu_star,hold)))
    print(f'mu*={mu_star} hold={hold}: spurious median {np.median(res):.1e} p95 {np.quantile(res,.95):.1e} max {np.max(res):.1e}')
