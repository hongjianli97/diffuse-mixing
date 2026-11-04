import numpy as np
from check import F_of, pq
from check2 import shock_positions
rng = np.random.default_rng(2)
def compose(p1,q1,p2,q2): return p2*p1+q2*np.conj(q1), p2*q1+q2*np.conj(p1)
viol=0; tot=0; res_merge=[]; res_inc=[]
for eta in (2.5, 6.0):
  for dt in (0.004, 0.01):
    for trial in range(200):
        X = rng.uniform(0,0.1,(3,2))
        if abs(np.linalg.det(np.c_[X[1]-X[0],X[2]-X[0]]))<2e-3: continue
        ts=np.arange(0,0.25,dt); P=[shock_positions(X,t,eta=eta) for t in ts]
        Pc,Qc=1+0j,0j; inc=0; bnd=0
        steps=[pq(F_of(P[k],P[k+1])) for k in range(len(ts)-1)]
        for p,q in steps:
            Mprev=Qc/Pc; bnd+=np.arcsin(min(1,abs(q/p)*abs(Mprev))); inc+=np.angle(p); Pc,Qc=compose(Pc,Qc,p,q)
        rem=np.angle(Pc)-inc; tot+=1; viol+= abs(rem)>bnd+1e-12
        # merge rule: steps with |mu|>mu* are composed into one block, others summed incrementally
        mu_star=0.05; phi=0; blkP,blkQ=1+0j,0j; inblk=False
        for p,q in steps:
            if abs(q/p)>mu_star:
                blkP,blkQ=compose(blkP,blkQ,p,q); inblk=True
            else:
                if inblk: phi+=np.angle(blkP); blkP,blkQ=1+0j,0j; inblk=False
                phi+=np.angle(p)
        if inblk: phi+=np.angle(blkP)
        res_merge.append(abs(phi)); res_inc.append(abs(inc))
print('cumulative-Beltrami bound violations:', viol, '/', tot)
print('spurious rotation, incremental: median %.3f p95 %.3f | merged-crossing: median %.1e p95 %.1e'%(np.median(res_inc),np.quantile(res_inc,.95),np.median(res_merge),np.quantile(res_merge,.95)))
# smooth check: Lamb-Oseen-like steady vortex, merged rule must equal incremental (no |mu|>mu*) and converge to 1/2 int omega
def vel(x):
    r2=(x**2).sum(1)+1e-12; ut=(1-np.exp(-r2))/np.sqrt(r2)/ (2*np.pi)*2*np.pi
    return np.c_[-x[:,1],x[:,0]]*(ut/np.sqrt(r2))[:,None]
def omega(x):
    r2=(x**2).sum(); return 2*np.exp(-r2)
for dt in (0.05,0.025,0.0125):
    X=np.array([[0.8,0.0],[0.83,0.0],[0.815,0.026]]); P=[X.copy()]; t=0; T=4.0; w_int=0
    while t<T-1e-12:
        x=P[-1]; c=x.mean(0); w_int+=omega(c[None,:][0])*dt
        k1=vel(x);k2=vel(x+0.5*dt*k1);k3=vel(x+0.5*dt*k2);k4=vel(x+dt*k3); P.append(x+dt/6*(k1+2*k2+2*k3+k4)); t+=dt
    inc=sum(np.angle(pq(F_of(P[k],P[k+1]))[0]) for k in range(len(P)-1)); mus=max(abs(pq(F_of(P[k],P[k+1]))[1]/pq(F_of(P[k],P[k+1]))[0]) for k in range(len(P)-1))
    print(f'dt {dt}: sum arg p {inc:.5f} vs 1/2 int omega(centroid) {0.5*w_int:.5f}; max step |mu| {mus:.3f}')
