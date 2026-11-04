import numpy as np
rng = np.random.default_rng(0)
def F_of(X, x):                      # affine map of a triangle from 3 points
    D0 = np.c_[X[1]-X[0], X[2]-X[0]]; D1 = np.c_[x[1]-x[0], x[2]-x[0]]
    return D1 @ np.linalg.inv(D0)
def pq(F):
    a,b,c,d = F[0,0],F[0,1],F[1,0],F[1,1]
    return 0.5*((a+d)+1j*(c-b)), 0.5*((a-d)+1j*(c+b))
R = lambda t: np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
# 1 exact relations
for _ in range(3):
    F = np.eye(2)+0.3*rng.normal(size=(2,2)); p,q = pq(F)
    print('det vs |p|^2-|q|^2', np.linalg.det(F), abs(p)**2-abs(q)**2, ' F21-F12 vs 2Im p', F[1,0]-F[0,1], 2*p.imag)
# 2 frame change: y = Q(t)^T x ; step map in y frame = Q1^T F Q0
for _ in range(3):
    F = np.eye(2)+0.3*rng.normal(size=(2,2)); b0, b1 = rng.uniform(-3,3,2)
    Fy = R(b1).T @ F @ R(b0); p,q = pq(F); py,qy = pq(Fy)
    print('arg shift', np.angle(py)-np.angle(p), ' expected', b0-b1, ' |mu| inv', abs(q/p), abs(qy/py))
# 3 composition
F1 = np.eye(2)+0.4*rng.normal(size=(2,2)); F2 = np.eye(2)+0.4*rng.normal(size=(2,2))
p1,q1 = pq(F1); p2,q2 = pq(F2); pt,qt = pq(F2@F1)
print('compose p', pt, p2*p1+q2*np.conj(q1), ' q', qt, p2*q1+q2*np.conj(p1))
mu1, mu2 = q1/p1, q2/p2
print('arg remainder', np.angle(pt)-np.angle(p1)-np.angle(p2), ' bound arcsin|mu1 mu2|', np.arcsin(abs(mu1*mu2)))
# 4 uniaxial: |mu| = tanh(|lnJ|/2)
for eta in (1.5, 2.0, 4.0):
    n = np.array([np.cos(0.7), np.sin(0.7)]); F = np.eye(2)-(1-1/eta)*np.outer(n,n); p,q = pq(F)
    print('eta',eta,' |mu|',abs(q/p),' tanh(lnJ/2)', np.tanh(0.5*np.log(eta)), ' r', np.log(np.linalg.det(F))-np.trace(F-np.eye(2)), 1-1/eta-np.log(eta))
