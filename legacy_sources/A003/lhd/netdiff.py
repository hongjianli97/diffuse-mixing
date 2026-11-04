"""Lagrangian advection-diffusion on the trajectory network:  M c_dot = -K(t) c,
mass form  : M = diag(m_i) (constant),          K = P1 stiffness with conductance rho_t * kappa (rho from material lnJ)
volume form: M = diag(A*_i(t)/3) (current area), K = P1 stiffness with conductance kappa
Advection is exact (tracers); implicit Euler per snapshot interval on the current Delaunay complex K(t)."""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import lhdx as X
def stiffness(P, tri, cond):
    G, A = X._grad_lambda(P, tri); N = len(P); r, c, v = [], [], []
    for i in range(3):
        for j in range(3):
            r.append(tri[:, i]); c.append(tri[:, j]); v.append(cond * A * (G[:, i] * G[:, j]).sum(1))
    return sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(N, N))
def run(frames, dt, c0, kappa, form='mass', m=None, rho=None, keep=()):
    S, N = frames.shape[:2]; c = np.asarray(c0, float).copy(); out = {0: c.copy()}
    for s in range(S - 1):
        P = frames[s]; tri = X.complex_peeled(P)['tri']
        if form == 'mass':
            cond = kappa * rho[s][tri].mean(1); M = m
        else:
            cond = kappa * np.ones(len(tri)); M = X._star(P, tri, N) / 3.0
        K = stiffness(P, tri, cond); used = M > 0
        Mx = np.where(used, M, 1.0)
        c = spsolve((sp.diags(Mx) + dt * K).tocsc(), Mx * c)
        if s + 1 in keep: out[s + 1] = c.copy()
    out[S - 1] = c
    return out
