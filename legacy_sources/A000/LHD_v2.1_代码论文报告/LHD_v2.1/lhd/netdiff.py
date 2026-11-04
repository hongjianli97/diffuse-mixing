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


# ------------------------------------------------------------------------------------------- extended driver
def _half_cot(P, tri):
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    def cot(u, v): return (u * v).sum(1) / np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    return 0.5 * np.stack([cot(a - c, b - c), cot(b - a, c - a), cot(c - b, a - b)], 1)   # edges (01),(12),(20)


def stiffness_edge(P, tri, cond_tri):
    """P1/cotan stiffness with an EDGE conductance (harmonic mean of the adjacent triangle conductances):
    K_ij = -k_e * max(0, sum_T 1/2 cot(theta_T^ij)).  On interior Delaunay edges the cot sum is >= 0, so only peeled
    boundary / degenerate edges are clipped.  Result: an M-matrix (K_ij <= 0), hence a discrete maximum principle and
    positive semi-definiteness for any conductance field.  (Replacing k_T by k_e without the clip is NOT PSD.)"""
    N = len(P); W = _half_cot(P, tri); E = ((0, 1), (1, 2), (2, 0))
    I = np.concatenate([tri[:, i] for i, j in E]); J = np.concatenate([tri[:, j] for i, j in E])
    w = np.concatenate([W[:, k] for k in range(3)]); kT = np.tile(cond_tri, 3)
    key = np.minimum(I, J) * N + np.maximum(I, J); u, inv = np.unique(key, return_inverse=True)
    ke = np.bincount(inv) / np.bincount(inv, weights=1.0 / kT); we = np.bincount(inv, weights=w)
    we = np.maximum(we, 0.0)          # negative cot sums only on peeled-boundary / degenerate edges: clip -> M-matrix,
    gi, gj = u // N, u % N; val = -ke * we   # unconditionally PSD weighted graph Laplacian
    K = sp.csr_matrix((np.r_[val, val], (np.r_[gi, gj], np.r_[gj, gi])), shape=(N, N))
    return (K - sp.diags(np.asarray(K.sum(1)).ravel())).tocsr()


def run2(frames, dt, c0, kappa, form="mass", m=None, rho=None, mode="elem_arith", nsub=1, theta=1.0, cond_fn=None):
    """Generalised driver (form 'mass'|'volume').  mode: 'elem_arith' (= run(), triangle conductance = kappa x mean
    node rho), 'elem_harm' (harmonic mean of node rho), 'edge_harm' (edge conductance, M-matrix on Delaunay edges).
    nsub: sub-steps per snapshot interval with linearly interpolated positions (log-linear rho).  theta: 1 = implicit
    Euler with the geometry of the left time (exactly run() for nsub=1), 0.5 = Crank-Nicolson.  cond_fn(P, tri, r, s)
    overrides the triangle conductance (s = fractional frame index).  Returns final c and diagnostics."""
    S, N, _ = frames.shape; c = np.asarray(c0, float).copy(); h = dt / nsub
    lo, hi = float(np.min(c0)), float(np.max(c0)); npos = []
    def state(s, f):
        if f == 0: return frames[s], (None if rho is None else rho[s])
        return (1 - f) * frames[s] + f * frames[s + 1], (None if rho is None else np.exp((1 - f) * np.log(rho[s]) + f * np.log(rho[s + 1])))
    def build(P, r, sf):
        tri = X.complex_peeled(P)["tri"]
        if cond_fn is not None: k = cond_fn(P, tri, r, sf)
        elif form == "mass": k = kappa * (3.0 / (1.0 / r[tri]).sum(1) if mode == "elem_harm" else r[tri].mean(1))
        else: k = kappa * np.ones(len(tri))
        K = stiffness_edge(P, tri, k) if mode == "edge_harm" else stiffness(P, tri, k)
        Kc = K.tocoo(); npos.append(int(((Kc.row != Kc.col) & (Kc.data > 1e-14)).sum()))
        M = m if form == "mass" else X._star(P, tri, N) / 3.0
        return K, np.where(M > 0, M, 1.0)
    Kl, Ml = build(*state(0, 0.0), 0.0)
    for s in range(S - 1):
        for j in range(nsub):
            f1 = (j + 1) / nsub
            Kr, Mr = build(*state(s, f1) if f1 < 1 else state(s + 1, 0.0), s + f1)
            if theta == 1.0:
                c = spsolve((sp.diags(Ml) + h * Kl).tocsc(), Ml * c)
            else:
                c = spsolve((sp.diags(Mr) + theta * h * Kr).tocsc(), Ml * c - (1 - theta) * h * (Kl @ c))
            Kl, Ml = Kr, Mr
    diag = dict(overshoot=float(max(c.max() - hi, lo - c.min(), 0.0)), pos_offdiag_mean=float(np.mean(npos)))
    return c, diag
