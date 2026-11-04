"""Frequency-cluster hierarchy (merge tree of the triangle rotation phase) as a network strength field.
P_i = height of node i above the level at which its frequency cluster merges into a cluster with a higher peak
(the cluster that first covers half the domain is the background and dies there).  Signed: P = p+(phi) - p+(-phi).
Invariant to phi -> phi + const, i.e. to the ensemble reference rotation."""
import sys, json, numpy as np, tcx, cases, afe_cases as AC, tcx_cases as TC
from scipy.stats import spearmanr
def edge_nodes(tri, mask):
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    u, cnt = np.unique(e, axis=0, return_counts=True); bd = np.zeros(len(mask), bool); bd[u[cnt == 1].ravel()] = True
    for x, y in u:                                       # nodes of the mask touching the outside of the mask
        if mask[x] != mask[y]: bd[x] |= mask[x]; bd[y] |= mask[y]
    return bd & mask
def prominence(f, E, mask, bd):
    """height above the level at which the node's frequency cluster either merges into a cluster with a higher
    peak, or first reaches the edge of the observed material (then it is no longer an enclosed vortex)."""
    N = len(f); nb = [[] for _ in range(N)]
    for a_, b_ in E:
        if mask[a_] and mask[b_]: nb[a_].append(b_); nb[b_].append(a_)
    comp = -np.ones(N, int); peak = {}; mem = {}; saddle = np.full(N, np.nan); dead = set()
    for i in np.argsort(-f):
        if not mask[i]: continue
        cs = {comp[j] for j in nb[i] if comp[j] >= 0}
        if not cs:
            comp[i] = i; peak[i] = f[i]; mem[i] = [i]
        else:
            cs = sorted(cs, key=lambda c: (c not in dead, peak[c]), reverse=True)   # an open (edge) cluster absorbs
            top = cs[0]
            for c in cs[1:]:
                for j in mem[c]:
                    if np.isnan(saddle[j]): saddle[j] = f[i]
                    comp[j] = top
                mem[top].extend(mem.pop(c)); peak.pop(c); dead.discard(c)
            comp[i] = top; mem[top].append(i)
        r = comp[i]
        if bd[i] and r not in dead:
            for j in mem[r]:
                if np.isnan(saddle[j]): saddle[j] = f[i]
            dead.add(r)
        if r in dead and np.isnan(saddle[i]): saddle[i] = f[i]
    out = np.maximum(0.0, f - np.where(np.isnan(saddle), f, saddle)); out[~mask] = 0.0; return out
def signedP(f, E, mask, tri):
    bd = edge_nodes(tri, mask); return prominence(f, E, mask, bd) - prominence(-f, E, mask, bd)
def chain_phi(frx):
    S = len(frx); mus, _ = tcx.mu_threshold(frx); ch = tcx.run_chain(frx, mus, Ks=[tcx.cplx(frx[s])["tri"] for s in range(S-1)])
    return tcx.rotation_fields(ch)[0]
def smooth(fv, E0, N, n=2):
    deg = np.bincount(E0.ravel(), minlength=N).astype(float); g = fv.copy()
    for _ in range(n):
        s_ = np.zeros(N); np.add.at(s_, E0[:, 0], g[E0[:, 1]]); np.add.at(s_, E0[:, 1], g[E0[:, 0]]); g = .5*g + .5*s_/np.maximum(deg, 1)
    return g
