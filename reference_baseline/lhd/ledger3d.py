"""Three-dimensional material ledger on the tetrahedral Delaunay complex of the tracers (the 3D form of Sec. 3):
star volume V*_i = sum of the volumes of the tetrahedra incident to i, ln J_i = sum over anchor segments of
ln V*_i(t_end)/V*_i(t_anchor) with the anchor tetrahedra kept fixed in between, re-anchoring every `block` snapshots
and deferred while a vertex of the star is hot (single-step |ln V* ratio| > tau, compression only, plus a margin of one
step).  Proposition 2 (telescoping across a planar shock) holds unchanged: the map between a fully pre-shock and a fully
post-shock time is affine on the star, and the volume ratio of a polyhedron under an affine map is det F."""
import numpy as np
from scipy.spatial import Delaunay


def tets(P):
    T = Delaunay(P).simplices
    a, b, c, d = (P[T[:, k]] for k in range(4)); v = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
    T = np.where((v < 0)[:, None], T[:, [0, 2, 1, 3]], T)
    return T


def vol(P, T):
    a, b, c, d = (P[T[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def star_vol(P, T, N):
    v = vol(P, T); s = np.zeros(N)
    for k in range(4):
        np.add.at(s, T[:, k], v)
    return s


def star_nb(T, N):
    nb = [set() for _ in range(N)]
    for t in T:
        for i in t:
            nb[i].update(t)
    return nb


def hot_steps(fr, tau=0.2):
    S, N, _ = fr.shape; hot = np.zeros((N, S - 1), bool)
    for s in range(S - 1):
        T = tets(fr[s]); a0 = star_vol(fr[s], T, N); a1 = star_vol(fr[s + 1], T, N)
        with np.errstate(all="ignore"):
            hot[:, s] = np.log(a1 / a0) < -tau
    return hot


def ledger(fr, block=5, defer=True, defer_cap=20, hot=None):
    S, N, _ = fr.shape
    if hot is None:
        hot = hot_steps(fr)
    bad = np.zeros((N, S), bool); bad[:, :-1] |= hot; bad[:, 1:] |= hot
    T = {0: tets(fr[0])}; anchor = np.zeros(N, int); V0 = star_vol(fr[0], T[0], N); base = np.zeros(N)
    for t in range(1, S - 1):
        due = (t - anchor) >= block
        if not due.any():
            continue
        Tt = tets(fr[t])
        if defer:
            near = bad[:, max(t - 1, 0):t + 2].any(1); sb = np.zeros(N, bool)
            for k in range(4):
                np.logical_or.at(sb, Tt[:, k], near[Tt].any(1))
            ok = due & (~sb | ((t - anchor) >= defer_cap * block))
        else:
            ok = due
        Vt = star_vol(fr[t], Tt, N)
        for a in np.unique(anchor[ok]):
            ii = np.where(ok & (anchor == a))[0]
            base[ii] += np.log(star_vol(fr[t], T[a], N)[ii] / V0[ii])
        V0[ok] = Vt[ok]; anchor[ok] = t; T[t] = Tt
        for a in [k for k in T if k not in set(anchor.tolist())]:
            del T[a]
    out = np.empty(N)
    for a in np.unique(anchor):
        ii = np.where(anchor == a)[0]; out[ii] = base[ii] + np.log(star_vol(fr[-1], T[a], N)[ii] / V0[ii])
    return out, hot


def lattice3(h, seed=0, jit=0.25):
    g = np.arange(h / 2, 1, h); X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return X + np.random.default_rng(seed).uniform(-jit, jit, X.shape) * h


def planar_shock3(h=0.1, eta=3.0, n=(0.8, 0.36, 0.48), Us=1.0, slip=(0.0, 0.3, -0.225), Cs=1.0, stop=1.0, seed=0):
    n = np.asarray(n, float); n /= np.linalg.norm(n); X = lattice3(h, seed); xi0 = X @ n
    s0 = xi0.min() - 0.15; T = stop * (xi0.max() + 0.15 - s0) / Us; dt = Cs * h / Us; S = int(np.ceil(T / dt)) + 1
    u2 = Us * (1 - 1 / eta); sl = np.asarray(slip, float); sl = sl - (sl @ n) * n
    fr = np.empty((S, len(X), 3)); lnJ = np.zeros(len(X))
    for s in range(S):
        t = s * dt; tc = (xi0 - s0) / Us; behind = t > tc
        fr[s] = X + (np.where(behind, u2 * (t - tc), 0.0))[:, None] * n + t * sl
        lnJ = np.where(behind, -np.log(eta), 0.0)
    return fr, lnJ, X


def smooth3(h=0.1, eps=0.04, S=21, T=1.0, seed=0):
    X = lattice3(h, seed); fr = np.empty((S, len(X), 3)); k = 2 * np.pi
    for s in range(S):
        a = eps * s * T / (S - 1)
        for i in range(3):
            j = (i + 1) % 3
            fr[s][:, i] = X[:, i] + a * (np.sin(k * X[:, j]) + 0.5 * np.sin(k * X[:, i]))
    a = eps * T; F = np.zeros((len(X), 3, 3))
    for i in range(3):
        j = (i + 1) % 3; F[:, i, i] = 1 + a * 0.5 * k * np.cos(k * X[:, i]); F[:, i, j] += a * k * np.cos(k * X[:, j])
    return fr, np.log(np.linalg.det(F)), X
