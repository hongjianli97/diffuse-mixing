"""Interface tangential-stretch estimators on a Lagrangian trajectory network (positions only).

  pair     : per-tracer median of ln(|r(t)|/|r(0)|) over mutual-kNN edges within 22.5 deg of the tangent  (old 'edge' method)
  polyline : band tracers sorted by initial parameter, per-segment ln ratio                            (old 'polyline' method)
  vector   : chained deformation-gradient tensor F (anchored like the ln J ledger, deferred across shocks),
             lambda = |F t0|;  anisotropy A = sigma_max(F) / lambda  (condition number of the estimate)
  covector : Nanson,  lambda = J |grad_x phi0| / |grad_X phi0|, P1 gradient of the Lagrangian label phi0 on K(t),
             J from the material ledger.  No time integration of the tensor.
  combined : vector where A < A*, covector otherwise.
Also: adaptive material marker curves (label-space insertion) used as references.
"""
import numpy as np
from scipy.spatial import cKDTree
import lhd as L, lhdx as X, afrc_h_engine as AH

R90 = np.array([[0.0, -1.0], [1.0, 0.0]])


# ----------------------------------------------------------------------------------------------- P1 gradient (covector)
def p1_grad(P, phi, tri):
    """Area-weighted nodal average of the P1 gradient of nodal field phi on triangles tri (current positions P)."""
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    M = np.stack([b - a, c - a], 1)
    A = 0.5 * np.abs(np.linalg.det(M))
    ok = A > 1e-300
    rhs = np.stack([phi[tri[:, 1]] - phi[tri[:, 0]], phi[tri[:, 2]] - phi[tri[:, 0]]], 1)
    g = np.zeros((len(tri), 2)); g[ok] = np.linalg.solve(M[ok], rhs[ok][..., None])[..., 0]
    G = np.zeros((len(P), 2)); W = np.zeros(len(P))
    for j in range(3):
        np.add.at(G, tri[ok, j], A[ok, None] * g[ok]); np.add.at(W, tri[ok, j], A[ok])
    return G / np.maximum(W, 1e-300)[:, None]


def covector_lnlam(P, phi0, gX, lnJ, tri=None):
    """ln lambda of the phi0 level set through every tracer (Nanson): ln J + ln|grad_x phi0| - ln|grad_X phi0|."""
    tri = X.complex_peeled(P)["tri"] if tri is None else tri
    g = p1_grad(P, phi0, tri)
    with np.errstate(divide="ignore"):
        return lnJ + np.log(np.linalg.norm(g, axis=1)) - np.log(gX), g


# ----------------------------------------------------------------------------------------------- chained tensor (vector)
def _fitF(A, B, h, quad=True, err=False):
    """least-squares B ~ F A (+ quadratic terms); A,B (n,k,2) neighbour offsets; returns linear part F (n,2,2)
    and, if err, the relative standard error of F from the fit residual (LSQ covariance)."""
    if quad:
        x, y = A[..., 0] / h, A[..., 1] / h
        Phi = np.concatenate([A, np.stack([x * x, x * y, y * y], -1) * h], -1)
    else:
        Phi = A
    M = np.einsum("nki,nkj->nij", Phi, Phi); b = np.einsum("nki,nkj->nij", Phi, B)
    M += 1e-12 * np.trace(M, axis1=1, axis2=2)[:, None, None] * np.eye(M.shape[1])
    S = np.linalg.solve(M, b)
    F = np.transpose(S[:, :2, :], (0, 2, 1))
    if not err:
        return F
    res = B - np.einsum("nkp,npj->nkj", Phi, S); k, p = Phi.shape[1], Phi.shape[2]
    s2 = (res ** 2).sum((1, 2)) / (2 * max(k - p, 1))
    Minv = np.linalg.inv(M); var = s2 * (Minv[:, 0, 0] + Minv[:, 1, 1]) * 2
    return F, var / np.maximum((F ** 2).sum((1, 2)), 1e-300)


def chained_F(frames, h, bad=None, block=5, max_defer=4, k=12, quad=True, idx=None, keep=None):
    """Chained deformation gradient with per-tracer anchors (same schedule as lhdx.chained_lnJ_v2):
    F(t) = F_inc(anchor -> t) F(anchor); re-anchor every `block` frames, deferred while any kNN neighbour is in a
    shock stencil (bad (N,S)), up to max_defer*block.  Returns dict frame -> F (n,2,2) for tracers idx."""
    S, N, _ = frames.shape
    idx = np.arange(N) if idx is None else np.asarray(idx)
    keep = set(range(S)) if keep is None else set(keep)
    n = len(idx)
    def knn(t):
        _, nb = cKDTree(frames[t]).query(frames[t][idx], k + 1); return nb[:, 1:]
    Fc = np.tile(np.eye(2), (n, 1, 1)); anchor = np.zeros(n, int); nb = knn(0); e2 = np.zeros(n)
    out = {0: (Fc.copy(), e2.copy())} if 0 in keep else {}
    for t in range(1, S):
        cur = np.empty_like(Fc); ce = np.empty(n)
        for a in np.unique(anchor):
            m = anchor == a; ii = idx[m]; nbm = nb[m]
            A = frames[a][nbm] - frames[a][ii][:, None]; B = frames[t][nbm] - frames[t][ii][:, None]
            Fi, ei = _fitF(A, B, h, quad, err=True)
            cur[m] = np.einsum("nij,njk->nik", Fi, Fc[m]); ce[m] = e2[m] + ei
        if t in keep:
            out[t] = (cur.copy(), ce.copy())
        due = (t - anchor) >= block
        if due.any() and t < S - 1:
            nbt = knn(t)
            if bad is not None:
                nbad = bad[:, max(t - 1, 0):t + 2].any(1)
                touched = nbad[nbt].any(1) | nbad[idx]
                ok = due & (~touched | ((t - anchor) >= max_defer * block))
            else:
                ok = due
            Fc[ok] = cur[ok]; e2[ok] = ce[ok]; anchor[ok] = t; nb[ok] = nbt[ok]
    return out


def vector_lnlam(F, t0):
    v = np.einsum("nij,nj->ni", F, t0); lam = np.linalg.norm(v, axis=1)
    smax = np.linalg.svd(F, compute_uv=False)[:, 0]
    return np.log(lam), smax / lam


# ----------------------------------------------------------------------------------------------- old estimators
def pair_lnlam(fr, idx, t0, s, k=12, E=None):
    """old edge method: per-tracer median over mutual-kNN edges whose initial direction is within 22.5 deg of t0."""
    E = AH._mutual_knn(fr[0], k) if E is None else E
    E = np.r_[E, E[:, ::-1]]; pos = {v: j for j, v in enumerate(idx)}
    E = E[np.isin(E[:, 0], idx)]
    r0 = fr[0][E[:, 1]] - fr[0][E[:, 0]]; rt = fr[s][E[:, 1]] - fr[s][E[:, 0]]
    tt = t0[np.array([pos[v] for v in E[:, 0]])]
    ok = np.abs((r0 * tt).sum(1)) / np.linalg.norm(r0, axis=1) > np.cos(np.pi / 8)
    E, ls = E[ok], np.log(np.linalg.norm(rt[ok], axis=1) / np.linalg.norm(r0[ok], axis=1))
    out = np.full(len(idx), np.nan)
    order = np.argsort(E[:, 0]); E, ls = E[order], ls[order]
    u, st = np.unique(E[:, 0], return_index=True); en = np.r_[st[1:], len(E)]
    for v, a, b in zip(u, st, en):
        out[pos[v]] = np.median(ls[a:b])
    return out


def polyline_lnlam(fr, ring_sorted, s, closed):
    """old polyline method: segments between consecutive band tracers (sorted by initial parameter);
    per-tracer value = mean of the ln ratios of its two segments.  Also returns ln(L/L0) of the polyline."""
    P0, P = fr[0][ring_sorted], fr[s][ring_sorted]
    if closed:
        P0, P = np.r_[P0, P0[:1]], np.r_[P, P[:1]]
    d0 = np.linalg.norm(np.diff(P0, axis=0), axis=1); d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    sg = np.log(d / d0)
    if closed:
        st = 0.5 * (sg + np.roll(sg, 1))
    else:
        st = np.r_[sg[0], 0.5 * (sg[1:] + sg[:-1]), sg[-1]]
    return st, float(np.log(d.sum() / d0.sum()))


# ----------------------------------------------------------------------------------------------- reference marker curves
class MarkerCurve:
    """Material curve x(s,t) carried by markers with labels s; x0(s) analytic.  Adaptive insertion in label space
    (cubic Lagrange interpolation through the 4 neighbouring markers) whenever a segment exceeds dmax or the turning
    angle at either end exceeds amax (segments longer than dmin)."""

    def __init__(self, x0fun, s_lo, s_hi, n0, closed, dmax, amax=0.15, dmin=None, cap=400000):
        self.x0fun, self.closed, self.dmax, self.amax = x0fun, closed, dmax, amax
        self.dmin = dmax / 8 if dmin is None else dmin; self.cap = cap
        self.s = np.linspace(s_lo, s_hi, n0 + 1)[: n0 if closed else None]
        self.period = (s_hi - s_lo) if closed else None
        self.X = x0fun(self.s)

    def _nbr(self, i):
        n = len(self.s)
        if self.closed:
            j = i % n; return self.s[j] + self.period * ((i - j) // n), self.X[j]
        j = np.clip(i, 0, n - 1); return self.s[j], self.X[j]

    def refine(self, passes=6):
        for _ in range(passes):
            n = len(self.s)
            if n >= self.cap:
                return
            nseg = n if self.closed else n - 1
            i = np.arange(nseg)
            s_a, Xa = self._nbr(i); s_b, Xb = self._nbr(i + 1)
            d = np.linalg.norm(Xb - Xa, axis=1)
            # turning angles at nodes
            sp, Xp = self._nbr(np.arange(n) - 1); sn, Xn = self._nbr(np.arange(n) + 1)
            e1 = self.X - Xp; e2 = Xn - self.X
            cosang = (e1 * e2).sum(1) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-300)
            ang = np.arccos(np.clip(cosang, -1, 1))
            if not self.closed:
                ang[0] = ang[-1] = 0.0
            angb = ang[(i + 1) % n] if self.closed else ang[np.minimum(i + 1, n - 1)]
            flag = (d > self.dmax) | ((np.maximum(ang[i], angb) > self.amax) & (d > self.dmin))
            if not flag.any():
                return
            ii = i[flag]
            sm = 0.5 * (s_a[flag] + s_b[flag])
            # 4-point Lagrange (3-point at open ends)
            S4 = np.stack([self._nbr(ii + o)[0] for o in (-1, 0, 1, 2)], 1)
            X4 = np.stack([self._nbr(ii + o)[1] for o in (-1, 0, 1, 2)], 1)
            W = np.ones((len(ii), 4))
            with np.errstate(all='ignore'):
              for a in range(4):
                for b in range(4):
                    if a != b:
                        W[:, a] *= (sm - S4[:, b]) / (S4[:, a] - S4[:, b])
              Xm = np.einsum("na,nak->nk", W, X4)
            if not self.closed:   # quadratic at the two end segments
                for sel, cols in ((ii == 0, (1, 2, 3)), (ii + 1 == n - 1, (0, 1, 2))):
                    if sel.any():
                        S3, X3 = S4[sel][:, cols], X4[sel][:, cols]; W3 = np.ones((sel.sum(), 3))
                        for a in range(3):
                            for b in range(3):
                                if a != b:
                                    W3[:, a] *= (sm[sel] - S3[:, b]) / (S3[:, a] - S3[:, b])
                        Xm[sel] = np.einsum("na,nak->nk", W3, X3)
            if self.closed:
                sm = np.where(sm >= self.s[0] + self.period, sm - self.period, sm)
            self.s = np.r_[self.s, sm]; self.X = np.r_[self.X, Xm]
            o = np.argsort(self.s, kind="stable"); self.s, self.X = self.s[o], self.X[o]

    def stats(self):
        Xs, X0 = self.X, self.x0fun(self.s)
        if self.closed:
            Xs, X0 = np.r_[Xs, Xs[:1]], np.r_[X0, X0[:1]]
        l = np.linalg.norm(np.diff(Xs, axis=0), axis=1); l0 = np.linalg.norm(np.diff(X0, axis=0), axis=1)
        ln = np.log(l / l0); sm = 0.5 * (self.s[:-1] + self.s[1:]) if not self.closed else np.r_[0.5 * (self.s[:-1] + self.s[1:]), self.s[-1]]
        return dict(mean_ln=float(np.sum(l0 * ln) / np.sum(l0)), ln_total=float(np.log(l.sum() / l0.sum())), s_mid=sm, ln=ln, l0=l0)


def rk4_curve(mc, vel, t0, t1, nsub, refine_every=1):
    h = (t1 - t0) / nsub; t = t0
    for j in range(nsub):
        Xq = mc.X
        k1 = vel(Xq, t); k2 = vel(Xq + .5 * h * k1, t + .5 * h); k3 = vel(Xq + .5 * h * k2, t + .5 * h); k4 = vel(Xq + h * k3, t + h)
        mc.X = Xq + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6; t += h
        if (j + 1) % refine_every == 0:
            mc.refine()
    return mc


def wmean(v, w=None):
    ok = np.isfinite(v)
    w = np.ones_like(v) if w is None else w
    return float(np.sum(w[ok] * v[ok]) / np.sum(w[ok]))
