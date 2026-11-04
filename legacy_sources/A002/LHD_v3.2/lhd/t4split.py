"""T4 on merged blocks: impulsive / smooth split of material-region circulation, and the DNS baroclinic torque.

Position-form circulation of a material triangle over the half step s+1/2:  G_t^{s+1/2} = Re(d1 c)_t / dt
(Theorem 1).  Its change at snapshot n, dG_t^n = G_t^{n+1/2} - G_t^{n-1/2}, equals (d1 g^n)_t / dt exactly
(Theorem 2).  A triangle is 'hot' at snapshot n when one of its vertices is hot on step n-1 or n (shock detection of
ledger_v4); maximal hot runs are the triangle's merged blocks.  Over a block, Theorem 2 telescopes:
    impulsive deposition of t on block [n0,n1] = G_t^{n1+1/2} - G_t^{n0-1/2},
a finite-step quantity of the block end frames (no per-step pseudo-rotation).  The smooth part is the remainder, so
impulsive + smooth = change of the region circulation EXACTLY, for any region, because sum_t G_t = G_R at every
half step.  The previous split (v2.1 report) assigned to the impulsive part every snapshot at which the shock was
anywhere in the region, which also counted the smooth deposition of all other triangles at those snapshots.

DNS reference: baroclinic torque  B = (grad rho x grad p) / rho^2  from the solver fields interpolated to the tracers,
with P1 gradients, integrated (midpoint rule in time) either on the same material triangles or on the current
Delaunay triangles whose three vertices carry the region label (robust once material triangles are stretched), and
split with the same per-tracer hot flags.

Every kept block is widened by `margin` snapshots on either side (default 1): a captured shock is a few cells wide,
so its leading edge reaches a triangle about one snapshot before the star compression crosses the detection
threshold.  Without the margin the split depends on the block-end convention (position form 0.1728 against 0.1612
for the DNS velocity circulation of the same triangles, shock-bubble wide band); with one snapshot both give
0.179 to 0.4%, and a second snapshot changes the position form by 1% (scripts/v10b_t4_dt.py).

A hot block counts as impulsive only if it compresses its triangle, ln(A(n1+1)/A(n0-1)) < -comp_min: hot flags in a
rolled-up sheet (where material triangles exchange circulation with their neighbours through the numerical or
physical viscous flux, a transfer that sums to ~0 over the region) would otherwise be booked as impulsive."""
import numpy as np
import lhdx as X
from lhdp import disp_cochain_tri, tri_area


def p1_cross(P, tri, rho, p):
    """per-triangle (grad rho x grad p)/rho_c^2 * area"""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    e1 = P[b] - P[a]; e2 = P[c] - P[a]; det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    def grad(f):
        d1 = f[b] - f[a]; d2 = f[c] - f[a]
        return np.c_[(d1 * e2[:, 1] - d2 * e1[:, 1]) / det, (-d1 * e2[:, 0] + d2 * e1[:, 0]) / det]
    gr, gp = grad(rho), grad(p); rc = (rho[a] + rho[b] + rho[c]) / 3
    return (gr[:, 0] * gp[:, 1] - gr[:, 1] * gp[:, 0]) / rc ** 2 * 0.5 * np.abs(det)


def split(fr, t, rho, p, hot, reg, tri0, comp_min=0.15, margin=1):
    """fr (S,N,2), t (S,), rho/p (S,N) DNS at tracers, hot (N,S-1), reg (N,) bool region label, tri0 CCW triangles at
    t[0].  Returns dict of cumulative series (length S-2, at interior snapshots n=1..S-2) and totals."""
    S = len(t); dt = np.diff(t)
    triR = tri0[reg[tri0].all(1)]
    G = np.array([disp_cochain_tri(fr[s], fr[s + 1], triR)[0] / dt[s] for s in range(S - 1)])      # (S-1, nT)
    dG = np.diff(G, axis=0)                                                                     # n = 1..S-2
    hv = hot.T                                                                                   # (S-1, N)
    th = np.array([(hv[n - 1][triR] | hv[n][triR]).any(1) for n in range(1, S - 1)])         # (S-2, nT)
    nblk = [0, 0]
    if comp_min is not None:                 # keep only blocks that compress the triangle (genuine shock crossings)
        d = np.diff(np.vstack([np.zeros((1, th.shape[1]), int), th.astype(int), np.zeros((1, th.shape[1]), int)]), axis=0)
        k0, j0 = np.where(d == 1); k1, j1 = np.where(d == -1); o0 = np.lexsort((k0, j0)); o1 = np.lexsort((k1, j1))
        k0, j0, k1 = k0[o0], j0[o0], k1[o1] - 1                        # block rows k0..k1 (n = k+1)
        Pa = fr[k0[:, None], triR[j0]]; Pb = fr[k1[:, None] + 2, triR[j0]]
        ar = lambda P: 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
        with np.errstate(all="ignore"):
            lr = np.log(ar(Pb) / ar(Pa))
        drop = ~(lr < -comp_min); nblk = [int(len(lr)), int(drop.sum())]
        for a, b, j in zip(k0[drop], k1[drop], j0[drop]):
            th[a:b + 1, j] = False
    for _ in range(int(margin)):              # widen every kept block by one snapshot on either side
        th = th | np.vstack([th[1:], np.zeros((1, th.shape[1]), bool)]) | np.vstack([np.zeros((1, th.shape[1]), bool), th[:-1]])
    imp = (dG * th).sum(1); smo = (dG * ~th).sum(1)
    reg_hot = th.any(1); imp_old = np.where(reg_hot, dG.sum(1), 0.0)
    # DNS torque, material triangles and current Delaunay triangles of the labelled region
    Bm = np.empty((S - 2, len(triR))); Bd_s = np.zeros(S - 2); Bd_i = np.zeros(S - 2)
    for k, n in enumerate(range(1, S - 1)):
        h = 0.5 * (t[n + 1] - t[n - 1])
        Bm[k] = p1_cross(fr[n], triR, rho[n], p[n]) * h
        tc = X.complex_peeled(fr[n])["tri"]; tc = tc[reg[tc].all(1)]
        A = tri_area(fr[n], tc); tc = np.where((A < 0)[:, None], tc[:, [0, 2, 1]], tc)
        b = p1_cross(fr[n], tc, rho[n], p[n]) * h; hh = (hv[n - 1][tc] | hv[n][tc]).any(1)
        Bd_s[k] = b[~hh].sum(); Bd_i[k] = b[hh].sum()
    o = dict(t=t[1:-1], G0=float(G[0].sum()), G_end=float(G[-1].sum()), dG=dG.sum(1), imp=imp, smo=smo, imp_old=imp_old,
             dns_tri_imp=(Bm * th).sum(1), dns_tri_smo=(Bm * ~th).sum(1), dns_del_imp=Bd_i, dns_del_smo=Bd_s,
             hot_tri_frac=float(th.mean()), blocks=nblk[0], blocks_dropped=nblk[1], th=th, triR=triR,
             imp_tri=(dG * th).sum(0), smo_tri=(dG * ~th).sum(0))
    o["closure"] = float(abs(imp.sum() + smo.sum() - (G[-1].sum() - G[0].sum())))
    return o
