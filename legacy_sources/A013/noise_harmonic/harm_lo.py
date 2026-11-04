"""Harmonic channel: diffusing Lamb-Oseen vortex, tracers in an annulus r_in < r < r_out (the core is a hole).
Exact kinematics: r = const, dtheta/dt = Gamma(r,t)/(2 pi r^2),  Gamma(r,t) = G0 (1 - exp(-r^2/(4 nu t))).
On the annulus complex (beta_1 = 1) the velocity cochain u = grad + curl + harmonic.  Continuum prediction for the
harmonic part (stream function of the co-exact part vanishes on both rims):
    h = C/(2 pi r) e_theta,   C(t) = int_{r_in}^{r_out} Gamma(r,t)/r dr / ln(r_out/r_in)
and for the acceleration a = d/dt u (azimuthal part = viscous):  harmonic period = dC/dt."""
import numpy as np, json, sys
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import lhdx as X, netmix as NM
G0, nu, h, r_in, r_out, t1, dtf, nF = 1.0, 1e-3, 0.005, 0.05, 0.3, 1.0, 0.005, 21
rng = np.random.default_rng(1); g = np.arange(-r_out, r_out + h, h)
Xg, Yg = np.meshgrid(g, g); P0 = np.c_[Xg.ravel(), Yg.ravel()] + rng.uniform(-0.25, 0.25, (Xg.size, 2)) * h
r = np.hypot(P0[:, 0], P0[:, 1]); keep = (r > r_in) & (r < r_out); P0, r = P0[keep], r[keep]; th0 = np.arctan2(P0[:, 1], P0[:, 0])
Gam = lambda rr, tt: G0 * (1 - np.exp(-rr ** 2 / (4 * nu * tt)))
tf = t1 + np.arange(nF) * dtf; tfine = np.linspace(t1, tf[-1], 4001)
om = Gam(r[None], tfine[:, None]) / (2 * np.pi * r[None] ** 2)
cum = np.concatenate([np.zeros((1, len(r))), np.cumsum(0.5 * (om[1:] + om[:-1]) * np.diff(tfine)[:, None], 0)])
idx = np.searchsorted(tfine, tf - 1e-12); theta = th0[None] + cum[idx]
fr = np.stack([r[None] * np.cos(theta), r[None] * np.sin(theta)], -1); N = len(r)
rmin, rmax = r.min(), r.max()
from scipy.integrate import quad
Cth = lambda tt: quad(lambda rr: Gam(rr, tt) / rr, rmin, rmax)[0] / np.log(rmax / rmin)
res = dict(N=int(N), frames={})
for s in range(1, nF - 1):
    P = fr[s]; tri = X.complex_peeled(P)['tri']
    tri = tri[np.hypot(*P[tri].mean(1).T) > rmin - 0.2 * h]      # triangles over the hole
    for _ in range(20):                                         # peel the hole rim like the outer rim: long triangles on the inner boundary
        Ls = np.linalg.norm(P[tri] - P[tri][:, [1, 2, 0]], axis=2)
        ed = np.sort(np.stack([tri, tri[:, [1, 2, 0]]], -1), -1).reshape(-1, 2); kk_ = ed[:, 0] * N + ed[:, 1]
        u_, inv_, cnt_ = np.unique(kk_, return_inverse=True, return_counts=True); bnd = (cnt_[inv_] == 1).reshape(-1, 3)
        midr = np.hypot(*(0.5 * (P[tri] + P[tri][:, [1, 2, 0]])).transpose(2, 0, 1))
        kill = (bnd & (midr < 0.5 * (rmin + rmax))).any(1) & (Ls.max(1) > 2.5 * h)
        if not kill.any(): break
        tri = tri[~kill]
    a = 0.5 * ((P[tri[:, 1], 0] - P[tri[:, 0], 0]) * (P[tri[:, 2], 1] - P[tri[:, 0], 1]) - (P[tri[:, 1], 1] - P[tri[:, 0], 1]) * (P[tri[:, 2], 0] - P[tri[:, 0], 0]))
    tri[a < 0] = tri[a < 0][:, [0, 2, 1]]; A = np.abs(a)
    E, w = NM.edge_weights(P, tri); nE = len(E)
    kk = E[:, 0] * N + E[:, 1]; order = np.argsort(kk); ks = kk[order]; rows, cols, vals = [], [], []
    for (p_, q_) in ((0, 1), (1, 2), (2, 0)):
        i, j = tri[:, p_], tri[:, q_]; k = np.minimum(i, j) * N + np.maximum(i, j)
        rows.append(order[np.searchsorted(ks, k)]); cols.append(np.arange(len(tri))); vals.append(np.where(i < j, 1.0, -1.0))
    B2 = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nE, len(tri)))
    B1 = sp.csr_matrix((np.r_[-np.ones(nE), np.ones(nE)], (np.r_[E[:, 0], E[:, 1]], np.r_[np.arange(nE), np.arange(nE)])), shape=(N, nE))
    used = np.zeros(N, bool); used[tri.ravel()] = True
    ncomp = connected_components(sp.csr_matrix((np.ones(nE), (E[:, 0], E[:, 1])), shape=(N, N)), directed=False)[0] - int((~used).sum())
    beta1 = ncomp - int(used.sum()) + nE - len(tri)
    V = (fr[s + 1] - fr[s - 1]) / (2 * dtf); Acc = (fr[s + 1] - 2 * fr[s] + fr[s - 1]) / dtf ** 2
    co = lambda Vv: 0.5 * ((Vv[E[:, 0]] + Vv[E[:, 1]]) * (P[E[:, 1]] - P[E[:, 0]])).sum(1)
    Wi = sp.diags(1 / np.maximum(w, 0.02 * np.median(w)))
    L0 = (B1 @ sp.diags(w) @ B1.T).tocsc() + sp.diags(1e-12 * np.ones(N)); Ld = (B2.T @ Wi @ B2).tocsc()
    def hd(c):
        cg = B1.T @ spsolve(L0, B1 @ (w * c)); cc = Wi @ (B2 @ spsolve(Ld, B2.T @ c)); return cg, cc, c - cg - cc
    ue, ae = co(V), co(Acc); ug, uc, uh = hd(ue); ag, ac, ah = hd(ae)
    ntri = np.asarray(abs(B2).sum(1)).ravel(); bE = np.where(ntri == 1)[0]
    mid = np.hypot(*(0.5 * (P[E[bE, 0]] + P[E[bE, 1]])).T); inner = bE[mid < 0.5 * (rmin + rmax)]
    sgn = np.asarray(B2[inner].sum(1)).ravel()                      # induced orientation (clockwise around the hole)
    per = lambda c: float(-(sgn * c[inner]).sum())                   # counter-clockwise period around the hole
    en = lambda c: float((w * c * c).sum())
    tt = tf[s]; dC = (Cth(tt + 1e-4) - Cth(tt - 1e-4)) / 2e-4
    rr = dict(t=float(tt), beta1=int(beta1), harm_share_u=en(uh) / en(ue), curl_share_u=en(uc) / en(ue),
              period_h_u=per(uh), C_theory=Cth(tt), circ_inner_u=per(ue), Gamma_inner_theory=float(Gam(rmin, tt)),
              period_h_a=per(ah), dCdt_theory=dC, circ_inner_a=per(ae), dGamma_inner_dt_theory=float(G0 * rmin ** 2 / (4 * nu * tt ** 2) * -np.exp(-rmin ** 2 / (4 * nu * tt))) * -1,
              harm_share_a=en(ah) / en(ae))
    res['frames'][s] = rr
    print(s, {k: ('%.5g' % v if isinstance(v, float) else v) for k, v in rr.items()}, flush=True)
json.dump(res, open('/home/claude/w/harm/res_harm_lo.json', 'w'), indent=1)
np.savez_compressed('/home/claude/w/harm/maps_harm.npz', P=fr[10], tri=tri, uh=uh, E=E)
