"""hodge_dyn.py -- kinematics on the trajectory complex K(t) and its dynamical counterparts (shock-heavy bubble, inviscid DNS).

Input actually used: tracer positions (+ density at t=0 for the node masses).  DNS fields are read only as references.
Cochains on the mirrored, outside-peeled Delaunay complex (ghosts across y=0):
  u_e = (u_i+u_j)/2 . (x_j-x_i)          velocity 1-cochain, u_i = central difference of positions
  a_e = (a_i+a_j)/2 . (x_j-x_i)          acceleration 1-cochain, a_i = second difference of positions
  f_e = (rho_i a_i + rho_j a_j)/2 . dx   force-density 1-cochain, rho_i = rho0_i / J_i (material ledger)
Operators: B1 (node-edge), B2 (edge-face), Hodge stars *0 = A*/3, *1 = cot weights W, *2 = 1/A.
Identities / equivalences tested
  (K) Kelvin: B2^T a = d/dt circulation of each material face  ->  = int (grad rho x grad p)/rho^2 dA  (inviscid)
  (P) momentum: rho a = -grad p (inviscid) -> f is exact; its gradient potential is -p; curl part = viscous/numerical
  (H) Helmholtz/Kovasznay: u = grad phi (dilatational) + W^-1 B2 chi (vortical) + harmonic, orthogonal in the mass metric
"""
import numpy as np, pickle, sys, json, time
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import lhdx as X

t0 = time.time()
z = np.load('/home/claude/w/sb_lite_m.npz'); d = pickle.load(open('/home/claude/w/sb_net_m.pkl', 'rb'))
o = pickle.load(open('/home/claude/w/LHD_v3.2/sb_dns_dx0.005.pkl', 'rb'))
fr = z['frames']; N0 = int(z['N0']); S = fr.shape[0]; dt = float(z['dt']); t = z['t']; N = fr.shape[1]
E_dns = o['E']; rho0 = np.r_[E_dns['rho'][0], E_dns['rho'][0]]

# ---------------- material ledger (positions only): one-step star-area shock detection + chained anchors
fr0 = fr[:, :N0]
import os
if os.path.exists('/home/claude/w/hd/ledger_sb.npz'):
    _l = np.load('/home/claude/w/hd/ledger_sb.npz'); lnJ, hot = _l['lnJ'], _l['hot']
else:
  tris = [X.complex_peeled(fr0[s])['tri'] for s in range(S)]
  dl = np.zeros((S - 1, N0))
  for s in range(S - 1):
    A0, A1 = X._star(fr0[s], tris[s], N0), X._star(fr0[s + 1], tris[s], N0)
    with np.errstate(all='ignore'): dl[s] = np.where(A0 > 0, np.log(A1 / A0), 0.0)
  det = dl < -0.2; det[:-1] |= (dl[:-1] + dl[1:]) < -0.2
  cs = det.T.copy(); cs[:, 1:] |= det.T[:, :-1]; cs[:, :-1] |= det.T[:, 1:]
  bad = np.zeros((N0, S), bool); bad[:, :-1] |= cs; bad[:, 1:] |= cs
  lnJ = X.chained_lnJ_v2(fr0, bad)
  hot = np.zeros((S, N0), bool); hot[:-1] |= det; hot[1:] |= det
  np.savez_compressed('/home/claude/w/hd/ledger_sb.npz', lnJ=lnJ, hot=hot)
print('ledger %.0fs' % (time.time() - t0), flush=True)


def ops(s):
    P = fr[s]; tri = d['tri'][s].copy(); E = d['E'][s]; w = d['w'][s]
    a = 0.5 * ((P[tri[:, 1], 0] - P[tri[:, 0], 0]) * (P[tri[:, 2], 1] - P[tri[:, 0], 1]) - (P[tri[:, 1], 1] - P[tri[:, 0], 1]) * (P[tri[:, 2], 0] - P[tri[:, 0], 0])); flip = a < 0; tri[flip] = tri[flip][:, [0, 2, 1]]; A = np.abs(a)
    nE = len(E); key = {}
    kk = E[:, 0] * N + E[:, 1]; order = np.argsort(kk); ks = kk[order]
    def eidx(i, j):
        lo, hi = np.minimum(i, j), np.maximum(i, j); k = lo * N + hi; pos = np.searchsorted(ks, k)
        return order[pos], np.where(i < j, 1.0, -1.0)
    rows, cols, vals = [], [], []
    for (p_, q_) in ((0, 1), (1, 2), (2, 0)):
        e, sg = eidx(tri[:, p_], tri[:, q_]); rows.append(e); cols.append(np.arange(len(tri))); vals.append(sg)
    B2 = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nE, len(tri)))
    B1 = sp.csr_matrix((np.r_[-np.ones(nE), np.ones(nE)], (np.r_[E[:, 0], E[:, 1]], np.r_[np.arange(nE), np.arange(nE)])), shape=(N, nE))
    Ast = X._star(P, tri, N) / 3.0
    return P, tri, A, E, w, B1, B2, Ast


def cochain(P, E, V):
    return 0.5 * ((V[E[:, 0]] + V[E[:, 1]]) * (P[E[:, 1]] - P[E[:, 0]])).sum(1)


def hodge(c, B1, B2, w, Wfloor):
    """orthogonal decomposition of the 1-cochain c in the metric diag(w)."""
    W = sp.diags(w); Wi = sp.diags(1.0 / np.maximum(w, Wfloor))
    L0 = (B1 @ W @ B1.T).tocsc() + sp.diags(1e-12 * np.ones(B1.shape[0]))
    phi = spsolve(L0, B1 @ (w * c)); cg = B1.T @ phi
    Ld = (B2.T @ Wi @ B2).tocsc(); chi = spsolve(Ld, B2.T @ c); cc = Wi @ (B2 @ chi)
    ch = c - cg - cc
    en = lambda x: float((w * x * x).sum())
    return dict(phi=phi, chi=chi, g=cg, c=cc, h=ch, E=en(c), Eg=en(cg), Ec=en(cc), Eh=en(ch))


def bar_grid(W):
    dx = float(o['dx']); Wp = np.pad(W, ((0, 0), (1, 1), (1, 1)), mode='edge'); Wp[2, 0, :] = -Wp[2, 1, :]
    ddx = lambda q: (q[1:-1, 2:] - q[1:-1, :-2]) / (2 * dx); ddy = lambda q: (q[2:, 1:-1] - q[:-2, 1:-1]) / (2 * dx)
    return (ddx(Wp[0]) * ddy(Wp[3]) - ddy(Wp[0]) * ddx(Wp[3])) / W[0] ** 2


res = dict(series={}, frames={}); maps = {}
grid_s = {round(float(k) / dt): k for k in o['grids'].keys()}
series_s = list(range(3, S - 1, 4)) + [30, 50]
for s in sorted(set(series_s + [15, 69])):
    P, tri, A, E, w, B1, B2, Ast = ops(s)
    V = (fr[s + 1] - fr[s - 1]) / (2 * dt); Acc = (fr[s + 1] - 2 * fr[s] + fr[s - 1]) / dt ** 2
    rho = rho0 * np.exp(-np.r_[lnJ[s], lnJ[s]])
    ue, ae = cochain(P, E, V), cochain(P, E, Acc); fe = cochain(P, E, rho[:, None] * Acc)
    rhoe = 0.5 * (rho[E[:, 0]] + rho[E[:, 1]]); wf = 0.02 * np.median(w)
    Hu = hodge(ue, B1, B2, w * rhoe, wf)                       # mass metric: Helmholtz split of kinetic energy
    Ha = hodge(ae, B1, B2, w, wf); Hf = hodge(fe, B1, B2, w, wf)
    p_rec = -Hf['phi']; pd = np.r_[E_dns['p'][s], E_dns['p'][s]]
    org = np.arange(N) < N0; p_rec += (pd[org] - p_rec[org]).mean()      # one additive constant
    err = p_rec - pd; hs = np.r_[hot[s], hot[s]]
    r = dict(t=float(t[s]), KE_nodes=float(0.5 * (d['m'] * (V ** 2).sum(1)).sum()), KE_cochain=0.5 * Hu['E'],
             frac_u_grad=Hu['Eg'] / Hu['E'], frac_u_curl=Hu['Ec'] / Hu['E'], frac_u_harm=Hu['Eh'] / Hu['E'],
             frac_a_curl=Ha['Ec'] / Ha['E'], frac_f_curl=Hf['Ec'] / Hf['E'],
             p_relrms=float(np.sqrt((err[org] ** 2).mean()) / pd[org].std()),
             p_relrms_offshock=float(np.sqrt((err[org & ~hs] ** 2).mean()) / pd[org & ~hs].std()),
             p_corr=float(np.corrcoef(p_rec[org], pd[org])[0, 1]), n_hot=int(hot[s].sum()))
    # node divergence / face vorticity vs DNS at tracers
    theta = -(B1 @ (w * ue)) / Ast; omT = (B2.T @ ue) / A
    omN = np.zeros(N); cnt = np.zeros(N)
    for k in range(3): np.add.at(omN, tri[:, k], omT * A); np.add.at(cnt, tri[:, k], A)
    omN = omN / np.maximum(cnt, 1e-30)
    inner = org & (Ast > 0) & ~hs
    r['corr_theta'] = float(np.corrcoef(theta[inner], np.r_[E_dns['th'][s], E_dns['th'][s]][inner])[0, 1])
    r['corr_omega'] = float(np.corrcoef(omN[inner], np.r_[E_dns['om'][s], E_dns['om'][s]][inner])[0, 1])
    # Kelvin: curl of the acceleration cochain = circulation rate of each material face
    tq = (B2.T @ ae) / A; cen = P[tri].mean(1); orgT = (tri < N0).all(1)
    if s in grid_s:
        g = o['grids'][grid_s[s]]; bar = bar_grid(g['W'].astype(np.float64))
        itp = RegularGridInterpolator((o['yc'], o['xc']), bar, bounds_error=False, fill_value=0.0)
        bT = itp(np.c_[np.abs(cen[:, 1]), cen[:, 0]]) * np.sign(cen[:, 1] + 1e-30)   # bar is odd in y
        # region integrals: heavy gas (label at t=0) and whole half-cloud
        heavyT = orgT & ((np.hypot(fr[0][tri][:, :, 0] - 0.75, fr[0][tri][:, :, 1]) < 0.25).all(1))
        bandT = orgT & ~heavyT & ((np.hypot(fr[0][tri][:, :, 0] - 0.75, fr[0][tri][:, :, 1]) < 0.25).any(1))
        dxg = float(o['dx'])
        # DNS integral over the same material region: integrate bar on the grid inside the triangles (centroid rule
        # at grid resolution is unreliable for interface-concentrated torque) -> compare dGamma/dt of region boundary
        r['kelvin_heavy_net'] = float((tq * A)[heavyT].sum()); r['kelvin_heavy_dnsbar_centroid'] = float((bT * A)[heavyT].sum())
        r['kelvin_heavyband_net'] = float((tq * A)[heavyT | bandT].sum()); r['kelvin_heavyband_dns'] = float((bT * A)[heavyT | bandT].sum())
        r['kelvin_cloud_net'] = float((tq * A)[orgT].sum()); r['kelvin_cloud_dns'] = float((bT * A)[orgT].sum())
        # coarse-grained field comparison (bins of 0.02)
        nb = 0.02; ix = np.floor(cen[orgT, 0] / nb).astype(int); iy = np.floor(cen[orgT, 1] / nb).astype(int)
        key = ix * 1000 + iy; u_, inv = np.unique(key, return_inverse=True)
        Bn = np.bincount(inv, weights=(tq * A)[orgT]); Bd = np.bincount(inv, weights=(bT * A)[orgT])
        r['torque_bin_corr'] = float(np.corrcoef(Bn, Bd)[0, 1])
        r['torque_bin_slope'] = float((Bn * Bd).sum() / (Bd * Bd).sum())
        maps['bar_dns_%d' % s] = bT; maps['tq_net_%d' % s] = tq
    if s in (30, 50, 69):
        maps['p_rec_%d' % s] = p_rec; maps['p_dns_%d' % s] = pd; maps['phi_%d' % s] = Hu['phi']; maps['chi_%d' % s] = Hu['chi']
        maps['tq_net_%d' % s] = tq; maps['omT_%d' % s] = omT; maps['theta_%d' % s] = theta
        maps['fcurlT_%d' % s] = (B2.T @ fe) / A
    res['frames'][s] = r
    print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, '%.0fs' % (time.time() - t0), flush=True)
json.dump(res, open('/home/claude/w/hd/res_hodge_dyn.json', 'w'), indent=1)
np.savez_compressed('/home/claude/w/hd/maps_hodge_dyn.npz', **maps)
