"""visc_hodge.py <inv|visc> -- momentum and energy on the trajectory complex, positions (+ density at t=0) only.

f = rho a cochain (rho from the material ledger).  Continuum: rho a = -grad p + div tau, and for constant mu in 2D
    div tau = (4/3) mu grad(theta) - mu curl(omega)          (curl(w) = (dw/dy, -dw/dx))
so the Hodge split of f is:  gradient part  <->  -grad p + (4/3) mu grad theta
                             co-exact part  <->  -mu curl(omega)   (purely viscous; pressure and baroclinic drop out)
Two kinematic estimators of mu (no DNS field used):
    mu_E = -<f_c, u>_W / sum_T Gamma_T^2/A_T      (vortical-channel power / enstrophy:  <f_c,u> = -mu int omega^2)
    mu_T = LSQ fit of curl(f)_T/A_T = mu (lap omega)_T   (Kelvin form of the viscous torque)
Energy: <f,u>_W = <f_g,u_g>_W + <f_c,u_c>_W  (orthogonal channels) ; dissipation Phi = mu int(omega^2 + 4/3 theta^2) + bdry
DNS references (only for comparison): p, div tau, Phi at the tracers."""
import numpy as np, pickle, sys, json, time, os
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import lhdx as X, netmix as NM
case = sys.argv[1]; t0 = time.time()
if case == 'visc':
    R = pickle.load(open('/home/claude/w/visc/sbv_ck.pkl', 'rb'))['R']; fr0 = np.array(R['frames']); Ed = {k: np.array(R[k]) for k in ('rho', 'u', 'v', 'p', 'om', 'th', 'fvx', 'fvy', 'Phi')}
    MU_TRUE = 5e-4
else:
    o = pickle.load(open('/home/claude/w/LHD_v3.2/sb_dns_dx0.005.pkl', 'rb')); fr0 = o['frames']; Ed = {k: o['E'][k] for k in ('rho', 'u', 'v', 'p', 'om', 'th')}
    MU_TRUE = 0.0
S, N0, _ = fr0.shape; dt = 0.02; t = np.arange(S) * dt; N = 2 * N0
fr = np.concatenate([fr0, fr0 * np.array([1, -1])], axis=1)
# ledger (positions only)
LC = '/home/claude/w/visc/ledger_%s.npz' % case
if case == 'inv' and os.path.exists('/home/claude/w/hd/ledger_sb.npz'): LC = '/home/claude/w/hd/ledger_sb.npz'
if os.path.exists(LC):
    _l = np.load(LC); lnJ, hot = _l['lnJ'], _l['hot']
else:
    tris = [X.complex_peeled(fr0[s])['tri'] for s in range(S)]; dl = np.zeros((S - 1, N0))
    for s in range(S - 1):
        A0, A1 = X._star(fr0[s], tris[s], N0), X._star(fr0[s + 1], tris[s], N0)
        with np.errstate(all='ignore'): dl[s] = np.where(A0 > 0, np.log(A1 / A0), 0.0)
    det = dl < -0.2; det[:-1] |= (dl[:-1] + dl[1:]) < -0.2
    cs = det.T.copy(); cs[:, 1:] |= det.T[:, :-1]; cs[:, :-1] |= det.T[:, 1:]
    bad = np.zeros((N0, S), bool); bad[:, :-1] |= cs; bad[:, 1:] |= cs
    lnJ = X.chained_lnJ_v2(fr0, bad); hot = np.zeros((S, N0), bool); hot[:-1] |= det; hot[1:] |= det
    np.savez_compressed(LC, lnJ=lnJ, hot=hot)
rho0 = np.r_[Ed['rho'][0], Ed['rho'][0]]
print('ledger %.0fs' % (time.time() - t0), flush=True)


def ops(s):
    P = fr[s]; tri = X.complex_peeled(P)['tri']; E, w = NM.edge_weights(P, tri)
    a = 0.5 * ((P[tri[:, 1], 0] - P[tri[:, 0], 0]) * (P[tri[:, 2], 1] - P[tri[:, 0], 1]) - (P[tri[:, 1], 1] - P[tri[:, 0], 1]) * (P[tri[:, 2], 0] - P[tri[:, 0], 0]))
    tri[a < 0] = tri[a < 0][:, [0, 2, 1]]; A = np.abs(a); nE = len(E)
    kk = E[:, 0] * N + E[:, 1]; order = np.argsort(kk); ks = kk[order]
    rows, cols, vals = [], [], []
    for (p_, q_) in ((0, 1), (1, 2), (2, 0)):
        i, j = tri[:, p_], tri[:, q_]; k = np.minimum(i, j) * N + np.maximum(i, j)
        rows.append(order[np.searchsorted(ks, k)]); cols.append(np.arange(len(tri))); vals.append(np.where(i < j, 1.0, -1.0))
    B2 = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nE, len(tri)))
    B1 = sp.csr_matrix((np.r_[-np.ones(nE), np.ones(nE)], (np.r_[E[:, 0], E[:, 1]], np.r_[np.arange(nE), np.arange(nE)])), shape=(N, nE))
    return P, tri, A, E, w, B1, B2, X._star(P, tri, N) / 3.0


def cochain(P, E, V): return 0.5 * ((V[E[:, 0]] + V[E[:, 1]]) * (P[E[:, 1]] - P[E[:, 0]])).sum(1)


def hodge(c, B1, B2, w, Wi):
    L0 = (B1 @ sp.diags(w) @ B1.T).tocsc() + sp.diags(1e-12 * np.ones(B1.shape[0]))
    phi = spsolve(L0, B1 @ (w * c)); cg = B1.T @ phi
    chi = spsolve((B2.T @ Wi @ B2).tocsc(), B2.T @ c); cc = Wi @ (B2 @ chi)
    return phi, cg, cc


res = {}; maps = {}
frames_list = sorted(set(list(range(3, S - 1, 4)) + [15, 30, 50, 69]))
for s in frames_list:
    P, tri, A, E, w, B1, B2, Ast = ops(s); Wi = sp.diags(1.0 / np.maximum(w, 0.02 * np.median(w)))
    V = (fr[s + 1] - fr[s - 1]) / (2 * dt); Acc = (fr[s + 1] - 2 * fr[s] + fr[s - 1]) / dt ** 2
    rho = rho0 * np.exp(-np.r_[lnJ[s], lnJ[s]])
    ue = cochain(P, E, V); fe = cochain(P, E, rho[:, None] * Acc)
    phif, fg, fc = hodge(fe, B1, B2, w, Wi); phiu, ug, uc = hodge(ue, B1, B2, w, Wi)
    ip = lambda x, y: float((w * x * y).sum())
    Gam = B2.T @ ue; enst = float((Gam ** 2 / A).sum())                    # = int omega^2 (whole mirrored cloud)
    # material-area-rate divergence (ledger form) for the dilatational part
    Am, Ap = X._star(fr[s - 1], tri, N), X._star(fr[s + 1], tri, N)
    with np.errstate(all='ignore'): thL = np.where((Am > 0) & (Ap > 0), (np.log(Ap) - np.log(Am)) / (2 * dt), 0.0)
    hs = np.r_[hot[s], hot[s]]; ntri = np.asarray(abs(B2).sum(1)).ravel(); bnode = np.zeros(N, bool); bnode[E[ntri == 1].ravel()] = True
    smooth = ~hs & ~bnode
    dil2 = float((thL[smooth] ** 2 * Ast[smooth]).sum())
    r = dict(t=float(t[s]), P_total=ip(fe, ue), P_grad=ip(fg, ug), P_curl=ip(fc, uc), P_cross=ip(fe, ue) - ip(fg, ug) - ip(fc, uc),
             enstrophy=enst, dil2_smooth=dil2, mu_E=-ip(fc, uc) / enst)
    # torque route: curl(f)/A = mu lap(omega), dual-graph (cotangent) Laplacian of the face vorticity
    om = Gam / A; lap = -(B2.T @ (Wi @ (B2 @ om))) / A; cf = (B2.T @ fe) / A
    hotT = hs[tri].any(1); bT = np.asarray(abs(B2[ntri == 1]).sum(0)).ravel() > 0
    okT = ~hotT & ~bT
    r['mu_T'] = float((cf[okT] * lap[okT] * A[okT]).sum() / (lap[okT] ** 2 * A[okT]).sum())
    cen = P[tri].mean(1); nb = 0.02; key = np.floor(cen[okT, 0] / nb).astype(int) * 1000 + np.floor((cen[okT, 1] + 0.5) / nb).astype(int)
    _, inv = np.unique(key, return_inverse=True); Cb = np.bincount(inv, weights=(cf * A)[okT]); Lb = np.bincount(inv, weights=(lap * A)[okT])
    r['mu_T_binned'] = float((Cb * Lb).sum() / (Lb * Lb).sum()); r['torque_bin_corr'] = float(np.corrcoef(Cb, Lb)[0, 1])
    # pressure from positions (with and without the (4/3) mu theta correction, using the TRUE mu for the correction)
    pd = np.r_[Ed['p'][s], Ed['p'][s]]; org = np.arange(N) < N0
    for tag, prec in (('p', -phif), ('p_corr', -phif + 4 / 3 * MU_TRUE * thL)):
        prec = prec + (pd[org] - prec[org]).mean(); e = prec - pd
        r[tag + '_relrms_smooth'] = float(np.sqrt((e[org & smooth] ** 2).mean()) / pd[org & smooth].std())
    # DNS references
    if case == 'visc':
        FV = np.c_[np.r_[Ed['fvx'][s], Ed['fvx'][s]], np.r_[Ed['fvy'][s], -Ed['fvy'][s]]]; fve = cochain(P, E, FV)
        _, fvg, fvc = hodge(fve, B1, B2, w, Wi)
        r['dns_visc_curlpart_corr_with_fc'] = float(ip(fvc, fc) / np.sqrt(ip(fvc, fvc) * ip(fc, fc)))
        r['dns_visc_power_curl'] = ip(fvc, uc); r['dns_visc_power_total'] = ip(fve, ue)
        Phi = np.r_[Ed['Phi'][s], Ed['Phi'][s]]
        r['dns_Phi_int'] = float((Phi * Ast).sum()); r['dns_Phi_int_smooth'] = float((Phi * Ast)[smooth].sum())
        r['net_Phi_true_mu'] = MU_TRUE * (enst + 4 / 3 * dil2)
        r['dns_enstrophy_nodes'] = float((np.r_[Ed['om'][s], Ed['om'][s]] ** 2 * Ast).sum())
    if s in (30, 50):
        maps['cf_%d' % s] = cf; maps['lap_%d' % s] = lap; maps['tri_%d' % s] = tri; maps['P_%d' % s] = P
        maps['fcurlpot_%d' % s] = spsolve((B2.T @ Wi @ B2).tocsc(), B2.T @ fe); maps['om_%d' % s] = om
    r['n_hot'] = int(hot[s].sum()); res[s] = r
    print(s, {k: ('%.4g' % v if isinstance(v, float) else v) for k, v in r.items()}, '%.0fs' % (time.time() - t0), flush=True)
json.dump(res, open('/home/claude/w/visc/res_visc_hodge_%s.json' % case, 'w'), indent=1)
np.savez_compressed('/home/claude/w/visc/maps_visc_%s.npz' % case, **maps)
