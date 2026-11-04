"""Split-sample (cross) estimator of a positive-definite network form.
Tracers are split into two interleaved sets A, B (checkerboard on the seeding lattice).  Each set gets its own complex,
ledger, pressure reconstruction and temperature T = p/rho -- so their noises are independent.  On A's complex
    D_cross = k sum_e w_e (dlnT_A)_e (dlnT_B->A)_e        (lnT_B linearly interpolated to A's nodes)
is free of the noise bias that inflates  D_raw = k sum_e w_e (dlnT_A)_e^2  (E[n_A n_B] = 0).  No smoothing length.
Reference: the same form with DNS temperature at the tracers (DNS T is smooth at tracer scale)."""
import numpy as np, json, sys, pickle, time
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.interpolate import LinearNDInterpolator
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import lhdx as X, netmix as NM
t0 = time.time(); R = pickle.load(open('/home/claude/w/visc/sbv_ck.pkl', 'rb'))['R']
fr0 = np.array(R['frames']); Ed = {k: np.array(R[k]) for k in ('rho', 'p')}; S, N0, _ = fr0.shape; dt = 0.02
MU = 5e-4; G = 1.4; K = MU * G / ((G - 1) * 0.72)
nx, ny = 190, 84; col = np.arange(N0) % nx; row = np.arange(N0) // nx; parity = (col + row) % 2
probe_all = np.where((fr0[0][:, 0] < 0.36) & (fr0[0][:, 1] > 0.36))[0]


def ledger(frs):
    n = frs.shape[1]; tris = [X.complex_peeled(frs[s])['tri'] for s in range(S)]; dl = np.zeros((S - 1, n))
    for s in range(S - 1):
        A0, A1 = X._star(frs[s], tris[s], n), X._star(frs[s + 1], tris[s], n)
        with np.errstate(all='ignore'): dl[s] = np.where(A0 > 0, np.log(A1 / A0), 0.0)
    det = dl < -0.2; det[:-1] |= (dl[:-1] + dl[1:]) < -0.2
    cs = det.T.copy(); cs[:, 1:] |= det.T[:, :-1]; cs[:, :-1] |= det.T[:, 1:]
    bad = np.zeros((n, S), bool); bad[:, :-1] |= cs; bad[:, 1:] |= cs
    hot = np.zeros((S, n), bool); hot[:-1] |= det; hot[1:] |= det
    return X.chained_lnJ_v2(frs, bad), hot


def graph(P):
    n = len(P); tri = X.complex_peeled(P)['tri']; E, w = NM.edge_weights(P, tri); nE = len(E)
    B1 = sp.csr_matrix((np.r_[-np.ones(nE), np.ones(nE)], (np.r_[E[:, 0], E[:, 1]], np.r_[np.arange(nE), np.arange(nE)])), shape=(n, nE))
    return tri, E, w, B1


def temperature(frs_m, rho, s, idx_probe, n0):
    """pressure from positions on the mirrored set, T = p / rho (original nodes)."""
    P = frs_m[s]; tri, E, w, B1 = graph(P)
    Acc = (frs_m[s + 1] - 2 * frs_m[s] + frs_m[s - 1]) / dt ** 2
    fe = 0.5 * (((rho[:, None] * Acc)[E[:, 0]] + (rho[:, None] * Acc)[E[:, 1]]) * (P[E[:, 1]] - P[E[:, 0]])).sum(1)
    L0 = (B1 @ sp.diags(w) @ B1.T).tocsc() + sp.diags(1e-12 * np.ones(len(P))); p = -spsolve(L0, B1 @ (w * fe))
    return p, (tri, E, w)


subs = {}
for tag, sel in (('A', np.where(parity == 0)[0]), ('B', np.where(parity == 1)[0])):
    f0 = fr0[:, sel]; lnJ, hot = ledger(f0); fm = np.concatenate([f0, f0 * np.array([1, -1])], axis=1)
    subs[tag] = dict(sel=sel, f0=f0, fm=fm, lnJ=lnJ, hot=hot, rho0=Ed['rho'][0][sel], probe=np.isin(sel, probe_all))
    print(tag, len(sel), 'ledger %.0fs' % (time.time() - t0), flush=True)
out = {}
for s in (20, 30, 40, 50, 60):
    lnT = {}; geo = {}
    for tag, d in subs.items():
        n = len(d['sel']); rho = np.r_[d['rho0'], d['rho0']] * np.exp(-np.r_[d['lnJ'][s], d['lnJ'][s]])
        p, g = temperature(d['fm'], rho, s, d['probe'], n)
        pr = np.r_[d['probe'], d['probe']]; p = p + (np.r_[Ed['p'][s][d['sel']], Ed['p'][s][d['sel']]][pr].mean() - p[pr].mean())
        lnT[tag] = np.log(p / rho); geo[tag] = g
    PA = subs['A']['fm'][s]; PB = subs['B']['fm'][s]; triA, EA, wA = geo['A']
    lnTB_A = LinearNDInterpolator(PB, lnT['B'])(PA); ok_n = np.isfinite(lnTB_A)
    nA = len(subs['A']['sel']); orgE = (EA < nA).all(1) & ok_n[EA].all(1)
    hotA = np.r_[subs['A']['hot'][s], subs['A']['hot'][s]]; cleanE = orgE & ~hotA[EA].any(1)
    dA = lnT['A'][EA[:, 1]] - lnT['A'][EA[:, 0]]; dB = lnTB_A[EA[:, 1]] - lnTB_A[EA[:, 0]]
    Td = np.log(np.r_[Ed['p'][s][subs['A']['sel']], Ed['p'][s][subs['A']['sel']]] / np.r_[Ed['rho'][s][subs['A']['sel']], Ed['rho'][s][subs['A']['sel']]])
    dD = Td[EA[:, 1]] - Td[EA[:, 0]]
    r = {}
    for nm, msk in (('all', orgE), ('smooth', cleanE)):
        r['D_raw_' + nm] = float(K * (wA * dA * dA)[msk].sum()); r['D_cross_' + nm] = float(K * (wA * dA * dB)[msk].sum())
        r['D_dns_' + nm] = float(K * (wA * dD * dD)[msk].sum())
    r['t'] = s * dt; out[s] = r
    print(s, {k: '%.4g' % v for k, v in r.items()}, '%.0fs' % (time.time() - t0), flush=True)
json.dump(out, open('/home/claude/w/visc/res_cross_dirichlet.json', 'w'), indent=1)
