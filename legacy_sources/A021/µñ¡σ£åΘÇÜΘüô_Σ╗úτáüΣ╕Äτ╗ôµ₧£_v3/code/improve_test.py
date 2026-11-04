"""(a) user's elliptic metric on our LO run; (b) AFRC_H variants; (c) LHD-dome AFRC_E with absolute rotation gate."""
import json, pickle, time, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import paper_ell as PE, afrc_h_engine as AH, lhd as L, lhdx as X, cases
from dgbench import make_dg, rk4_traj, ftle_from_positions, AFRC_H_primitive, ridge_strength, _dgb_nz
R = {}
def gkcorr(P, a, b, box, h):
    ZA, _ = PE.gk_grid(P, a, box, n=(230, 205), sigph=2 * h); ZB, _ = PE.gk_grid(P, b, box, n=(230, 205), sigph=2 * h)
    m = np.isfinite(ZA) & np.isfinite(ZB); return float(np.corrcoef(ZA[m], ZB[m])[0, 1])
def box_of(P): return (P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max())
def spacing(P): return float(np.sqrt((np.ptp(P[:, 0]) * np.ptp(P[:, 1])) / len(P)))
# ---------------------------------------------------------------- C-channel events (positions only)
def events(fr):
    S, N = fr.shape[:2]; dl = np.zeros((S - 1, N))
    for s in range(S - 1):
        tri = X.complex_peeled(fr[s])['tri']; A0, A1 = X._star(fr[s], tri, N), X._star(fr[s + 1], tri, N)
        with np.errstate(all='ignore'): dl[s] = np.where(A0 > 0, np.log(A1 / A0), 0.0)
    det = dl < -0.2; det[:-1] |= (dl[:-1] + dl[1:]) < -0.2
    cs = det.T.copy(); cs[:, 1:] |= det.T[:, :-1]; cs[:, :-1] |= det.T[:, 1:]
    return cs                                                   # (N, S-1) censored steps
# ---------------------------------------------------------------- (b) AFRC_H variants
def robust_stretch(traj, cens, k=12):
    """node mean over ALL mutual-kNN edges (endpoints) of |median censored dlog d| / dt_step: no gates, no curvature."""
    E = np.unique(np.vstack([AH._mutual_knn(traj[:, 0], k), AH._mutual_knn(traj[:, -1], k)]), axis=0)
    AH._FORMAN_CENSOR_CURRENT = cens; AH._GRID_CELL = 0.5 * float(np.median(np.linalg.norm(traj[E[:, 0], 0] - traj[E[:, 1], 0], axis=1)))
    net, TV, med, J, ss, T, neff = AH._edge_logdist_stats_v2(traj, E); AH._FORMAN_CENSOR_CURRENT = None
    w = np.abs(np.nan_to_num(med)); N = traj.shape[0]; acc = np.zeros(N); cnt = np.zeros(N)
    np.add.at(acc, E[:, 0], w); np.add.at(acc, E[:, 1], w); np.add.at(cnt, E[:, 0], 1); np.add.at(cnt, E[:, 1], 1)
    return acc / np.maximum(cnt, 1)
def engineB(traj, cens):
    per = []
    for k in (12, 16, 24):
        B, _, _ = AH.afrc_h_field(traj, k=k, dt=1.0, censor=cens); B = np.maximum(B - np.median(B), 0); pos = B[B > 0]
        per.append(B / (np.quantile(pos, 0.95) if pos.size else 1))
    return np.max(np.vstack(per), 0)
T, dt, nx, ny = 15.0, 0.1, 200, 100; vel = make_dg(); xa, ya = np.linspace(0, 2, nx), np.linspace(0, 1, ny)
GX, GY = np.meshgrid(xa, ya); P0 = np.c_[GX.ravel(), GY.ravel()]; tr = rk4_traj(vel, P0, T, dt, 'fwd'); trg = tr.reshape(ny, nx, -1, 2)
RSF = ridge_strength(ftle_from_positions(trg[:, :, -1, :], xa[1] - xa[0], ya[1] - ya[0], T))
cor = lambda a, b: float(np.corrcoef(_dgb_nz(a[5:-5, 5:-5]).ravel(), _dgb_nz(b[5:-5, 5:-5]).ravel())[0, 1])
trs = np.ascontiguousarray(tr[:, ::2, :])
R['DG_RS_corr_vs_FTLE'] = dict(proxy=cor(ridge_strength(AFRC_H_primitive(trg)), RSF),
                               robust_stretch=cor(ridge_strength(robust_stretch(trs, None).reshape(ny, nx)), RSF),
                               engine_curvature=cor(ridge_strength(engineB(trs, None).reshape(ny, nx)), RSF))
print('DG', R['DG_RS_corr_vs_FTLE'], flush=True)
S = pickle.load(open('/home/claude/bench/sv_dns.pkl', 'rb')); x0 = S['P0'][:, 0]; y0 = S['P0'][:, 1]; lam = S['Ms'] * S['dt']; xe = S['xs0'] + S['Ms'] * S['t'][10]
quiet = (np.abs(y0 - S['yv']) > 0.22) & (x0 > 0.47) & (x0 < 1.03)
def stripe_edge(v, sc):
    bins = np.arange(0.47, 1.03, 0.005); xb = 0.5 * (bins[1:] + bins[:-1])
    pr = np.array([np.nanmean(v[(x0 >= bins[i]) & (x0 < bins[i + 1])]) for i in range(len(xb))]) / sc
    r_ = (pr - np.convolve(pr, np.ones(11) / 11, 'same'))[6:-6]; amp = np.abs(np.sum(r_ * np.exp(-2j * np.pi * xb[6:-6] / lam))) * 2 / len(r_)
    lo = quiet & (x0 > xe - 0.06) & (x0 < xe); hi = quiet & (x0 > xe) & (x0 < xe + 0.06)
    return float(amp), float((np.nanmean(v[hi]) - np.nanmean(v[lo])) / sc)
R['SV'] = {}; F_sv = {}
for mode in ('in', 'off'):
    fr = S['fr_' + mode][10:31]; cs = events(fr); trb = np.ascontiguousarray(np.transpose(fr, (1, 0, 2))[:, ::-1]); csb = cs[:, ::-1]
    flds = dict(FTLE_bwd=X.ls_ftle(fr[-1], fr[0], 0.4), proxy_bwd=AFRC_H_primitive(trb.reshape(120, 120, -1, 2)).ravel(),
                robust_stretch=robust_stretch(trb, csb), engine_curvature=engineB(trb, csb))
    F_sv[mode] = flds
    for k, v in flds.items():
        sc = np.nanquantile(np.abs(F_sv['in'][k]), 0.99); st, ed = stripe_edge(v, sc)
        R['SV'].setdefault(k, {})[mode] = dict(stripe=st, edge_step=ed)
print('SV', json.dumps(R['SV']), flush=True)
json.dump(R, open('res_improve_H.json', 'w'), indent=1); pickle.dump(dict(F_sv=F_sv), open('res_improve_H.pkl', 'wb'))
