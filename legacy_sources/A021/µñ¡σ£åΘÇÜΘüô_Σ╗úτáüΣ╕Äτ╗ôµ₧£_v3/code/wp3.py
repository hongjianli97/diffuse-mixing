"""WP3: joint Dirac-Kelvin denoising of the torque 2-cochain.
T0 raw: d1 g (acceleration cochain, second differences)
T1 time (Kelvin): Savitzky-Golay (5 frames) derivative of material-triangle circulation d1 f (first-order signal)
T2 space (Dirac): low-pass with the Dirac-squared triangle block, argmin ||c-c_obs||^2_{W M2} + l^2 ||delta2 c||^2_{M1}
   (censored triangles = missing data, weight 0); T3 = T2 applied to T1."""
import json, pickle, numpy as np, scipy.sparse.linalg as sla, lhd as L, lhdx as X, dirac as DR
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.stats import spearmanr
from trajnet import TrajectoryNetwork as TN
def torques(net, s, ell_fac=1.5):
    P, cx = net.fr[s], net.K[s]; tri = cx['tri']; A = L.tri_signed_area(P, tri); N = net.N
    acc = DR.acceleration(net.fr, net.dt); c0 = DR.tri_circ(P, acc[s], tri)
    ok0 = ~net.bad[tri, s - 1:s + 2].any((1, 2))
    G = np.stack([DR.tri_circ(net.fr[q], net.V[q], tri) for q in range(s - 2, s + 3)]); c1 = (np.array([-2, -1, 0, 1, 2]) @ G) / (10 * net.dt)
    sg_ok = ~net.bad[tri, max(s - 3, 0):s + 4].any((1, 2)) & np.all([np.sign(L.tri_signed_area(net.fr[q], tri)) == np.sign(A) for q in range(s - 2, s + 3)], 0)
    B = DR.blocks(P, cx); h = np.median(np.linalg.norm(P[cx['edges'][:, 1]] - P[cx['edges'][:, 0]], axis=1)); l2 = (ell_fac * h) ** 2
    def lowpass(c, w):
        op = sla.LinearOperator((len(c), len(c)), matvec=lambda x: w * B['M2'] * x + l2 * B['M2'] * (B['d1'] @ B['lu'].solve(B['d1'].T @ (B['M2'] * x))))
        x, _ = sla.cg(op, w * B['M2'] * np.nan_to_num(c), x0=np.where(w > 0, np.nan_to_num(c), 0.0), maxiter=400); return x
    c2 = lowpass(c0, ok0.astype(float)); c3 = lowpass(c1, sg_ok.astype(float))
    allok = np.ones(len(tri), bool); node = lambda c, ok: DR.node_avg(N, c / A, np.abs(A), tri, ok)
    return dict(T0=node(c0, ok0), T1=node(c1, sg_ok), T2=node(c2, allok), T3=node(c3, allok))
R = {}
S = pickle.load(open('/home/claude/bench/sv_dns.pkl', 'rb')); TQ = {}
for mode in ('in', 'off'):
    n2 = TN(S['fr_' + mode][10:31], S['dt']); n2.events(); TQ[mode] = torques(n2, 10)
R['SV_in_vs_off_spearman'] = {}
for k in ('T0', 'T1', 'T2', 'T3'):
    a, b = TQ['in'][k], TQ['off'][k]; ok = np.isfinite(a) & np.isfinite(b)
    R['SV_in_vs_off_spearman'][k] = dict(spearman=float(spearmanr(a[ok], b[ok])[0]), coverage=float(ok.mean()))
print('SV', json.dumps(R['SV_in_vs_off_spearman']), flush=True)
D = pickle.load(open('/home/claude/bench/sb_dns.pkl', 'rb')); fr, dt, xc, yc, grids = D['frames'], D['dt'], D['xc'], D['yc'], D['grids']; del D
net = TN(fr, dt); net.events(); R['SB_vs_DNS'] = {}; SBT = {}
for tk, s in ((0.6, 30), (1.0, 50)):
    T = torques(net, s); W = grids[tk]['W'].astype(float); gyr, gxr = np.gradient(W[0], yc, xc); gyp, gxp = np.gradient(W[3], yc, xc)
    bt = RGI((yc, xc), (gxr * gyp - gyr * gxp) / W[0] ** 2, bounds_error=False, fill_value=0.0)(np.c_[fr[s][:, 1], fr[s][:, 0]])
    R['SB_vs_DNS']['t=%.1f' % tk] = {k: dict(corr=float(np.corrcoef(v[np.isfinite(v)], bt[np.isfinite(v)])[0, 1]), coverage=float(np.isfinite(v).mean())) for k, v in T.items()}
    SBT[tk] = dict(P=fr[s], dns=bt, **T)
print('SB', json.dumps(R['SB_vs_DNS']), flush=True)
json.dump(R, open('res_wp3.json', 'w'), indent=1); pickle.dump(dict(SV=TQ, SB=SBT, Pin=S['fr_in'][20], Poff=S['fr_off'][20]), open('res_wp3.pkl', 'wb'))
