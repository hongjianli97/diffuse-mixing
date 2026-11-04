import json, pickle, numpy as np, lhd as L, lhdx as X, dirac as DR
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.stats import spearmanr, rankdata
from trajnet import TrajectoryNetwork as TN
R = {}
def auc(pos, neg):
    v = np.r_[pos, neg]; r = rankdata(v); n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
# ---------------- WP1 (a): algebra and linear-field accuracy on a random cloud
rng = np.random.default_rng(0); P = rng.uniform(0, 1, (3000, 2)); cx = X.complex_peeled(P); B = DR.blocks(P, cx)
Am = np.array([[0.3, -0.7], [0.4, 0.5]]); U = P @ Am.T + [0.2, -0.1]; f = DR.cochain(P, U, cx['edges'])
y0, y1, y2 = DR.apply_D(B, np.zeros(len(P)), f, np.zeros(len(cx['tri'])))
inn = ((P > 0.1) & (P < 0.9)).all(1) & B['used']; div, om = np.trace(Am), Am[1, 0] - Am[0, 1]
x = [rng.normal(size=len(P)), rng.normal(size=len(f)), rng.normal(size=len(cx['tri']))]; y = [rng.normal(size=len(P)), rng.normal(size=len(f)), rng.normal(size=len(cx['tri']))]
Dx, Dy = DR.apply_D(B, *x), DR.apply_D(B, *y)
ip = lambda a, b: float(a[0] @ (B['M0'] * b[0]) + a[1] @ (B['M1'] @ b[1]) + a[2] @ (B['M2'] * b[2]))
gd = X.galerkin_div(P, cx, U)
R['WP1_algebra'] = dict(d1d0_max=float(np.abs((B['d1'] @ B['d0']).toarray()).max()),
                        selfadjoint_relerr=abs(ip(x, Dy) - ip(Dx, y)) / abs(ip(x, Dy)),
                        circulation_relerr=float(np.max(np.abs(y2 / B['A'] - om)) / abs(om)),
                        whitney_delta1_div_relerr_median=float(np.median(np.abs(-y0[inn] - div)) / abs(div)),
                        galerkin_div_relerr_median=float(np.median(np.abs(gd[inn] - div)) / abs(div)))
print('WP1 algebra', json.dumps(R['WP1_algebra']), flush=True)
# ---------------- WP1 (b): Dirac residual as contamination mask (shock-bubble)
D = pickle.load(open('/home/claude/bench/sb_dns.pkl', 'rb')); fr, Ed, dt = D['frames'], D['E'], D['dt']; xc, yc = D['xc'], D['yc']
GamE = D['Gam']; r0 = np.hypot(fr[0][:, 0] - D['xb'], fr[0][:, 1]); Rb = D['R']; grids = D['grids']; del D
net = TN(fr, dt); net.events(); Acc = DR.acceleration(fr, dt)
R['WP1_residual'] = {}
for s in (15, 30, 50):
    rD, rG, thC = DR.residual(net, s); badn = net.bad[:, s - 1:s + 2].any(1)
    for nm, r in (('dirac_whitney', rD), ('galerkin', rG)):
        ok = np.isfinite(r); R['WP1_residual'].setdefault(nm, {})['t=%.1f' % (s * dt)] = dict(
            AUC_shock_vs_clean=auc(np.abs(r[ok & badn]), np.abs(r[ok & ~badn])),
            median_rel_clean=float(np.median(np.abs(r[ok & ~badn])) / np.median(np.abs(thC[ok & ~badn]))))
print('WP1 residual', json.dumps(R['WP1_residual']), flush=True)
# ---------------- WP2 (a): torque maps vs DNS baroclinic
R['WP2_torque'] = {}; TQ = {}
for tk, s in ((0.6, 30), (1.0, 50)):
    tq = DR.torque_map(net, s, Acc); W = grids[tk]['W'].astype(float)
    gyr, gxr = np.gradient(W[0], yc, xc); gyp, gxp = np.gradient(W[3], yc, xc); bar = (gxr * gyp - gyr * gxp) / W[0] ** 2
    bt = RGI((yc, xc), bar, bounds_error=False, fill_value=0.0)(np.c_[fr[s][:, 1], fr[s][:, 0]])
    ok = np.isfinite(tq) & ~net.bad[:, s - 1:s + 2].any(1); st = ok & (np.abs(bt) > np.quantile(np.abs(bt[ok]), 0.9))
    R['WP2_torque']['t=%.1f' % tk] = dict(corr=float(np.corrcoef(tq[ok], bt[ok])[0, 1]), sign_agree_top10=float(np.mean(np.sign(tq[st]) == np.sign(bt[st]))))
    TQ[tk] = dict(P=fr[s], lhd=tq, dns=bt)
print('WP2 torque', json.dumps(R['WP2_torque']), flush=True)
# ---------------- WP2 (b): Kelvin split for the bubble + sheet region over [0.1, 0.7]
s0, s1 = 5, 35; ks = DR.kelvin_split(net, Acc, r0 < Rb + 0.05, s0, s1)
gE = lambda key, t: float(np.interp(t, GamE['t'], GamE[key]))
R['WP2_kelvin'] = dict(triangles_kept=ks['kept'], total_LHD=float(ks['total'].sum()), impulsive_LHD=float(ks['impulsive'].sum()), smooth_LHD=float(ks['smooth'].sum()),
                       dGamma_Euler=gE('Y2', s1 * dt) - gE('Y2', s0 * dt), baroclinic_integral_Euler=gE('bar', s1 * dt) - gE('bar', s0 * dt))
print('WP2 kelvin', json.dumps(R['WP2_kelvin']), flush=True)
# per-tracer deposition maps (on positions at s0)
tri = ks['tri']; N = net.N
imp = DR.node_avg(N, ks['impulsive'] / ks['A0'], np.abs(ks['A0']), tri, np.ones(len(tri), bool))
smo = DR.node_avg(N, ks['smooth'] / ks['A0'], np.abs(ks['A0']), tri, np.ones(len(tri), bool))
# ---------------- WP2 (c): robustness of the torque to tracking mode (shock-vortex, t=0.4)
S = pickle.load(open('/home/claude/bench/sv_dns.pkl', 'rb')); tqs = {}
for mode in ('in', 'off'):
    n2 = TN(S['fr_' + mode][10:31], S['dt']); n2.events(); tqs[mode] = DR.torque_map(n2, 10, DR.acceleration(n2.fr, n2.dt))
ok = np.isfinite(tqs['in']) & np.isfinite(tqs['off'])
R['WP2_tracking'] = dict(torque_in_vs_off_spearman=float(spearmanr(tqs['in'][ok], tqs['off'][ok])[0]))
print('WP2 tracking', json.dumps(R['WP2_tracking']), flush=True)
json.dump(R, open('res_dirac.json', 'w'), indent=1)
pickle.dump(dict(TQ=TQ, P0=fr[s0], imp=imp, smo=smo, GamE=GamE, s0=s0, s1=s1), open('res_dirac.pkl', 'wb'))
