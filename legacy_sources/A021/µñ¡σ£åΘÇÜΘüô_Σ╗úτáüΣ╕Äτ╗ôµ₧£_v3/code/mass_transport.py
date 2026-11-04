"""Mass-weighted material transport on the trajectory network: validation.
V1 exact: mass flux through a fixed section (Sod) / circle (blast); reference = exact trajectories at very fine resolution.
V2 DNS : heavy-gas mass transport in the shock-bubble case vs Eulerian int rho*Y1 dA.
V3 dyn. systems: mass- vs area-weighted transfer operator (Ulam matrix) and its leading coherent pair."""
import json, pickle, numpy as np, cases, lhdx as X
R = {}
# ---------------- V1a Sod: per-unit-height mass flux in +x through x = Xs over [t0, t1]
def flux_x(fr, mpu, Xs):
    a, b = fr[0][:, 0], fr[-1][:, 0]
    return float((mpu * ((a <= Xs) & (b > Xs))).sum() - (mpu * ((a > Xs) & (b <= Xs))).sum())
ref = cases.case_sod(h=1e-5, Ly=1e-5, S=2)                              # one row of 1e5 labels: exact trajectories
R['sod'] = {}
for Xs in (0.0, 0.3):
    exact = flux_x(ref['frames'], ref['mass'] / 1e-5, Xs)
    row = {'exact': exact}
    for h in (0.008, 0.004, 0.002):
        c = cases.case_sod(h=h, S=2); row['h=%g' % h] = flux_x(c['frames'], c['mass'] / 0.12, Xs) / exact - 1
    R['sod']['X=%g' % Xs] = row
print('sod', json.dumps(R['sod']), flush=True)
# ---------------- V1b blast: outward mass flux through r = Rc over [t0, t1]
p, R0, t0, t1, eta = 1 / 2.4, 0.2, 1.0, 25.0, 6.0
def rad(rs, t):
    tc = t0 * (rs / R0) ** 2
    return np.where(rs <= R0, rs * (t / t0) ** p, np.where(t >= tc, rs * (np.maximum(t, tc) / tc) ** p, rs))
rs = np.linspace(1e-6, 1.2, 2_000_001); drs = rs[1] - rs[0]; dm = 2 * np.pi * rs * np.where(rs <= R0, eta, 1.0) * drs
R['blast'] = {}
for Rc in (0.6, 0.9):
    exact = float(dm[(rad(rs, t0) <= Rc) & (rad(rs, t1) > Rc)].sum() - dm[(rad(rs, t0) > Rc) & (rad(rs, t1) <= Rc)].sum())
    row = {'exact': exact}
    for h in (0.044, 0.022, 0.011):
        c = cases.case_blast(h=h, S=2); r0_, r1_ = (np.hypot(*c['frames'][k].T) for k in (0, -1))
        m = np.where(c['rs'] <= R0, eta, 1.0) * h * h
        row['h=%g' % h] = float((m * ((r0_ <= Rc) & (r1_ > Rc))).sum() - (m * ((r0_ > Rc) & (r1_ <= Rc))).sum()) / exact - 1
    R['blast']['R=%g' % Rc] = row
print('blast', json.dumps(R['blast']), flush=True)
# ---------------- V2 shock-bubble: heavy-gas mass to the right of x = X
D = pickle.load(open('/home/claude/bench/sb_dns.pkl', 'rb')); fr, Ed, grids, xc, yc, dx = D['frames'], D['E'], D['grids'], D['xc'], D['yc'], D['dx']; del D
mh = Ed['rho'][0] * X.star_area(fr[0]) / 3.0 * Ed['Y1'][0]               # heavy-gas mass carried by each tracer
R['sb'] = {'total_heavy_Lagr': float(mh.sum())}
for tk, s in ((0.6, 30), (1.0, 50), (1.4, 70)):
    W = grids[tk]['W'].astype(float); hv = W[0] * W[4] * dx * dx; colsum = hv.sum(0); cum = np.cumsum(colsum[::-1])[::-1]
    R['sb']['total_heavy_Euler_t=%.1f' % tk] = float(hv.sum())
    for q in (0.25, 0.5, 0.75):
        Xq = xc[np.argmin(np.abs(cum - q * hv.sum()))]
        lag = float(mh[fr[s][:, 0] > Xq].sum()) / float(mh.sum()); eul = float(hv[:, xc > Xq].sum()) / float(hv.sum())
        R['sb']['t=%.1f q=%.2f' % (tk, q)] = dict(X=float(Xq), lagrangian_fraction=lag, eulerian_fraction=eul)
print('sb', json.dumps(R['sb']), flush=True)
# ---------------- V3 transfer operator (Ulam) with mass vs area weights; leading coherent pair
def boxes(P, nx=24, ny=12):
    x0, x1, y0, y1 = P[:, 0].min(), P[:, 0].max() + 1e-9, P[:, 1].min(), P[:, 1].max() + 1e-9
    return np.minimum((ny * (P[:, 1] - y0) / (y1 - y0)).astype(int), ny - 1) * nx + np.minimum((nx * (P[:, 0] - x0) / (x1 - x0)).astype(int), nx - 1)
b0, b1 = boxes(fr[0]), boxes(fr[-1]); r0 = np.hypot(fr[0][:, 0] - 0.75, fr[0][:, 1]); bubble = r0 < 0.25
mass = Ed['rho'][0] * X.star_area(fr[0]) / 3.0; area = X.star_area(fr[0]) / 3.0
R['coherent'] = {}; CP = {}
for nm, w in (('mass', mass), ('area', area)):
    Pm = np.zeros((288, 288)); np.add.at(Pm, (b0, b1), w)
    pr, qc = Pm.sum(1), Pm.sum(0); ok_r, ok_c = pr > 0, qc > 0
    Q = Pm[np.ix_(ok_r, ok_c)] / np.sqrt(np.outer(pr[ok_r], qc[ok_c]))
    U, sv, Vt = np.linalg.svd(Q); u2 = np.zeros(288); u2[np.where(ok_r)[0]] = U[:, 1] / np.sqrt(pr[ok_r])
    f = u2[b0]; f = f if (f[bubble].mean() > 0) else -f; cut = f > 0
    R['coherent'][nm] = dict(sigma2=float(sv[1]), jaccard_with_bubble=float((cut & bubble).sum() / (cut | bubble).sum()),
                             heavy_mass_captured=float(mh[cut].sum() / mh.sum()))
    CP[nm] = f
print('coherent', json.dumps(R['coherent']), flush=True)
json.dump(R, open('res_mass_transport.json', 'w'), indent=1); pickle.dump(dict(P0=fr[0], P1=fr[-1], CP=CP, bubble=bubble), open('res_mass_transport.pkl', 'wb'))
