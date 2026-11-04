"""Is the offline-tracking density error numerical or theoretical?
(1) 1D model: smooth moving shock (tanh, half-width delta), exact vs offline trajectories (snapshots every dt_s, linear in time
    and space, RK4 substeps) and shock-fitted interpolation; scan the snapshot shock Courant number C_s = U_s dt_s / dx.
(2) shock-vortex DNS: first-order identity  delta lnJ = div(delta x)  between offline and in-loop trajectories."""
import json, pickle, numpy as np, lhdx as X
from trajnet import TrajectoryNetwork as TN
eta = 2.4 * 1.69 / (0.4 * 1.69 + 2); Us = 1.0; up = Us * (1 - 1 / eta); dx = 0.0025; h = dx; delta = 1.0 * dx; s0, T = 0.25, 1.0
Xl = np.arange(0.3, 1.2, h); xg = np.arange(0, 2.0, dx)
u_ex = lambda x, t: up * 0.5 * (1 - np.tanh((x - (s0 + Us * t)) / delta))
def rk4(f, x, t0, t1, n):
    dt = (t1 - t0) / n; t = t0
    for _ in range(n):
        k1 = f(x, t); k2 = f(x + .5 * dt * k1, t + .5 * dt); k3 = f(x + .5 * dt * k2, t + .5 * dt); k4 = f(x + dt * k3, t + dt); x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6; t += dt
    return x
x_true = rk4(u_ex, Xl.copy(), 0.0, T, 20000)
lnJ = lambda x: np.log((x[2:] - x[:-2]) / (2 * h))
crossed = Xl[1:-1] < s0 + Us * T - 0.05
R = {'eta': eta, 'up_over_Us': up / Us, 'model': {}}
for Cs in (0.25, 0.5, 1.0, 2.0, 5.2, 10.0, 20.0):
    dts = Cs * dx / Us; nsnap = int(np.ceil(T / dts)) + 1; tn = np.arange(nsnap) * dts; U = np.array([u_ex(xg, t) for t in tn])
    def u_lin(x, t):
        n = min(int(t / dts), nsnap - 2); th = t / dts - n
        return (1 - th) * np.interp(x, xg, U[n]) + th * np.interp(x, xg, U[n + 1])
    def u_fit(x, t):                      # shock-fitted: interpolate in the frame attached to the shock
        n = min(int(t / dts), nsnap - 2); th = t / dts - n; st = s0 + Us * t
        return (1 - th) * np.interp(x - (st - (s0 + Us * tn[n])), xg, U[n]) + th * np.interp(x - (st - (s0 + Us * tn[n + 1])), xg, U[n + 1])
    row = {}
    for nm, f in (('linear', u_lin), ('shock_fitted', u_fit)):
        x_off = rk4(f, Xl.copy(), 0.0, T, max(int(np.ceil(T / (dts / 8))), 2000))
        d = (lnJ(x_off) - lnJ(x_true))[crossed]; dxe = (x_off - x_true)[1:-1][crossed]
        row[nm] = dict(median_abs_dlnJ=float(np.median(np.abs(d))), mean_dlnJ=float(d.mean()), rms_dx_over_h=float(np.sqrt(np.mean(dxe ** 2)) / h))
    R['model']['Cs=%g' % Cs] = row
    print('Cs=%5.2f  linear: med|dlnJ| %.4f mean %.1e  rms dx/h %.3f | shock-fitted: med|dlnJ| %.5f' % (Cs, row['linear']['median_abs_dlnJ'], row['linear']['mean_dlnJ'], row['linear']['rms_dx_over_h'], row['shock_fitted']['median_abs_dlnJ']), flush=True)
# ---------------- (2) DNS identity
S = pickle.load(open('/home/claude/bench/sv_dns.pkl', 'rb')); s = 30; L = {}
for mode in ('in', 'off'):
    fr = S['fr_' + mode][:31]; n2 = TN(fr, S['dt']); n2.events(); L[mode] = X.chained_lnJ_v2(fr, n2.bad)[s]
P = S['fr_in'][s]; cx = X.complex_peeled(P); dxv = S['fr_off'][s] - P
div = X.galerkin_div(P, cx, dxv); dl = L['off'] - L['in']; ok = np.isfinite(div) & np.isfinite(dl); ok[X.open_star_nodes(cx)] = False
A = X.star_area(P); a = np.polyfit(div[ok], dl[ok], 1)
R['DNS'] = dict(pearson_dlnJ_vs_div_dx=float(np.corrcoef(div[ok], dl[ok])[0, 1]), regression_slope=float(a[0]),
                median_abs_dlnJ=float(np.median(np.abs(dl[ok]))), area_weighted_mean_dlnJ=float(np.sum(A[ok] * dl[ok]) / np.sum(A[ok])),
                median_abs_dx_over_h=float(np.median(np.linalg.norm(dxv, axis=1)) / 0.005))
print('DNS', json.dumps({k: round(v, 4) for k, v in R['DNS'].items()}), flush=True)
json.dump(R, open('res_offline_theory.json', 'w'), indent=1)
