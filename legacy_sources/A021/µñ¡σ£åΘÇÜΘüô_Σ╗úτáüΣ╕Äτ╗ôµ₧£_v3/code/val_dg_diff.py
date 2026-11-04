import json, numpy as np, cases, lhdx as X, netdiff as ND
from scipy.ndimage import map_coordinates
from scipy.fft import dctn, idctn
vel, vort, grad = cases.dg_fields()
def eulerian(kappa, T=10.0, nx=512, ny=256, dt=0.01, keep=(5.0, 10.0)):
    dxg = 2.0 / nx; xs = (np.arange(nx) + 0.5) * dxg; ys = (np.arange(ny) + 0.5) * dxg; XX, YY = np.meshgrid(xs, ys)
    c = 0.5 * (1 + np.tanh((XX - 1) / 0.1)); kx = np.pi * np.arange(nx) / 2.0; ky = np.pi * np.arange(ny) / 1.0
    damp = 1.0 / (1 + kappa * dt * (kx[None, :] ** 2 + ky[:, None] ** 2)); t = 0.0; out = {}
    P = np.c_[XX.ravel(), YY.ravel()]
    for n in range(int(round(T / dt))):
        um = vel(P - 0.5 * dt * vel(P, t + 0.5 * dt), t + 0.5 * dt); Pd = P - dt * um
        c = map_coordinates(c, [Pd[:, 1] / dxg - 0.5, Pd[:, 0] / dxg - 0.5], order=3, mode='nearest').reshape(ny, nx)
        c = idctn(dctn(c, norm='ortho') * damp, norm='ortho'); t += dt
        if any(abs(t - k) < 1e-9 for k in keep): out[round(t, 3)] = c.copy()
    return out, dxg
c = cases.case_dg(); fr, dt = c['frames'], c['dt']; S, N = fr.shape[:2]
c0 = 0.5 * (1 + np.tanh((fr[0][:, 0] - 1) / 0.1)); m = X.star_area(fr[0]) / 3.0; rho = np.ones((S, N))
R = {}
for kappa in (1e-3, 3e-4):
    E, dxg = eulerian(kappa)
    net = ND.run(fr, dt, c0, kappa, 'mass', m=m, rho=rho, keep=(25,))
    row = {}
    for tk, s in ((5.0, 25), (10.0, 50)):
        P = fr[s]; ref = map_coordinates(E[tk], [P[:, 1] / dxg - 0.5, P[:, 0] / dxg - 0.5], order=1, mode='nearest')
        inn = (P[:, 0] > .03) & (P[:, 0] < 1.97) & (P[:, 1] > .03) & (P[:, 1] < .97)
        e = lambda v: float(np.sqrt(np.mean((v[inn] - ref[inn]) ** 2)) / np.std(ref[inn]))
        var = lambda v: float(np.sum(m * (v - np.sum(m * v) / m.sum()) ** 2) / m.sum())
        row['t=%g' % tk] = dict(r_network=float(np.corrcoef(net[s][inn], ref[inn])[0, 1]), relL2_network=e(net[s]),
                                relL2_pure_advection=e(c0), variance_ratio_network_over_euler=var(net[s]) / var(ref))
    R['kappa=%g' % kappa] = row; print('kappa=%g' % kappa, json.dumps(row), flush=True)
json.dump(R, open('res_val_dg_diff.json', 'w'), indent=1)
