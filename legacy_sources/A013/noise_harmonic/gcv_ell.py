"""Noise-aware choice of the low-pass length for positive-definite network forms (conduction entropy production).
Smoother S_l = (M + l^2 L)^-1 M on the trajectory complex.  Generalised cross-validation
   GCV(l) = N ||(I - S_l) T||^2 / tr(I - S_l)^2          (trace by Hutchinson, 8 Rademacher probes)
needs no noise level and no DNS.  sigma^2 estimate = ||(I-S)T||^2 / tr(I-S) at the optimum."""
import numpy as np, json, sys
from scipy.sparse.linalg import splu
sys.argv = [sys.argv[0], 'visc']
src = open('visc_hodge.py').read(); exec(src[:src.index('res = {}; maps = {}')])
G = 1.4; K = MU_TRUE * G / ((G - 1) * 0.72)
z = np.load('entropy_visc.npz'); pr, rl, phiN, hotv = z['pr'], z['rl'], z['phiN'], z['hot']
C = json.load(open('res_entropy_cond.json')); tt = np.array(C['meas']['t'])
tri0 = X.complex_peeled(fr0[0])['tri']; m = Ed['rho'][0] * X._star(fr0[0], tri0, N0) / 3.0
heavy = np.hypot(fr0[0][:, 0] - 0.75, fr0[0][:, 1]) < 0.25
ells = np.array([0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025])
rng = np.random.default_rng(0); out = {}
for s in (15, 20, 30, 40, 50, 60, 65):
    P, tri, A, E, w, B1, B2, Ast = ops(s)
    T = np.r_[pr[s] / rl[s], pr[s] / rl[s]]; L0 = (B1 @ sp.diags(w) @ B1.T).tocsc(); Mv = np.maximum(Ast, 1e-12); Mm = sp.diags(Mv)
    Zp = rng.choice([-1.0, 1.0], size=(8, N)); g = []; sig = []; preds = []
    clean = ~hotv[s - 1:s + 2].any(0)
    for l in ells:
        lu = splu((Mm + l ** 2 * L0).tocsc()); Ts = lu.solve(Mv * T); r = T - Ts
        tr_ = np.mean([zz @ zz - zz @ lu.solve(Mv * zz) for zz in Zp])
        g.append(N * (r @ r) / tr_ ** 2); sig.append(float(np.sqrt((r @ r) / tr_)))
        cond = -K * (L0 @ Ts) / Mv; pred = (np.r_[phiN[s], phiN[s]] + cond) / (np.r_[rl[s], rl[s]] * Ts)
        preds.append(float((m * pred[:N0])[heavy & clean].sum()))
    k = int(np.argmin(g)); i = int(np.argmin(abs(tt - s * dt)))
    Td = Ed['p'][s] / Ed['rho'][s]; far = heavy & clean & (np.abs(np.hypot(fr0[0][:, 0] - 0.75, fr0[0][:, 1]) - 0.25) > 0.03)
    out[s] = dict(t=s * dt, ell_gcv=float(ells[k]), gcv=[float(x) for x in g], sigma_hat=sig[k], sigma_true=float(np.sqrt(np.mean((T[:N0] - Td)[far] ** 2))),
                  pred_gcv=preds[k], preds=preds, dns=C['meas']['dns_heavy'][i], meas=C['meas']['meas_heavy'][i])
    print(s, 'ell*=%.3f sigma_hat=%.4f sigma_true=%.4f  pred(ell*)=%.4f  dns=%.4f  | pred(0.01)=%.4f' % (ells[k], sig[k], out[s]['sigma_true'], preds[k], out[s]['dns'], preds[6]), flush=True)
json.dump(dict(ells=ells.tolist(), frames=out), open('res_gcv.json', 'w'), indent=1)
