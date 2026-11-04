"""conduction term with a graph low-pass on T: T_s = (M + l^2 L)^-1 M T  (heat-kernel smoothing over length l).
The conduction entropy production is the Dirichlet form k sum_e w (ln T_j - ln T_i)^2 >= 0, so tracer noise in T
shows up as spurious positive production; smoothing removes it.  Also reports the noise level of T_rec vs DNS."""
import numpy as np, json, sys
sys.argv = [sys.argv[0], 'visc']
src = open('visc_hodge.py').read(); exec(src[:src.index('res = {}; maps = {}')])
G = 1.4; CV = 2.5; K = MU_TRUE * G / ((G - 1) * 0.72)
z = np.load('entropy_visc.npz'); pr, rl, phiN, hotv = z['pr'], z['rl'], z['phiN'], z['hot']
tri0 = X.complex_peeled(fr0[0])['tri']; m = Ed['rho'][0] * X._star(fr0[0], tri0, N0) / 3.0
heavy = np.hypot(fr0[0][:, 0] - 0.75, fr0[0][:, 1]) < 0.25
s_rec = CV * (np.log(pr) - G * np.log(rl)); s_dns = CV * (np.log(Ed['p']) - G * np.log(Ed['rho']))
out = {ell: dict(t=[], pred_heavy=[], cond_heavy=[], prod_cond=[], corr_field=[]) for ell in (0.0, 0.005, 0.01, 0.02)}
meas = dict(t=[], meas_heavy=[], dns_heavy=[], meas_all=[], dns_all=[], Tnoise_heavy=[])
for s in range(3, S - 3):
    P, tri, A, E, w, B1, B2, Ast = ops(s)
    p2 = np.r_[pr[s], pr[s]]; r2 = np.r_[rl[s], rl[s]]; T = p2 / r2
    L0 = (B1 @ sp.diags(w) @ B1.T).tocsc(); Mm = sp.diags(np.maximum(Ast, 1e-12))
    clean = ~hotv[s - 1:s + 2].any(0) & np.isfinite(s_rec[s - 1]) & np.isfinite(s_rec[s + 1])
    rm = (s_rec[s + 1] - s_rec[s - 1]) / (2 * dt); rd_ = (s_dns[s + 1] - s_dns[s - 1]) / (2 * dt)
    meas['t'].append(s * dt); meas['meas_heavy'].append(float((m * rm)[heavy & clean].sum())); meas['dns_heavy'].append(float((m * rd_)[heavy & clean].sum()))
    meas['meas_all'].append(float((m * rm)[clean].sum())); meas['dns_all'].append(float((m * rd_)[clean].sum()))
    Td = Ed['p'][s] / Ed['rho'][s]; sel = heavy & clean & (np.abs(np.hypot(fr0[0][:, 0] - 0.75, fr0[0][:, 1]) - 0.25) > 0.03)
    meas['Tnoise_heavy'].append(float(np.sqrt(np.mean(((T[:N0] - Td) / Td)[sel] ** 2))))
    orgE = (E < N0).all(1)
    for ell, d_ in out.items():
        Ts = T if ell == 0 else spsolve((Mm + ell ** 2 * L0).tocsc(), Mm @ T)
        cond = -K * (L0 @ Ts) / np.maximum(Ast, 1e-30); pred = (np.r_[phiN[s], phiN[s]] + cond) / (r2 * Ts)
        d_['t'].append(s * dt); d_['pred_heavy'].append(float((m * pred[:N0])[heavy & clean].sum()))
        d_['cond_heavy'].append(float((m * cond[:N0] / (rl[s] * Ts[:N0]))[heavy & clean].sum()))
        d_['prod_cond'].append(float(K * (w * (np.log(Ts[E[:, 1]]) - np.log(Ts[E[:, 0]])) ** 2)[orgE].sum()))
        if s in (40, 50, 60):
            key = np.floor(P[:N0][clean, 0] / 0.02).astype(int) * 1000 + np.floor(P[:N0][clean, 1] / 0.02).astype(int)
            _, inv = np.unique(key, return_inverse=True); W_ = np.bincount(inv, weights=m[clean]); b = lambda v: np.bincount(inv, weights=(m * v)[clean]) / W_
            d_['corr_field'].append([s, float(np.corrcoef(b(pred[:N0]), b(rm))[0, 1]), float(np.corrcoef(b(pred[:N0]), b(rd_))[0, 1])])
json.dump(dict(meas=meas, smooth={str(k): v for k, v in out.items()}), open('res_entropy_cond.json', 'w'), indent=1)
tt = np.array(meas['t'])
for tq in (0.4, 0.6, 0.8, 1.0, 1.2, 1.3):
    i = np.argmin(abs(tt - tq))
    print('t=%.2f meas %.4f dns %.4f | ' % (tt[i], meas['meas_heavy'][i], meas['dns_heavy'][i]) + ' '.join('l=%g:%.4f' % (l, out[l]['pred_heavy'][i]) for l in out) + ' | Tnoise %.4f' % meas['Tnoise_heavy'][i])
for l in out: print('ell', l, 'field corr', out[l]['corr_field'], 'prod_cond@1.0', out[l]['prod_cond'][np.argmin(abs(tt - 1.0))])
print('meas_all@1.0', meas['meas_all'][np.argmin(abs(tt - 1.0))], 'dns_all', meas['dns_all'][np.argmin(abs(tt - 1.0))])
