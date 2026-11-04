import json, pickle, numpy as np, cases, lhd as L, lhdx as X, evo_lib as EV
def courant(fr, censor, coord, spacing):
    det = censor.any(1); fd = np.argmax(censor, axis=1); i = np.where(det)[0]; pos = coord(fr[fd[i] + 1][np.arange(len(i)), :] if False else fr[fd[i], i])
    a = np.polyfit(fd[i], pos, 1); return float(abs(a[0]) / spacing)
out = {}
# ---------------- blast
cb = cases.case_blast(); fr, t = cb['frames'], cb['t']; P0 = fr[0]; S, N = fr.shape[:2]; p, R0 = 1 / 2.4, 0.2; kap = 2e-4
rho0 = np.where(cb['rs'] <= R0, 6.0, 1.0); side = lambda P: np.hypot(*P.T) > 0.6
def kin(rs, tq):
    tc = (rs / R0) ** 2; ins = rs <= R0; post = (~ins) & (tq >= tc)
    r_ = np.where(ins, rs * tq ** p, np.where(post, rs * (np.maximum(tq, tc) / tc) ** p, rs))
    lj = np.where(ins, 2 * p * np.log(tq), np.where(post, -np.log(6.0) + 2 * p * np.log(np.maximum(tq, tc) / tc), 0.0)); return r_, np.where(ins, 6.0, 1.0) * np.exp(-lj)
rsF = np.linspace(1e-4, 1.2, 6000); dr = rsF[1] - rsF[0]; dmu = 2 * np.pi * rsF * np.where(rsF <= R0, 6.0, 1.0) * dr
rsd = np.linspace(1e-6, 1.2, 1_000_001); dmd = 2 * np.pi * rsd * np.where(rsd <= R0, 6.0, 1.0) * (rsd[1] - rsd[0]); r00 = kin(rsd, 1.0)[0]
fref = np.array([float(dmd[(r00 <= 0.6) & (kin(rsd, tq)[0] > 0.6)].sum()) for tq in t]) * (np.sum(rho0 * X.star_area(P0) / 3.0) / float(dmd[r00 <= 1.0].sum()) if False else 1.0)
def condb(tq):
    r_, rh = kin(rsF, tq); k = 4 * np.pi ** 2 * r_ ** 2 * rh ** 2; return 2 * k[1:] * k[:-1] / (k[1:] + k[:-1])
c0f = lambda rs: 0.5 * (1 + np.tanh((rs - 0.5) / 0.04)); tsF = np.linspace(1.0, 25.0, 2401); cr = EV.lag1d(dmu, condb, c0f(rsF), tsF, kap, rec=set(range(0, 2401, 40)))
vref = (np.array([tsF[k] for k in sorted(cr)]), [EV.wvar(cr[k], dmu) for k in sorted(cr)])
idx = np.where(np.abs(cb['rs'] - 0.5) < cb['h'] / 2)[0]; ang = np.arctan2(P0[:, 1], P0[:, 0]); tc5 = (0.5 / R0) ** 2
r, aux = EV.run_case(dict(name='BLAST', title='Cylindrical blast wave (compressible, exact): transport, diffusion ($\\kappa=2\\times10^{-4}$) and mixing', fr=fr, t=t, tidx=(0, 15, 30, 60),
                          rho0=rho0, compressible=True, kappa=kap, c0=c0f(cb['rs']), label=cb['rs'] < 0.5, label_names=('outer gas', 'inner gas (r0<0.5)'),
                          side=side, side_name='r > 0.6', iface=(idx, np.c_[-np.sin(ang), np.cos(ang)]),
                          render='scatter', refs=dict(flux=(t, fref), var=vref, stretch=(t, np.where(t > tc5, p * np.log(np.maximum(t, tc5) / tc5), 0.0)), comp=(t, -cb['lnJ_ex'][:, idx].mean(1)),
                                    rho=rho0[None, :] * np.exp(-cb['lnJ_ex']))))
out['BLAST'] = dict(windows=r['windows'], C_s_tracer_spacing=courant(fr, aux['censor'], lambda P: np.hypot(*P.T), cb['h'])); print('BLAST', json.dumps(out['BLAST']), flush=True)
