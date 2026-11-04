import json, pickle, numpy as np, cases, lhd as L, lhdx as X, evo_lib as EV
def courant(fr, censor, coord, spacing):
    det = censor.any(1); fd = np.argmax(censor, axis=1); i = np.where(det)[0]; pos = coord(fr[fd[i] + 1][np.arange(len(i)), :] if False else fr[fd[i], i])
    a = np.polyfit(fd[i], pos, 1); return float(abs(a[0]) / spacing)
out = {}
# ---------------- Sod
cs = cases.case_sod(); fr, t = cs['frames'], cs['t']; P0 = fr[0]; S, N = fr.shape[:2]; side = lambda P: P[:, 0] > 0.0; kap = 1e-3
fine = cases.case_sod(h=1e-5, Ly=1e-5, S=S); a0 = side(fine['frames'][0]); mf = fine['mass'] / 1e-5 * 0.12
fref = np.array([np.sum(mf * (~a0 & side(fine['frames'][s]))) for s in range(S)])
c0f = lambda x0: 0.5 * (1 + np.tanh(x0 / 0.02)); F2 = cases.case_sod(h=2e-4, Ly=2e-4, S=401); o = np.argsort(F2['x0']); x0f = F2['x0'][o]
dmu = np.where(x0f > 0, 0.125, 1.0) * 2e-4; rF = F2['rho'][:, o]; tF = F2['t']
cond = lambda tq: (lambda r_: 2 * r_[1:] ** 2 * r_[:-1] ** 2 / (r_[1:] ** 2 + r_[:-1] ** 2))(rF[min(int(round((tq - tF[0]) / (tF[1] - tF[0]))), len(tF) - 1)])
cr = EV.lag1d(dmu, cond, c0f(x0f), tF, kap, rec=set(range(0, 401, 10))); vref = (np.array([tF[k] for k in sorted(cr)]), [EV.wvar(cr[k], dmu) for k in sorted(cr)])
idx = np.where(np.abs(cs['x0'] - 0.1) < cs['h'] / 2)[0]
r, aux = EV.run_case(dict(name='SOD', title='Sod shock tube (compressible, exact): transport, diffusion ($\\kappa=10^{-3}$) and mixing', fr=fr, t=t, tidx=(0, 10, 20, 40),
                          rho0=cs['rho'][0], compressible=True, kappa=kap, c0=c0f(cs['x0']), label=cs['x0'] < 0, label_names=('driven gas', 'driver gas'),
                          side=side, side_name='x > 0', iface=(idx, np.tile([0.0, 1.0], (N, 1))),
                          refs=dict(flux=(t, fref), var=vref, stretch=(t, np.zeros(S)), comp=(t, -cs['lnJ_ex'][:, idx].mean(1)), rho=cs['rho'])))
out['SOD'] = dict(windows=r['windows'], C_s_tracer_spacing=courant(fr, aux['censor'], lambda P: P[:, 0], cs['h'])); print('SOD', json.dumps(out['SOD']), flush=True)
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
                          refs=dict(flux=(t, fref), var=vref, stretch=(t, np.where(t > tc5, p * np.log(np.maximum(t, tc5) / tc5), 0.0)), comp=(t, -cb['lnJ_ex'][:, idx].mean(1)),
                                    rho=rho0[None, :] * np.exp(-cb['lnJ_ex']))))
out['BLAST'] = dict(windows=r['windows'], C_s_tracer_spacing=courant(fr, aux['censor'], lambda P: np.hypot(*P.T), cb['h'])); print('BLAST', json.dumps(out['BLAST']), flush=True)
# ---------------- shock-bubble (diffusive-scalar DNS run: same flow and seeds)
D = pickle.load(open('/home/claude/bench/sb_diff_dns.pkl', 'rb')); fr, t, Y, rhoD = D['frames'], D['t'], D['Y'], D['rho']; P0 = fr[0]; N = len(P0)
G = pickle.load(open('/home/claude/bench/sb_dns.pkl', 'rb')); grids, xc, dxg = G['grids'], G['xc'], G['dx']; Y1 = G['E']['Y1'][0]; del G
m = rhoD[0] * X.star_area(P0) / 3.0; vref = (t, [EV.wvar(Y[s], m) for s in range(len(t))])
fref_t, fref_v = [t[0]], [0.0]
for tk in (0.6, 1.0, 1.4):
    W = grids[tk]['W'].astype(float); fref_t.append(tk); fref_v.append(float((W[0] * W[4])[:, xc > 1.0].sum() * dxg * dxg))
d0 = P0 - [D['xb'], 0.0]; idx = np.where(np.abs(np.hypot(*d0.T) - D['R']) < 0.003)[0]; ang = np.arctan2(d0[:, 1], d0[:, 0])
r, aux = EV.run_case(dict(name='SB', title='Shock-heavy-bubble (compressible DNS): transport, diffusion ($\\kappa=10^{-3}$) and mixing', fr=fr, t=t, tidx=(0, 30, 50, 70),
                          rho0=rhoD[0], compressible=True, kappa=1e-3, c0=Y[0], label=Y1 > 0.5, label_names=('ambient gas', 'heavy (bubble) gas'),
                          side=lambda P: P[:, 0] > 1.0, side_name='x > 1 (heavy-gas mass)', flux_weight=Y1, iface=(idx, np.c_[-np.sin(ang), np.cos(ang)]),
                          refs=dict(flux=(np.array(fref_t), np.array(fref_v)), var=vref, comp=(t, np.log(rhoD[:, idx] / rhoD[0, idx]).mean(1)), rho=rhoD), coh_nb=(24, 12)))
out['SB'] = dict(windows=r['windows'], C_s_grid=courant(fr, aux['censor'], lambda P: P[:, 0], dxg)); print('SB', json.dumps(out['SB']), flush=True)
# ---------------- shock-vortex: in-loop (figure) with the offline density error overlaid
S = pickle.load(open('/home/claude/bench/sv_dns.pkl', 'rb')); fr, t = S['fr_in'][:31], S['t'][:31]; P0 = S['P0']; Ev = S['Ev']; rhoE = Ev['rho'][:31]
from trajnet import TrajectoryNetwork as TN
no = TN(S['fr_off'][:31], S['dt']); no.events(); rho_off = rhoE[0][None, :] * np.exp(-X.chained_lnJ_v2(S['fr_off'][:31], no.bad))
off_err = np.array([float(np.median(np.abs(rho_off[s] / rhoE[s] - 1))) for s in range(31)])
d0 = P0 - [S['xv'], S['yv']]; idx = np.where(np.abs(np.hypot(*d0.T) - 2 * S['Rv']) < 0.0025)[0]; ang = np.arctan2(d0[:, 1], d0[:, 0])
r, aux = EV.run_case(dict(name='SV', title='Shock-vortex (compressible DNS, in-loop tracking; offline density error overlaid): transport, diffusion and mixing', fr=fr, t=t, tidx=(0, 10, 20, 30),
                          rho0=rhoE[0], compressible=True, kappa=2e-4, c0=0.5 * (1 + np.tanh((P0[:, 0] - 0.75) / 0.03)), label=np.hypot(*d0.T) < 2 * S['Rv'],
                          label_names=('ambient', 'vortex core (r0<2Rv)'), side=lambda P: P[:, 0] > 0.9, side_name='x > 0.9', iface=(idx, np.c_[-np.sin(ang), np.cos(ang)]),
                          refs=dict(comp=(t, np.log(rhoE[:, idx] / rhoE[0, idx]).mean(1)), rho=rhoE, rho_extra=('offline tracking', off_err))))
out['SV'] = dict(windows=r['windows'], C_s_grid=courant(fr, aux['censor'], lambda P: P[:, 0], S.get('dx', 0.005)), offline_density_err_final=float(off_err[-1]))
print('SV', json.dumps(out['SV']), flush=True)
json.dump(out, open('res_evo_comp.json', 'w'), indent=1)
