"""Sod and cylindrical blast: exact compressible references.  Density/ln J, divergence, Hodge leakage,
spurious vorticity/LAVD (raw vs censored), position-only shock detection, edge-weight ablation."""
import sys, json, time, numpy as np, cases, lhd as L, lhdx as X
name = sys.argv[1]
c = cases.case_sod() if name == 'sod' else cases.case_blast()
fr, dt, t = c['frames'], c['dt'], c['t']; S, N = fr.shape[:2]
bad = L.vertex_bad_mask(c['ev_pidx'], c['ev_step'], N, S, halo=1)
kinds = ('cotan', 'bary', 'unit', 'gauss', 'invlen2')
t0 = time.time()
o = X.run_lhd(fr, dt, kinds=kinds, bad=bad, hodge_every=2, cplx='peel')
g = X.run_lhd(fr, dt, kinds=('galerkin',), op='galerkin', cplx='peel', bad=bad, hodge_every=1)
o['div']['galerkin'] = g['div']['galerkin']; o['Efrac']['galerkin'] = g['Efrac']['galerkin']; o['lnJ_dec'] = o['lnJ']; o['lnJ'] = g['lnJ']
kinds = kinds + ('galerkin',)
R = dict(N=N, S=S, t_lhd=time.time() - t0)
om_ls, th_ls = X.ls_vort_div(fr, dt, k=12)
inner = c['interior'](fr[0]) if name == 'blast' else c['interior'](fr[0])
s = S - 1
# ---------------- density at the last frame
if name == 'sod':
    rho_ex = c['rho']; rho0 = c['rho'][0]; m = c['mass']; m2 = rho0 * X.star_area(fr[0]) / 3.0
else:
    eta = 6.0
    rho0 = np.ones(N)                                      # pre-shock gas density 1 (inside R0: shocked at t0)
    rho0[c['rs'] <= c['R0']] = eta
    rho_ex = rho0[None, :] * np.exp(-c['lnJ_ex'])
    m = rho0 * c['h'] ** 2; m2 = rho0 * X.star_area(fr[0]) / 3.0
rho_lhd = rho0[None, :] * np.exp(-o['lnJ'])
fixed = L.complex_from_delaunay(fr[0])
lnJ_fix = L.log_jacobian(np.transpose(fr, (1, 0, 2)), fixed).T
rho_fix = rho0[None, :] * np.exp(-lnJ_fix)
rho_dtfe = X.dtfe_density(fr[s], m); rho_sph = X.sph_density(fr[s], m); rho_dtfe2 = X.dtfe_density(fr[s], m2); rho_sph2 = X.sph_density(fr[s], m2)
P = fr[s]; mI = inner & np.isfinite(rho_lhd[s])
if name == 'sod':
    xs = c['info']['Ssh'] * t[s]; xc = c['info']['us'] * t[s]
    reg = dict(all=mI, shock=mI & (np.abs(P[:, 0] - xs) < 4 * c['h']), contact=mI & (np.abs(P[:, 0] - xc) < 4 * c['h']),
               fan=mI & (np.abs(c['div_ex'][s]) > 0), uniform=mI & (np.abs(P[:, 0] - xs) > 4 * c['h']) & (np.abs(P[:, 0] - xc) > 4 * c['h']) & (c['div_ex'][s] == 0))
else:
    Rs = c['R0'] * np.sqrt(t[s] / t[0]); r = np.hypot(P[:, 0], P[:, 1])
    reg = dict(all=mI, shock=mI & (np.abs(r - Rs) < 3 * c['h']), post=mI & (r < Rs - 3 * c['h']), pre=mI & (r > Rs + 3 * c['h']))
R['rho_relerr'] = {}
for nm, est in [('LHD_chained', rho_lhd[s]), ('fixed_complex', rho_fix[s]), ('DTFE', rho_dtfe), ('SPH', rho_sph), ('DTFE_starmass', rho_dtfe2), ('SPH_starmass', rho_sph2)]:
    R['rho_relerr'][nm] = {k: float(np.nanmedian(np.abs(est[v] / rho_ex[s][v] - 1))) for k, v in reg.items()}
# ---------------- divergence in smooth region (exclude shock stencils)
okd = inner & ~bad[:, s - 1:s + 2].any(1) & np.isfinite(o['div']['cotan'][s - 1])
sm = okd & ((c['div_ex'][s - 1] > 0) if name == 'sod' else (np.hypot(fr[s - 1][:, 0], fr[s - 1][:, 1]) < c['R0'] * np.sqrt(t[s - 1] / t[0]) - 3 * c['h']))
R['div_smooth'] = {k: dict(r=X.corr(o['div'][k][s - 1], c['div_ex'][s - 1], sm), relL2=X.relL2(o['div'][k][s - 1], c['div_ex'][s - 1], sm),
                           relL2_fit=X.relL2(o['div'][k][s - 1], c['div_ex'][s - 1], sm, fit_scale=True)) for k in kinds}
R['div_smooth']['LS'] = dict(r=X.corr(th_ls[s - 1], c['div_ex'][s - 1], sm), relL2=X.relL2(th_ls[s - 1], c['div_ex'][s - 1], sm))
# ---------------- Hodge leakage (exact flow is a pure gradient): coexact energy fraction
R['curl_frac_max'] = {k: float(np.nanmax(o['Efrac'][k][:, 1])) for k in kinds}
R['curl_frac_med'] = {k: float(np.nanmedian(o['Efrac'][k][:, 1])) for k in kinds}
# ---------------- spurious vorticity / LAVD (exact = 0)
scale = float(np.nanmax(c['div_ex']))
R['spurious_vort_p99_over_maxdiv'] = dict(raw=float(np.nanquantile(np.abs(o['vort'][1:-1][:, inner]), 0.99) / scale),
                                          censored=float(np.nanquantile(np.abs(o['vort_c'][1:-1][:, inner]), 0.99) / scale),
                                          LS=float(np.nanquantile(np.abs(om_ls[1:-1][:, inner]), 0.99) / scale))
R['spurious_lavd_p99_rad'] = dict(raw=float(np.quantile(o['lavd'][inner], 0.99)), censored=float(np.quantile(o['lavd_c'][inner], 0.99)),
                                  LS=float(np.quantile(X.lavd_from_series(om_ls, dt)[inner], 0.99)))
# ---------------- position-only shock detection from the node channel (material-area ln J increments)
dlnJ = np.diff(o['lnJ'], axis=0)                     # (S-1, N)
d2 = dlnJ[:-1] + dlnJ[1:]                            # 2-step window
truth = np.zeros((S - 1, N), bool); truth[c['ev_step'], c['ev_pidx']] = True
hit = np.zeros(N, bool); fa = np.zeros(N, bool)
for tau in (0.25,):
    det = (dlnJ < -tau); det[:-1] |= (d2 < -tau); det[1:] |= (d2 < -tau)
    near = truth.copy(); near[1:] |= truth[:-1]; near[:-1] |= truth[1:]
    cr = np.zeros(N, bool); cr[c['ev_pidx']] = True
    R['detect'] = dict(tau=tau, rate=float((det & near).any(0)[cr & inner].mean()),
                       false_alarm_particles=float((det & ~near).any(0)[inner].mean()), false_alarm_never_crossed=float(det.any(0)[inner & ~cr & ~((np.abs(c['frames'][-1][:, 0]) < 1) if False else np.zeros(N, bool))].mean()),
                       lnJ_jump_median=float(np.median(o['lnJ'][-1][cr & inner] - o['lnJ'][0][cr & inner])),
                       lnJ_exact_median=float(np.median(c['lnJ_ex'][-1][cr & inner])))
R['lnJ_err'] = dict(chained_median=float(np.nanmedian(np.abs(o['lnJ'][-1] - c['lnJ_ex'][-1])[inner])),
                    chained_p95=float(np.nanquantile(np.abs(o['lnJ'][-1] - c['lnJ_ex'][-1])[inner], 0.95)),
                    fixed_median=float(np.nanmedian(np.abs(lnJ_fix[-1] - c['lnJ_ex'][-1])[inner])),
                    fixed_p95=float(np.nanquantile(np.abs(lnJ_fix[-1] - c['lnJ_ex'][-1])[inner], 0.95)),
                    old_chained_median=float(np.nanmedian(np.abs(o['lnJ_dec'][-1] - c['lnJ_ex'][-1])[inner])),
                    old_chained_p95=float(np.nanquantile(np.abs(o['lnJ_dec'][-1] - c['lnJ_ex'][-1])[inner], 0.95)),
                    chained_p95_off_shock=float(np.nanquantile(np.abs(o['lnJ'][-1] - c['lnJ_ex'][-1])[reg['all'] & ~reg['shock']], 0.95)))
print(json.dumps(R, indent=1, default=float))
json.dump(R, open('res_%s.json' % name, 'w'), indent=1, default=float)
np.savez_compressed('res_%s.npz' % name, P=fr[s], P0=fr[0], Pm=fr[s - 1], rho_ex=rho_ex[s], rho_lhd=rho_lhd[s], rho_fix=rho_fix[s],
                    rho_dtfe=rho_dtfe, rho_sph=rho_sph, div_ex=c['div_ex'][s - 1], div_cot=o['div']['cotan'][s - 1],
                    div_unit=o['div']['unit'][s - 1], div_gal=o['div']['galerkin'][s - 1], div_gauss=o['div']['gauss'][s - 1], div_ls=th_ls[s - 1],
                    vort_raw=o['vort'][s - 1], vort_c=o['vort_c'][s - 1], om_ls=om_ls[s - 1], lavd=o['lavd'], lavd_c=o['lavd_c'],
                    lnJ=o['lnJ'], lnJ_old=o['lnJ_dec'][-1], lnJ_ex=c['lnJ_ex'], lnJ_fix=lnJ_fix[-1], t=t, inner=inner, frames_sub=fr[:, ::7],
                    **{'Ef_' + k: o['Efrac'][k] for k in kinds})
