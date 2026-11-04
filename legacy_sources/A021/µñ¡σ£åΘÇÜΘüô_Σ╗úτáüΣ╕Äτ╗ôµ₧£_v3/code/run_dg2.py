"""DG with the rebuilt operator (Galerkin inner product, outside-peeled complex) + DEC-star ablation on the same complex."""
import numpy as np, json, time, cases, lhd as L, lhdx as X, afrc_h_engine as AH
c = cases.case_dg(); fr, dt, S = c['frames'], c['dt'], len(c['t']); N = fr.shape[1]
R = json.load(open('res_dg.json')); old = dict(np.load('res_dg.npz'))
E = np.unique(np.vstack([AH._mutual_knn(fr[0], 24), AH._mutual_knn(fr[-1], 24)]), axis=0)
rho_var = np.where(fr[0][:, 0] < 1.0, 3.0, 1.0)
t0 = time.time()
g = X.run_lhd(fr, dt, kinds=('galerkin',), op='galerkin', cplx='peel', pairs=E, rho_frames=np.tile(rho_var, (S, 1)),
              rho_kinds=('galerkin',), hodge_every=1)
R['v2_time_per_snap'] = g['t_snap']
d = X.run_lhd(fr, dt, kinds=('cotan', 'bary', 'unit', 'gauss', 'invlen2'), cplx='peel', hodge_every=5)
tm = S // 2; Pm = fr[tm]; innm = (Pm[:, 0] > .05) & (Pm[:, 0] < 1.95) & (Pm[:, 1] > .05) & (Pm[:, 1] < .95)
rms = np.sqrt(np.mean(c['om_ex'][tm][innm] ** 2))
nz = lambda a: float(np.sqrt(np.nanmean(a[innm] ** 2)) / rms)
R['v2'] = dict(div_noise=nz(g['div']['galerkin'][tm]), Egrad_med=float(np.nanmedian(g['Efrac']['galerkin'][1:, 0])),
               Egrad_max=float(np.nanmax(g['Efrac']['galerkin'][1:, 0])), Egrad_t0=float(g['Efrac']['galerkin'][0, 0]),
               Egrad_rho_med=float(np.nanmedian(g['Efrac']['rho_galerkin'][1:, 0])), Egrad_rho_max=float(np.nanmax(g['Efrac']['rho_galerkin'][1:, 0])),
               vort_r=X.corr(g['vort'][tm], c['om_ex'][tm], innm))
R['ablation_peel'] = {k: dict(div_noise=nz(d['div'][k][tm]), Egrad_med=float(np.nanmedian(d['Efrac'][k][1:, 0])),
                              Egrad_max=float(np.nanmax(d['Efrac'][k][1:, 0]))) for k in d['div']}
ok = np.isfinite(g['s_g']).all(1)
Ag, Ac = np.abs(g['s_g'][ok]).sum(0), np.abs(g['s_c'][ok]).sum(0); share = Ag / (Ag + Ac + 1e-30)
R['v2']['pair_share_g_median'] = float(np.median(share))
inner = (fr[0][:, 0] > .05) & (fr[0][:, 0] < 1.95) & (fr[0][:, 1] > .05) & (fr[0][:, 1] < .95)
keep = AH.coherent_edge_keep(np.transpose(fr, (1, 0, 2)), E); Ek = E[keep]
sce = np.abs(np.nanmedian(g['s_c'][ok][:, keep], 0)); sce = np.maximum(sce, 1e-3 * np.median(sce[sce > 0]))
B_rb = AH._forman_node_from_weights(N, Ek, sce, AH._node_weight(N, Ek, sce)); B_rb[np.bincount(Ek.ravel(), minlength=N) < 3] = 0
R['afrcH']['AFRC_H_rebuilt_v2'] = dict(sp=X.spearman(B_rb, c['ftle_ex'], inner), J=X.jaccard_top(B_rb, c['ftle_ex'], 0.9, inner))
ang = np.deg2rad(10) * (c['t'] - c['t'][0]) / (c['t'][-1] - c['t'][0])
Q = np.stack([np.stack([np.cos(ang), -np.sin(ang)], -1), np.stack([np.sin(ang), np.cos(ang)], -1)], 1)
frq = np.einsum('sij,snj->sni', Q, fr) + (0.3 * c['t'])[:, None, None] * np.array([1.0, 0.5])
oa = X.run_lhd(fr, dt, kinds=('galerkin',), op='galerkin', cplx='peel', objective=True, hodge_every=5, pairs=E)
ob = X.run_lhd(frq, dt, kinds=('galerkin',), op='galerkin', cplx='peel', objective=True, hodge_every=5, pairs=E)
rel = lambda a, b: float(np.nanmax(np.abs(a - b)) / np.nanmax(np.abs(a)))
okk = np.isfinite(oa['s_c'][5:-5:5]) & np.isfinite(ob['s_c'][5:-5:5])
R['v2']['objectivity'] = dict(div=rel(oa['div']['galerkin'][1:-1], ob['div']['galerkin'][1:-1]), vort=rel(oa['vort'][1:-1], ob['vort'][1:-1]),
                              lavd=rel(oa['lavd'], ob['lavd']), s_c=float(np.max(np.abs(oa['s_c'][5:-5:5][okk] - ob['s_c'][5:-5:5][okk])) / np.max(np.abs(oa['s_c'][5:-5:5][okk]))))
R['v2_total'] = time.time() - t0
json.dump(R, open('res_dg.json', 'w'), indent=1, default=float)
old.update(div_gal=g['div']['galerkin'][tm], div_cot_peel=d['div']['cotan'][tm], afrcH_rb=B_rb, share_g=share,
           Efrac_gal=g['Efrac']['galerkin'], Efrac_gal_rho=g['Efrac']['rho_galerkin'], **{'Efrac_peel_' + k: v for k, v in d['Efrac'].items()})
np.savez_compressed('res_dg.npz', **old)
print(json.dumps({k: R[k] for k in ('v2', 'ablation_peel', 'afrcH')}, indent=1, default=float))
