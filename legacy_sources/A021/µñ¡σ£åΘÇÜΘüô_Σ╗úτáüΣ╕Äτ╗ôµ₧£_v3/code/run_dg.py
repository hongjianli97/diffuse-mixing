"""Double gyre: LHD vs exact vs LS-kNN vs AFRC (orig / rebuilt weights); weight ablation; variable-density test."""
import numpy as np, time, json, pickle, cases, lhd as L, lhdx as X
import afrc_engine as AF, afrc_h_engine as AH
R = {}
t0 = time.time()
c = cases.case_dg()
fr, dt, S = c['frames'], c['dt'], len(c['t'])
N = fr.shape[1]; print('N', N, 'gen %.0fs' % (time.time() - t0), flush=True)
# fixed material pairs (AFRC_H graph: mutual 24-NN at both ends) for attribution
E = np.unique(np.vstack([AH._mutual_knn(fr[0], 24), AH._mutual_knn(fr[-1], 24)]), axis=0)
rho_var = np.where(fr[0][:, 0] < 1.0, 3.0, 1.0)            # material density label (variable-density test)
t1 = time.time()
o = X.run_lhd(fr, dt, kinds=('cotan', 'bary', 'unit', 'gauss', 'invlen2'), pairs=E,
              rho_frames=np.tile(rho_var, (S, 1)), rho_kinds=('cotan',), hodge_every=5, verbose=True)
R['lhd_time_per_snap'] = o['t_snap']; print('lhd %.0fs' % (time.time() - t1), flush=True)
# baselines
om_ls, th_ls = X.ls_vort_div(fr, dt, k=12)
lavd_ls = X.lavd_from_series(om_ls, dt)
ftle_ls = X.ls_ftle(fr[0], fr[-1], c['t'][-1] - c['t'][0])
# interior mask (labels), away from walls
P0 = fr[0]; inner = (P0[:, 0] > 0.05) & (P0[:, 0] < 1.95) & (P0[:, 1] > 0.05) & (P0[:, 1] < 0.95)
tm = S // 2
Pm = fr[tm]; innm = (Pm[:, 0] > 0.05) & (Pm[:, 0] < 1.95) & (Pm[:, 1] > 0.05) & (Pm[:, 1] < 0.95)
R['vort'] = dict(stokes=dict(r=X.corr(o['vort'][tm], c['om_ex'][tm], innm), relL2=X.relL2(o['vort'][tm], c['om_ex'][tm], innm)),
                 ls=dict(r=X.corr(om_ls[tm], c['om_ex'][tm], innm), relL2=X.relL2(om_ls[tm], c['om_ex'][tm], innm)))
rms_om = np.sqrt(np.nanmean(c['om_ex'][tm][innm] ** 2))
R['div_noise_over_rms_vort'] = {k: float(np.sqrt(np.nanmean(o['div'][k][tm][innm] ** 2)) / rms_om) for k in o['div']}
R['div_noise_over_rms_vort']['ls'] = float(np.sqrt(np.mean(th_ls[tm][innm] ** 2)) / rms_om)
R['Efrac_grad_max'] = {k: float(np.nanmax(v[:, 0])) for k, v in o['Efrac'].items()}
R['Efrac_grad_med'] = {k: float(np.nanmedian(v[:, 0])) for k, v in o['Efrac'].items()}
R['lavd'] = dict(stokes=dict(r=X.corr(o['lavd'], c['lavd_ex'], inner), sp=X.spearman(o['lavd'], c['lavd_ex'], inner),
                             J=X.jaccard_top(o['lavd'], c['lavd_ex'], 0.9, inner)),
                 ls=dict(r=X.corr(lavd_ls, c['lavd_ex'], inner), sp=X.spearman(lavd_ls, c['lavd_ex'], inner),
                         J=X.jaccard_top(lavd_ls, c['lavd_ex'], 0.9, inner)))
# attribution: incompressible -> gradient share must be ~0
sg, sc = o['s_g'], o['s_c']
ok = np.isfinite(sg).all(1)
Ag, Ac = np.abs(sg[ok]).sum(0), np.abs(sc[ok]).sum(0)
R['pair_share_g_median'] = float(np.median(Ag / (Ag + Ac + 1e-300)))
# ---------------- AFRC_H: original weights vs rebuilt (coexact-part stretch) weights
t1 = time.time()
B_orig, win_orig, kf = AH.afrc_h_field(np.transpose(fr, (1, 0, 2)), k=24, dt=c['t'][-1] - c['t'][0])
R['afrcH_time'] = time.time() - t1
# rebuilt: same graph & coherence gate, edge weight = |median_t s_c| (s_c on fixed pairs E)
keep = AH.coherent_edge_keep(np.transpose(fr, (1, 0, 2)), E)
Ek = E[keep]
sc_e = np.abs(np.nanmedian(sc[ok][:, keep], 0)); sc_e = np.maximum(sc_e, 1e-3 * np.median(sc_e[sc_e > 0]))
w_v = AH._node_weight(N, Ek, sc_e)
B_rb = AH._forman_node_from_weights(N, Ek, sc_e, w_v)
deg = np.bincount(Ek.ravel(), minlength=N); B_rb[deg < 3] = 0.0
R['afrcH'] = {}
for nm, F in [('FTLE_ls', ftle_ls), ('AFRC_H', B_orig), ('AFRC_H_rebuilt', B_rb)]:
    R['afrcH'][nm] = dict(r=X.corr(F, c['ftle_ex'], inner), sp=X.spearman(F, c['ftle_ex'], inner),
                          J=X.jaccard_top(F, c['ftle_ex'], 0.9, inner))
R['afrcH']['orig_vs_rebuilt_r'] = X.corr(B_orig, B_rb, inner)
# ---------------- AFRC_E: original input vs Stokes (Dirac B2^T f) node rotation input
t1 = time.time()
FE = AF.afrc_fields(np.transpose(fr, (1, 0, 2)))
R['afrcE_time'] = time.time() - t1
orig_turn = AF._frc5_edge_turning
TH = 0.5 * np.nan_to_num(o['vort'][:-1] + o['vort'][1:]) * 0.5 * dt   # node rotation per step = omega dt / 2
def stokes_turning(Pg, Ee, chunk=AF.FRC5_CHUNK_E):
    _, rln = orig_turn(Pg, Ee, chunk)
    d = 0.5 * (TH[:, Ee[:, 0]] + TH[:, Ee[:, 1]])
    return np.ascontiguousarray(d.T.astype(np.float32)), rln
AF._frc5_edge_turning = stokes_turning
FE2 = AF.afrc_fields(np.transpose(fr, (1, 0, 2)))
AF._frc5_edge_turning = orig_turn
R['afrcE'] = {}
for nm, F in [('Stokes_LAVD', o['lavd']), ('LS_LAVD', lavd_ls), ('AFRC_E', FE['core_med']), ('AFRC_E_stokes', FE2['core_med']),
              ('AFRC_E_zint', FE['zint_or_med'])]:
    R['afrcE'][nm] = dict(sp=X.spearman(F, c['lavd_ex'], inner), J=X.jaccard_top(F, c['lavd_ex'], 0.9, inner))
# ---------------- noise robustness (PTV-like, 0.05 h)
rng = np.random.default_rng(9)
frn = fr + 0.05 * c['h'] * rng.normal(size=fr.shape)
on = X.run_lhd(frn, dt, kinds=('cotan',), hodge_every=10)
om_lsn, _ = X.ls_vort_div(frn, dt, k=12)
R['noise05h'] = dict(stokes_vort_r=X.corr(on['vort'][tm], c['om_ex'][tm], innm), ls_vort_r=X.corr(om_lsn[tm], c['om_ex'][tm], innm),
                     stokes_lavd_sp=X.spearman(on['lavd'], c['lavd_ex'], inner),
                     ls_lavd_sp=X.spearman(X.lavd_from_series(om_lsn, dt), c['lavd_ex'], inner))
# ---------------- objectivity: rotation 10 deg over the window + boost
ang = np.deg2rad(10) * (c['t'] - c['t'][0]) / (c['t'][-1] - c['t'][0])
Q = np.stack([np.stack([np.cos(ang), -np.sin(ang)], -1), np.stack([np.sin(ang), np.cos(ang)], -1)], 1)
frq = np.einsum('sij,snj->sni', Q, fr) + (0.3 * c['t'])[:, None, None] * np.array([1.0, 0.5])
oa = X.run_lhd(fr, dt, kinds=('cotan',), objective=True, hodge_every=5, pairs=E)
ob = X.run_lhd(frq, dt, kinds=('cotan',), objective=True, hodge_every=5, pairs=E)
rel = lambda a, b: float(np.nanmax(np.abs(a - b)) / np.nanmax(np.abs(a)))
R['objectivity_relmax'] = dict(div=rel(oa['div']['cotan'][1:-1], ob['div']['cotan'][1:-1]),
                               vort=rel(oa['vort'][1:-1], ob['vort'][1:-1]), lavd=rel(oa['lavd'], ob['lavd']),
                               s_c=rel(oa['s_c'][1:-1:5], ob['s_c'][1:-1:5]))
R['total_time'] = time.time() - t0
json.dump(R, open('res_dg.json', 'w'), indent=1, default=float)
np.savez_compressed('res_dg.npz', P0=P0, Pm=Pm, tm=tm, om_ex=c['om_ex'][tm], om_st=o['vort'][tm], om_ls=om_ls[tm],
                    div_cot=o['div']['cotan'][tm], div_unit=o['div']['unit'][tm], div_ls=th_ls[tm],
                    lavd_ex=c['lavd_ex'], lavd_st=o['lavd'], lavd_ls=lavd_ls, afrcE=FE['core_med'], afrcE2=FE2['core_med'],
                    ftle_ex=c['ftle_ex'], ftle_ls=ftle_ls, afrcH=B_orig, afrcH_rb=B_rb,
                    Efrac_cot=o['Efrac']['cotan'], Efrac_rho=o['Efrac']['rho_cotan'], Efrac_unit=o['Efrac']['unit'],
                    t=c['t'])
print(json.dumps(R, indent=1, default=float))
