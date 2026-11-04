import json, pickle, time, numpy as np, cases, lhdx as X, paper_ell as PE
from scipy.stats import spearmanr
from trajnet import TrajectoryNetwork as TN
from improve_test_helpers import gkcorr, spacing, box_of
R, F = {}, {}
def line(name, d):
    R[name] = d; print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)
def full(fr, dt, merge, censor=True, attribution=True):
    t0 = time.time(); net = TN(fr, dt)
    if censor: net.events()
    net.lhd(attribution=attribution); net.rotation(); net.robust_stretch(); net.curvature_topology(); net.vortex_dome(merge)
    net.sec = time.time() - t0; return net
# ---------------- 1) Lamb-Oseen (paper setup)
G, a = PE._ELL_G, 1.0; uta = (G / (2 * np.pi * a)) * (1 - np.exp(-1)); Tc = 2 * np.pi * a / uta
vel, _ = PE.lamb_oseen([(-3., 0.), (3., 0.)], G, a); Pc = PE.seeds_disk(4.5, 2000)
trc = PE.rk4_traj(vel, Pc, 3 * Tc, 3 * Tc / 60, False, "fwd"); lv = PE.LAVD_from_traj(trc, vel, 3 * Tc, 3 * Tc / 60, False, "fwd")
net = full(np.transpose(trc, (1, 0, 2)), 3 * Tc / 60, 2.0, censor=False)
line('LO', dict(dome_corr_LAVD=gkcorr(Pc, net.dome, lv, box_of(Pc), spacing(Pc)), cores=len(net.cores), edges_G=len(net.E), edges_afrc=len(net.E_afrc), sec=net.sec))
F['LO'] = dict(P=Pc, dome=net.dome, cores=net.cores)
# ---------------- 2) double gyre
c = cases.case_dg(); net = full(c['frames'], c['dt'], 0.3, censor=False); P0 = c['frames'][0]; tm = c['frames'].shape[0] // 2
inn = (P0[:, 0] > .05) & (P0[:, 0] < 1.95) & (P0[:, 1] > .05) & (P0[:, 1] < .95); Pm = c['frames'][tm]
innm = (Pm[:, 0] > .05) & (Pm[:, 0] < 1.95) & (Pm[:, 1] > .05) & (Pm[:, 1] < .95); rms = np.sqrt(np.mean(c['om_ex'][tm][innm] ** 2))
sg, sc_ = net.L_['s_g'], net.L_['s_c']; ok = np.isfinite(sg).all(1); Ag, Ac = np.abs(sg[ok]).sum(0), np.abs(sc_[ok]).sum(0)
turn = net.turnover()
line('DG', dict(vort_r=float(np.corrcoef(net.L_['vort'][tm][innm], c['om_ex'][tm][innm])[0, 1]),
                div_noise=float(np.sqrt(np.nanmean(net.L_['div'][tm][innm] ** 2)) / rms), grad_leak_med=float(np.nanmedian(net.L_['Efrac'][1:, 0])),
                attribution_share_g_med=float(np.median(Ag / (Ag + Ac + 1e-30))),
                stretch_sp_FTLE=float(spearmanr(net.stretch[inn], c['ftle_ex'][inn])[0]), bottleneck_sp_FTLE=float(spearmanr(net.bottleneck[inn], c['ftle_ex'][inn])[0]),
                turnover_sp_FTLE=float(spearmanr(turn[inn], c['ftle_ex'][inn])[0]),
                stokesLAVD_sp_LAVD=float(spearmanr(net.lavd_stokes[inn], c['lavd_ex'][inn])[0]),
                dome_corr_LAVD=gkcorr(P0, net.dome, c['lavd_ex'], box_of(P0), spacing(P0)), cores=len(net.cores), sec=net.sec))
F['DG'] = dict(P=P0, stretch=net.stretch, bottleneck=net.bottleneck, dome=net.dome, cores=net.cores, turnover=turn, ftle=c['ftle_ex'])
# ---------------- 3) Sod, 4) blast (exact references; irrotational)
for nm, cc in (('SOD', cases.case_sod()), ('BLAST', cases.case_blast())):
    fr = cc['frames']; net = full(fr, cc['dt'], 0.2, attribution=False); S_, N_ = fr.shape[:2]; inner = cc['interior'](fr[0])
    truth = np.zeros(N_, bool); truth[cc['ev_pidx']] = True
    det = net.censor.any(1)
    lnJerr = np.abs(net.L_['lnJ'][-1] - cc['lnJ_ex'][-1])[inner]
    line(nm, dict(lnJ_err_med=float(np.median(lnJerr)), detect_rate=float(det[truth & inner].mean()), false_alarm=float(det[~truth & inner].mean()),
                  spurious_LAVD_p99=float(np.quantile(net.lavd_stokes[inner], 0.99)), Hodge_coexact_leak_med=float(np.nanmedian(net.L_['Efrac'][:, 1])),
                  cores=len(net.cores), max_rotation_rad=float(np.abs(net.phi).max()), sec=net.sec))
# ---------------- 5) shock-bubble (three windows as in the paper protocol)
D = pickle.load(open('/home/claude/bench/sb_dns.pkl', 'rb')); fr, Ed, dt = D['frames'], D['E'], D['dt']; GamE = D['Gam']; xb, Rb = D['xb'], D['R']
r0 = np.hypot(fr[0][:, 0] - xb, fr[0][:, 1]); del D
net = full(fr, dt, 0.06); s = 50; inner0 = (fr[0][:, 0] > 0.32) & (fr[0][:, 0] < 1.23) & (fr[0][:, 1] < 0.40)
m = inner0 & ~net.bad[:, s] & np.isfinite(net.L_['vort_c'][s])
rho_l = Ed['rho'][0] * np.exp(-net.L_['lnJ'][s]); kel = net.kelvin(r0 < Rb + 0.05)
iE = np.argmin(np.abs(GamE['t'] - 0.6))
pres = (np.diff(np.log(Ed['p']), axis=0) > 0.4).T.any(1)
line('SB_full', dict(vort_r=float(np.corrcoef(net.L_['vort_c'][s][m], Ed['om'][s][m])[0, 1]), div_r=float(np.corrcoef(np.nan_to_num(net.L_['div'][s][m]), Ed['th'][s][m])[0, 1]),
                     rho_err_med=float(np.nanmedian(np.abs(rho_l[inner0] / Ed['rho'][s][inner0] - 1))),
                     kelvin_L_t06=float(kel[30, 0]), kelvin_E_t06=float(GamE['Y2'][iE]),
                     detect_rate_vs_pressure=float(net.censor.any(1)[pres & inner0].mean()), sec=net.sec))
Wst = np.array([X.star_area(fr[q]) for q in range(30, 71)]); lavd_sb = X.lavd_from_series(Ed['om'][30:71], dt, A_weights=Wst)
net_e = full(fr[30:71], dt, 0.06, attribution=False)
line('SB_elliptic', dict(dome_corr_DNSLAVD=gkcorr(fr[30], net_e.dome, lavd_sb, box_of(fr[30]), spacing(fr[30])), cores=len(net_e.cores), sec=net_e.sec))
Z = np.load('/home/claude/bench/res_sb_phys.npz'); quiet, x0 = Z['quiet'], fr[0][:, 0]
net_b = TN(fr[20:51][::-1], dt); net_b.events(); net_b.robust_stretch(); net_b.curvature_topology()
def step(v):
    sc = np.nanquantile(v[Z['struct']], 0.95); lo = quiet & (x0 > 0.79) & (x0 < 0.85); hi = quiet & (x0 > 0.85) & (x0 < 0.91)
    return float((np.nanmean(v[hi]) - np.nanmean(v[lo])) / sc)
line('SB_backward', dict(edge_step_stretch=step(net_b.stretch), edge_step_bottleneck=step(net_b.bottleneck), edge_step_FTLE=step(Z['FTLE_bwd'])))
F['SB'] = dict(P=fr[50], stretch=net_b.stretch, bottleneck=net_b.bottleneck, dome_e=net_e.dome, cores_e=net_e.cores, P_e=fr[30])
# ---------------- 6) shock-vortex, in-loop / offline
S = pickle.load(open('/home/claude/bench/sv_dns.pkl', 'rb'))
Wst = np.array([X.star_area(S['fr_in'][q]) for q in range(10, 31)]); lavd_sv = X.lavd_from_series(S['Ev']['om'][10:31], S['dt'], A_weights=Wst)
lav = {}
for mode in ('in', 'off'):
    fr = S['fr_' + mode][10:31]; net = full(fr, S['dt'], 0.1, attribution=False); lav[mode] = net.lavd_stokes
    nb = TN(fr[::-1], S['dt']); nb.events(); nb.robust_stretch()
    x0 = S['P0'][:, 0]; y0 = S['P0'][:, 1]; xe = S['xs0'] + S['Ms'] * S['t'][10]; q = (np.abs(y0 - S['yv']) > 0.22) & (x0 > 0.47) & (x0 < 1.03)
    scl = np.nanquantile(nb.stretch, 0.99); st = float((np.nanmean(nb.stretch[q & (x0 > xe) & (x0 < xe + 0.06)]) - np.nanmean(nb.stretch[q & (x0 > xe - 0.06) & (x0 < xe)])) / scl)
    line('SV_' + mode, dict(dome_corr_DNSLAVD=gkcorr(fr[0], net.dome, lavd_sv, box_of(fr[0]), spacing(fr[0])), cores=len(net.cores), edge_step_stretch_bwd=st, sec=net.sec))
R['SV_stokesLAVD_mode_spearman'] = float(spearmanr(lav['in'], lav['off'])[0]); print('SV Stokes-LAVD in vs off spearman %.4f' % R['SV_stokesLAVD_mode_spearman'])
json.dump(R, open('res_unified.json', 'w'), indent=1); pickle.dump(F, open('res_unified.pkl', 'wb'))
