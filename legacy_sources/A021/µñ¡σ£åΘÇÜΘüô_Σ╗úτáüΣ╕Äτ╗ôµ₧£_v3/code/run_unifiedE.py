"""Paper's unified AFRC_E (dome E, cells 99-100) on: paper Lamb-Oseen config (reproduction), then DG, SB, SV, blast."""
import json, pickle, time, numpy as np, paper_ell as PE
R, OUT = {}, {}
def run(name, trc, lavd, cfg, K=1):
    t0 = time.time()
    res = PE.unified_field(trc, K, cfg)
    E = res['E']; ok = np.isfinite(lavd)
    r = dict(pearson=float(np.corrcoef(E[ok], lavd[ok])[0, 1]),
             spearman=float(__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(E[ok], lavd[ok])[0]),
             n_cores=len(res['cores']), support_frac=float((E > 0).mean()), sec=time.time() - t0)
    R[name] = r; OUT[name] = dict(P0=trc[:, 0, :], E=E, lavd=lavd, Phi=res['Phi'], phi_core=res['phi_core'], cores=np.array(res['cores']))
    print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
# ---- 1) paper's own Lamb-Oseen pair (equal), exactly as cell 101
G, a = PE._ELL_G, 1.0
uta = (G / (2 * np.pi * a)) * (1 - np.exp(-1)); Tcore = 2 * np.pi * a / uta
vel, _ = PE.lamb_oseen([(-3., 0.), (3., 0.)], G, a); Pc = PE.seeds_disk(4.5, 2000)
Te, dte = 3 * Tcore, 3 * Tcore / 60
trc = PE.rk4_traj(vel, Pc, Te, dte, False, "fwd")
lv = PE.LAVD_from_traj(trc, vel, Te, dte, False, "fwd")
run('LO_paper', trc, lv, dict(merge=2.0, interior=None))
# ---- 2) our cases (trajectories as (N,S,2); LAVD references as before)
import cases, lhdx as X
c = cases.case_dg(); run('DG', np.ascontiguousarray(np.transpose(c['frames'], (1, 0, 2))), c['lavd_ex'], dict(merge=0.3, interior=None))
D = pickle.load(open('/home/claude/bench/sb_dns.pkl', 'rb')); fr = D['frames'][30:71]; Ed = D['E']; dt = D['dt']; del D
Wst = np.array([X.star_area(fr[s]) for s in range(len(fr))]); lavd_sb = X.lavd_from_series(Ed['om'][30:71], dt, A_weights=Wst)
run('SB', np.ascontiguousarray(np.transpose(fr, (1, 0, 2))), lavd_sb, dict(merge=0.06, interior=None))
S = pickle.load(open('/home/claude/bench/sv_dns.pkl', 'rb'))
Wst = np.array([X.star_area(S['fr_in'][s]) for s in range(10, 31)]); lavd_sv = X.lavd_from_series(S['Ev']['om'][10:31], S['dt'], A_weights=Wst)
for mode in ('in', 'off'):
    run('SV_' + mode, np.ascontiguousarray(np.transpose(S['fr_' + mode][10:31], (1, 0, 2))), lavd_sv, dict(merge=0.1, interior=None))
cb = cases.case_blast(); run('BLAST', np.ascontiguousarray(np.transpose(cb['frames'], (1, 0, 2))), np.zeros(cb['frames'].shape[1]) + 1e-12, dict(merge=0.2, interior=None))
json.dump(R, open('res_unifiedE.json', 'w'), indent=1); pickle.dump(OUT, open('res_unifiedE.pkl', 'wb'))
