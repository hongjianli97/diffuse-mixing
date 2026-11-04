import json, pickle, time, numpy as np, cases, paper_ell as PE
from lhd_dome import lhd_rotation, dome
from improve_test_helpers import events, gkcorr, spacing, box_of
U = pickle.load(open('res_unifiedE.pkl', 'rb')); R = {}; OUT = {}
def go(name, fr, dt, merge, censor=True):
    t0 = time.time(); trc = np.ascontiguousarray(np.transpose(fr, (1, 0, 2))); P0 = fr[0]
    bad = None
    if censor:
        cs = events(fr); bad = np.zeros((fr.shape[1], len(fr)), bool); bad[:, :-1] |= cs; bad[:, 1:] |= cs
    Phi = lhd_rotation(fr, dt, bad); E, cores = dome(trc, Phi, merge)
    u = U[name]; lv = u['lavd']; h = spacing(P0); bx = box_of(P0)
    r = dict(n_cores_LHDdome=len(cores), n_cores_paperdome=len(u['cores']), Phi_max_rad=float(Phi.max()), sec=round(time.time() - t0, 1))
    if name != 'BLAST':
        r.update(corr_LHDdome=gkcorr(P0, E, lv, bx, h), corr_paperdome=gkcorr(P0, u['E'], lv, bx, h), corr_LHD_Phi=gkcorr(P0, Phi, lv, bx, h))
    R[name] = r; OUT[name] = dict(P0=P0, E=E, Phi=Phi, cores=cores)
    print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
G, a = PE._ELL_G, 1.0; uta = (G / (2 * np.pi * a)) * (1 - np.exp(-1)); Tc = 2 * np.pi * a / uta
vel, _ = PE.lamb_oseen([(-3., 0.), (3., 0.)], G, a); Pc = PE.seeds_disk(4.5, 2000)
trc = PE.rk4_traj(vel, Pc, 3 * Tc, 3 * Tc / 60, False, "fwd")
go('LO_paper', np.transpose(trc, (1, 0, 2)), 3 * Tc / 60, 2.0, censor=False)
c = cases.case_dg(); go('DG', c['frames'], c['dt'], 0.3, censor=False)
D = pickle.load(open('/home/claude/bench/sb_dns.pkl', 'rb')); go('SB', D['frames'][30:71], D['dt'], 0.06); del D
S = pickle.load(open('/home/claude/bench/sv_dns.pkl', 'rb'))
for mode in ('in', 'off'): go('SV_' + mode, S['fr_' + mode][10:31], S['dt'], 0.1)
cb = cases.case_blast(); go('BLAST', cb['frames'], cb['dt'], 0.2)
json.dump(R, open('res_lhd_dome.json', 'w'), indent=1); pickle.dump(OUT, open('res_lhd_dome.pkl', 'wb'))
