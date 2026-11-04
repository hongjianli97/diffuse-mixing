"""Why does AFRC_E differ from LAVD?  Stage-by-stage agreement and the excess baseline (scene median vs W0)."""
import json, pickle, time, numpy as np, cases, lhdx as X, afrc_engine as AF
def run_case(name, fr, lavd, inner):
    traj = np.ascontiguousarray(np.transpose(fr, (1, 0, 2))); out = {}
    AF._CAP_W = []; AF.FRC6_BASELINE = 'median'; t0 = time.time()
    Fm = AF.afrc_fields(traj); w = np.nanmean(np.stack(AF._CAP_W), 0); AF._CAP_W = None
    AF.FRC6_BASELINE = 'floor'; Ff = AF.afrc_fields(traj); AF.FRC6_BASELINE = 'median'
    f = float((lavd[inner] >= 0.25 * np.quantile(lavd[inner], 0.99)).mean())
    flds = dict(Phi_n=Fm['Phi_n'], weight=w, AFRC_E_median=Fm['core_med'], AFRC_E_floor=Ff['core_med'])
    out = dict(vortex_fraction=f, time=time.time() - t0,
               **{k: dict(sp=X.spearman(v, lavd, inner), J=X.jaccard_top(v, lavd, 0.9, inner)) for k, v in flds.items()})
    print(name, json.dumps({k: (v if not isinstance(v, dict) else {q: round(x, 3) for q, x in v.items()}) for k, v in out.items()}), flush=True)
    return out, dict(P=fr[0], lavd=lavd, **flds)
R, F = {}, {}
for nm, kw in [('LO_Rd12', dict(a=1.0, Rd=12.0)), ('LO_Rd6.5', dict(a=1.0, Rd=6.5)), ('LO_a1.4_Rd5.6', dict(a=1.4, Rd=5.6))]:
    c = cases.case_lo(**kw); inner = np.hypot(c['frames'][0][:, 0], c['frames'][0][:, 1]) < c['Rd'] - 2 * c['h']
    R[nm], F[nm] = run_case(nm, c['frames'], c['lavd_ex'], inner)
D = pickle.load(open('sb_dns.pkl', 'rb')); fr = D['frames'][30:71]; Ed = D['E']; dt = D['dt']; del D
Wst = np.array([X.star_area(fr[s]) for s in range(len(fr))])
lavd = X.lavd_from_series(Ed['om'][30:71], dt, A_weights=Wst); P0 = fr[0]
inner = np.ones(len(P0), bool)
R['SB'], F['SB'] = run_case('SB', fr, lavd, inner)
c = cases.case_dg(); P0 = c['frames'][0]
inner = (P0[:, 0] > .05) & (P0[:, 0] < 1.95) & (P0[:, 1] > .05) & (P0[:, 1] < .95)
R['DG'], F['DG'] = run_case('DG', c['frames'], c['lavd_ex'], inner)
json.dump(R, open('res_afrcE_study.json', 'w'), indent=1); pickle.dump(F, open('res_afrcE_study.pkl', 'wb'))
