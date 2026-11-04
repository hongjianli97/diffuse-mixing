"""AFRC improvements: (1) direction-aware (signed) AFRC_H; (2) AFRC_E core from the co-rotation weight, rim from curvature."""
import json, pickle, numpy as np, cases, lhdx as X, afrc_h_engine as AH, afrc_engine as AF
def ridge3(traj, cens, T, signed):
    AH.FORMAN_SIGNED = signed; out = []
    for k in (12, 16, 24):
        B, _, _ = AH.afrc_h_field(traj, k=k, dt=T, censor=cens); B = np.maximum(B - np.median(B), 0); out.append(B / max(np.quantile(B, 0.99), 1e-30))
    AH.FORMAN_SIGNED = False
    return np.mean(out, 0)
R, F = {}, {}
c = cases.case_dg(); fr = c['frames']; traj = np.ascontiguousarray(np.transpose(fr, (1, 0, 2))); T = c['t'][-1]
P0 = fr[0]; inner = (P0[:, 0] > .05) & (P0[:, 0] < 1.95) & (P0[:, 1] > .05) & (P0[:, 1] < .95)
ftle_b = X.ls_ftle(fr[-1], fr[0], T)
F['DG'] = dict(P0=P0, P1=fr[-1], ftle_f=c['ftle_ex'], ftle_b=ftle_b,
               H_abs=ridge3(traj, None, T, False), H_sf=ridge3(traj, None, T, True), H_sb=ridge3(np.ascontiguousarray(traj[:, ::-1]), None, T, True))
R['DG'] = {k: dict(sp_fwdFTLE=X.spearman(F['DG'][k], c['ftle_ex'], inner), sp_bwdFTLE=X.spearman(F['DG'][k], ftle_b, inner)) for k in ('H_abs', 'H_sf', 'H_sb')}
print('DG', json.dumps(R['DG']), flush=True)
D = pickle.load(open('sv_dns.pkl', 'rb')); fr = D['fr_off'][10:31]; traj = np.ascontiguousarray(np.transpose(fr, (1, 0, 2))); T = 0.4
F['SV'] = dict(Pa=fr[0], Pb=fr[-1], H_sf=ridge3(traj, None, T, True), H_sb=ridge3(np.ascontiguousarray(traj[:, ::-1]), None, T, True))
pickle.dump(F, open('res_afrc_improve.pkl', 'wb')); json.dump(R, open('res_afrc_improve.json', 'w'), indent=1)
