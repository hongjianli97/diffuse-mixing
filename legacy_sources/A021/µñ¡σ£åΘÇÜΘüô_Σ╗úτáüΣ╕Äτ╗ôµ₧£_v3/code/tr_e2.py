"""Edge-based tangential stretch of material interfaces: median over interface tracers of ln(|r_e(T)|/|r_e(0)|) for network edges
whose initial direction is within 22.5 deg of the interface tangent (no polyline ordering, insensitive to seeding jitter)."""
import json, pickle, numpy as np, cases, afrc_h_engine as AH
def tang_stretch(fr, idx, tangent, s, k=12, w=None):
    E = AH._mutual_knn(fr[0], k); E = np.r_[E, E[:, ::-1]]; E = E[np.isin(E[:, 0], idx)]
    r0 = fr[0][E[:, 1]] - fr[0][E[:, 0]]; rt = fr[s][E[:, 1]] - fr[s][E[:, 0]]; t0 = tangent[E[:, 0]]
    ok = np.abs((r0 * t0).sum(1)) / np.linalg.norm(r0, axis=1) > np.cos(np.pi / 8); E, ls = E[ok], np.log(np.linalg.norm(rt[ok], axis=1) / np.linalg.norm(r0[ok], axis=1))
    vals = np.array([np.median(ls[E[:, 0] == i]) if (E[:, 0] == i).any() else np.nan for i in idx]); okv = np.isfinite(vals)
    ww = np.ones(len(idx)) if w is None else w[idx]; return float(np.sum(ww[okv] * vals[okv]) / np.sum(ww[okv]))
R = {}
c = cases.case_sod(); fr = c['frames']; col = np.where(np.abs(c['x0'] - 0.1) < c['h'] / 2)[0]; T = np.tile([0.0, 1.0], (len(c['x0']), 1))
R['SOD'] = dict(edge_stretch=tang_stretch(fr, col, T, -1), exact=0.0)
c = cases.case_blast(); fr = c['frames']; ring = np.where(np.abs(c['rs'] - 0.5) < c['h'] / 2)[0]; th = np.arctan2(fr[0][:, 1], fr[0][:, 0]); T = np.c_[-np.sin(th), np.cos(th)]
p = 1 / 2.4; tc = (0.5 / 0.2) ** 2; R['BLAST'] = dict(edge_stretch=tang_stretch(fr, ring, T, -1), exact=float(p * np.log(25.0 / tc)))
c = cases.case_dg(); fr = c['frames']; line = np.where(np.abs(fr[0][:, 0] - 0.5) < c['h'] / 2)[0]; T = np.tile([0.0, 1.0], (len(fr[0]), 1))
R['DG'] = {('t=%g' % t): tang_stretch(fr, line, T, s) for t, s in ((5.0, 25), (10.0, 50))}
c = cases.case_lo(a=1.0, Rd=6.5, N=8000); fr = c['frames']; d0 = fr[0] - [-3.0, 0.0]; ring = np.where(np.abs(np.hypot(*d0.T) - 1.5) < c['h'] / 2)[0]
th = np.arctan2(d0[:, 1], d0[:, 0]); T = np.c_[-np.sin(th), np.cos(th)]
R['LO'] = {('t=%.1f' % (c['t'][s])): tang_stretch(fr, ring, T, s) for s in (5, 10, 20, 40)}
D = pickle.load(open('/home/claude/bench/sb_dns.pkl', 'rb')); fr = D['frames']; d0 = fr[0] - [D['xb'], 0.0]; ring = np.where(np.abs(np.hypot(*d0.T) - D['R']) < 0.003)[0]
th = np.arctan2(d0[:, 1], d0[:, 0]); T = np.c_[-np.sin(th), np.cos(th)]; m = D['E']['rho'][0]
R['SB'] = {('t=%.1f' % (s * 0.02)): tang_stretch(fr, ring, T, s, w=m) for s in (15, 30, 50, 70)}; del D
S = pickle.load(open('/home/claude/bench/sv_dns.pkl', 'rb')); d0 = S['P0'] - [S['xv'], S['yv']]; ring = np.where(np.abs(np.hypot(*d0.T) - 2 * S['Rv']) < 0.0025)[0]
th = np.arctan2(d0[:, 1], d0[:, 0]); T = np.c_[-np.sin(th), np.cos(th)]
R['SV'] = {mode: tang_stretch(S['fr_' + mode], ring, T, 30) for mode in ('in', 'off')}
print(json.dumps(R)); json.dump(R, open('res_tr_e2.json', 'w'), indent=1)
