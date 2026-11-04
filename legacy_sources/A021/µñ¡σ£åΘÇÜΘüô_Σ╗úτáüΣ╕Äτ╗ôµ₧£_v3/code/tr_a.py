"""Transport report, part A-C on all cases: density reconstruction, mass flux, transfer operator / coherent sets / composition."""
import json, pickle, numpy as np, cases, lhd as L, lhdx as X, paper_ell as PE
from trajnet import TrajectoryNetwork as TN
R = {'A': {}, 'B': {}, 'C': {}}
def side_flux(fr, m, side):       # net mass from side False -> True over the window
    a, b = side(fr[0]), side(fr[-1]); return float((m * (~a & b)).sum() - (m * (a & ~b)).sum())
def coherent(P0, P1, w, nb=(20, 10)):
    def boxes(P):
        x0, x1, y0, y1 = P[:, 0].min(), P[:, 0].max() + 1e-9, P[:, 1].min(), P[:, 1].max() + 1e-9
        return np.minimum((nb[1] * (P[:, 1] - y0) / (y1 - y0)).astype(int), nb[1] - 1) * nb[0] + np.minimum((nb[0] * (P[:, 0] - x0) / (x1 - x0)).astype(int), nb[0] - 1)
    b0, b1 = boxes(P0), boxes(P1); n = nb[0] * nb[1]; Pm = np.zeros((n, n)); np.add.at(Pm, (b0, b1), w)
    pr, qc = Pm.sum(1), Pm.sum(0); r_, c_ = pr > 0, qc > 0
    U, sv, _ = np.linalg.svd(Pm[np.ix_(r_, c_)] / np.sqrt(np.outer(pr[r_], qc[c_])))
    u2 = np.zeros(n); u2[np.where(r_)[0]] = U[:, 1] / np.sqrt(pr[r_]); return float(sv[1]), u2[b0]
def jac(a, b): return float((a & b).sum() / max((a | b).sum(), 1))
# ---------------- double gyre (incompressible, exact): density, flux ladder vs dense reference, coherent pair
dg = {n: cases.case_dg(nx=n, ny=n // 2, S=26 if n == 640 else 51) for n in (80, 160, 320, 640)}
mid = lambda P: P[:, 0] > 1.0
ref = side_flux(dg[640]['frames'], X.star_area(dg[640]['frames'][0]) / 3.0, mid)
R['B']['DG_midline_flux'] = {'reference_640x320': ref}
for n in (80, 160, 320):
    fr = dg[n]['frames']; R['B']['DG_midline_flux']['N=%dx%d' % (n, n // 2)] = side_flux(fr, X.star_area(fr[0]) / 3.0, mid) / ref - 1
c = dg[160]; fr = c['frames']; net = TN(fr, c['dt']); lnJ = X.chained_lnJ_v2(fr, None)
R['A']['DG'] = dict(rho_median_abs_err=float(np.median(np.abs(np.exp(-lnJ[-1]) - 1))))
m = X.star_area(fr[0]) / 3.0; s2, f = coherent(fr[0], fr[-1], m); left = fr[0][:, 0] < 1
cut = f > 0 if f[left].mean() > 0 else f < 0
R['C']['DG'] = dict(sigma2_mass=s2, sigma2_area=s2, jaccard_with_left_gyre=jac(cut, left))
del dg
print('DG', json.dumps({k: R[k].get('DG', R[k].get('DG_midline_flux')) for k in 'ABC'}), flush=True)
# ---------------- Lamb-Oseen pair (incompressible): flux across x=0 vs dense reference, coherent set vs cores
lo = {N: cases.case_lo(a=1.0, Rd=6.5, N=N) for N in (2000, 8000, 32000, 128000)}
side = lambda P: P[:, 0] > 0.0
ref = side_flux(lo[128000]['frames'], X.star_area(lo[128000]['frames'][0]) / 3.0, side)
R['B']['LO_centreline_flux'] = {'reference_N128000': ref}
for N in (2000, 8000, 32000):
    fr = lo[N]['frames']; R['B']['LO_centreline_flux']['N=%d' % N] = side_flux(fr, X.star_area(fr[0]) / 3.0, side) / ref - 1
c = lo[8000]; fr = c['frames']; lnJ = X.chained_lnJ_v2(fr, None); R['A']['LO'] = dict(rho_median_abs_err=float(np.median(np.abs(np.exp(-lnJ[-1]) - 1))))
m = X.star_area(fr[0]) / 3.0; s2, f = coherent(fr[0], fr[-1], m, nb=(16, 16)); cores = c['lavd_ex'] >= 0.25 * np.quantile(c['lavd_ex'], 0.99)
cut = (f > 0) if f[cores].mean() > 0 else (f < 0)
R['C']['LO'] = dict(sigma2=s2, jaccard_with_cores=jac(cut, cores))
del lo
print('LO', json.dumps(R['B']['LO_centreline_flux']), json.dumps(R['C']['LO']), flush=True)
# ---------------- Sod and blast (exact): density, flux (from exact trajectories), resolution ladder
for nm in ('SOD', 'BLAST'):
    cc = cases.case_sod() if nm == 'SOD' else cases.case_blast(); fr = cc['frames']; S, N = fr.shape[:2]
    bad = L.vertex_bad_mask(cc['ev_pidx'], cc['ev_step'], N, S, halo=1); lnJ = X.chained_lnJ_v2(fr, bad); inner = cc['interior'](fr[0])
    err = np.abs(np.exp(-(lnJ[-1] - cc['lnJ_ex'][-1])) - 1)[inner]
    R['A'][nm] = dict(rho_median_abs_err=float(np.median(err)), rho_p95=float(np.quantile(err, 0.95)))
R['B']['SOD_BLAST'] = json.load(open('res_mass_transport.json'))
R['B']['SOD_BLAST'] = {k: R['B']['SOD_BLAST'][k] for k in ('sod', 'blast')}
print('SOD/BLAST', json.dumps(R['A']['SOD']), json.dumps(R['A']['BLAST']), flush=True)
# ---------------- shock-bubble (DNS): density, heavy-gas transport, composition by origin (mass vs count), coherent pair
D = pickle.load(open('/home/claude/bench/sb_dns.pkl', 'rb')); fr, Ed, grids, xc, yc, dx = D['frames'], D['E'], D['grids'], D['xc'], D['yc'], D['dx']; del D
net = TN(fr, 0.02); net.events(); lnJ = X.chained_lnJ_v2(fr, net.bad); inner0 = (fr[0][:, 0] > 0.32) & (fr[0][:, 0] < 1.23) & (fr[0][:, 1] < 0.40)
rho = Ed['rho'][0][None, :] * np.exp(-lnJ)
R['A']['SB'] = {('t=%.1f' % (s * 0.02)): float(np.median(np.abs(rho[s][inner0] / Ed['rho'][s][inner0] - 1))) for s in (30, 50, 70)}
mass = Ed['rho'][0] * X.star_area(fr[0]) / 3.0; Y0 = Ed['Y1'][0]
W = grids[1.4]['W'].astype(float); om = grids[1.4]['om']; jy, ix = np.unravel_index(np.argmin(om), om.shape); xv, yv = xc[ix], yc[jy]
GX, GY = np.meshgrid(xc, yc); disk = np.hypot(GX - xv, GY - yv) < 0.06; tin = np.hypot(fr[70][:, 0] - xv, fr[70][:, 1] - yv) < 0.06
R['C']['SB_composition_vortex_t1.4'] = dict(DNS_mass_fraction=float((W[0] * W[4])[disk].sum() / W[0][disk].sum()), DNS_volume_fraction=float(W[4][disk].mean()),
                                            tracers_mass_based=float((mass * Y0)[tin].sum() / mass[tin].sum()), tracers_count_based=float(Y0[tin].mean()), n_tracers=int(tin.sum()))
area = X.star_area(fr[0]) / 3.0
s2m, fm = coherent(fr[0], fr[-1], mass, nb=(24, 12)); s2a, fa = coherent(fr[0], fr[-1], area, nb=(24, 12)); bub = np.hypot(fr[0][:, 0] - 0.75, fr[0][:, 1]) < 0.25
R['C']['SB'] = dict(sigma2_mass=s2m, sigma2_area=s2a, jaccard_bubble_mass=jac((fm > 0) if fm[bub].mean() > 0 else (fm < 0), bub),
                    jaccard_bubble_area=jac((fa > 0) if fa[bub].mean() > 0 else (fa < 0), bub))
print('SB', json.dumps(R['A']['SB']), json.dumps(R['C']['SB_composition_vortex_t1.4']), json.dumps(R['C']['SB']), flush=True)
# ---------------- shock-vortex (DNS, in-loop vs offline): density vs DNS, transport consistency, coherent set vs core
S = pickle.load(open('/home/claude/bench/sv_dns.pkl', 'rb')); P0 = S['P0']; r0 = np.hypot(P0[:, 0] - S['xv'], P0[:, 1] - S['yv']); core = r0 < 2 * S['Rv']
R['A']['SV'] = {}; flux = {}; cuts = {}
for mode in ('in', 'off'):
    fr = S['fr_' + mode][:31]; n2 = TN(fr, S['dt']); n2.events(); lnJ = X.chained_lnJ_v2(fr, n2.bad)
    rho = S['Ev']['rho'][0][None, :] * np.exp(-lnJ); R['A']['SV'][mode] = float(np.median(np.abs(rho[30] / S['Ev']['rho'][30] - 1)))
    m = S['Ev']['rho'][0] * X.star_area(fr[0]) / 3.0; flux[mode] = side_flux(fr, m, lambda P: P[:, 0] > 0.9)
    s2, f = coherent(fr[0], fr[-1], m, nb=(16, 16)); cuts[mode] = ((f > 0) if f[core].mean() > 0 else (f < 0)); R['C']['SV_' + mode] = dict(sigma2=s2, jaccard_core=jac(cuts[mode], core))
R['B']['SV_flux_x0.9'] = dict(inloop=flux['in'], offline=flux['off'], rel_diff=abs(flux['off'] / flux['in'] - 1))
print('SV', json.dumps(R['A']['SV']), json.dumps(R['B']['SV_flux_x0.9']), json.dumps({k: R['C'][k] for k in ('SV_in', 'SV_off')}), flush=True)
json.dump(R, open('res_tr_a.json', 'w'), indent=1)
