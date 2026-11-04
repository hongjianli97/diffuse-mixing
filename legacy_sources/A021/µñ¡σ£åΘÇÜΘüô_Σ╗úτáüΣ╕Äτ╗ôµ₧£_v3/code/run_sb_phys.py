"""Shock-bubble, physics-based reading at t_c = 1.0 following the paper's protocol.
Forward layer [1.0,1.4] (repelling, plotted at x(t_c)); backward layer [1.0 -> 0.4] (attracting, at x(t_c));
elliptic window [0.6,1.4].  Method-independent test: shock-dose response on non-vortical tracers."""
import pickle, json, time, numpy as np, lhd as L, lhdx as X, afrc_engine as AF, afrc_h_engine as AH
D = pickle.load(open('sb_dns.pkl', 'rb')); fr, t, dt, Ed = D['frames'], D['t'], D['dt'], D['E']
S, N = fr.shape[:2]; P0 = fr[0]; r0 = np.hypot(P0[:, 0] - D['xb'], P0[:, 1]); R0 = D['R']
gridW = D['grids'][1.0]; xc, yc = D['xc'], D['yc']; del D
tc, fw, bw, el = 50, (50, 70), (20, 50), (30, 70)
dlp = np.diff(np.log(Ed['p']), axis=0)                        # (S-1, N) step s: frames s -> s+1
cst = np.zeros((N, S - 1), bool); es, ep = np.where(dlp > 0.15); cst[ep, es] = True
cs = cst.copy(); cs[:, 1:] |= cst[:, :-1]; cs[:, :-1] |= cst[:, 1:]
strong = dlp > 0.4
dose_b = strong[bw[0]:bw[1]].sum(0); dose_f = strong[fw[0]:fw[1]].sum(0)
Tf, Tb = t[fw[1]] - t[fw[0]], t[bw[1]] - t[bw[0]]
ftle_f = X.ls_ftle(fr[tc], fr[fw[1]], Tf); ftle_b = X.ls_ftle(fr[tc], fr[bw[0]], Tb)
def afrcH(trajw, censor, T):
    out = []
    for k in (12, 16, 24):
        B, _, _ = AH.afrc_h_field(trajw, k=k, dt=T, censor=censor)
        B = np.maximum(B - np.median(B), 0); out.append(B / max(np.quantile(B, 0.99), 1e-30))
    return np.mean(out, 0)
t0 = time.time()
trf = np.ascontiguousarray(np.transpose(fr[fw[0]:fw[1] + 1], (1, 0, 2)))
trb = np.ascontiguousarray(np.transpose(fr[bw[0]:bw[1] + 1][::-1], (1, 0, 2)))
H_f = afrcH(trf, cs[:, fw[0]:fw[1]], Tf)
H_b = afrcH(trb, cs[:, bw[0]:bw[1]][:, ::-1], Tb)
H_b_raw = afrcH(trb, None, Tb)                                 # no censoring: shows what censoring removes
print('afrcH %.0fs' % (time.time() - t0), flush=True)
# elliptic window
W = np.array([X.star_area(fr[s]) for s in range(el[0], el[1] + 1)])
om = Ed['om'][el[0]:el[1] + 1]
lavd_dns = X.lavd_from_series(om, dt, A_weights=W)
vc = np.load('sb_stage1.npz')['vort_c'][el[0]:el[1] + 1]
wb = np.array([np.nansum(vc[s] * W[s]) / np.nansum(W[s] * np.isfinite(vc[s])) for s in range(len(vc))])
g = np.abs(vc - wb[:, None]); ti = np.arange(len(vc))
for i in range(N):
    ok = np.isfinite(g[:, i])
    if ok.sum() >= 2 and (~ok).any():
        g[~ok, i] = np.interp(ti[~ok], ti[ok], g[ok, i])
lavd_st = np.trapezoid(np.nan_to_num(g), dx=dt, axis=0)
t0 = time.time(); FE = AF.afrc_fields(np.ascontiguousarray(np.transpose(fr[el[0]:el[1] + 1], (1, 0, 2)))); print('afrcE %.0fs' % (time.time() - t0))
# ---- method-independent test: dose response on non-vortical tracers (outside bubble + margin)
inner0 = (P0[:, 0] > 0.32) & (P0[:, 0] < 1.23) & (P0[:, 1] < 0.40)
quiet = inner0 & (r0 > R0 + 0.08) & (lavd_dns < np.quantile(lavd_dns[inner0], 0.5))
struct = inner0 & (lavd_dns >= np.quantile(lavd_dns[inner0], 0.95))
fields = dict(FTLE_fwd=ftle_f, AFRC_H_fwd=H_f, FTLE_bwd=ftle_b, AFRC_H_bwd=H_b, AFRC_H_bwd_uncensored=H_b_raw,
              DNS_LAVD=lavd_dns, Stokes_LAVD=lavd_st, AFRC_E=FE['core_med'])
R = dict(n_quiet=int(quiet.sum()), dose_b_counts={int(d): int(((dose_b == d) & quiet).sum()) for d in range(4)})
R['dose_response'] = {}
for k, F in fields.items():
    sc_ = np.nanquantile(F[struct], 0.95)
    R['dose_response'][k] = {int(d): float(np.nanmedian(F[quiet & (np.minimum(dose_b, 2) == d)]) / sc_) for d in (0, 1, 2)
                             if ((np.minimum(dose_b, 2) == d) & quiet).sum() > 30}
# physical check: backward FTLE jump per incident-shock crossing vs ln(eta)/Tb
m0 = quiet & (dose_b == 0); m1 = quiet & (dose_b == 1)
R['ftle_bwd_jump'] = float(np.nanmedian(ftle_b[m1]) - np.nanmedian(ftle_b[m0]))
R['ftle_bwd_jump_theory'] = float(np.log(1.8620689655) / Tb)
R['ftle_fwd_jump'] = float(np.nanmedian(ftle_f[quiet & (dose_f == 1)]) - np.nanmedian(ftle_f[quiet & (dose_f == 0)])) if (quiet & (dose_f == 1)).sum() > 30 else None
print(json.dumps(R, indent=1))
json.dump(R, open('res_sb_phys.json', 'w'), indent=1)
np.savez_compressed('res_sb_phys.npz', P=fr[tc], P0=P0, r0=r0, dose_b=dose_b, dose_f=dose_f, quiet=quiet, struct=struct,
                    om_tc=Ed['om'][tc], rho_tc=Ed['rho'][tc], **fields)
pickle.dump(dict(W=gridW, xc=xc, yc=yc), open('sb_grid_tc.pkl', 'wb'))
