"""Shock-vortex: in-loop vs offline tracking.  Window [0.2,0.6]; forward fields at x(0.2), backward at x(0.6)."""
import json, pickle, time, numpy as np, lhd as L, lhdx as X, afrc_h_engine as AH
D = pickle.load(open('sv_dns.pkl', 'rb')); t = D['t']; dt = D['dt']; a, b = 10, 30; Tw = t[b] - t[a]
P0 = D['P0']; x0, y0 = P0[:, 0], P0[:, 1]; Ms, xs0 = D['Ms'], D['xs0']; lam = Ms * dt
r0 = np.hypot(x0 - D['xv'], y0 - D['yv']); quiet = (np.abs(y0 - D['yv']) > 0.22) & (x0 > 0.47) & (x0 < 1.03)
xe = xs0 + Ms * t[a]                                       # window-edge line (shock position at window start)
def events(fr):
    """position-only shock detection: one-step material star-area ratio (C channel), 2-step window."""
    S, N = fr.shape[:2]; dl = np.zeros((S - 1, N))
    for s in range(S - 1):
        tri = X.complex_peeled(fr[s])['tri']
        A0, A1 = X._star(fr[s], tri, N), X._star(fr[s + 1], tri, N)
        with np.errstate(all='ignore'):
            dl[s] = np.where(A0 > 0, np.log(A1 / A0), 0.0)
    det = dl < -0.2; det[:-1] |= (dl[:-1] + dl[1:]) < -0.2
    return det.T                                           # (N, S-1)
def ridge3(trajw, cens, T):
    out = []
    for k in (12, 16, 24):
        B, _, _ = AH.afrc_h_field(trajw, k=k, dt=T, censor=cens)
        B = np.maximum(B - np.median(B), 0); out.append(B / max(np.quantile(B, 0.99), 1e-30))
    return np.mean(out, 0)
def stokes_lavd(fr, bad):
    S, N = fr.shape[:2]; o = X.run_lhd(fr, dt, kinds=('galerkin',), op='galerkin', cplx='peel', bad=bad, hodge_every=10 ** 6)
    W = np.array([X.star_area(fr[s]) for s in range(S)]); vc = o['vort_c']
    wb = np.array([np.nansum(vc[s] * W[s]) / np.nansum(W[s] * np.isfinite(vc[s])) for s in range(S)])
    g = np.abs(vc - wb[:, None]); ti = np.arange(S)
    for i in np.where(~np.isfinite(g).all(0))[0]:
        ok = np.isfinite(g[:, i])
        if ok.sum() >= 2:
            g[~ok, i] = np.interp(ti[~ok], ti[ok], g[ok, i])
    return np.trapezoid(np.nan_to_num(g), dx=dt, axis=0), o['vort'], W
res, F = {}, {}
for mode in ('in', 'off'):
    fr = D['fr_' + mode][a:b + 1]; N = fr.shape[1]; t0 = time.time()
    det = events(fr); cs = det.copy(); cs[:, 1:] |= det[:, :-1]; cs[:, :-1] |= det[:, 1:]
    bad = np.zeros((N, len(fr)), bool); bad[:, :-1] |= cs; bad[:, 1:] |= cs
    f = dict(FTLE_fwd=X.ls_ftle(fr[0], fr[-1], Tw), FTLE_bwd=X.ls_ftle(fr[-1], fr[0], Tw))
    trf = np.ascontiguousarray(np.transpose(fr, (1, 0, 2))); trb = np.ascontiguousarray(trf[:, ::-1])
    f['AFRC_H_fwd'] = ridge3(trf, cs, Tw); f['AFRC_H_bwd'] = ridge3(trb, cs[:, ::-1], Tw)
    f['Stokes_LAVD'], vort, W = stokes_lavd(fr, bad)
    om_ls, _ = X.ls_vort_div(fr, dt, k=12); f['LS_LAVD'] = X.lavd_from_series(om_ls, dt, A_weights=W)
    F[mode] = dict(P_a=fr[0], P_b=fr[-1], ndet=det.sum(1), **f)
    print(mode, 'done %.0fs' % (time.time() - t0), 'detected crossings/tracer (median)', np.median(det.sum(1)), flush=True)
F['DNS_LAVD'] = X.lavd_from_series(D['Ev']['om'][a:b + 1], dt, A_weights=np.array([X.star_area(D['fr_in'][s]) for s in range(a, b + 1)]))
struct = F['DNS_LAVD'] >= np.quantile(F['DNS_LAVD'], 0.95)
for k in ('FTLE_fwd', 'FTLE_bwd', 'AFRC_H_fwd', 'AFRC_H_bwd', 'Stokes_LAVD', 'LS_LAVD'):
    vi, vo = F['in'][k], F['off'][k]; sc = np.nanquantile(np.abs(vi[struct]), 0.95) if 'LAVD' in k else np.nanquantile(np.abs(vi), 0.99)
    d = dict(mode_diff=float(np.nanmedian(np.abs(vi - vo)[quiet]) / sc), mode_sp=X.spearman(vi, vo))
    for mode, v in (('in', vi), ('off', vo)):
        bins = np.arange(0.47, 1.03, 0.005); prof = np.array([np.nanmean(v[quiet & (x0 >= bins[i]) & (x0 < bins[i + 1])]) for i in range(len(bins) - 1)])
        tr_ = np.convolve(prof, np.ones(11) / 11, 'same'); r_ = (prof - tr_)[6:-6]; xb = 0.5 * (bins[1:] + bins[:-1])[6:-6]
        amp = np.abs(np.sum(r_ * np.exp(-2j * np.pi * xb / lam))) * 2 / len(r_)
        d['stripe_' + mode] = float(amp / sc)
        lo = quiet & (x0 > xe - 0.06) & (x0 < xe); hi = quiet & (x0 > xe) & (x0 < xe + 0.06)
        d['edge_step_' + mode] = float((np.nanmean(v[hi]) - np.nanmean(v[lo])) / sc)
    res[k] = d
    print(k, {q: round(w, 4) for q, w in d.items()}, flush=True)
pd = np.linalg.norm(D['fr_in'][b] - D['fr_off'][b], axis=1)
res['pos_diff_t06'] = dict(median=float(np.median(pd)), p99=float(np.quantile(pd, 0.99)), h=0.005)
json.dump(res, open('res_sv.json', 'w'), indent=1)
pickle.dump(dict(F=F, quiet=quiet, xe=xe, lam=lam, P0=P0, posdiff=pd, grids=D['grids'], xc=D['xc'], yc=D['yc']), open('res_sv.pkl', 'wb'))
