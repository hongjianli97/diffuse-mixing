"""Material ledger with (i) a per-tracer a-posteriori non-affinity indicator and (ii) loud failure diagnostics.

Same anchoring schedule as lhdx.chained_lnJ_v2 (re-anchor every `block` frames, deferred while the star touches a
shock stencil), so ln J is bit-identical to v2.  In addition, for every anchor segment a -> t:

    delta_i(a,t) = ln[A*_i(t) / A*_i(a)] - ln det F_i(a,t)

where F_i is the least-squares affine map of the star vertices (centre fixed).  Because polygon area is a quadratic
form of the vertices, A(x_hat + e) = A(x_hat) + grad A(x_hat).e + A(e) exactly (tests/test_identities_exact.py), so
delta is exactly the part of the ledger increment that is NOT explained by an affine motion of the star.
The indicator   I_i(t) = sqrt( sum_closed delta^2 + delta_open(t)^2 )   accumulates it over segments.
It is zero when every segment is affine (planar shocks included, Proposition 'telescoping') and large for stars
that straddle a shock or deform non-affinely (vortex cores).  It is an indicator, calibrated in scripts/v3_*.py,
not a rigorous bound on |ln J_hat - ln J|.

noise_sigma(frames): robust estimate of the per-coordinate position noise from third time differences.
"""
import numpy as np
import lhdx as X


def star_nb(tri, N):
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]], tri[:, [1, 0]], tri[:, [2, 1]], tri[:, [0, 2]]])
    e = np.unique(e, axis=0); cnt = np.bincount(e[:, 0], minlength=N); K = max(int(cnt.max()), 1)
    nb = np.full((N, K), -1); st = np.r_[0, np.cumsum(cnt)[:-1]]; pos = np.arange(len(e)) - st[e[:, 0]]
    nb[e[:, 0], pos] = e[:, 1]
    return nb


def _affine_F(Pa, Pt, idx, nb):
    """least-squares F (centre fixed) mapping star offsets at anchor -> current"""
    ok = nb >= 0; nbc = np.where(ok, nb, idx[:, None])
    Xa = Pa[nbc] - Pa[idx][:, None]; Xt = Pt[nbc] - Pt[idx][:, None]; w = ok[:, :, None].astype(float)
    G = np.einsum("nki,nkj->nij", Xa * w, Xa) + 1e-30 * np.eye(2); H = np.einsum("nki,nkj->nij", Xa * w, Xt)
    Ft = np.linalg.solve(G, H)
    return np.transpose(Ft, (0, 2, 1))


def _affine_det(Pa, Pt, idx, nb):
    return np.linalg.det(_affine_F(Pa, Pt, idx, nb))


def chained_lnJ_ind(frames, bad=None, block=5, max_defer=4, cplx_fn=None):
    """returns lnJ (S,N) [identical to chained_lnJ_v2], indicator I (S,N), diag dict"""
    cplx_fn = X.complex_peeled if cplx_fn is None else cplx_fn
    S, N, _ = frames.shape
    tris = {0: cplx_fn(frames[0])["tri"]}; nbs = {0: star_nb(tris[0], N)}
    anchor = np.zeros(N, int); A0 = X._star(frames[0], tris[0], N); base = np.zeros(N)
    out = np.zeros((S, N)); ind = np.zeros((S, N)); closed2 = np.zeros(N)
    diag = dict(fallback_events=0, fallback_tracers=0, nonpositive_anchor_area=int((A0 <= 0).sum()), reanchors=0)
    fb_any = np.zeros(N, bool)
    for t in range(1, S):
        cur = np.full(N, np.nan); dlt = np.zeros(N)
        for a in np.unique(anchor):
            idx = np.where(anchor == a)[0]
            cur[idx] = X._star(frames[t], tris[a], N)[idx]
            with np.errstate(all="ignore"):
                dF = _affine_det(frames[a], frames[t], idx, nbs[a][idx])
                dlt[idx] = np.log(cur[idx] / A0[idx]) - np.log(dF)
        with np.errstate(all="ignore"):
            inc = np.log(cur / A0)
        bad_inc = ~np.isfinite(inc)
        if bad_inc.any():
            diag["fallback_events"] += int(bad_inc.sum()); fb_any |= bad_inc
        out[t] = np.where(bad_inc, out[t - 1], base + inc)
        dlt = np.where(np.isfinite(dlt), dlt, 0.0)
        ind[t] = np.sqrt(closed2 + dlt ** 2)
        due = (t - anchor) >= block
        if due.any() and t < S - 1:
            trit = cplx_fn(frames[t])["tri"]
            if bad is not None:
                nbd = bad[:, max(t - 1, 0):t + 2].any(1); sb = np.zeros(N, bool)
                for k in range(3):
                    np.logical_or.at(sb, trit[:, k], nbd[trit].any(1))
                ok = due & (~sb | ((t - anchor) >= max_defer * block))
            else:
                ok = due
            if ok.any():
                tris[t] = trit; nbs[t] = star_nb(trit, N)
                At = X._star(frames[t], trit, N)
                good = ok & (At > 0)
                closed2[good] += dlt[good] ** 2
                A0[good] = At[good]; base[good] = out[t][good]; anchor[good] = t; diag["reanchors"] += int(good.sum())
        live = set(np.unique(anchor))
        for a in list(tris):
            if a not in live:
                del tris[a]; del nbs[a]
    diag["fallback_tracers"] = int(fb_any.sum())
    return out, ind, diag


def hot_steps_multi(frames, tau_min=0.2, kfac=4.0, W=6, bg=3.0, sv_tol=0.1, confirm=True):
    """C channel extended to shocks that need several snapshots to cross a star (snapshot Courant Cs < ~2, weak or
    viscous shocks).  Single-step hot steps are kept.  In addition, steps s0..s0+w-1 (w <= W) are marked hot when
      (a) the star area drops:  ln A*(s0+w)/A*(s0) < -max(tau, bg * w * median_t|R_i|)   (compression only: entropy
          condition; per-tracer smooth background), and
      (b) confirm=True: the least-squares affine map of the star over the window is a uniaxial compression, i.e. one
          principal stretch equals 1 within sv_tol in ln (the block signature of Sec. 2.3, rotation-invariant form).
    (b) rejects non-affine roll-up / shear deformation, which must NOT extend the anchor deferral."""
    import classify as C
    from lhdp import pq
    hot1, R, tau = C.hot_steps(frames, tau_min=tau_min, kfac=kfac)
    S, N, _ = frames.shape; hot = hot1.copy()
    base = np.median(np.abs(R), axis=1)
    for s0 in range(S - 1):
        tri = X.complex_peeled(frames[s0])["tri"]; a0 = X._star(frames[s0], tri, N); nb = star_nb(tri, N) if confirm else None
        for w in range(2, W + 1):
            if s0 + w > S - 1:
                break
            a1 = X._star(frames[s0 + w], tri, N)
            with np.errstate(all="ignore"):
                D = np.where((a0 > 0) & (a1 > 0), np.log(a1 / a0), 0.0)
            cand = D < -np.maximum(tau, bg * w * base)
            if confirm and cand.any():
                idx = np.where(cand)[0]; F = _affine_F(frames[s0], frames[s0 + w], idx, nb[idx]); p, q = pq(F)
                s1, s2 = np.abs(p) + np.abs(q), np.abs(p) - np.abs(q)
                with np.errstate(all="ignore"):
                    uni = (s2 > 0) & (np.minimum(np.abs(np.log(s1)), np.abs(np.log(np.abs(s2)))) < sv_tol)
                cand[idx[~uni]] = False
            if cand.any():
                hot[cand, s0:s0 + w] = True
    return hot, R, tau


def ledger_ind(fr, tau_min=0.2, block=5, defer=4, kfac=4.0, multi=False):
    import classify as C
    if multi:
        hot, R, tau = hot_steps_multi(fr, tau_min=tau_min, kfac=kfac)
    else:
        hot, R, tau = C.hot_steps(fr, tau_min=tau_min, kfac=kfac)
    S, N, _ = fr.shape; bad = np.zeros((N, S), bool); bad[:, :-1] |= hot; bad[:, 1:] |= hot
    lnJ, I, diag = chained_lnJ_ind(fr, bad, block=block, max_defer=defer)
    return lnJ, I, diag, hot, bad, tau


def noise_sigma(frames):
    """robust per-coordinate noise std from third time differences (noise variance of the 3rd difference = 20 s^2);
    smooth motion contributes O(dt^3), shocks are outliers removed by the MAD."""
    d3 = frames[3:] - 3 * frames[2:-1] + 3 * frames[1:-2] - frames[:-3]
    v = d3.reshape(-1)
    return float(1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(20.0))


def tensor_lnJ(frames, bad=None, block=5, max_defer=4, k=12, keep=None):
    """point-wise ('tensor') ledger: ln det of the chained least-squares deformation gradient (quadratic fit on k
    neighbours, same anchoring/deferral as the polygon ledger).  Second order in smooth flow (MFM E3) and exact across
    planar shocks (affine segments); the polygon ledger remains the mass-consistent estimator (m_i = rho_i A*_i/3)."""
    import stretch_est as SE
    from scipy.spatial import cKDTree
    d, _ = cKDTree(frames[0]).query(frames[0], k=2); h = float(np.median(d[:, 1]))
    S = frames.shape[0]; keep = [S - 1] if keep is None else keep
    Fo = SE.chained_F(frames, h, bad=bad, block=block, max_defer=max_defer, k=k, quad=True, keep=keep)
    with np.errstate(all="ignore"):
        return {s: np.log(np.linalg.det(Fo[s][0])) for s in keep}


def bad_from_hot(hot):
    N, S1 = hot.shape; bad = np.zeros((N, S1 + 1), bool); bad[:, :-1] |= hot; bad[:, 1:] |= hot
    return bad


def auc(score, positive):
    """probability that a random positive has a higher score than a random negative (Mann-Whitney)"""
    from scipy.stats import rankdata
    r = rankdata(score); n1 = positive.sum(); n0 = (~positive).sum()
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def hot_runs_q(hot, q=0.99):
    """quantile of the length of contiguous hot runs (per tracer)"""
    L = []
    for row in hot:
        if not row.any():
            continue
        d = np.diff(np.r_[0, row.astype(int), 0]); st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        L.extend((en - st).tolist())
    return float(np.quantile(L, q)) if L else 0.0


def ledger_v3(fr, block=5, multi=True, auto_defer=True, defer=4, hybrid_tol=0.1, tensor=True):
    """recommended driver: multi-window shock detection, deferral cap derived from the measured hot-run length,
    polygon ledger (mass-consistent) + indicator, point-wise tensor ledger, and a hybrid that uses the tensor value
    where it agrees with the polygon value to within hybrid_tol in ln J (falls back to the polygon otherwise)."""
    import classify as C
    hot = hot_steps_multi(fr)[0] if multi else C.hot_steps(fr)[0]
    if auto_defer:
        run = hot_runs_q(hot); defer = int(np.clip(np.ceil((2 * run + 2) / block), defer, 10))
    bad = bad_from_hot(hot); S = fr.shape[0]
    lnJ, I, diag = chained_lnJ_ind(fr, bad, block=block, max_defer=defer)
    diag.update(max_defer=defer, hot_frac=float(hot.mean()))
    res = dict(lnJ=lnJ, ind=I, diag=diag, hot=hot)
    if tensor:
        lt = tensor_lnJ(fr, bad, block=block, max_defer=defer)[S - 1]
        ok = np.isfinite(lt) & (np.abs(lt - lnJ[-1]) <= hybrid_tol)
        res.update(lnJ_tensor=lt, lnJ_hybrid=np.where(ok, lt, lnJ[-1]), hybrid_used=float(ok.mean()))
    return res


def thermo_from_ledger(lnJ, hot, rho0, p0, gamma=1.4, eta_min=1.05):
    """Thermodynamic state from positions (calorically perfect gas, inviscid): entropy K = p/rho^gamma is constant
    along a tracer except at detected shock blocks, where the measured density ratio eta = exp(-[lnJ]) gives the
    normal Mach number and the entropy jump by Rankine-Hugoniot.  Returns p, T = p/rho (S,N)."""
    S, N = lnJ.shape; g = gamma; K = np.tile(p0 / rho0 ** g, (S, 1))
    for i in range(N):
        row = hot[i]
        if not row.any():
            continue
        d = np.diff(np.r_[0, row.astype(int), 0]); st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        for a, b in zip(st, en):
            eta = np.exp(-(lnJ[b, i] - lnJ[a, i]))
            if eta_min < eta < (g + 1) / (g - 1) - 1e-6:
                M2 = 2 * eta / ((g + 1) - (g - 1) * eta); pr = 1 + 2 * g / (g + 1) * (M2 - 1)
                K[b:, i] *= pr / eta ** g
    rho = rho0[None] * np.exp(-lnJ); p = K * rho ** g
    return p, p / rho
