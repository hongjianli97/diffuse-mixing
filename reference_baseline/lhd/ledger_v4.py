"""Material ledger v4.

Three changes with respect to ledger_ub.ledger_v3 (which used one tensor chain for the whole window and a single
final-time agreement test, so that one bad tensor increment propagated to the end and the hybrid fell back to the
polygon value everywhere downstream of it):

1. Geometric-window shock detection (hot_steps_geo): the multi-window C channel of ledger_ub.hot_steps_multi with
   windows w in (2,3,4,6,8,12,16,24,32) instead of w <= 6, so that resolved (viscous) shocks that need many snapshots
   to cross a star are detected; the same compression-only and uniaxial-confirmation tests reject roll-up and shear.
   The deferral cap is derived from the measured hot-run length (99th percentile) and may now reach `defer_cap`
   blocks (default 60 blocks = 300 snapshots).

2. Segment-wise guarded tensor increments.  ln J is a sum of increments over anchor segments a -> t.  For every
   segment both the polygon increment (star-area ratio, exact for affine star motion and across planar shocks) and the
   tensor increment (ln det of the quadratic least-squares deformation gradient on k neighbours, second order in
   smooth flow) are computed with the SAME anchor.  The tensor increment is used for that segment only if
     - the fit is well conditioned (relative standard error of F below fit_tol),
     - the k neighbours do not straddle a material interface (reference density jump),
     - |tensor - polygon| <= tol_smooth on segments without hot steps, and
       |tensor - polygon| <= tol_shock + delta_fac*|delta| on segments with hot steps, where delta is the polygon
       non-affinity of that segment (ledger_ub indicator): where the polygon is provably affine (planar shocks,
       delta ~ 0) the polygon value is kept unless the tensor agrees tightly; where the star moved non-affinely
       (vortex cores swept by a shock) the tensor may correct it; noise_fac*sigma/h is added for position noise
       (sigma estimated from third time differences, Proposition 'noise law').
   A rejected tensor increment is replaced by the polygon increment of that segment only, so outliers cannot
   propagate.

3. One-sided arc repair.  (a) Stars that straddle a material interface (reference density jump > iface_jump in ln):
   every segment increment is evaluated on the fan polygon of the largest contiguous arc of same-material ring
   neighbours (>= 2 triangles).  (b) Stars being swept by a shock at the evaluation time (open segment): the ring
   arc that moves affinely with the centre is searched (least-squares affine residual of the arc below
   arc_rel * residual of the full ring and below an absolute floor set by the position noise); the open increment is
   the fan-polygon area ratio of that arc.  For a sharp shock or contact both are exact (the map from the anchor to
   the current time is affine on either side).  For resolved shocks no arc is affine and nothing is changed.

Masses m_i = rho_i A*_i / 3 are not touched: v4 is a point-wise density estimator; the polygon ledger stays the
mass-consistent one.
"""
import numpy as np
from scipy.spatial import cKDTree
import lhdx as X
import classify as C
import stretch_est as SE
from ledger_ub import star_nb, _affine_F, bad_from_hot, hot_runs_q, noise_sigma
from lhdp import pq

GEO_W = (2, 3, 4, 6, 8, 12, 16, 24, 32)


# --------------------------------------------------------------------------------------------- detection
def hot_steps_geo(frames, tau_min=0.2, kfac=4.0, Ws=GEO_W, bg=3.0, sv_tol=0.1, confirm=True, trim=True, frac=0.1, pad=0.5, trim_span=True, tail="adaptive", c_noise=3.0, f_peak=0.02):
    """single-step C channel + windows s0..s0+w (w in Ws): compression-only area drop beyond max(tau, bg*w*median|R|)
    and (confirm) a uniaxial least-squares affine map of the star over the window (one principal stretch = 1 within
    sv_tol in ln).  Returns hot (N,S-1), R, tau."""
    hot1, R, tau = C.hot_steps(frames, tau_min=tau_min, kfac=kfac)
    S, N, _ = frames.shape; hot = hot1.copy(); base = np.median(np.abs(R), axis=1)
    bgR = 1.4826 * np.median(np.abs(R - np.median(R, axis=1, keepdims=True)), axis=1)   # robust background of R
    for s0 in range(S - 1):
        tri = X.complex_peeled(frames[s0])["tri"]; a0 = X._star(frames[s0], tri, N); nb = star_nb(tri, N)
        for w in Ws:
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
            if cand.any() and trim:
                # (c) a window is only needed where the drop is NOT already explained by shorter hot steps inside it,
                # and only the steps that carry the compression are marked (first..last step with R < -frac*D/w)
                idx = np.where(cand)[0]; Rw = R[idx, s0:s0 + w]; Dw = D[idx]
                expl = (Rw * hot[idx, s0:s0 + w]).sum(1)
                keep = expl > 0.5 * Dw                               # |explained| < half of the window drop
                sig = Rw < frac * Dw[:, None] / w
                for j in np.where(keep)[0]:
                    ss = np.where(sig[j])[0]
                    if len(ss) and not trim_span:
                        hot[idx[j], s0:s0 + w] = True
                    elif len(ss) and tail == "adaptive":
                        # extend the marked span on either side while the star keeps compressing measurably:
                        # R < -max(c_noise * robust background of R, f_peak * peak step compression)
                        i = idx[j]; a, b = s0 + ss[0], s0 + ss[-1]
                        thr = max(c_noise * bgR[i], f_peak * float(-Rw[j].min()))
                        while a > 0 and R[i, a - 1] < -thr: a -= 1
                        while b < S - 2 and R[i, b + 1] < -thr: b += 1
                        hot[i, max(a - 1, 0):min(b + 2, S - 1)] = True
                    elif len(ss):
                        pd = int(np.ceil(pad * (ss[-1] - ss[0] + 1)))
                        hot[idx[j], max(s0 + ss[0] - pd, 0):min(s0 + ss[-1] + 1 + pd, S - 1)] = True
            elif cand.any():
                hot[cand, s0:s0 + w] = True
    return hot, R, tau


# --------------------------------------------------------------------------------------------- rings and arcs
def rings(tri, centres):
    """ordered (CCW) one-ring of each centre from a CCW triangulation; None for open (boundary) fans"""
    cs = set(int(c) for c in centres); nxt = {c: {} for c in cs}
    for a, b, c in tri:
        a, b, c = int(a), int(b), int(c)
        if a in cs: nxt[a][b] = c
        if b in cs: nxt[b][c] = a
        if c in cs: nxt[c][a] = b
    out = {}
    for c in cs:
        d = nxt[c]
        if len(d) < 3:
            out[c] = None; continue
        start = next(iter(d)); r = [start]; cur = start; ok = True
        for _ in range(len(d)):
            cur = d.get(cur)
            if cur is None:
                ok = False; break
            if cur == start:
                break
            r.append(cur)
        out[c] = np.array(r) if ok and cur == start and len(r) == len(d) else None
    return out


def fan_area(P, c, arc):
    """area of the fan polygon (c, arc[0], ..., arc[-1]) (arc consecutive ring vertices)"""
    a = P[arc[:-1]] - P[c]; b = P[arc[1:]] - P[c]
    return 0.5 * float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))


def _arc_resid(Pa, Pt, c, arc):
    A = Pa[arc] - Pa[c]; B = Pt[arc] - Pt[c]
    Ft, *_ = np.linalg.lstsq(A, B, rcond=None); r = B - A @ Ft
    return float(np.sqrt((r ** 2).sum(1).mean()))


def best_arc(Pa, Pt, c, ring, rel=0.25, floor=0.0, Lmin=3):
    """longest contiguous ring arc (>= Lmin vertices) moving affinely with the centre: residual (normalised by the
    sqrt of the anchor star area) < rel * residual of the full ring and < floor.  Returns (arc, r_arc, r_full)."""
    n = len(ring); sA = np.sqrt(max(fan_area(Pa, c, np.r_[ring, ring[:1]]), 1e-300))
    rf = _arc_resid(Pa, Pt, c, ring) / sA
    for L in range(n - 1, Lmin - 1, -1):
        best = None
        for s in range(n):
            arc = ring[(s + np.arange(L)) % n]; ra = _arc_resid(Pa, Pt, c, arc) / sA
            if best is None or ra < best[1]:
                best = (arc, ra)
        if best[1] < rel * rf and best[1] < floor:
            return best[0], best[1], rf
    return None, np.nan, rf


def same_material_arc(ring, mat_ok):
    """largest cyclic run of ring vertices with mat_ok True (>= 3 vertices, i.e. >= 2 fan triangles)"""
    n = len(ring); m = mat_ok.astype(int)
    if m.all() or m.sum() < 3:
        return None
    k0 = int(np.where(m == 0)[0][0]); best = (0, 0); run = 0; st = 0
    for j in range(1, n + 1):
        i = (k0 + j) % n
        if m[i]:
            if run == 0: st = i
            run += 1
            if run > best[0]: best = (run, st)
        else:
            run = 0
    if best[0] < 3:
        return None
    return ring[(best[1] + np.arange(best[0])) % n]


# --------------------------------------------------------------------------------------------- driver
def ledger_v4(fr, rho0=None, block=5, geo=True, defer_min=4, defer_cap=60, tol_smooth=0.1, tol_shock=0.02,
              delta_fac=2.0, fit_tol=0.05, k=12, arc_shock=True, arc_iface=True, iface_jump=0.1, arc_rel=0.25,
              hot=None, Ws=GEO_W, arc_min=0.03, noise_fac=3.0, verbose=False):
    """returns dict(lnJ_poly (N,) polygon ledger at the final frame with the same detection, lnJ_v4 (N,), parts)"""
    S, N, _ = fr.shape
    if hot is None:
        hot = hot_steps_geo(fr, Ws=Ws)[0] if geo else C.hot_steps(fr)[0]
    run = hot_runs_q(hot); defer = int(np.clip(np.ceil((2 * run + 2) / block), defer_min, defer_cap))
    bad = bad_from_hot(hot)
    d, _ = cKDTree(fr[0]).query(fr[0], k=2); h = float(np.median(d[:, 1]))
    sig = noise_sigma(fr) if S >= 4 else 0.0
    floor = 0.01 + 3.0 * sig / h
    lr = np.zeros(N) if rho0 is None else np.log(np.asarray(rho0, float))

    tris = {0: X.complex_peeled(fr[0])["tri"]}; nbs = {0: star_nb(tris[0], N)}
    anchor = np.zeros(N, int); A0 = X._star(fr[0], tris[0], N)
    knn = cKDTree(fr[0]).query(fr[0], k + 1)[1][:, 1:]
    base_p = np.zeros(N); base_v = np.zeros(N)
    ring_c = {}                                            # tracer -> (anchor, ring) for interface stars
    D = dict(segments=0, tensor_segments=0, tensor_shock_segments=0, shock_segments=0, iface_arc_segments=0,
             arc_repairs=0, rejected_fit=0, rejected_tol=0, max_defer=defer, hot_frac=float(hot.mean()),
             noise_sigma_over_h=sig / h)

    def iface_mask(nb_rows, ii):
        okn = nb_rows >= 0; nbc = np.where(okn, nb_rows, ii[:, None])
        return (np.abs(lr[nbc] - lr[ii][:, None]) > iface_jump).any(1)

    def increments(ii, a, t, final=False):
        """polygon / chosen increments for tracers ii anchored at a, evaluated at t"""
        Pa, Pt = fr[a], fr[t]
        with np.errstate(all="ignore"):
            poly = np.log(X._star(Pt, tris[a], N)[ii] / A0[ii])
        nbst = nbs[a][ii]
        with np.errstate(all="ignore"):
            Fs = _affine_F(Pa, Pt, ii, nbst); delta = poly - np.log(np.linalg.det(Fs))
        delta = np.where(np.isfinite(delta), delta, 0.0)
        # interface stars: same-material arc (every segment)
        pol_v = poly.copy(); ifc = iface_mask(nbst, ii) if (arc_iface and rho0 is not None) else np.zeros(len(ii), bool)
        if ifc.any():
            need = [int(i) for i in ii[ifc] if ring_c.get(int(i), (None,))[0] != a]
            if need:
                rg = rings(tris[a], need)
                for i in need: ring_c[i] = (a, rg[i])
            for j in np.where(ifc)[0]:
                i = int(ii[j]); rg = ring_c[i][1]
                if rg is None: continue
                arc = same_material_arc(rg, np.abs(lr[rg] - lr[i]) <= iface_jump)
                if arc is None: continue
                a0, a1 = fan_area(Pa, i, arc), fan_area(Pt, i, arc)
                if a0 > 0 and a1 > 0:
                    pol_v[j] = np.log(a1 / a0); D["iface_arc_segments"] += 1
        # tensor increment on k neighbours (anchor-time kNN)
        nb = knn[ii]; A = Pa[nb] - Pa[ii][:, None]; B = Pt[nb] - Pt[ii][:, None]
        F, err = SE._fitF(A, B, h, True, err=True)
        with np.errstate(all="ignore"):
            ten = np.log(np.linalg.det(F))
        ifk = (np.abs(lr[nb] - lr[ii][:, None]) > iface_jump).any(1) if rho0 is not None else np.zeros(len(ii), bool)
        hs = bad[:, a:t + 1].any(1); hseg = hs[ii] | hs[nb].any(1) | np.where(nbst >= 0, hs[np.maximum(nbst, 0)], False).any(1)
        fit_ok = np.isfinite(ten) & (err < fit_tol ** 2) & ~ifk & ~ifc
        tol = np.where(hseg, tol_shock + delta_fac * np.abs(delta) + noise_fac * sig / h, tol_smooth)
        use = fit_ok & (np.abs(ten - poly) <= tol)
        ch = np.where(use, ten, pol_v)
        D["segments"] += len(ii); D["tensor_segments"] += int(use.sum()); D["shock_segments"] += int(hseg.sum())
        D["tensor_shock_segments"] += int((use & hseg).sum()); D["rejected_fit"] += int((~fit_ok).sum())
        D["rejected_tol"] += int((fit_ok & ~use).sum())
        # open segment at the evaluation time: one-sided arc for stars being swept by a shock now
        if final and arc_shock:
            now = bad[:, max(t - 1, 0):t + 1].any(1)
            sw = np.where(now[ii] | np.where(nbst >= 0, now[np.maximum(nbst, 0)], False).any(1))[0]
            if len(sw):
                rg = rings(tris[a], [int(ii[j]) for j in sw])
                for j in sw:
                    i = int(ii[j]); r = rg[i]
                    if r is None or len(r) < 5: continue
                    arc, ra, rf = best_arc(Pa, Pt, i, r, rel=arc_rel, floor=floor)
                    if arc is None or rf < arc_min: continue
                    a0, a1 = fan_area(Pa, i, arc), fan_area(Pt, i, arc)
                    if a0 > 0 and a1 > 0:
                        ch[j] = np.log(a1 / a0); D["arc_repairs"] += 1
        return poly, ch

    for t in range(1, S):
        due = (t - anchor) >= block
        if due.any() and t < S - 1:
            trit = X.complex_peeled(fr[t])["tri"]
            nbd = bad[:, max(t - 1, 0):t + 2].any(1); sb = np.zeros(N, bool)
            for kk in range(3):
                np.logical_or.at(sb, trit[:, kk], nbd[trit].any(1))
            ok = due & (~sb | ((t - anchor) >= defer * block))
            if ok.any():
                At = X._star(fr[t], trit, N); good = np.where(ok & (At > 0))[0]
                for a in np.unique(anchor[good]):
                    ii = good[anchor[good] == a]; p, c = increments(ii, a, t)
                    base_p[ii] += np.where(np.isfinite(p), p, 0.0); base_v[ii] += np.where(np.isfinite(c), c, p)
                tris[t] = trit; nbs[t] = star_nb(trit, N)
                A0[good] = At[good]; anchor[good] = t
                knn[good] = cKDTree(fr[t]).query(fr[t][good], k + 1)[1][:, 1:]
        live = set(np.unique(anchor).tolist())
        for a in list(tris):
            if a not in live:
                del tris[a]; del nbs[a]
        if verbose and t % 50 == 0:
            print("   t=%d/%d" % (t, S - 1), flush=True)
    lp = np.full(N, np.nan); lv = np.full(N, np.nan)
    for a in np.unique(anchor):
        ii = np.where(anchor == a)[0]; p, c = increments(ii, a, S - 1, final=True)
        lp[ii] = base_p[ii] + p; lv[ii] = base_v[ii] + np.where(np.isfinite(c), c, p)
    D["tensor_share"] = D["tensor_segments"] / max(D["segments"], 1)
    return dict(lnJ_poly=lp, lnJ_v4=lv, hot=hot, diag=D)
