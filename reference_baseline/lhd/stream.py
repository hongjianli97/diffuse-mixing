"""Streaming material ledger for broken tracks (PTV-like data).

Data model
----------
P : (S, M, 2) float array of TRACK positions, NaN where the track is absent.  A physical particle may produce
    several tracks (a break ends a track; the particle reappears later as a new track with a new id).
Each track keeps its OWN anchor: anchor frame a_i, the ordered link ring of its Delaunay star at a_i (track ids),
the ring positions relative to the track at a_i, and a validity flag for each consecutive ring pair.

Ledger rule
-----------
ln J_i(t) = base_i + ln( A_ring(t) / A_ring(a_i) ),  A_ring = signed fan area around i over the ring vertices that
are PRESENT at t (both areas use the same vertex subset, so a vanished vertex shrinks the polygon instead of breaking
the measurement; the anchor frame is not moved, so a crossing shock still telescopes).
Re-anchor (new ring from the current Delaunay of present tracks, base := ln J(t)) when the segment is 'due':
age >= block, or the affine residual of the ring motion > eps, or fewer than 3 ring vertices / less than
`min_frac` of the anchor area survive.  Re-anchoring is deferred while the ring touches a shock stencil, up to
`max_defer` extra frames; a ring that has become unmeasurable inside a shock is re-anchored anyway (counted).

Births
------
A track that appears at s0 > 0 starts its own ledger (ln J = 0 at s0).  Its absolute density is calibrated from its
established ring neighbours (inverse-distance mean of ln rho) either immediately at s0 ('immediate') or at the first
later frame at which neither it nor its ring is in a shock stencil ('deferred'); the ledger then carries it back/forward.
"""
import numpy as np
from scipy.spatial import cKDTree
import lhdx as X


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def tracks_from_particles(fr, p, gap=(1, 3), seed=0, forced=None):
    """Split particle trajectories fr (S,N,2) into tracks.  After a frame, each live track breaks with probability p
    (or deterministically where forced[n, s] is True); the particle is then absent for a random gap and reappears
    as a NEW track.  Returns P (S,M,2) with NaN, owner (M,), start (M,), end (M,)."""
    rng = np.random.default_rng(seed); S, N, _ = fr.shape
    cols, owner, start, end = [], [], [], []
    for n in range(N):
        s = 0
        while s < S:
            e = s
            while e < S-1:
                brk = (forced is not None and forced[n, e]) or (rng.random() < p)
                if brk: break
                e += 1
            col = np.full((S, 2), np.nan); col[s:e+1] = fr[s:e+1, n]
            cols.append(col); owner.append(n); start.append(s); end.append(e)
            s = e+1+int(rng.integers(gap[0], gap[1]+1))
    P = np.stack(cols, 1)
    return P, np.array(owner), np.array(start), np.array(end)


def _rings(Pt, ids, M):
    """Ordered link rings of the present tracks ids (positions Pt (M,2) with NaN for absent) from the peeled Delaunay.
    Returns ring (M,K) track ids (-1 pad), pairok (M,K) validity of pair (k, k+1 mod n), nring (M,)"""
    loc = X.complex_peeled(Pt[ids])["tri"]; tri = ids[loc]
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]], tri[:, [1, 0]], tri[:, [2, 1]], tri[:, [0, 2]]])
    e = np.unique(e, axis=0); cnt = np.bincount(e[:, 0], minlength=M); K = max(int(cnt.max()), 3)
    ring = np.full((M, K), -1); st = np.r_[0, np.cumsum(cnt)[:-1]]; pos = np.arange(len(e))-st[e[:, 0]]
    ring[e[:, 0], pos] = e[:, 1]
    rr = np.where(ring >= 0, ring, 0); d = Pt[rr]-Pt[np.arange(M)][:, None]
    ang = np.where(ring >= 0, np.arctan2(d[..., 1], d[..., 0]), np.inf)
    o = np.argsort(ang, axis=1); ring = np.take_along_axis(ring, o, 1)
    nring = (ring >= 0).sum(1)
    key = np.sort(tri, 1); key = (key[:, 0]*M+key[:, 1])*M+key[:, 2]; key = np.unique(key)
    nxt = np.take_along_axis(ring, (np.arange(K)[None, :]+1) % np.maximum(nring, 1)[:, None], 1)
    a = np.arange(M)[:, None]*np.ones((1, K), int)
    tk = np.sort(np.stack([a, np.where(ring >= 0, ring, 0), np.where(nxt >= 0, nxt, 0)], -1), -1)
    tk = (tk[..., 0]*M+tk[..., 1])*M+tk[..., 2]
    pairok = (ring >= 0) & (nxt >= 0) & np.isin(tk, key)
    return ring, pairok, nring


def _fan_area(center, ringpos, valid_v, pairok, inring):
    """signed fan area around center over the PRESENT ring vertices (valid_v), keeping the original angular order;
    a pair of consecutive present vertices counts if all original pairs between them were triangles."""
    M, K = valid_v.shape
    o = np.argsort(~valid_v, axis=1, kind="stable")                      # present first, order kept
    pv = np.take_along_axis(valid_v, o, 1); c = pv.sum(1)
    pos = np.take_along_axis(ringpos, o[..., None], 1)-center[:, None]
    # invalid-gap counter along the original ring
    gap = np.cumsum(~pairok, axis=1); gap = np.concatenate([np.zeros((M, 1), int), gap[:, :-1]], 1)
    gpos = np.take_along_axis(gap, o, 1); total = (~pairok & inring).sum(1)
    nxt_idx = (np.arange(K)[None, :]+1) % np.maximum(c, 1)[:, None]
    pos2 = np.take_along_axis(pos, nxt_idx[..., None], 1); g2 = np.take_along_axis(gpos, nxt_idx, 1)
    inside = np.arange(K)[None, :] < c[:, None]
    wrap = np.arange(K)[None, :] == (c-1)[:, None]
    ok_pair = np.where(wrap, (total[:, None]-gpos+g2) == 0, (g2-gpos) == 0) & inside & (c[:, None] >= 2)
    return 0.5*np.where(ok_pair, _cross(pos, pos2), 0.0).sum(1), c



def _rering(P, pres, b, a, t, k=12):
    """New ring for track b AT ITS ANCHOR FRAME a, built only from tracks present at both a and t (local Delaunay of
    the k nearest survivors).  Keeps the anchor (pre-shock) so the crossing still telescopes."""
    from scipy.spatial import Delaunay
    surv = np.where(pres[a] & pres[t])[0]; surv = surv[surv != b]
    if surv.size < 3: return None
    d, nb = cKDTree(P[a, surv]).query(P[a, b], k=min(k, surv.size)); nb = surv[np.atleast_1d(nb)]
    pts = np.vstack([P[a, b][None], P[a, nb]])
    try: tri = Delaunay(pts).simplices
    except Exception: return None
    inc = tri[(tri == 0).any(1)]
    if len(inc) == 0: return None
    nbrs = np.unique(inc[inc != 0]); rel = pts[nbrs]-pts[0]; o = np.argsort(np.arctan2(rel[:, 1], rel[:, 0])); nbrs = nbrs[o]
    tris = {tuple(sorted(x)) for x in inc.tolist()}
    n = len(nbrs); ok = np.array([tuple(sorted((0, nbrs[i], nbrs[(i+1) % n]))) in tris for i in range(n)])
    return nb[nbrs-1], ok, pts[nbrs]-pts[0]

def hot_tracks(P, tau_min=0.2, kfac=4.0):
    """C channel on tracks: star-area log ratio over each step, on the Delaunay of tracks present at s and s+1."""
    S, M, _ = P.shape; R = np.zeros((M, S-1)); have = np.zeros((M, S-1), bool)
    pres = ~np.isnan(P[..., 0])
    for s in range(S-1):
        ids = np.where(pres[s] & pres[s+1])[0]
        tri = X.complex_peeled(P[s, ids])["tri"]
        a0 = X._star(P[s, ids], tri, len(ids)); a1 = X._star(P[s+1, ids], tri, len(ids))
        with np.errstate(all="ignore"):
            r = np.where((a0 > 0) & (a1 > 0), np.log(a1/a0), 0.0)
        R[ids, s] = r; have[ids, s] = True
    v = R[have]; sig = 1.4826*np.median(np.abs(v-np.median(v))); tau = max(tau_min, kfac*sig)
    return (np.abs(R) > tau) & have, tau


def ledger_stream(P, rho_init, block=5, eps=0.01, max_defer=15, min_frac=0.3, calib="deferred", tau_min=0.2):
    """rho_init (M,) : known density at frame 0 for tracks present at frame 0 (NaN otherwise).
    Returns rho (S,M) estimated density (NaN where absent / unknown), diagnostics."""
    S, M, _ = P.shape; pres = ~np.isnan(P[..., 0])
    hot, tau = hot_tracks(P, tau_min)
    bad = np.zeros((M, S), bool); bad[:, :-1] |= hot; bad[:, 1:] |= hot
    first = np.argmax(pres, 0); last = S-1-np.argmax(pres[::-1], 0)
    K = 16
    ring = np.full((M, K), -1); pairok = np.zeros((M, K), bool); rrel = np.zeros((M, K, 2)); anchor = np.full(M, -1)
    base = np.zeros(M); lnJ = np.full((S, M), np.nan); A0full = np.ones(M)
    n_forced_shock = 0; n_reanchor = 0; n_shrunk = 0; n_rering = 0
    for t in range(S):
        ids = np.where(pres[t])[0]; Pt = np.where(pres[t][:, None], P[t], np.nan)
        # ---- measure current segments
        live = ids[anchor[ids] >= 0]
        if live.size:
            rg = ring[live]; v = (rg >= 0) & pres[t][np.where(rg >= 0, rg, 0)]
            cur_pos = P[t][np.where(rg >= 0, rg, 0)]; anc_pos = rrel[live]
            Acur, c = _fan_area(P[t][live], cur_pos, v, pairok[live], rg >= 0)
            Aanc, _ = _fan_area(np.zeros((len(live), 2)), anc_pos, v, pairok[live], rg >= 0)
            okm = (c >= 3) & (Aanc > min_frac*A0full[live]) & (Acur > 0) & (Aanc > 0)
            n_shrunk += int(((v.sum(1) < (rg >= 0).sum(1)) & okm).sum())
            val = np.where(okm, base[live]+np.log(np.where(okm, Acur, 1)/np.where(okm, Aanc, 1)), np.nan)
            prev = lnJ[t-1, live] if t > 0 else np.zeros(len(live))
            lnJ[t, live] = np.where(okm, val, prev)                        # unmeasurable: hold the last value
            broken = ~okm
            shock0 = bad[live, t] | (bad[np.where(rg >= 0, rg, 0), t] & (rg >= 0)).any(1)
            for q in np.where(broken)[0]:                                  # re-ring at the anchor frame, keep anchor
                b = live[q]; rr_ = _rering(P, pres, b, anchor[b], t)
                if rr_ is None: continue
                nbr, okp, rel = rr_; n = len(nbr)
                if n > ring.shape[1]: continue
                ring[b] = -1; ring[b, :n] = nbr; pairok[b] = False; pairok[b, :n] = okp; rrel[b] = 0; rrel[b, :n] = rel
                vv = np.zeros((1, ring.shape[1]), bool); vv[0, :n] = True
                fa, _ = _fan_area(np.zeros((1, 2)), rrel[b][None], vv, pairok[b][None], vv); A0full[b] = max(fa[0], 1e-30)
                cpos = np.zeros((1, ring.shape[1], 2)); cpos[0, :n] = P[t][nbr]
                ac, cc = _fan_area(P[t][b][None], cpos, vv, pairok[b][None], vv)
                if cc[0] >= 3 and ac[0] > 0 and fa[0] > 0:
                    lnJ[t, b] = base[b]+np.log(ac[0]/fa[0]); okm[q] = True; n_rering += 1
            broken = ~okm
            # affine residual of ring motion (present vertices)
            G = np.einsum("nki,nkj->nij", anc_pos*v[..., None], anc_pos)+1e-30*np.eye(2)
            rel = (cur_pos-P[t][live][:, None])*v[..., None]
            H = np.einsum("nki,nkj->nij", anc_pos*v[..., None], rel)
            Ft = np.linalg.pinv(G) @ H; r = (rel-np.einsum("nki,nij->nkj", anc_pos, Ft))*v[..., None]
            res = np.sqrt((r**2).sum((1, 2))/np.maximum(v.sum(1), 1))/np.sqrt(np.maximum(np.abs(Acur), 1e-30))
            age = t-anchor[live]
            shock = bad[live, t] | (bad[np.where(rg >= 0, rg, 0), t] & (rg >= 0)).any(1)
            lost = v.sum(1) < (rg >= 0).sum(1)
            due = (age >= block) | (res > eps) | broken | (lost & ~shock)
            okre = due & ((~shock) | (age >= block+max_defer) | broken)
            n_forced_shock += int((broken & shock).sum())
            reanchor = live[okre] if t < S-1 else live[:0]
        else:
            reanchor = ids[:0]
        newborn = ids[anchor[ids] < 0]
        lnJ[t, newborn] = 0.0
        todo = np.r_[reanchor, newborn]
        if todo.size and t < S-1:
            rg2, po2, nr2 = _rings(Pt, ids, M)
            if rg2.shape[1] > K:                                          # enlarge storage if needed
                pad = rg2.shape[1]-K; K = rg2.shape[1]
                ring = np.c_[ring, np.full((M, pad), -1)]; pairok = np.c_[pairok, np.zeros((M, pad), bool)]
                rrel = np.concatenate([rrel, np.zeros((M, pad, 2))], 1)
            kk = rg2.shape[1]
            ring[todo] = -1; ring[todo, :kk] = rg2[todo]; pairok[todo] = False; pairok[todo, :kk] = po2[todo]
            rrel[todo] = 0; rr = np.where(rg2[todo] >= 0, rg2[todo], 0)
            rrel[todo, :kk] = np.where((rg2[todo] >= 0)[..., None], P[t][rr]-P[t][todo][:, None], 0.0)
            full, _ = _fan_area(np.zeros((len(todo), 2)), rrel[todo], ring[todo] >= 0, pairok[todo], ring[todo] >= 0)
            A0full[todo] = np.maximum(full, 1e-30)
            base[todo] = lnJ[t, todo]; anchor[todo] = t; n_reanchor += len(reanchor)
    # ---- absolute densities
    rho = np.full((S, M), np.nan); known = np.zeros(M, bool)
    st0 = pres[0] & np.isfinite(rho_init)
    rho[:, st0] = rho_init[st0][None, :]*np.exp(-np.nan_to_num(lnJ[:, st0]-lnJ[0, st0]))
    known[st0] = True
    order = np.argsort(first)
    for j in order:
        if known[j]: continue
        s0 = first[j]; cand = range(s0, last[j]+1)
        tc = s0
        if calib == "deferred":
            for tt in cand:
                if not bad[j, tt] and tt+1 <= last[j]:
                    tc = tt; break
        # neighbours with known density at tc
        idk = np.where(known & pres[tc] & np.isfinite(rho[tc]))[0]
        if idk.size == 0: continue
        d, nb = cKDTree(P[tc, idk]).query(P[tc, j], k=min(8, idk.size)); w = 1/np.maximum(np.atleast_1d(d), 1e-12)
        lr = np.log(rho[tc, idk[np.atleast_1d(nb)]]); rc = np.exp((w*lr).sum()/w.sum())
        rho[:, j] = rc*np.exp(-(lnJ[:, j]-lnJ[tc, j]))
        rho[~pres[:, j], j] = np.nan; known[j] = True
    rho[~pres] = np.nan
    return rho, dict(tau=tau, n_rering=n_rering, n_reanchor=n_reanchor, n_forced_shock=n_forced_shock, n_shrunk=n_shrunk, lnJ=lnJ, bad=bad)


def ledger_windowed(P, rho_init, Lw=5):
    """previous method on the track representation: per window, only tracks complete in the window; new tracks get
    their density from the 8 nearest tracks of known density (inverse distance, ln rho)."""
    S, M, _ = P.shape; pres = ~np.isnan(P[..., 0]); rho = np.full((S, M), np.nan)
    D = np.where(pres[0], rho_init, np.nan)
    rho[0] = D
    for a in range(0, S-1, Lw):
        b = min(a+Lw, S-1); use = pres[a:b+1].all(0)
        known = use & np.isfinite(D); unk = use & ~known
        if unk.any() and known.any():
            d, nb = cKDTree(P[a, known]).query(P[a, unk], k=8); w = 1/np.maximum(d, 1e-12)
            D[unk] = np.exp((w*np.log(D[known][nb])).sum(1)/w.sum(1))
        sub = P[a:b+1][:, use]
        hot, _, tau = __import__("classify").hot_steps(sub)
        Sb, Nb, _ = sub.shape; bd = np.zeros((Nb, Sb), bool); bd[:, :-1] |= hot; bd[:, 1:] |= hot
        lnJ = X.chained_lnJ_v2(sub, bd)
        newD = np.full(M, np.nan)
        for k in range(1, b-a+1):
            rho[a+k, use] = D[use]*np.exp(-lnJ[k])
        newD[use] = D[use]*np.exp(-lnJ[-1]); D = newD
    rho[~pres] = np.nan
    return rho
