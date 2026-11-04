"""AFRC_E constructions transplanted onto the strictly objective triangle-chain rotation (tcx).

Three pieces, each the triangle-level form of one FRC6 ingredient:
  node_noise        S4 random-walk floor  -> per-node, per-segment noise sigma_seg from the node's own increments
  lavd_energy       S4 soft threshold     -> energy debias of each segment's |.|  (no global calibration)
  phi_shrink        S2 significance gate  -> phi soft-thresholded by its own noise envelope (absolute zero)
  coherent_boundary dispersion gate g     -> outermost closed level set whose MATERIAL perimeter does not grow
                                             (replaces Haller's convexity deficiency, which is a proxy for it)
Everything uses tracer positions only.
"""
import numpy as np
from matplotlib.path import Path


def _segment_sums(ch, psibar):
    inc = ch["inc"]; S1, N = inc.shape
    d = inc - psibar[:, None]
    cs = np.vstack([np.zeros((1, N)), np.cumsum(d, 0)])
    return d, cs


def node_noise(ch, psibar):
    """sigma_seg,i: noise std of one anchored segment sum at node i.
    Per-step increments of a fixed triangle telescope the position noise: d_s = signal + c(n_{s+1}-n_s), so
    Delta d carries a second difference of n (variance 6 sig^2) while a segment sum carries n_end-n_start (2 sig^2).
    Hence sigma_seg = sd(Delta d)/sqrt(3), estimated robustly from the node's own series (no global calibration)."""
    d, _ = _segment_sums(ch, psibar)
    dd = np.diff(d, axis=0)
    mad = np.median(np.abs(dd - np.median(dd, 0)), 0)
    return 1.4826 * mad / np.sqrt(3.0)


def lavd_energy(ch, psibar, sig_seg):
    """LAVD = sum_seg 2|x_seg|, with |x| estimated as sqrt(max(0, y^2 - sigma_seg^2)) per segment."""
    d, cs = _segment_sums(ch, psibar); N = d.shape[1]
    out = np.zeros(N); nseg = np.zeros(N)
    for n in range(N):
        s2 = sig_seg[n] ** 2
        for (s0, s1, rem) in ch["segs"][n]:
            y = cs[s1, n] - cs[s0, n] + rem
            out[n] += 2 * np.sqrt(max(0.0, y * y - s2)); nseg[n] += 1
    return out, nseg


def phi_shrink(phi, sig_seg, nseg, z):
    """phi soft-thresholded by z times its own noise envelope sigma_seg*sqrt(nseg): pure noise -> exactly 0."""
    env = sig_seg * np.sqrt(np.maximum(nseg, 1))
    return np.sign(phi) * np.maximum(0.0, np.abs(phi) - z * env), env


# ------------------------------------------------------------------ material perimeter of a level curve
def _edge_lookup(tri):
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    return np.unique(e, axis=0)


def contour_on_edges(P0, tri, f, level):
    """Closed contour pieces of P1 field f at 'level' on triangulation tri, each vertex given as
    (edge a, edge b, parameter t)  so that its image at any time is (1-t) x_a + t x_b.  Marching triangles."""
    fa = f[tri] - level
    sgn = fa > 0
    cut = (sgn[:, 0] != sgn[:, 1]).astype(int) + (sgn[:, 1] != sgn[:, 2]) + (sgn[:, 2] != sgn[:, 0])
    tt = np.where(cut == 2)[0]
    # each crossing triangle contributes one segment between its two cut edges
    key = {}; adj = {}
    def ekey(a, b):
        return (a, b) if a < b else (b, a)
    segs = []
    for t in tt:
        es = []
        for j in range(3):
            a, b = tri[t, j], tri[t, (j + 1) % 3]
            if sgn[t, j] != sgn[t, (j + 1) % 3]:
                es.append(ekey(a, b))
        segs.append(es)
        for e in es:
            adj.setdefault(e, []).append(len(segs) - 1)
    # chain segments into polylines through shared edges
    used = np.zeros(len(segs), bool); loops = []
    for s0 in range(len(segs)):
        if used[s0]:
            continue
        used[s0] = True; chain = [segs[s0][0], segs[s0][1]]; closed = False
        while True:
            last = chain[-1]; nxt = [s for s in adj.get(last, []) if not used[s]]
            if not nxt:
                break
            s = nxt[0]; used[s] = True
            e2 = segs[s][1] if segs[s][0] == last else segs[s][0]
            if e2 == chain[0]:
                closed = True; break
            chain.append(e2)
        if closed and len(chain) >= 6:
            A = np.array([c[0] for c in chain]); B = np.array([c[1] for c in chain])
            t = (level - f[A]) / (f[B] - f[A])
            loops.append((A, B, t))
    return loops


def _poly(fr_s, A, B, t):
    return (1 - t)[:, None] * fr_s[A] + t[:, None] * fr_s[B]


def _smooth_closed(Q, n=2):
    for _ in range(n):
        Q = 0.25 * np.roll(Q, 1, 0) + 0.5 * Q + 0.25 * np.roll(Q, -1, 0)
    return Q


def _perim(Q):
    return float(np.linalg.norm(np.diff(np.vstack([Q, Q[:1]]), axis=0), axis=1).sum())


def _area(Q):
    x, y = Q[:, 0], Q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _convdef(Q):
    from scipy.spatial import ConvexHull
    try:
        return (ConvexHull(Q).volume - _area(Q)) / max(_area(Q), 1e-300)
    except Exception:
        return np.inf


def _perim_debiased(Q, t, sig):
    """S4-style energy debias of each polygon side: E|d_obs|^2 = |d|^2 + 2(sv_k^2 + sv_k+1^2)."""
    sv2 = sig ** 2 * ((1 - t) ** 2 + t ** 2)
    D = np.roll(Q, -1, 0) - Q; d2 = (D ** 2).sum(1) - 2 * (sv2 + np.roll(sv2, -1))
    return float(np.sqrt(np.maximum(d2, 0.0)).sum())


def scan_levels(fr, tri, f, core, nlev=60, frames=None, others=(), sig=0.0):
    """For each level from 0.98 f[core] down to 0.02 f[core]: the closed contour around the core (at t0), its
    convexity deficiency, and its material perimeter growth max_t ln(L(t)/L(0)) through the tracer flow map."""
    P0 = fr[0]; fmax = f[core]; pc = P0[core]
    S = len(fr); frames = np.unique(np.linspace(0, S - 1, 11).round().astype(int)) if frames is None else frames
    rec = []
    for lev in np.linspace(0.98 * fmax, 0.02 * fmax, nlev):
        loops = contour_on_edges(P0, tri, f, lev); hit = None
        for (A, B, t) in loops:
            Q0 = _poly(P0, A, B, t)
            if Path(Q0).contains_point(pc):
                if hit is None or _area(Q0) < _area(hit[3]):
                    hit = (A, B, t, Q0)
        if hit is None:
            rec.append(dict(level=lev, found=False)); continue
        A, B, t, Q0 = hit; L0 = _perim(Q0)
        pth = Path(Q0); merged = any(pth.contains_point(P0[o]) for o in others)
        g = max(np.log(_perim(_poly(fr[s], A, B, t)) / L0) for s in frames)
        L0d = max(_perim_debiased(Q0, t, sig), 1e-300)
        gs = max(np.log(max(_perim_debiased(_poly(fr[s], A, B, t), t, sig), 1e-300) / L0d) for s in frames)
        rec.append(dict(level=lev, found=True, merged=merged, area=_area(Q0), cdef=_convdef(Q0), lnL=float(g), lnLd=float(gs), poly=Q0, A=A, B=B, t=t))
    return rec


def pick(rec, crit, thr):
    """outermost level (scanning downward) that satisfies crit<=thr, stopping when the contour is lost or merges
    (area jumps by >4x, i.e. it swallowed a neighbour)."""
    best = None; started = False; prevA = None
    for r in rec:
        if not r["found"]:
            if started:
                break
            continue
        if r["merged"]:
            break
        started = True; prevA = r["area"]
        if r[crit] <= thr:
            best = r
    return best


def inside(P0, poly):
    return Path(poly).contains_points(P0)
