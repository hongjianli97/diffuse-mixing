"""
lhd2.py -- Lagrangian Hodge-Dirac (LHD) decomposition on a trajectory network, v2.
v2 (benchmark revision): complex_peeled (no internal holes), Galerkin P1/Crouzeix-Raviart inner product
(galerkin_div, galerkin_hodge; replaces the diagonal cotan star, which is pointwise inconsistent on deformed
tracer clouds), chained_lnJ_v2 (per-tracer anchors, re-anchoring deferred inside shock stencils).
The v1 functions are kept for comparison.

Companion to "Discrete trajectory network curvature for objective identification of
coherent structures in compressible flows" (AFRC).  Uses tracer positions only.

Design (report, section 3)
--------------------------
* Operators live on a fresh Delaunay complex of the CURRENT tracer positions at every
  snapshot, with cotan Hodge stars.  DEC on the fixed t_c material complex degrades in
  shear zones (synthetic test: vortex->gradient leakage 4.5% -> 93% over one turn).
* Material bookkeeping lives on LABELS: fixed tracer pairs (AFRC_H edges) for stretching
  attribution, tracer sets (AFRC_E cores) for Kelvin budgets, and piecewise-material
  complexes re-anchored every `block` snapshots for ln J.
* Objective fields use the relative velocity v_d (Aksamit et al. 2024), so every Hodge
  part is frame-indifferent.  Kelvin budgets use the inertial-frame velocity.
* Shock crossings: RH-event censoring with the same +-1 logic as the hyperbolic channel.

  D = [[0, B1, 0], [B1^T, 0, B2], [0, B2^T, 0]],   D^2 = L0 (+) L1 (+) L2
  D (0, f, 0) = (B1 f, 0, B2^T f)  ->  node dilatation (with Hodge stars), triangle circulation
  Gershgorin margin of row e of L1 = 4 - deg u - deg v + 3 #tri(e) = struct(e) (AFRC Eq. 4)
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay, cKDTree

# --------------------------------------------------------------------------- geometry


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def tri_signed_area(P, tri):
    return 0.5 * _cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])


def _tri_edge_lengths(P, tri):
    return np.stack([np.linalg.norm(P[tri[:, j]] - P[tri[:, (j + 1) % 3]], axis=1)
                     for j in range(3)], 1)

# --------------------------------------------------------------------------- complex


def _assemble(tri, N, edges=None):
    """tri (F,3) vertex ids, CCW at the reference time; edges oriented u<v.
    B1 (N x M): -1 tail, +1 head.  B2 (M x F): +-1 induced boundary orientation."""
    F = len(tri)
    loc = np.stack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1)   # (F,3,2)
    lo, hi = loc.min(2), loc.max(2)
    sgn = np.where(loc[..., 0] < loc[..., 1], 1.0, -1.0)
    key_t = lo.astype(np.int64) * N + hi
    if edges is None:
        keys, inv = np.unique(key_t.ravel(), return_inverse=True)
        edges = np.stack([keys // N, keys % N], 1)
        te = inv.reshape(F, 3)
    else:
        edges = np.sort(np.asarray(edges, np.int64), 1)
        keys = edges[:, 0] * N + edges[:, 1]
        order = np.argsort(keys)
        pos = np.minimum(np.searchsorted(keys[order], key_t.ravel()), len(keys) - 1)
        if not np.all(keys[order][pos] == key_t.ravel()):
            raise ValueError("a triangle edge is missing from the edge list")
        te = order[pos].reshape(F, 3)
    M = len(edges)
    ar = np.arange(M)
    B1 = sp.csr_matrix((np.r_[-np.ones(M), np.ones(M)],
                        (np.r_[edges[:, 0], edges[:, 1]], np.r_[ar, ar])), shape=(N, M))
    B2 = sp.csr_matrix((sgn.ravel(), (te.ravel(), np.repeat(np.arange(F), 3))), shape=(M, F))
    if F and abs(B1 @ B2).max() != 0:
        raise RuntimeError("B1 B2 != 0")
    return dict(N=N, edges=edges, tri=tri, tri_e=te, tri_s=sgn, B1=B1, B2=B2)


def complex_from_delaunay(P, max_len_fac=3.0, local_k=None, edge_whitelist=None, keep_fn=None):
    """Planar complex of the positions P (N,2): Delaunay, CCW, long hull slivers removed.
    local_k=None : drop triangles whose longest edge > max_len_fac * global median edge;
    local_k=int  : compare with the local k-th neighbour distance (robust to the strong
                   density contrast behind shocks).
    edge_whitelist (M,2): keep only triangles whose 3 edges are in it (e.g. your mutual-kNN E).
    keep_fn(P, tri) -> bool (F,): extra filter, e.g. segments crossing a wall mask."""
    P = np.asarray(P, np.float64)
    tri = Delaunay(P).simplices.astype(np.int64)
    neg = tri_signed_area(P, tri) < 0
    tri[neg] = tri[neg][:, [0, 2, 1]]
    L = _tri_edge_lengths(P, tri)
    if local_k is None:
        keep = L.max(1) <= max_len_fac * np.median(L)
    else:
        d, _ = cKDTree(P).query(P, k=int(local_k) + 1)
        keep = L.max(1) <= max_len_fac * d[:, -1][tri].max(1)
    if edge_whitelist is not None:
        N = len(P)
        w = np.sort(np.asarray(edge_whitelist, np.int64), 1)
        wk = np.sort(w[:, 0] * N + w[:, 1])
        for j in range(3):
            a, b = tri[:, j], tri[:, (j + 1) % 3]
            k = np.minimum(a, b) * N + np.maximum(a, b)
            p = np.minimum(np.searchsorted(wk, k), len(wk) - 1)
            keep &= wk[p] == k
    if keep_fn is not None:
        keep &= np.asarray(keep_fn(P, tri), bool)
    return _assemble(tri[keep], len(P))


def complex_from_clique(E, tri_edges, P0):
    """Wrap the FRC6 clique complex (E, tri_edges from _frc_triangles: (e_uv, e_uz, e_vz),
    u<v<z) as an oriented 2-complex, triangles CCW at P0."""
    E = np.sort(np.asarray(E, np.int64), 1)
    e = np.asarray(tri_edges, np.int64)
    tri = np.stack([E[e[:, 0], 0], E[e[:, 0], 1], E[e[:, 1], 1]], 1)
    neg = tri_signed_area(np.asarray(P0, np.float64), tri) < 0
    tri[neg] = tri[neg][:, [0, 2, 1]]
    return _assemble(tri, len(P0), edges=E)


def dirac(cx):
    B1, B2 = cx["B1"], cx["B2"]
    return sp.bmat([[None, B1, None], [B1.T, None, B2], [None, B2.T, None]]).tocsr()


def mutual_knn_pairs(P_list, k=8):
    """Union over snapshots of mutual-kNN pairs (u<v) -- same construction as the
    elliptic engine's _frc_build_topology; used as the FIXED material pairs."""
    out = []
    for P in P_list:
        P = np.asarray(P, np.float64)
        n = len(P)
        _, nb = cKDTree(P).query(P, k=k + 1)
        nb = nb[:, 1:]
        A = sp.csr_matrix((np.ones(n * k, np.int8), (np.repeat(np.arange(n), k), nb.ravel())),
                          shape=(n, n))
        Mu = A.multiply(A.T).tocoo()
        s = Mu.row < Mu.col
        out.append(np.stack([Mu.row[s], Mu.col[s]], 1))
    return np.unique(np.concatenate(out), axis=0).astype(np.int64)

# --------------------------------------------------------------------------- metric


def hodge_stars(P, cx, kind="cotan", eps=1e-3):
    """Diagonal Hodge stars from the CURRENT positions.  kind: 'cotan' (DEC; non-negative on
    interior Delaunay edges, clipped below by eps*barycentric) | 'barycentric' | 'unit'."""
    tri, te, e = cx["tri"], cx["tri_e"], cx["edges"]
    N, M = cx["N"], len(e)
    Aa = np.abs(tri_signed_area(P, tri)) + 1e-300
    s0 = np.zeros(N)
    np.add.at(s0, tri.ravel(), np.repeat(Aa / 3.0, 3))
    if kind == "unit":
        return s0, np.ones(M), 1.0 / Aa
    le = np.linalg.norm(P[e[:, 1]] - P[e[:, 0]], axis=1) + 1e-300
    c = P[tri].mean(1)
    mid = 0.5 * (P[e[:, 0]] + P[e[:, 1]])
    dual = np.zeros(M)
    np.add.at(dual, te.ravel(), np.linalg.norm(mid[te] - c[:, None, :], axis=2).ravel())
    s1 = dual / le
    if kind == "cotan":
        cot = np.zeros(M)
        for j in range(3):
            o = tri[:, (j + 2) % 3]
            a, b = P[tri[:, j]] - P[o], P[tri[:, (j + 1) % 3]] - P[o]
            np.add.at(cot, te[:, j], 0.5 * (a * b).sum(1) / (np.abs(_cross(a, b)) + 1e-300))
        s1 = np.maximum(cot, eps * s1)
    return s0, s1, 1.0 / Aa

# --------------------------------------------------------------------------- signals


def velocity_from_positions(traj, dt):
    """Tracer velocity by central differences of positions (as Part-1 §OBJ-ROT)."""
    return np.gradient(np.asarray(traj, np.float64), dt, axis=1)


def relative_velocity(P, V):
    """Aksamit et al. (2024) relative velocity with MEAN references (exactly objective):
    v_d = v - v_bar - omega_rb J (x - x_bar).  (Part-1 uses component-wise medians,
    which are only quasi-objective.)"""
    xb, vb = P.mean(0), V.mean(0)
    r, dv = P - xb, V - vb
    om = (r[:, 0] * dv[:, 1] - r[:, 1] * dv[:, 0]).sum() / (r * r).sum()
    return dv - om * np.stack([-r[:, 1], r[:, 0]], 1)


def edge_cochain(P, V, edges):
    """Trapezoidal line integral along each edge: f_e = 1/2 (v_u + v_v) . (x_v - x_u)."""
    u, v = edges[:, 0], edges[:, 1]
    return 0.5 * ((V[u] + V[v]) * (P[v] - P[u])).sum(1)


def circulation(f, cx):
    """Circulation of every triangle (CCW-positive) = B2^T f  (Stokes, no grad u)."""
    return cx["B2"].T @ f


def node_divergence(f, cx, s0, s1):
    """DEC divergence  div_i = -(B1 *1 f)_i / *0_i  (NaN on isolated tracers)."""
    num = -(cx["B1"] @ (s1 * f))
    with np.errstate(all="ignore"):
        return np.where(s0 > 0, num / s0, np.nan)


def node_vorticity(P, cx, f, tri_ok=None):
    """Vorticity at tracers from triangle circulations, area-weighted (NaN if no valid
    incident triangle).  tri_ok masks out triangles touching a shock-crossing stencil."""
    tri = cx["tri"]
    A = tri_signed_area(P, tri)
    w_t = circulation(f, cx) / np.where(np.abs(A) > 1e-300, A, 1e-300)
    w = np.abs(A) if tri_ok is None else np.abs(A) * tri_ok
    num, den = np.zeros(cx["N"]), np.zeros(cx["N"])
    for j in range(3):
        np.add.at(num, tri[:, j], w_t * w)
        np.add.at(den, tri[:, j], w)
    with np.errstate(all="ignore"):
        return np.where(den > 0, num / den, np.nan)

# --------------------------------------------------------------------------- Hodge


class Hodge:
    """Orthogonal decomposition in <.,.>_{*1}:  f = B1^T phi + *1^{-1} B2 chi + h.
    phi : (B1 *1 B1^T) phi = B1 *1 f      (cotan Laplacian, natural BC)
    chi : (B2^T *1^{-1} B2) chi = B2^T f  (dual Laplacian, chi = 0 outside -> the
          coexact part is tangent to the cloud boundary; the gradient part carries
          all boundary normal flux)."""

    def __init__(self, cx, s1):
        B1, B2 = cx["B1"], cx["B2"]
        L0 = (B1 @ sp.diags(s1) @ B1.T).tocsc()
        self.lu0 = splu((L0 + 1e-12 * L0.diagonal().mean() * sp.eye(L0.shape[0])).tocsc())
        L2 = (B2.T @ sp.diags(1.0 / s1) @ B2).tocsc()
        self.lu2 = splu((L2 + 1e-12 * L2.diagonal().mean() * sp.eye(L2.shape[0])).tocsc())
        self.B1, self.B2, self.s1 = B1, B2, s1

    def __call__(self, f):
        phi = self.lu0.solve(self.B1 @ (self.s1 * f))
        fg = self.B1.T @ phi
        chi = self.lu2.solve(self.B2.T @ f)
        fc = (self.B2 @ chi) / self.s1
        fh = f - fg - fc
        en = lambda g: float((self.s1 * g * g).sum())
        return dict(phi=phi, chi=chi, fg=fg, fc=fc, fh=fh,
                    Eg=en(fg), Ec=en(fc), Eh=en(fh), E=en(f))


def node_vectors(P, cx, g):
    """Sharp map: per-triangle least-squares constant vector from its 3 edge values,
    area-weighted to vertices (NaN on isolated tracers)."""
    tri, te, e = cx["tri"], cx["tri_e"], cx["edges"]
    R = (P[e[:, 1]] - P[e[:, 0]])[te]                                     # (F,3,2)
    G = np.einsum("fki,fkj->fij", R, R)
    G = G + (1e-12 * np.trace(G, axis1=1, axis2=2))[:, None, None] * np.eye(2)
    b = np.einsum("fki,fk->fi", R, g[te])
    ut = np.linalg.solve(G, b[..., None])[..., 0]
    w = np.abs(tri_signed_area(P, tri))
    U, W = np.zeros((cx["N"], 2)), np.zeros(cx["N"])
    for j in range(3):
        np.add.at(U, tri[:, j], ut * w[:, None])
        np.add.at(W, tri[:, j], w)
    with np.errstate(all="ignore"):
        return np.where(W[:, None] > 0, U / W[:, None], np.nan)


def stretch_rate(P, U, pairs):
    """d ln|r|/dt of tracer pairs for a node velocity field U (objective input -> objective)."""
    r = P[pairs[:, 1]] - P[pairs[:, 0]]
    return (r * (U[pairs[:, 1]] - U[pairs[:, 0]])).sum(1) / (r * r).sum(1)


def attribute_stretch(P, cx, dec, pairs, V=None):
    """Split pair stretching (the AFRC_H input) into gradient / coexact / harmonic parts,
    all reconstructed with the SAME sharp operator; resid = what the reconstruction misses."""
    sg = stretch_rate(P, node_vectors(P, cx, dec["fg"]), pairs)
    sc = stretch_rate(P, node_vectors(P, cx, dec["fc"]), pairs)
    sh = stretch_rate(P, node_vectors(P, cx, dec["fh"]), pairs)
    out = dict(s_g=sg, s_c=sc, s_h=sh)
    if V is not None:
        out["s_tot"] = stretch_rate(P, V, pairs)
        out["resid"] = out["s_tot"] - sg - sc - sh
    return out

# --------------------------------------------------------------------------- material bookkeeping


def log_jacobian(traj, cx):
    """ln J on ONE fixed material complex (kept only to show why it fails in shear)."""
    N, T = traj.shape[:2]
    tri = cx["tri"]
    out, ref = np.zeros((N, T)), None
    for t in range(T):
        Aa = np.abs(tri_signed_area(np.asarray(traj[:, t], np.float64), tri))
        s = np.zeros(N)
        np.add.at(s, tri.ravel(), np.repeat(Aa / 3.0, 3))
        if ref is None:
            ref = np.where(s > 0, s, np.nan)
        with np.errstate(all="ignore"):
            out[:, t] = np.log(s / ref)
    return out


def chained_log_jacobian(traj, block=5, max_len_fac=3.0, local_k=None, return_star0=False):
    """ln J_i(t) from MATERIAL star areas on a piecewise-material complex re-anchored every
    `block` snapshots.  Area is an integral quantity: the jump across a shock thinner than
    the tracer spacing is captured exactly; re-anchoring keeps triangles well shaped."""
    N, T = traj.shape[:2]
    out, star0 = np.zeros((N, T)), None
    a = 0
    while a < T - 1:
        b = min(a + block, T - 1)
        tri = complex_from_delaunay(traj[:, a], max_len_fac, local_k=local_k)["tri"]
        ref = None
        for t in range(a, b + 1):
            Aa = np.abs(tri_signed_area(np.asarray(traj[:, t], np.float64), tri))
            s = np.zeros(N)
            np.add.at(s, tri.ravel(), np.repeat(Aa / 3.0, 3))
            if ref is None:
                ref = np.where(s > 0, s, np.nan)
                if star0 is None:
                    star0 = s.copy()
            with np.errstate(all="ignore"):
                inc = np.log(s / ref)
            out[:, t] = np.where(np.isfinite(inc), out[:, a] + inc, out[:, a])
        a = b
    return (out, star0) if return_star0 else out


def vertex_bad_mask(ev_pidx, ev_step, N, T, halo=1):
    """(N,T) True where a tracer's central-difference velocity spans a shock crossing.
    ev_step = crossing increment s (between snapshots s and s+1); flagged snapshots are
    s-halo ... s+1+halo (halo=0 is the minimal stencil of the central difference)."""
    bad = np.zeros((N, T), bool)
    p, s = np.asarray(ev_pidx, np.int64), np.asarray(ev_step, np.int64)
    ok = (p >= 0) & (p < N) & (s >= 0)
    p, s = p[ok], s[ok]
    for d in range(-halo, 2 + halo):
        t = s + d
        m = (t >= 0) & (t < T)
        bad[p[m], t[m]] = True
    return bad


def nodes_of(cx, tri_mask):
    m = np.zeros(cx["N"], bool)
    if np.any(tri_mask):
        m[np.unique(cx["tri"][tri_mask].ravel())] = True
    return m


def censored_increments(G, bad):
    """Increments of a per-object series between consecutive CLEAN samples only.  The jump
    across a censored run is kept (assigned to the run), so sums close."""
    F, T = G.shape
    inc = np.zeros((F, T - 1))
    last = np.full(F, -1)
    for t in range(T):
        ok = ~bad[:, t]
        idx = np.where(ok & (last >= 0))[0]
        inc[idx, t - 1] = G[idx, t] - G[idx, last[idx]]
        last[ok] = t
    return inc


def region_circulation(P, f, cx, mask, max_len_fac=3.0):
    """Circulation of a material region (tracer set) = sum over the current triangles whose 3
    vertices belong to it.  Returns (Gamma, perimeter, area, n_tri).  A collapsing perimeter
    means the region has filamented and the budget is no longer reliable."""
    tri = cx["tri"]
    L = _tri_edge_lengths(P, tri)
    inreg = mask[tri].all(1) & (L.max(1) < max_len_fac * np.median(L))
    G = float(circulation(f, cx)[inreg].sum())
    cnt = np.bincount(cx["tri_e"][inreg].ravel(), minlength=len(cx["edges"]))
    be = cx["edges"][cnt == 1]
    per = float(np.linalg.norm(P[be[:, 1]] - P[be[:, 0]], axis=1).sum())
    area = float(np.abs(tri_signed_area(P, tri[inreg])).sum())
    return G, per, area, int(inreg.sum())

# --------------------------------------------------------------------------- Weitzenboeck


def weitzenbock(cx):
    """L1 = B1^T B1 + B2 B2^T = (D^2)|_{C1};  F(e) = Gershgorin margin of row e.
    Returns L1, F, struct(e) = 4 - deg u - deg v + 3 #tri(e)  (identical to F)."""
    B1, B2, e = cx["B1"], cx["B2"], cx["edges"]
    L1 = (B1.T @ B1 + B2 @ B2.T).tocsr()
    d = L1.diagonal()
    F = d - (np.asarray(abs(L1).sum(1)).ravel() - np.abs(d))
    deg = np.asarray(abs(B1).sum(1)).ravel()
    ntri = np.asarray(abs(B2).sum(1)).ravel()
    return L1, F, 4.0 - deg[e[:, 0]] - deg[e[:, 1]] + 3.0 * ntri


def knn_struct_stats(P, ks=(8, 12, 16, 32, 64, 128), margin=0.1):
    """Bulk value of struct(e) on mutual-kNN clique complexes (interior edges)."""
    P = np.asarray(P, np.float64)
    n = len(P)
    tree = cKDTree(P)
    lo, hi = P.min(0) + margin * np.ptp(P, 0), P.max(0) - margin * np.ptp(P, 0)
    rows = []
    for k in ks:
        _, nb = tree.query(P, k=k + 1)
        nb = nb[:, 1:]
        A = sp.csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), nb.ravel())), shape=(n, n))
        Am = A.multiply(A.T).tocsr()
        Am.data[:] = 1
        deg = np.asarray(Am.sum(1)).ravel()
        T2 = (Am @ Am).multiply(Am).tocoo()
        m = T2.row < T2.col
        Ac = sp.triu(Am, 1).tocoo()
        ka = Ac.row.astype(np.int64) * n + Ac.col
        order = np.argsort(ka)
        cnt = np.zeros(len(ka))
        cnt[order[np.searchsorted(ka[order], T2.row[m].astype(np.int64) * n + T2.col[m])]] = T2.data[m]
        st = 4 - deg[Ac.row] - deg[Ac.col] + 3 * cnt
        inner = ((P[Ac.row] > lo) & (P[Ac.row] < hi)).all(1)
        rows.append((k, float(deg.mean()), float(np.median(st[inner])), float((st[inner] > 0).mean())))
    return rows


# =========================================================================== v2 operators


def complex_peeled(P, fac=2.5, k=8):
    """Delaunay complex peeled ONLY from the outside: a triangle is removed when it touches the current
    outer boundary through an edge longer than fac x the local k-NN spacing; repeat.  Interior triangles are
    never removed, so no internal holes (open stars) appear inside stretched regions."""
    from scipy.spatial import Delaunay
    P = np.asarray(P, np.float64)
    tri = Delaunay(P).simplices.astype(np.int64)
    neg = tri_signed_area(P, tri) < 0
    tri[neg] = tri[neg][:, [0, 2, 1]]
    d, _ = cKDTree(P).query(P, k=k + 1); loc = d[:, -1]
    Ne = len(P)
    ek = np.stack([np.minimum(tri[:, j], tri[:, (j + 1) % 3]) * Ne + np.maximum(tri[:, j], tri[:, (j + 1) % 3]) for j in range(3)], 1)
    elen = np.stack([np.linalg.norm(P[tri[:, j]] - P[tri[:, (j + 1) % 3]], axis=1) for j in range(3)], 1)
    thr = fac * np.stack([np.maximum(loc[tri[:, j]], loc[tri[:, (j + 1) % 3]]) for j in range(3)], 1)
    longe = elen > thr
    cand = longe.any(1)                       # only triangles with a long edge can ever be peeled
    alive = np.ones(len(tri), bool)
    # edge -> count of alive triangles
    keys, inv = np.unique(ek.ravel(), return_inverse=True); inv = inv.reshape(-1, 3)
    cnt = np.bincount(inv.ravel(), minlength=len(keys))
    for _ in range(500):
        onb = cnt[inv] == 1
        rm = alive & cand & (onb & longe).any(1)
        if not rm.any():
            break
        alive &= ~rm
        np.subtract.at(cnt, inv[rm].ravel(), 1)
    return _assemble(tri[alive], len(P))


def open_star_nodes(cx):
    cnt = np.bincount(cx["tri_e"].ravel(), minlength=len(cx["edges"]))
    m = np.zeros(cx["N"], bool); m[cx["edges"][cnt == 1].ravel()] = True
    return m


def _grad_lambda(P, tri):
    """per-face gradients of the three barycentric coordinates (F,3,2) and areas (F,)."""
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    A = 0.5 * _cross(b - a, c - a)
    rot = lambda d: np.stack([-d[:, 1], d[:, 0]], 1)
    G = np.stack([rot(c - b), rot(a - c), rot(b - a)], 1) / (2 * A)[:, None, None]
    return G, A


def galerkin_div(P, cx, V):
    """Weak P1 divergence with exact quadrature: div_i = -3 sum_t A_t u_t . grad(lambda_i) / A*_i.
    Exact for linear velocity on ANY triangulation (the diagonal cotan star is not)."""
    tri = cx["tri"]; G, A = _grad_lambda(P, tri); ub = V[tri].mean(1)
    b = np.zeros(len(P)); st = np.zeros(len(P))
    for k in range(3):
        np.add.at(b, tri[:, k], A * (ub * G[:, k]).sum(1)); np.add.at(st, tri[:, k], A)
    with np.errstate(all="ignore"):
        return np.where(st > 0, -3.0 * b / st, np.nan)


def galerkin_hodge(P, cx, V, rho_t=None):
    """Polthier-Preuss decomposition of the face-constant field u_t (mean of vertex velocities):
    u = grad(phi) [P1 conforming, natural BC] + rot(psi) [Crouzeix-Raviart, psi=0 on boundary edges] + h.
    L2(A_t [rho_t]) orthogonal.  Returns face fields and energy fractions."""
    from scipy.sparse.linalg import spsolve
    tri, te = cx["tri"], cx["tri_e"]; N, M = cx["N"], len(cx["edges"])
    G, A = _grad_lambda(P, tri); u = V[tri].mean(1)
    w = A if rho_t is None else A * rho_t
    F = len(tri)
    # P1 part
    r, c, v = [], [], []
    for i in range(3):
        for j in range(3):
            r.append(tri[:, i]); c.append(tri[:, j]); v.append(w * (G[:, i] * G[:, j]).sum(1))
    K = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(N, N))
    b = np.zeros(N)
    for i in range(3):
        np.add.at(b, tri[:, i], w * (u * G[:, i]).sum(1))
    used = np.zeros(N, bool); used[tri.ravel()] = True
    K = K + sp.diags(np.where(used, 1e-12 * K.diagonal().mean(), 1.0))
    phi = spsolve(K.tocsc(), b - (b[used].mean() if used.any() else 0) * used)
    ug = np.einsum("fk,fkd->fd", phi[tri], G)
    # CR part: basis chi_e = 1 - 2 lambda_k (k opposite e); local edge j joins vertices j, j+1 -> opposite (j+2)%3
    rotm = lambda d: np.stack([d[..., 1], -d[..., 0]], -1)
    Gc = np.stack([-2 * G[:, (j + 2) % 3] for j in range(3)], 1)          # grad chi for local edge j
    Rc = rotm(Gc)
    cnt = np.bincount(te.ravel(), minlength=M); interior = cnt == 2
    r, c, v = [], [], []
    for i in range(3):
        for j in range(3):
            r.append(te[:, i]); c.append(te[:, j]); v.append(w * (Gc[:, i] * Gc[:, j]).sum(1))
    Kc = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(M, M)).tocsr()
    bc = np.zeros(M)
    for i in range(3):
        np.add.at(bc, te[:, i], w * (u * Rc[:, i]).sum(1))
    idx = np.where(interior)[0]
    psi = np.zeros(M)
    psi[idx] = spsolve(Kc[idx][:, idx].tocsc(), bc[idx])
    uc = np.einsum("fk,fkd->fd", psi[te], Rc)
    uh = u - ug - uc
    en = lambda x: float((w * (x * x).sum(1)).sum())
    E = en(u)
    return dict(u=u, ug=ug, uc=uc, uh=uh, Eg=en(ug) / E, Ec=en(uc) / E, Eh=en(uh) / E, A=A, orth=float((w * (ug * uc).sum(1)).sum()) / E)


def face_to_node(P, cx, Fv):
    tri = cx["tri"]; A = np.abs(tri_signed_area(P, tri))
    U = np.zeros((cx["N"], Fv.shape[1])); W = np.zeros(cx["N"])
    for k in range(3):
        np.add.at(U, tri[:, k], Fv * A[:, None]); np.add.at(W, tri[:, k], A)
    with np.errstate(all="ignore"):
        return np.where(W[:, None] > 0, U / W[:, None], np.nan)


def _star(P, tri, N):
    A = np.abs(tri_signed_area(P, tri)); s = np.zeros(N)
    np.add.at(s, tri.ravel(), np.repeat(A, 3)); return s


def chained_lnJ_v2(frames, bad=None, block=5, max_defer=4, cplx_fn=None):
    """Material ln J with PER-TRACER anchors.  A tracer is re-anchored to the current Delaunay star every
    `block` frames, but the re-anchor is deferred while any vertex of its star lies in a shock stencil
    (bad (N,S)), up to max_defer*block frames.  The anchor star is then pre-shock at one end and post-shock at
    the other, so the area ratio telescopes across the crossing (integral form, exact for thin shocks)."""
    cplx_fn = complex_peeled if cplx_fn is None else cplx_fn
    S, N, _ = frames.shape
    tris = {0: cplx_fn(frames[0])["tri"]}
    anchor = np.zeros(N, int); A0 = _star(frames[0], tris[0], N); base = np.zeros(N)
    out = np.zeros((S, N))
    for t in range(1, S):
        cur = np.full(N, np.nan)
        for a in np.unique(anchor):
            idx = anchor == a
            cur[idx] = _star(frames[t], tris[a], N)[idx]
        with np.errstate(all="ignore"):
            inc = np.log(cur / A0)
        out[t] = np.where(np.isfinite(inc), base + inc, out[t - 1])
        due = (t - anchor) >= block
        if due.any() and t < S - 1:
            trit = cplx_fn(frames[t])["tri"]
            if bad is not None:
                nb = bad[:, max(t - 1, 0):t + 2].any(1)
                sb = np.zeros(N, bool)
                for k in range(3):
                    np.logical_or.at(sb, trit[:, k], nb[trit].any(1))
                ok = due & (~sb | ((t - anchor) >= max_defer * block))
            else:
                ok = due
            if ok.any():
                tris[t] = trit
                At = _star(frames[t], trit, N)
                good = ok & (At > 0)
                A0[good] = At[good]; base[good] = out[t][good]; anchor[good] = t
        for a in list(tris):
            if a not in set(np.unique(anchor)):
                del tris[a]
    return out
