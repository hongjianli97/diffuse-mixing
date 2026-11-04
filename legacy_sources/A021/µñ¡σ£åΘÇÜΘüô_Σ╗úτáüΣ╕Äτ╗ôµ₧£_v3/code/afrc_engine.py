"""AFRC (FRC6) elliptic engine, extracted VERBATIM from part2_tc0_90.ipynb cells 37, 39, 41
(user's v27 'true zint / fused-core' engine).  Only the parameter block is reproduced here
(values copied from that notebook's MASTER CONFIG).  Nothing in the engine bodies is modified."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags, eye as sparse_eye
from scipy.sparse.linalg import eigsh
# ---- MASTER CONFIG values (part2_tc0_90.ipynb) ----
FRC6_BASELINE = 'median'; _CAP_W = None
CAND_KNN_K = 16; CAND_SUBSAMPLE_TOPO = 1600000; CAND_LIFT_K = 8
FRC5_BETA_D = 1.0; FRC5_BND_L1 = 1.0; FRC5_BND_L2 = 1.0; FRC5_CHUNK_E = 200000; FRC5_CLIP_MADS = 3.0
FRC5_DEL_ALT = 0.35; FRC5_DEL_SFAC = 4.0; FRC5_EPS_TH = 0.0001; FRC5_KAPPA = 1.0
FRC5_THQ = 0.7853981633974483; FRC5_THQ_TOT = 1.5707963267948966; FRC5_W0 = 0.05; FRC5_WIN_L = 7
FRC6_BND_Z0 = 1.0; FRC6_HAMPEL_K = 4.5; FRC6_KSCALES = (8, 12); FRC6_MIN_KEEP = 4; FRC6_PHI_Z0 = 1.7
FRC6_SFLOOR_FAC = 0.05; FRC6_SKIP_EIG = True; FRC6_TH_MIN = 0.0001; FRC6_TSIG_Z0 = 1.0; FRC6_TSIG_Z1 = 1.5
FRC6_WIN_L = 5; FRC6_ZINT_MINV = 3; FRC6_ZWINS = 8.0; FRC6_ZX_W = 0.7; FRC6_ZX_Z0 = 2.0
FRC_BETA_FAC = 1.0; FRC_DELTA_C = 1.0; FRC_DWELL_D0 = 0.5; FRC_DWELL_S = 0.15; FRC_EIG_MAXITER = 3000
FRC_EIG_SIGMA = -0.001; FRC_EIG_TOL = 0.001; FRC_GRAPH_K = 32; FRC_GSCALE = 1.0; FRC_KSCALES = (32,)
FRC_NEARZERO = 0.001; FRC_NEVEC = 8; FRC_NODECAP = 160000; FRC_NSNAP = 5; FRC_PERSIST_P = 3
FRC_PIPELINE_A = True; FRC_SIGMA_FAC = 1.0; FRC_TAU_Q = 0.75; FRC_WFLOOR = 0.001; FRC_STRIP_DISC = True

# ======================= part2_tc0_90.ipynb cell 37 (verbatim) =======================
# ============================================================================
#  §B-FRC  Augmented Forman-Ricci curvature-flip elliptic detector  (REV)
#  ---------------------------------------------------------------------------
#  Strictly follows the revised report's dependency order:
#    §1 topology (ONCE): time-averaged mutual-kNN UNION over FRC_NSNAP snapshots
#       -> fixed weighted undirected G=(V,E); triangles enumerated once.
#    §2 per-snapshot OBJECTIVE instantaneous weights on the fixed topology:
#         w_e(t)=exp(-||dx||^2/sigma^2)*[tanh(b dw_i)tanh(b dw_j)]_+ ;
#         w_v(t)=g(|dw_v|)*mean_{e in v}[tanh tanh]_+ ,  dw=omega-omega_bar.
#    §3 augmented Forman-Ricci curvature F# = (plain weighted Sreejith Forman)
#         + 3*sum_{f>e} w_e/w_f   [report Def 3.3 form; reduces to
#         4-deg(u)-deg(v)+3*#tri unweighted, App C].
#    §4 project edge curvature -> vertex (degree-normalised MEAN) BEFORE counting
#         -> vertex-curvature time series F#_i(t)   (pipeline B).
#    §5 stabilized flip functional N_{delta,p}: hysteresis dead-band + persistence
#         -> nu_i (flips), P_i = <1{F#_i>delta}>_t (continuous), Phi (net flow).
#    §6 Pi = P_i*(1+nu_i)^-1 ;  pipeline A: per-edge nu_e -> boundary node field.
#    §9 Def-2.1 three-factor STATIC weight -> near-zero sheaf spectrum -> #vortices.
#  All curvature inputs are objective scalars (pairwise distance, omega-omega_bar)
#  so F# and its flips are frame-invariant (report Sec 7).  The curvature path
#  never uses grad(u)'s spin part; disc<0 is never reintroduced (FRC_STRIP_DISC).
#  D2 (by request) NOT done: omega=c-b comes from the windowed LS Jacobian.
# ============================================================================

def _frc_mad(x):
    x = np.asarray(x, float); med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)

def _frc_subsample_nodes(M, cap, seed=17):
    if M <= cap: return np.arange(M)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(M, size=cap, replace=False))

def _frc_dwell_gate(x, x0, s):
    return 1.0 / (1.0 + np.exp(-(x - x0) / max(s, 1e-9)))

def _frc_mutual_knn_edges(P, k):
    """Mutual-kNN undirected edge array (|E|,2) with u<v, on points P (n,2)."""
    n = len(P); kk = min(k, n - 1)
    tree = cKDTree(np.ascontiguousarray(P))
    _, idx = tree.query(np.ascontiguousarray(P), k=kk + 1, workers=-1)
    nbr = idx[:, 1:]
    rows = np.repeat(np.arange(n), kk); cols = nbr.reshape(-1)
    A = coo_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1
    Mu = A.minimum(A.T).tocoo()            # mutual: present in BOTH directions (symmetric)
    sel = Mu.row < Mu.col
    return np.stack([Mu.row[sel], Mu.col[sel]], axis=1).astype(np.int64)

def _seg_crosses_wall(A, B, wall, ext, S):
    """Vectorised: does the straight segment A->B (END config) cross any barrier-wall pixel? S samples/edge."""
    x0, x1, y0, y1 = ext; H, W = wall.shape
    t = np.linspace(0.0, 1.0, S)[None, :]
    X = A[:, 0][:, None]*(1-t) + B[:, 0][:, None]*t
    Y = A[:, 1][:, None]*(1-t) + B[:, 1][:, None]*t
    ix = np.clip(((X-x0)/(x1-x0+1e-30)*W).astype(np.int64), 0, W-1)
    iy = np.clip(((Y-y0)/(y1-y0+1e-30)*H).astype(np.int64), 0, H-1)
    return wall[iy, ix].any(axis=1)

def _knn_edges_barrier(P, Pend, k, wall, ext, fan, S):
    """Within-SIDE mutual-kNN: for each node take its k nearest neighbours whose END-config segment does
       NOT cross the barrier (re-route across-barrier candidates to the next same-side neighbour). Keeps
       degree ~k (no collapse -> no Forman artifact) AND works for OPEN barriers (line test, not flood-fill)."""
    n = len(P); kk = min(k, n-1)
    if n < 4 or kk < 1: return np.zeros((0, 2), np.int64)
    Kc = min(n-1, max(kk*fan, kk+1))
    tree = cKDTree(P); _, idx = tree.query(P, k=Kc+1, workers=-1); nbr = idx[:, 1:]
    src = np.repeat(np.arange(n), Kc); dst = nbr.reshape(-1)
    cross = _seg_crosses_wall(Pend[src], Pend[dst], wall, ext, S).reshape(n, Kc)
    keep = ~cross; cc = np.cumsum(keep, axis=1); sel = (keep & (cc <= kk)).reshape(-1)
    rows = src[sel]; cols = dst[sel]
    A = coo_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(n, n)).tocsr(); A.data[:] = 1
    Mu = A.minimum(A.T).tocoo(); s = Mu.row < Mu.col
    E = np.stack([Mu.row[s], Mu.col[s]], axis=1).astype(np.int64)
    deg = np.bincount(E.reshape(-1), minlength=n) if E.size else np.zeros(n, np.int64)
    iso = np.where(deg == 0)[0]
    if iso.size:                                   # reconnect isolated nodes -> no spurious +4 curvature
        ex = []
        for u in iso:
            ok = nbr[u][~cross[u]]
            v = int(ok[0]) if ok.size else int(nbr[u][0])   # prefer same-side; else nearest (avoid isolated +4)
            ex.append((min(int(u), v), max(int(u), v)))
        if ex: E = np.unique(np.concatenate([E, np.array(ex, np.int64)], axis=0), axis=0)
    return E

def _frc_build_topology(P_snaps, k):
    """Union of per-snapshot mutual-kNN edge sets (material-continuity tolerant)."""
    allE = [_frc_mutual_knn_edges(P, k) for P in P_snaps]
    return np.unique(np.concatenate(allE, axis=0), axis=0)

def _frc_triangles(E, n):
    """Enumerate triangles ONCE on the fixed topology -> (n_tri, tri_edge_ids (|T|,3))."""
    nbrs = [[] for _ in range(n)]
    for u, v in E:
        nbrs[int(u)].append(int(v)); nbrs[int(v)].append(int(u))
    nbrs = [np.array(sorted(s), np.int64) for s in nbrs]
    eid = {(int(u), int(v)): i for i, (u, v) in enumerate(E)}
    tri_edges = []
    for (u, v) in E:
        u = int(u); v = int(v)
        common = np.intersect1d(nbrs[u], nbrs[v], assume_unique=True)
        common = common[common > v]        # u<v<z -> each triangle counted once
        for z in common:
            z = int(z)
            tri_edges.append((eid[(u, v)], eid[(u, z)], eid[(v, z)]))
    tri_edges = np.asarray(tri_edges, np.int64) if tri_edges else np.zeros((0, 3), np.int64)
    return tri_edges.shape[0], tri_edges

def _frc_augmented_curvature(E, tri_edges, we, wv, eps=1e-12):
    """Augmented Forman-Ricci curvature per edge (report Def 3.3 'plain + triangle
       bonus' form):  F#(e) = F_plain_weighted(e) + 3*sum_{f>e} w_e/w_f .
       Implemented vectorised as
         F#(e) = 2(w_u+w_v) - sqrt(w_e)*(w_u*S[u]+w_v*S[v]) + 3*sum_{t in e} w_e/w_f,
       with S[x]=sum_{e' in x} 1/sqrt(w_{e'}); the +w_u,+w_v from including e in S
       are cancelled by 2(w_u+w_v) so `base` equals the weighted plain Sreejith
       Forman exactly.  w_f = geomean of the triangle's edge weights.  Reduces
       EXACTLY to 4-deg(u)-deg(v)+3*#tri when unweighted (report App C)."""
    u = E[:, 0]; v = E[:, 1]
    we = np.maximum(we, eps); wv = np.maximum(wv, eps)
    inv = 1.0 / np.sqrt(we); nN = wv.shape[0]
    S = np.zeros(nN); np.add.at(S, u, inv); np.add.at(S, v, inv)
    wu_ = wv[u]; wv_ = wv[v]
    base = 2.0 * (wu_ + wv_) - np.sqrt(we) * (wu_ * S[u] + wv_ * S[v])   # = plain weighted Forman
    B = np.zeros(E.shape[0])
    if len(tri_edges):
        w3 = we[tri_edges]
        wf = np.cbrt(np.prod(w3, axis=1) + eps)
        contrib = we[tri_edges] / wf[:, None]
        np.add.at(B, tri_edges.reshape(-1), contrib.reshape(-1))
    return base + 3.0 * B

def _frc_sliding_median_sign(h, p):
    """Majority sign of the +/-1 latch over a centred odd window p (= median); ties
       (window sum 0) are forward-filled.  Implements the persistence filter."""
    half = p // 2
    Hp = np.pad(h.astype(np.float32), ((0, 0), (half, half)), mode='edge')
    cs = np.cumsum(np.concatenate([np.zeros((h.shape[0], 1)), Hp], axis=1), axis=1)
    wsum = cs[:, p:] - cs[:, :-p]
    c = np.sign(wsum).astype(np.int8)
    for t in range(1, c.shape[1]):
        z = c[:, t] == 0; c[z, t] = c[z, t - 1]
    return c

def _frc_Ndp(series, delta, p):
    """Stabilized flip functional N_{delta,p} on ANY (n,T) scalar series (report
       Def 5.3).  Schmitt hysteresis dead-band [-delta,+delta] + persistence p.
       Returns:
         nu       (n,)  confirmed sign-flip count,
         posdwell (n,)  <1{series>delta}>_t  (continuous positive-curvature dwell),
         Phi      (n,)  net signed flow = 1/2 (c_last - c_first) in {-1,0,+1}."""
    n, T = series.shape
    h = np.zeros((n, T), np.int8); st = np.zeros(n, np.int8)
    for t in range(T):
        up = series[:, t] >  delta; dn = series[:, t] < -delta
        st = np.where(up, 1, np.where(dn, -1, st)).astype(np.int8)
        h[:, t] = st
    c = _frc_sliding_median_sign(h, p) if p > 1 else h
    fl = (c[:, 1:] != c[:, :-1]) & (c[:, 1:] != 0) & (c[:, :-1] != 0)
    nu = fl.sum(axis=1).astype(np.float32)
    posdwell = (series > delta).mean(axis=1).astype(np.float32)
    Phi = (0.5 * (c[:, -1].astype(np.float32) - c[:, 0].astype(np.float32))).astype(np.float32)
    return nu, posdwell, Phi

def _frc_median_edge_len(Pg, E, snaps):
    u = E[:, 0]; v = E[:, 1]; acc = np.zeros(len(E))
    for s in snaps:
        dx = Pg[u, s, 0] - Pg[v, s, 0]; dy = Pg[u, s, 1] - Pg[v, s, 1]
        acc += np.sqrt(dx * dx + dy * dy)
    return float(np.median(acc / max(len(snaps), 1)) + 1e-12)

def _frc_vortex_count(E, m, w_static, nevec, nearzero):
    """Near-zero spectrum of the normalised sheaf(=weighted graph) Laplacian on the
       Def-2.1 three-factor TIME-AVERAGED graph.  #(lambda<nearzero) ~ # decoupled
       co-rotating communities = #elliptic vortices (report Sec 9.1)."""
    rows = np.concatenate([E[:, 0], E[:, 1]]); cols = np.concatenate([E[:, 1], E[:, 0]])
    W = coo_matrix((np.concatenate([w_static, w_static]), (rows, cols)), shape=(m, m)).tocsr()
    deg = np.asarray(W.sum(1)).ravel(); di = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    Lsym = sparse_eye(m) - diags(di) @ W @ diags(di)
    kq = max(1, min(nevec, m - 2))
    vals = None
    try:                                       # SHIFT-INVERT: fast for the smallest eigenvalues on a
        vals = eigsh(Lsym, k=kq, sigma=FRC_EIG_SIGMA, which='LM',      # spatially-local mutual-kNN graph
                     tol=FRC_EIG_TOL, return_eigenvectors=False)       # (sigma<0 avoids a singular L - sigma*I)
    except Exception:
        try:                                   # fallback: smallest-algebraic Lanczos with a loose tol
            vals = eigsh(Lsym, k=kq, which='SA', tol=FRC_EIG_TOL,
                         maxiter=FRC_EIG_MAXITER, return_eigenvectors=False)
        except Exception:
            vals = np.array([0.0])
    vals = np.sort(vals)
    return int((vals < nearzero).sum()), vals

def _frc_extract_boundary(F, grid, level_q=0.6, cd_max=0.20):
    """Best-effort outermost convexity-deficiency-compliant CLOSED contour of a
       gridded field F (report Sec 9.2).  Scans level quantiles high->low and
       returns the OUTERMOST set of near-closed, low-deficiency contours (a list
       of (x,y) polylines), or [] on failure.  Defensive rendering-stage helper;
       not validated end-to-end here."""
    try:
        from scipy.spatial import ConvexHull
        xg = np.asarray(grid['xg']); yg = np.asarray(grid['yg']); Ff = np.asarray(F, float)
        finite = Ff[np.isfinite(Ff) & (Ff > 1e-6)]
        if finite.size < 16: return []
        span = float(xg.max() - xg.min())
        for q in (0.8, 0.7, float(level_q), 0.5, 0.4):       # high -> low: outermost closed first
            lvl = float(np.quantile(finite, q))
            fig = plt.figure(); cs = plt.contour(xg, yg, Ff, levels=[lvl]); plt.close(fig)
            segs = cs.allsegs[0] if (hasattr(cs, "allsegs") and cs.allsegs) else []
            out = []
            for seg in segs:
                if len(seg) < 8: continue
                if np.hypot(*(seg[0] - seg[-1])) > 0.05 * span:    # require near-closed loop
                    continue
                x = seg[:, 0]; y = seg[:, 1]
                Apoly = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                Ahull = float(ConvexHull(seg).volume)              # 2D hull area
                cd = (Ahull - Apoly) / max(Ahull, 1e-12)           # convexity deficiency
                if cd <= cd_max: out.append(seg)
            if out:
                out.sort(key=lambda s: -len(s)); return out        # outermost ~ longest
        return []
    except Exception:
        return []


def _frc_graph_score(Pg, dev, snaps, kg):
    """One scale.  Builds topology+triangles ONCE, evaluates per-snapshot augmented
       Forman curvature on the fixed topology, projects to vertices (pipeline B),
       runs the stabilized flip functional, and -- separately -- the Def-2.1 static
       three-factor weight for the near-zero vortex count."""
    m = Pg.shape[0]; T1 = Pg.shape[1]
    E = _frc_build_topology([np.ascontiguousarray(Pg[:, s, :]) for s in snaps], kg)
    n_tri, tri_edges = _frc_triangles(E, m)
    sigma = FRC_SIGMA_FAC * _frc_median_edge_len(Pg, E, snaps)
    mad = _frc_mad(dev.reshape(-1)); beta = 1.0 / max(FRC_BETA_FAC * mad, 1e-9)
    tau = float(np.quantile(np.abs(dev), FRC_TAU_Q))
    u = E[:, 0]; v = E[:, 1]; nE = len(E)

    Fv = np.empty((m, T1), np.float32)                               # vertex curvature series (pipeline B)
    Fe = np.empty((nE, T1), np.float32) if FRC_PIPELINE_A else None   # edge series (pipeline A) [O(nE*T1)]
    deg_node = (np.bincount(u, minlength=m) + np.bincount(v, minlength=m)).astype(np.float64)  # STATIC degree
    deg_safe = np.maximum(deg_node, 1.0)                             # hoisted: degree is fixed, not per-t
    # Def-2.1 three-factor STATIC weight accumulators (report Sec 3 / D3)
    sdist2 = np.zeros(nE); ssign = np.zeros(nE); dwell = np.zeros(m)

    for t in range(T1):
        d = dev[:, t]
        dx = Pg[u, t, 0] - Pg[v, t, 0]; dy = Pg[u, t, 1] - Pg[v, t, 1]
        d2 = dx * dx + dy * dy
        prox = np.exp(-d2 / (sigma * sigma))
        cco = np.maximum(np.tanh(beta * d[u]) * np.tanh(beta * d[v]), 0.0)   # objective co-rotation
        we = prox * cco
        pos = we[we > 0]; we = np.maximum(we, FRC_WFLOOR * (np.median(pos) if pos.size else 1.0))
        g = _frc_dwell_gate(np.abs(d), tau, FRC_GSCALE * mad)               # instantaneous dwell gate
        cn = np.bincount(u, cco, minlength=m) + np.bincount(v, cco, minlength=m)   # sum incident co-rotation
        wv = g * (cn / deg_safe)                                           # degree hoisted out of the loop
        posv = wv[wv > 0]; wv = np.maximum(wv, FRC_WFLOOR * (np.median(posv) if posv.size else 1.0))
        fe_t = _frc_augmented_curvature(E, tri_edges, we, wv)             # edge curvature at t
        if Fe is not None: Fe[:, t] = fe_t.astype(np.float32)             # only store if pipeline A is on
        # pipeline B: project edge curvature -> vertex (degree-normalised mean) BEFORE counting
        fv = np.bincount(u, fe_t, minlength=m) + np.bincount(v, fe_t, minlength=m)
        Fv[:, t] = (fv / deg_safe).astype(np.float32)
        # static three-factor weight accumulation (report Def 2.1)
        sdist2 += d2
        ssign  += np.sign(d[u]) * np.sign(d[v])
        dwell  += (np.abs(d) > tau).astype(np.float64)

    # --- pipeline B: vertex flips -> Pi (composite) + P_i (continuous) + Phi (net) ---
    delta_v = FRC_DELTA_C * _frc_mad(Fv.reshape(-1))
    nu, posdwell, Phi = _frc_Ndp(Fv, delta_v, FRC_PERSIST_P)
    Pi = posdwell / (1.0 + nu)
    Pi = np.clip(Pi / (np.quantile(Pi, 0.99) + 1e-12), 0.0, 1.0)
    # --- pipeline A: per-edge flips nu_e -> boundary node field (report Sec 6-A, 9.2) ---
    if FRC_PIPELINE_A:
        delta_e = FRC_DELTA_C * _frc_mad(Fe.reshape(-1))
        nu_e, _, _ = _frc_Ndp(Fe, delta_e, FRC_PERSIST_P)
        bnd = np.zeros(m)
        np.maximum.at(bnd, u, nu_e); np.maximum.at(bnd, v, nu_e)   # node near boundary if an incident edge churns
        bnd = np.clip(bnd / (np.quantile(bnd, 0.99) + 1e-12), 0.0, 1.0).astype(np.float32)
    else:
        nu_e = np.zeros(nE, np.float32); bnd = np.zeros(m, np.float32)   # pipeline A skipped
    # --- Def-2.1 three-factor STATIC weight -> near-zero sheaf spectrum -> #vortices (D3) ---
    we_static = np.exp(-(sdist2 / T1) / (sigma * sigma)) * np.maximum(ssign / T1, 0.0)
    d_i = dwell / T1
    gd = _frc_dwell_gate(d_i, FRC_DWELL_D0, FRC_DWELL_S)        # STATIC dwell gate on the time-fraction
    we_static = np.maximum(we_static * gd[u] * gd[v], 0.0)
    rvort, vals = _frc_vortex_count(E, m, we_static, FRC_NEVEC, FRC_NEARZERO)

    diag = dict(n_edges=nE, n_tri=n_tri, m_nodes=m, delta=float(delta_v),
                vortex_count=rvort, eig=vals[:FRC_NEVEC], k_graph=kg,
                nu_mean=float(nu.mean()), posdwell_mean=float(posdwell.mean()),
                phi_net_mean=float(np.abs(Phi).mean()), nu_e_mean=float(nu_e.mean()))
    return (Pi.astype(np.float32), posdwell.astype(np.float32),
            Phi.astype(np.float32), bnd, nu.astype(np.float32), diag)

def compute_candidate_frc(STATE, layer_name='fwd', k=CAND_KNN_K):
    """Augmented Forman-Ricci elliptic-coherence detector (report-strict).
       Returns per-particle fields:
         phi          = Pi   (composite elliptic-core coherence, P_i*(1+nu)^-1),
         phi_posdwell = P_i  (continuous positive-curvature dwell -- the report's
                              RECOMMENDED primary read-out, Prop 8.4),
         phi_boundary = pipeline-A edge-flip nu_e aggregated to nodes (vortex
                        boundary / spiral filaments, report Sec 6-A, 9.2),
       plus diagnostics (vortex count, edges, triangles, delta, <|Phi|>, <nu_e>)."""
    S = _jacobian_series_fh(STATE, layer_name, k)        # memoized; shared with BEFORE/E2
    traj_sub = S['traj_sub']; T1 = S['T1']
    omega = np.asarray(S['omega'], np.float64)           # windowed vorticity c-b  (D2: fit-derived, by request)
    M = omega.shape[0]
    gsel = _frc_subsample_nodes(M, FRC_NODECAP, seed=17)
    Pg = np.ascontiguousarray(traj_sub[gsel])            # (m,T1,2)
    # --- D4: robust spatial-MEDIAN vorticity reference (replaces the per-slice mean) ---
    ombar_med = np.median(omega, axis=0)                 # spatial median over all nodes (shock-pulse robust)
    dev = (omega[gsel] - ombar_med[None, :])             # objective IVD-signed deviation (m,T1)
    m = len(gsel)
    snaps = np.unique(np.linspace(0, T1 - 1, min(FRC_NSNAP, T1)).round().astype(int))
    scales = FRC_KSCALES if len(FRC_KSCALES) > 0 else (FRC_GRAPH_K,)

    Pi_acc = np.zeros(m); Pd_acc = np.zeros(m); Phi_acc = np.zeros(m)
    Bnd_acc = np.zeros(m); Nu_acc = np.zeros(m)
    last = None
    for kg in scales:
        Pi_s, Pd_s, Phi_s, Bnd_s, Nu_s, diag = _frc_graph_score(Pg, dev, snaps, kg)
        Pi_acc += Pi_s; Pd_acc += Pd_s; Phi_acc += Phi_s; Bnd_acc += Bnd_s; Nu_acc += Nu_s
        last = diag
    ns = len(scales)
    Pi  = (Pi_acc  / ns).astype(np.float32)
    Pd  = (Pd_acc  / ns).astype(np.float32)              # continuous P_i (primary)
    Phi = (Phi_acc / ns).astype(np.float32)
    Bnd = (Bnd_acc / ns).astype(np.float32)
    Nu  = (Nu_acc  / ns).astype(np.float32)              # stabilized flip count nu_i = N_{delta,p}[F#_i]

    traj_full = STATE['layers'][layer_name]['traj']
    src = traj_sub[gsel, 0, :]; dst = traj_full[:, 0, :]
    def _lift(val):
        x = _smooth_lift_idw(src, dst, val, k_lift=CAND_LIFT_K)
        return np.where(np.isfinite(x), x, 0.0).astype(np.float32)

    out = dict(phi=_lift(Pi), phi_posdwell=_lift(Pd), phi_boundary=_lift(Bnd),
               phi_nu=_lift(Nu), layer=layer_name)
    out.update(last)
    out['n_scales'] = ns
    return out

def compute_candidate_FRC(STATE, layer_name='fwd'):
    """Alias matching the compute_candidate_* naming used by the other detectors."""
    return compute_candidate_frc(STATE, layer_name)

print("[cand-FRC] augmented Forman-Ricci detector ready (report-strict: Def-2.1 "
      "static spectral weight, median omega_bar, Phi + edge nu_e + continuous P_i)")


# ======================= part2_tc0_90.ipynb cell 39 (verbatim) =======================
# ============================================================================
#  §B-FRC5 engine — design notes in the header of §B-FRC5-params;
#  reuses §B-FRC topology/triangle builders and §9 lift; positions only.
# ============================================================================
# ----------------------------- basis ---------------------------------------
def _frc5_edge_turning(Pg, E, chunk=FRC5_CHUNK_E):
    """Per-edge separation-vector geometry, positions only.
       Returns dth (nE,T1-1) raw signed turning increments [rad] and
               rln (nE,T1)   separation lengths.  Chunked over edges."""
    nE = len(E); T1 = Pg.shape[1]
    dth = np.empty((nE, T1 - 1), np.float32)
    rln = np.empty((nE, T1), np.float32)
    P32 = np.ascontiguousarray(Pg, dtype=np.float32)
    for a in range(0, nE, chunk):
        b = min(a + chunk, nE)
        R = P32[E[a:b, 1]] - P32[E[a:b, 0]]                  # (c,T1,2)
        L = np.sqrt(np.maximum((R * R).sum(2), 0.0))
        cross = R[:, :-1, 0] * R[:, 1:, 1] - R[:, :-1, 1] * R[:, 1:, 0]
        dot   = (R[:, :-1, :] * R[:, 1:, :]).sum(2)
        d = np.arctan2(cross, dot)
        d[(L[:, :-1] * L[:, 1:]) < 1e-18] = 0.0              # coincident guard
        dth[a:b] = d; rln[a:b] = L
    return dth, rln

def _frc5_grouped_quantile(idx, vals, n, q=0.5):
    """median of vals within groups idx (0..n-1), vectorised (lexsort once)."""
    o = np.lexsort((vals, idx)); iv = idx[o]; vv = vals[o]
    starts = np.searchsorted(iv, np.arange(n), side='left')
    ends   = np.searchsorted(iv, np.arange(n), side='right')
    Lg = ends - starts
    pos = q * (np.maximum(Lg, 1) - 1)
    lo = np.clip(starts + np.floor(pos).astype(np.int64), 0, max(len(vv) - 1, 0))
    hi = np.clip(starts + np.ceil(pos).astype(np.int64),  0, max(len(vv) - 1, 0))
    if len(vv) == 0: return np.zeros(n)
    frac = (pos - np.floor(pos))
    med = (1.0 - frac) * vv[lo] + frac * vv[hi]
    med[Lg == 0] = 0.0
    return med

def _frc5_grouped_median(idx, vals, n):
    return _frc5_grouped_quantile(idx, vals, n, q=0.5)

def _frc5_masked_curvature(E, tri_edges, we, wv, act, eps=1e-12, return_ntri_act=False):
    """Augmented Forman (same 'plain + 3*triangle bonus' form as §B-FRC) with an
       ACTIVE-EDGE MASK: deleted edges (act=0) are removed from every sum and
       get F#=NaN; a triangle contributes only if all three edges are active.
       All-active unweighted limit == 4 - deg u - deg v + 3*#tri (self-check)."""
    u = E[:, 0]; v = E[:, 1]
    we = np.maximum(we, eps); wv = np.maximum(wv, eps)
    inv = np.where(act, 1.0 / np.sqrt(we), 0.0)
    nN = wv.shape[0]
    S = np.zeros(nN); np.add.at(S, u, inv); np.add.at(S, v, inv)
    wu_ = wv[u]; wv_ = wv[v]
    base = 2.0 * (wu_ + wv_) - np.sqrt(we) * (wu_ * S[u] + wv_ * S[v])
    B = np.zeros(E.shape[0]); nta = np.zeros(E.shape[0])
    if len(tri_edges):
        a3 = act[tri_edges].all(axis=1)
        if a3.any():
            te = tri_edges[a3]
            w3 = we[te]
            wf = np.cbrt(np.prod(w3, axis=1) + eps)
            contrib = (w3 * w3) / wf[:, None]      # Forman-2003 face term w(a)*[w(a)/w(b)] = we^2/wf
            np.add.at(B, te.reshape(-1), contrib.reshape(-1))
            np.add.at(nta, te.reshape(-1), 1.0)
    F = base + 3.0 * B
    F[~act] = np.nan
    if return_ntri_act:
        return F, nta
    return F

# ----------------------------- one scale ------------------------------------
def _frc5_graph_score(Pg, snaps, kg):
    """One graph scale, POSITIONS ONLY.  Fixed topology+triangles (reused §B-FRC
       builders), objective turning basis, two weight suites, masked augmented
       curvature, continuous read-outs.  Returns per-node fields + diagnostics."""
    m, T1 = Pg.shape[0], Pg.shape[1]
    E = _frc_build_topology([np.ascontiguousarray(Pg[:, s, :]) for s in snaps], kg)
    n_tri, tri_edges = _frc_triangles(E, m)
    u = E[:, 0]; v = E[:, 1]; nE = len(E)
    deg_node = (np.bincount(u, minlength=m) + np.bincount(v, minlength=m)).astype(np.float64)
    sigma = FRC_SIGMA_FAC * _frc_median_edge_len(Pg, E, snaps)

    # ---- objective relative rotation (frame-median debias) ------------------
    dth, rln = _frc5_edge_turning(Pg, E)
    # Frame reference, wrap-safe AND robust (two-pass):
    #   ref0 = circular mean  arg mean(e^{i dth})   -- EXACTLY equivariant under a
    #          frame increment (+da rotates the complex sum), but vortex-biased;
    #   d    = wrap(dth - ref0)                     -- exactly frame-INVARIANT;
    #   ref  = ref0 + median(d)                     -- re-centres on the bulk
    #          (median robustness of the old edge-median, no wrap-rank pathology
    #          because the bulk sits near 0 in the ref0 chart).
    ref0 = np.angle(np.exp(1j * dth.astype(np.float64)).mean(axis=0))
    d0 = (dth - ref0[None, :].astype(np.float32) + np.pi) % (2.0 * np.pi) - np.pi
    ref = ref0 + np.median(d0, axis=0)
    dthc = (dth - ref[None, :].astype(np.float32) + np.pi) % (2.0 * np.pi) - np.pi
    lw = np.log(rln + 1e-12)
    madg = _frc_mad(dthc.reshape(-1))

    tot_e = np.clip(dthc, -FRC5_CLIP_MADS * madg, FRC5_CLIP_MADS * madg).sum(axis=1)
    q_tot = np.tanh(np.abs(tot_e) / FRC5_THQ_TOT)           # demonstrated net co-rotation
    Lw = max(3, int(FRC5_WIN_L) | 1)
    t_out = np.arange(Lw - 1, T1 - 1)                       # window-end increments
    T_out = len(t_out)
    Fv  = np.full((m, T_out), np.nan, np.float32)           # core vertex curvature
    Vv  = np.zeros((m, T_out), bool)                        # validity
    Fb_e_acc = np.zeros(nE); Fb_e_cnt = np.zeros(nE)        # boundary edge curvature (time agg)
    lock_acc = np.zeros(m); act_frac = 0.0
    sdist2 = np.zeros(nE)                                   # static spectral weight accumulators
    locke_acc = np.zeros(nE)

    s_all = None                                            # global jitter scale (pass 1)
    Sbuf = np.empty((nE, T_out), np.float32)
    Mbuf = np.empty((nE, T_out), np.float32)
    for ti, t in enumerate(t_out):
        win = dthc[:, t - Lw + 1: t + 1]
        mm = np.median(win, axis=1)
        ss = 1.4826 * np.median(np.abs(win - mm[:, None]), axis=1)
        Mbuf[:, ti] = mm; Sbuf[:, ti] = ss
    s_med = float(np.median(Sbuf) + 1e-12)
    sgn = np.sign(dthc); alt_step = (sgn[:, 1:] * sgn[:, :-1] < 0).astype(np.float32)

    for ti, t in enumerate(t_out):
        mm = Mbuf[:, ti].astype(np.float64); ss = Sbuf[:, ti].astype(np.float64)
        c = np.abs(mm) / (np.abs(mm) + ss + FRC5_EPS_TH)            # locking
        lwin = lw[:, t - Lw + 2: t + 2]                              # distances in window
        medr = np.median(lwin, axis=1)
        lnD  = medr - lw[:, 0]                                       # net separation vs t0
        g = np.exp(-FRC5_BETA_D * np.maximum(0.0, lnD))              # dispersion gate
        q = np.tanh(np.abs(mm) * Lw / FRC5_THQ)                      # window winding factor
        alt = alt_step[:, max(t - Lw + 1, 0): t].mean(axis=1)        # sign-alternation rate
        act = ~((ss > FRC5_DEL_SFAC * s_med) & (alt > FRC5_DEL_ALT)) # DELETE jitter (alternating) edges
        cg = c * g * q * q_tot
        we_core = FRC5_W0 + FRC5_KAPPA * cg
        cn = np.zeros(m); np.add.at(cn, u, np.where(act, cg, 0.0)); np.add.at(cn, v, np.where(act, cg, 0.0))
        an = np.zeros(m); np.add.at(an, u, act.astype(float));      np.add.at(an, v, act.astype(float))
        wv_core = FRC5_W0 + cn / np.maximum(an, 1.0)
        fe = _frc5_masked_curvature(E, tri_edges, we_core, wv_core, act)
        good = np.isfinite(fe)
        sv = np.zeros(m); np.add.at(sv, u, np.where(good, fe, 0.0)); np.add.at(sv, v, np.where(good, fe, 0.0))
        cv = np.zeros(m); np.add.at(cv, u, good.astype(float));     np.add.at(cv, v, good.astype(float))
        ok = cv > 0
        Fv[ok, ti] = (sv[ok] / cv[ok]).astype(np.float32); Vv[:, ti] = ok

        # ---- boundary suite (same mask, separation-coherence weights) -------
        cvL = 1.4826 * np.median(np.abs(lwin - medr[:, None]), axis=1)
        s1 = cvL / (np.median(cvL) + 1e-12)
        pos = np.maximum(0.0, lnD); s2 = pos / (np.quantile(pos, 0.75) + 1e-9)
        we_b = FRC5_W0 + FRC5_KAPPA / (1.0 + FRC5_BND_L1 * s1 + FRC5_BND_L2 * s2)
        bn = np.zeros(m); np.add.at(bn, u, np.where(act, we_b, 0.0)); np.add.at(bn, v, np.where(act, we_b, 0.0))
        wv_b = FRC5_W0 + bn / np.maximum(an, 1.0)
        fb, nta = _frc5_masked_curvature(E, tri_edges, we_b, wv_b, act, return_ntri_act=True)
        # combinatorial-free CONTRAST: subtract the exact locally-uniform-weight
        # prediction (for ANY mask):  pred = 2(wu+wv) - (wu*an_u + wv*an_v) + 3*ntri_act*we.
        # If all weights in the 1-ring are equal, F# == pred identically, so the
        # residual R kills deg/#tri (density) fluctuations and keeps ONLY the
        # weight-contrast signal -- bridge/rim edges have R << 0.
        wbu = wv_b[u]; wbv = wv_b[v]
        pred = 2.0 * (wbu + wbv) - (wbu * an[u] + wbv * an[v]) + 3.0 * nta * we_b
        R = fb - pred
        gb = np.isfinite(R)
        Fb_e_acc[gb] += R[gb]; Fb_e_cnt[gb] += 1.0

        lock_acc += cn / np.maximum(an, 1.0)
        locke_acc += np.where(act, cg, 0.0)
        act_frac += float(act.mean())
        d2 = (Pg[u, t, 0] - Pg[v, t, 0])**2 + (Pg[u, t, 1] - Pg[v, t, 1])**2
        sdist2 += d2

    # ---- core read-outs (continuous primary; flips optional) ----------------
    delta_v = FRC_DELTA_C * _frc_mad(Fv[np.isfinite(Fv)])
    Fv0 = np.where(np.isfinite(Fv), Fv, 0.0)
    nu, _, Phi_net = _frc_Ndp(Fv0, delta_v, FRC_PERSIST_P)
    sig = 1.0 / (1.0 + np.exp(-np.clip((np.where(np.isfinite(Fv), Fv, -np.inf) - delta_v)
                                       / (0.25 * delta_v + 1e-12), -60, 60)))
    posdwell = (sig * Vv).sum(1) / np.maximum(Vv.sum(1), 1)
    posdwell = posdwell.astype(np.float32)          # smooth dwell: sigmoid((F#-delta)/(delta/4))
    Pi = posdwell / (1.0 + nu)
    Pi = np.clip(Pi / (np.quantile(Pi, 0.99) + 1e-12), 0.0, 1.0)

    # ---- vorticity-free LAVD signature Phi_i --------------------------------
    idx2 = np.concatenate([u, v]); val2 = np.concatenate([tot_e, tot_e])
    Phi_i = np.abs(_frc5_grouped_median(idx2, val2, m)).astype(np.float32)
    Phi_n = np.clip(Phi_i / (np.quantile(Phi_i, 0.99) + 1e-12), 0.0, 1.0).astype(np.float32)

    # ---- boundary node field (negative time-median curvature = rim/bridge) --
    Fb_med = np.where(Fb_e_cnt > 0, Fb_e_acc / np.maximum(Fb_e_cnt, 1.0), np.nan)
    fin = Fb_med[np.isfinite(Fb_med)]
    scl = 1.4826 * float(np.median(np.abs(fin))) + 1e-12    # scale around the NULL R=0:
    z = np.where(np.isfinite(Fb_med),                       # uniform 1-ring weights give R==0
                 np.maximum(0.0, -Fb_med / scl), 0.0)       # identically, so 0 is the principled
                                                            # reference; only curvature DEFICITS
                                                            # (R<0, bridges/rim) count.
    idxb = np.concatenate([u, v]); valb = np.concatenate([z, z])
    Bnd = _frc5_grouped_quantile(idxb, valb, m, q=0.75)     # robust (max is fluctuation-prone)
    Bnd = np.clip(Bnd / (np.quantile(Bnd, 0.99) + 1e-12), 0.0, 1.0).astype(np.float32)

    # ---- static spectral weight (positions-only Def-2.1 analogue) -> #vortices
    lockdwell = lock_acc / max(T_out, 1)
    gd = _frc_dwell_gate(lockdwell, FRC_DWELL_D0 * float(np.median(lockdwell) * 2.0 + 1e-12), FRC_DWELL_S)
    we_static = np.exp(-(sdist2 / max(T_out, 1)) / (sigma * sigma)) * (locke_acc / max(T_out, 1))
    we_static = np.maximum(we_static * gd[u] * gd[v], 0.0)
    rvort, vals = _frc_vortex_count(E, m, we_static, FRC_NEVEC, FRC_NEARZERO)

    diag = dict(n_edges=nE, n_tri=n_tri, m_nodes=m, delta=float(delta_v), k_graph=kg,
                vortex_count=rvort, eig=vals[:FRC_NEVEC], nu_mean=float(nu.mean()),
                posdwell_mean=float(posdwell.mean()), phi_net_mean=float(np.abs(Phi_net).mean()),
                act_frac=float(act_frac / max(T_out, 1)), T_out=T_out, win_L=Lw)
    return (Pi.astype(np.float32), posdwell, Phi_n, Bnd, nu.astype(np.float32), diag)

# ----------------------------- entry point -----------------------------------
def compute_candidate_frc5(STATE, layer_name='fwd'):
    """VORTICITY-FREE augmented Forman-Ricci elliptic detector (positions only;
       no _jacobian_series_fh, no omega anywhere).  Returns per-particle fields:
         phi          = Pi   composite core coherence  (P_i*(1+nu)^-1),
         phi_posdwell = P_i  continuous positive-curvature dwell (primary),
         phi_objrot   = Phi_i vorticity-free LAVD signature (net relative rotation),
         phi_boundary = boundary/rim strength (negative boundary-suite curvature),
         phi_nu       = stabilized flip count (optional channel),
       plus diagnostics (vortex count, |E|, #tri, delta, active-edge fraction)."""
    layer = STATE['layers'][layer_name]
    traj = layer['traj']; T1 = traj.shape[1]
    sub_idx, _ = topo_subsample_and_lift(traj[:, 0, :], max_n=CAND_SUBSAMPLE_TOPO)
    gsel = _frc_subsample_nodes(len(sub_idx), FRC_NODECAP, seed=17)
    Pg = np.ascontiguousarray(traj[sub_idx][gsel].astype(np.float64))    # (m,T1,2)
    m = Pg.shape[0]
    snaps = np.unique(np.linspace(0, T1 - 1, min(FRC_NSNAP, T1)).round().astype(int))
    scales = FRC_KSCALES if len(FRC_KSCALES) > 0 else (FRC_GRAPH_K,)

    acc = [np.zeros(m) for _ in range(5)]; last = None
    for kg in scales:
        Pi_s, Pd_s, Ph_s, Bd_s, Nu_s, diag = _frc5_graph_score(Pg, snaps, kg)
        for a, s in zip(acc, (Pi_s, Pd_s, Ph_s, Bd_s, Nu_s)): a += s
        last = diag
    ns = len(scales)
    Pi, Pd, Ph, Bd, Nu = [(a / ns).astype(np.float32) for a in acc]

    traj_full = STATE['layers'][layer_name]['traj']
    src = Pg[:, 0, :]; dst = traj_full[:, 0, :]
    def _lift(val):
        x = _smooth_lift_idw(src, dst, val, k_lift=CAND_LIFT_K)
        return np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    out = dict(phi=_lift(Pi), phi_posdwell=_lift(Pd), phi_objrot=_lift(Ph),
               phi_boundary=_lift(Bd), phi_nu=_lift(Nu), layer=layer_name)
    out.update(last); out['n_scales'] = ns
    return out

print("[cand-FRC5] vorticity-free objective-relative-rotation detector ready "
      "(positions-only basis; masked augmented Forman; core+boundary suites)")


# ======================= part2_tc0_90.ipynb cell 41 (verbatim) =======================
# ============================================================================
#  §B-FRC6  S1-S5 noise-hardened FRC5  (pure addition; FRC5 cells untouched)
#  ---------------------------------------------------------------------------
#  Five fixes, each tied to a failure mode verified on the cand11 run and
#  reproduced quantitatively on the synthetic surrogate (self-check below):
#   S1  per-edge temporal HAMPEL CENSORING of shock-crossing samples.
#       The FRC5 deletion rule (s > 4*median(s) AND alternation > 0.35) only
#       sees ALTERNATING chatter; a shock crossing is a coherent ONE-SIDED
#       dtheta pulse (low alternation) and slips through.  S1 flags per-edge
#       per-step outliers on BOTH channels,
#         |dtheta - med_e| > K*(mad_e + floor)  OR  |dlog r - med| > K*(...),
#       floor = max(0.05*globalMAD, 1e-3 rad)  (NOT 0.5*globalMAD: a soup-
#       dominated global scale would blunt censoring on clean edges), and
#       removes those SAMPLES from every window statistic and from Phi.
#       Purely temporal/Lagrangian -- no spatial shock mask (report Sec 6).
#   S2  significance-gated locking.  c = |m|/(|m|+s+eps) is ~0.4 for PURE
#       NOISE windows of L=7 (a 0/0 trap).  S2 multiplies by a t-gate on the
#       window median, t = |m|/(1.2533*s/sqrt(n_kept)), gate = clip((t-Z0)/Z1),
#       plus an absolute floor |m| > 1e-4 rad/slice.  Noise: t ~ |N(0,1)|.
#   S3  density-matched curvature excess.  On a uniform-weight 1-ring the
#       masked augmented Forman is EXACTLY  F# = w * struct,
#         struct = 4 - deg_act(u) - deg_act(v) + 3*#tri_act,
#       so raw F# carries graph DENSITY, not dynamics (the v4 inversion and
#       the v5 speckle bath).  S3 subtracts the per-slice uniform null
#       F0 = what(t)*struct (what = spatial median of w_e over active edges)
#       and standardises the vertex-projected excess by the per-slice spatial
#       MAD:  zX.  Dwell/flips run on zX in z-units (the single global-MAD
#       delta is GONE), with a floor-subtracted sigmoid: dwell(z<=0) == 0.
#   S4  random-walk noise-floor debias of Phi_i.  |sum dtheta| of zero-mean
#       jitter grows like mad_e*sqrt(n) -- a strictly positive haze largest
#       exactly where the flow is noisiest (the post-shock purple veil).  S4
#       winsorises per edge, sums KEPT samples only, and soft-thresholds
#       tot_hat = sign(tot)*max(0, |tot| - 1.7*mad_e*sqrt(n)).
#   S5  boundary dead-zone + absolute activity gate (mad_e > 1e-5 rad):
#       float-noise-only regions (seed-lattice strips) contribute nothing.
#   R   threshold-free, flip-free INTEGRAL readout (this revision): the
#       time-MEDIAN of the standardised excess zX over valid slices
#       (zint_med; winsorised-mean variant zint_wm in diag).  Replaces BOTH
#       the dwell threshold and the flip/hysteresis machinery as a primary
#       core map: soup integrates to ~0 (signed median -- no one-sided
#       E[z+]=0.4/slice floor), cores keep amplitude, rims show as graded
#       intermediate values the dwell threshold clips away.  It integrates
#       zX, NOT raw F#: by the uniform-ring identity F# = w*struct a raw
#       time-integral is a density map (trap 1 in integral form).  Building
#       it exposed a latent degeneracy the dwell's sigmoid had masked: S2 can
#       gate the majority to exactly W0, collapsing the per-slice spatial MAD
#       of X to 0 (z -> 1e8); the scale floor 0.5*mean|dev| keeps the
#       standardisation finite and incidentally cleans the dwell too.
#   PROFILES (this revision): FRC6_PROFILE = 'sensitive' (default) protects
#       weak/small structures -- right-wall vortex train, faint arms -- via
#       lighter censoring (K=4.5), permissive locking gate (t0=1.25), lower
#       dwell dead point (z0=2.0), shorter window (L=5 -> T_out up, transit
#       retention up), and two graph scales (8,12) (~2x candidate time).
#       'clean' = the conservative original set (max soup suppression).
#       Both pass every synthetic gate; A/B them on the real data.
#  Drop-in: _frc6_graph_score has the same signature/returns as
#  _frc5_graph_score (diag additionally carries cens_frac and the per-sample
#  S1 mask `keep` + edge list `E`); compute_candidate_frc6 mirrors
#  compute_candidate_frc5 (same subsample seed / topology / lift).
# ============================================================================
# ── FRC6 旋钮 + sensitive/clean 档位逻辑已移至 §0 MASTER CONFIG ──
_FRC6_CAPTURE = None   # 运行期 capture/调试句柄: viz/reflow 在调用引擎前置为 dict 抓取
                       #   we_st/zXe/srot 等中间量; 引擎读它决定是否回填。
                       #   必须在首次调用引擎(§FRC6 compute)之前于模块级存在。


def _nan_med_mad(win):
    """nan-aware per-row median + 1.4826*MAD + finite count for (nE, L) windows."""
    nk = np.isfinite(win).sum(axis=1)
    mm = np.nanmedian(np.where(np.isfinite(win), win, np.nan), axis=1)
    mm = np.where(nk > 0, mm, 0.0)
    ss = 1.4826*np.nanmedian(np.abs(win - mm[:, None]), axis=1)
    ss = np.where(nk > 0, ss, np.inf)
    return mm, ss, nk


def _frc6_graph_score(Pg, snaps, kg):
    m, T1 = Pg.shape[0], Pg.shape[1]
    E = _frc_build_topology([np.ascontiguousarray(Pg[:, s, :]) for s in snaps], kg)
    n_tri, tri_edges = _frc_triangles(E, m)
    u = E[:, 0]; v = E[:, 1]; nE = len(E)

    # ---- objective relative rotation (identical two-pass debias to FRC5) ----
    dth, rln = _frc5_edge_turning(Pg, E)
    ref0 = np.angle(np.exp(1j*dth.astype(np.float64)).mean(axis=0))
    d0 = (dth - ref0[None, :].astype(np.float32) + np.pi) % (2*np.pi) - np.pi
    ref = ref0 + np.median(d0, axis=0)
    dthc = (dth - ref[None, :].astype(np.float32) + np.pi) % (2*np.pi) - np.pi
    lw = np.log(rln + 1e-12)
    madg = _frc_mad(dthc.reshape(-1))

    # ================= S1: per-edge temporal Hampel censoring ================
    med_e = np.median(dthc, axis=1, keepdims=True)
    mad_e = 1.4826*np.median(np.abs(dthc - med_e), axis=1, keepdims=True)
    sfl = max(FRC6_SFLOOR_FAC*madg, 1e-3)
    zth = np.abs(dthc - med_e)/(mad_e + sfl)
    dlw = lw[:, 1:] - lw[:, :-1]
    med_l = np.median(dlw, axis=1, keepdims=True)
    mad_l = 1.4826*np.median(np.abs(dlw - med_l), axis=1, keepdims=True)
    lfl = max(FRC6_SFLOOR_FAC*_frc_mad(dlw.reshape(-1)), 1e-3)
    zlr = np.abs(dlw - med_l)/(mad_l + lfl)
    keep = (zth <= FRC6_HAMPEL_K) & (zlr <= FRC6_HAMPEL_K)
    cens_frac = float(1.0 - keep.mean())
    dthk = np.where(keep, dthc, np.nan).astype(np.float32)
    # ===== transport: per-edge robust net log-separation rate (BV-safe; u.n surrogate) =====
    #   dlw & keep both (nE, T1-1); zeta_e>0 => endpoints net-separating (detrain side).
    #   distance is O(2)-invariant INCLUDING reflection -> zeta_e (hence Psi) is a TRUE scalar.
    zeta_e = np.nanmedian(np.where(keep, dlw, np.nan), axis=1)
    zeta_e = np.where(np.isfinite(zeta_e), zeta_e, 0.0)

    # ================= S4: debiased winding (also feeds q_tot) ===============
    scl_e = (mad_e[:, 0] + sfl)
    wins = np.clip(dthc, -3.0*scl_e[:, None], 3.0*scl_e[:, None])
    tot = np.where(keep, wins, 0.0).sum(axis=1)
    nke = keep.sum(axis=1)
    floor = FRC6_PHI_Z0*scl_e*np.sqrt(np.maximum(nke, 1))
    tot_hat = np.sign(tot)*np.maximum(0.0, np.abs(tot) - floor)
    q_tot = np.tanh(np.abs(tot_hat)/FRC5_THQ_TOT)

    # activity gate (S5): pure float-noise edges contribute nothing anywhere
    edge_active_abs = (mad_e[:, 0] > 1e-5) | (np.abs(tot_hat) > 1e-4)
    tot_hat = np.where(edge_active_abs, tot_hat, 0.0)
    q_tot = np.where(edge_active_abs, q_tot, 0.0)

    Lw = max(3, int(FRC6_WIN_L) | 1)
    t_out = np.arange(Lw - 1, T1 - 1); T_out = len(t_out)
    Fv = np.full((m, T_out), np.nan, np.float32)            # standardised excess zX
    Vv = np.zeros((m, T_out), bool)
    _RAWv = np.full((m, T_out), np.nan, np.float32) if _FRC6_CAPTURE is not None else None
    _ZXe  = np.full((nE, T_out), np.nan, np.float32) if _FRC6_CAPTURE is not None else None
    Fb_e_acc = np.zeros(nE); Fb_e_cnt = np.zeros(nE)
    jit_e_acc = np.zeros(nE); jit_e_cnt = np.zeros(nE)   # mixing: per-edge time-mean of ln-r jitter (cvL)
    act_frac = 0.0
    sigma = FRC_SIGMA_FAC*_frc_median_edge_len(Pg, E, snaps)
    lock_acc = np.zeros(m); locke_acc = np.zeros(nE); sdist2 = np.zeros(nE)

    # pass 1: window scales for the (kept) alternation deletion threshold
    Sbuf = np.empty((nE, T_out), np.float32); Mbuf = np.empty((nE, T_out), np.float32)
    Nbuf = np.empty((nE, T_out), np.int32)
    for ti, t in enumerate(t_out):
        mm, ss, nk = _nan_med_mad(dthk[:, t - Lw + 1: t + 1])
        Mbuf[:, ti] = mm; Sbuf[:, ti] = np.where(np.isfinite(ss), ss, 0.0); Nbuf[:, ti] = nk
    s_med = float(np.median(Sbuf[Nbuf >= FRC6_MIN_KEEP]) + 1e-12) if (Nbuf >= FRC6_MIN_KEEP).any() else 1e-12
    sgnk = np.sign(np.where(keep, dthc, 0.0))
    alt_step = ((sgnk[:, 1:]*sgnk[:, :-1]) < 0).astype(np.float32)

    for ti, t in enumerate(t_out):
        mm = Mbuf[:, ti].astype(np.float64); ss = Sbuf[:, ti].astype(np.float64)
        nk = Nbuf[:, ti].astype(np.float64)
        # ---------------- S2: significance-normalised locking ----------------
        tstat = np.abs(mm)/(1.2533*ss/np.sqrt(np.maximum(nk, 1.0)) + 1e-12)
        sig = np.clip((tstat - FRC6_TSIG_Z0)/max(FRC6_TSIG_Z1, 1e-9), 0.0, 1.0)
        sig *= (np.abs(mm) > FRC6_TH_MIN) & edge_active_abs
        c = np.abs(mm)/(np.abs(mm) + ss + FRC5_EPS_TH)*sig
        lwin = np.where(keep[:, t - Lw + 2: t + 1], lw[:, t - Lw + 2: t + 1], np.nan)
        with np.errstate(all='ignore'):
            medr = np.nanmedian(np.concatenate([lwin, lw[:, t + 1:t + 2]], axis=1), axis=1)
        medr = np.where(np.isfinite(medr), medr, lw[:, 0])
        lnD = medr - lw[:, 0]
        g = np.exp(-FRC5_BETA_D*np.maximum(0.0, lnD))
        q = np.tanh(np.abs(mm)*Lw/FRC5_THQ)
        alt = alt_step[:, max(t - Lw + 1, 0): t].mean(axis=1)
        act = (nk >= FRC6_MIN_KEEP) & ~((ss > FRC5_DEL_SFAC*s_med) & (alt > FRC5_DEL_ALT))
        cg = c*g*q*q_tot
        we_core = FRC5_W0 + FRC5_KAPPA*cg
        cn = np.zeros(m); np.add.at(cn, u, np.where(act, cg, 0.0)); np.add.at(cn, v, np.where(act, cg, 0.0))
        an = np.zeros(m); np.add.at(an, u, act.astype(float));      np.add.at(an, v, act.astype(float))
        wv_core = FRC5_W0 + cn/np.maximum(an, 1.0)
        if _CAP_W is not None: _CAP_W.append((cn/np.maximum(an, 1.0)).astype(np.float32))
        fe, nta = _frc5_masked_curvature(E, tri_edges, we_core, wv_core, act, return_ntri_act=True)
        # ---------------- S3: density-matched curvature excess ---------------
        struct = 4.0 - an[u] - an[v] + 3.0*nta
        what = (float(np.median(we_core[act])) if act.any() else FRC5_W0) if FRC6_BASELINE == 'median' else FRC5_W0
        X = fe - what*struct                                  # ==0 on uniform rings
        goodX = np.isfinite(X) & act
        if _ZXe is not None:                                  # edge-level capture
            finE = X[goodX]
            if finE.size:
                dvE = np.abs(finE - np.median(finE))
                sE = max(1.4826*float(np.median(dvE)), 0.5*float(np.mean(dvE)), 1e-9)
                _ZXe[:, ti] = np.where(goodX, X/sE, np.nan).astype(np.float32)
        sv = np.zeros(m); np.add.at(sv, u, np.where(goodX, X, 0.0)); np.add.at(sv, v, np.where(goodX, X, 0.0))
        cv = np.zeros(m); np.add.at(cv, u, goodX.astype(float));    np.add.at(cv, v, goodX.astype(float))
        ok = cv > 0
        Xv = np.where(ok, sv/np.maximum(cv, 1.0), np.nan)
        fin = Xv[np.isfinite(Xv)]
        if fin.size:
            dv = np.abs(fin - np.median(fin))
            # scale floor: S2 gating the majority to exactly W0 can put >50% of
            # the spatial mass at X==0 (median=MAD=0, degenerate).  mean|dev|
            # stays finite there while remaining NON-binding on healthy slices
            # (Gaussian: 0.5*mean|dev| = 0.40 sigma < 1.4826*MAD = sigma).
            sX = max(1.4826*float(np.median(dv)), 0.5*float(np.mean(dv)), 1e-9)
        else:
            sX = 1.0
        Fv[:, ti] = (Xv/sX).astype(np.float32); Vv[:, ti] = ok
        if _RAWv is not None:                                  # raw vertex curvature (NO S3 subtraction)
            svr = np.zeros(m); np.add.at(svr, u, np.where(act & np.isfinite(fe), fe, 0.0))
            np.add.at(svr, v, np.where(act & np.isfinite(fe), fe, 0.0))
            _RAWv[:, ti] = np.where(ok, svr/np.maximum(cv, 1.0), np.nan).astype(np.float32)

        # ---------------- boundary suite (kept samples, dead-zone in S5) -----
        with np.errstate(all='ignore'):
            cvL = 1.4826*np.nanmedian(np.abs(lwin - medr[:, None]), axis=1)
        cvL = np.where(np.isfinite(cvL), cvL, 0.0)
        s1 = cvL/(np.median(cvL[act]) + 1e-12) if act.any() else cvL
        pos = np.maximum(0.0, lnD)
        s2 = pos/(np.quantile(pos[act], 0.75) + 1e-9) if act.any() else pos
        we_b = FRC5_W0 + FRC5_KAPPA/(1.0 + FRC5_BND_L1*s1 + FRC5_BND_L2*s2)
        bn = np.zeros(m); np.add.at(bn, u, np.where(act, we_b, 0.0)); np.add.at(bn, v, np.where(act, we_b, 0.0))
        wv_b = FRC5_W0 + bn/np.maximum(an, 1.0)
        fb, ntb = _frc5_masked_curvature(E, tri_edges, we_b, wv_b, act, return_ntri_act=True)
        wbu = wv_b[u]; wbv = wv_b[v]
        pred = 2.0*(wbu + wbv) - (wbu*an[u] + wbv*an[v]) + 3.0*ntb*we_b
        R = fb - pred
        gb = np.isfinite(R) & act
        Fb_e_acc[gb] += R[gb]; Fb_e_cnt[gb] += 1.0
        jit_e_acc[act] += cvL[act]; jit_e_cnt[act] += 1.0   # ln-r jitter on active edges (mixing)
        act_frac += float(act.mean())
        lock_acc += cn/np.maximum(an, 1.0)
        locke_acc += np.where(act, cg, 0.0)
        sdist2 += (Pg[u, t, 0] - Pg[v, t, 0])**2 + (Pg[u, t, 1] - Pg[v, t, 1])**2

    # ======================================================================
    #  σ⋆ GLOBAL ORIENTATION  (paper Eq.15-16) — THE SIGN FIX.
    #  Φ_i (objective rotation magnitude) is the orientation weight, so it is
    #  computed FIRST.  σ⋆ locks the rotation-locked lobe to +z and is applied
    #  to a COPY (Fo) used by the dwell and the fused core map.  Three signed
    #  read-outs come out of it:
    #    zint_canon = -σ⋆·zint  -> CANONICAL cores NEGATIVE, run-independent.
    #                 Fed to `phi_zint`, so the existing §S0 screening
    #                 (z < z_core) and the whole viz suite keep working
    #                 unchanged AND become dataset-robust (no more accidental
    #                 sign). In THIS run σ⋆=-1 so phi_zint is byte-identical.
    #    zint_or    =  σ⋆·zint  -> paper convention, cores POSITIVE (new field).
    #    phi_core   = relu(zint_or_median)·P_i  -> fused complete-core map.
    #  P_i (dwell) is now evaluated on Fo, so it fires on the CORE lobe instead
    #  of the rim (the original bug).
    # ======================================================================
    idx2 = np.concatenate([u, v]); val2 = np.concatenate([tot_hat, tot_hat])
    Phi_med_signed = _frc5_grouped_median(idx2, val2, m).astype(np.float32)   # signed objective rotation (chirality)
    Phi_i = np.abs(Phi_med_signed).astype(np.float32)
    with np.errstate(all='ignore'):
        _zagg_raw = np.nanmedian(np.where(Vv, Fv, np.nan), axis=1)            # un-oriented per-vertex z_agg
    sigstar = float(np.sign(np.nansum(np.where(np.isfinite(_zagg_raw), _zagg_raw, 0.0) * Phi_i)) or 1.0)
    Fo = (sigstar * Fv).astype(np.float32)                                    # ORIENTED z*X : core lobe -> +z
    if _ZXe is not None: _ZXe = (sigstar * _ZXe).astype(np.float32)           # keep edge-level capture consistent

    # ---- nu (flip count is sign-symmetric -> raw Fv is fine) ---------------
    Fv0 = np.where(np.isfinite(Fv), Fv, 0.0)
    nu, _, Phi_net = _frc_Ndp(Fv0, FRC_DELTA_C * _frc_mad(Fv0[Vv]) if Vv.any() else 1.0, FRC_PERSIST_P)
    nvalid = Vv.sum(1); minv = max(int(FRC6_ZINT_MINV), 1)

    # ---- R-channel: time-median of the standardised excess (raw + oriented) -
    with np.errstate(all='ignore'):
        zmed    = np.nanmedian(np.clip(np.where(Vv, Fv, np.nan), -FRC6_ZWINS, FRC6_ZWINS), axis=1)
        zwm     = np.nanmean (np.clip(np.where(Vv, Fv, np.nan), -FRC6_ZWINS, FRC6_ZWINS), axis=1)
        zmed_or = np.nanmedian(np.clip(np.where(Vv, Fo, np.nan), -FRC6_ZWINS, FRC6_ZWINS), axis=1)
    zint_med   = np.where(np.isfinite(zmed)    & (nvalid >= minv), zmed,    0.0).astype(np.float32)  # RAW sign
    zint_wm    = np.where(np.isfinite(zwm)     & (nvalid >= minv), zwm,     0.0).astype(np.float32)
    zint_or    = np.where(np.isfinite(zmed_or) & (nvalid >= minv), zmed_or, 0.0).astype(np.float32)  # cores +z
    zint_canon = (-sigstar * zint_med).astype(np.float32)                     # CANONICAL cores -z (run-indep.)

    # ---- dwell on the ORIENTED excess -> CORE lobe; fused core map ----------
    sig_d = 1.0/(1.0 + np.exp(-np.clip((np.where(np.isfinite(Fo), Fo, -np.inf) - FRC6_ZX_Z0)/FRC6_ZX_W, -60, 60)))
    s0 = 1.0/(1.0 + np.exp(FRC6_ZX_Z0/FRC6_ZX_W))          # sigmoid value at the z=0 null
    sig_d = np.clip((sig_d - s0)/(1.0 - s0), 0.0, 1.0)     # exact zero at z<=0  (now keyed to the CORE lobe)
    posdwell = ((sig_d*Vv).sum(1)/np.maximum(Vv.sum(1), 1)).astype(np.float32)
    Pi = posdwell/(1.0 + nu)
    Pi = np.clip(Pi/(np.quantile(Pi, 0.99) + 1e-12), 0.0, 1.0)
    phi_core = (np.clip(zint_or, 0.0, None) * posdwell).astype(np.float32)     # paper Sec II.C as one number

    Phi_n = np.clip(Phi_i/(np.quantile(Phi_i, 0.99) + 1e-12), 0.0, 1.0).astype(np.float32)

    # ---- boundary node field with S5 dead-zone ------------------------------
    Fb_med = np.where(Fb_e_cnt > 0, Fb_e_acc/np.maximum(Fb_e_cnt, 1.0), np.nan)
    fin = Fb_med[np.isfinite(Fb_med)]
    scl = 1.4826*float(np.median(np.abs(fin))) + 1e-12 if fin.size else 1.0
    z = np.where(np.isfinite(Fb_med),
                 np.maximum(0.0, (-Fb_med) - FRC6_BND_Z0*scl)/scl, 0.0)
    z = np.where(edge_active_abs, z, 0.0)
    idxb = np.concatenate([u, v]); valb = np.concatenate([z, z])
    Bnd = _frc5_grouped_quantile(idxb, valb, m, q=0.75)
    Bnd = np.clip(Bnd/(np.quantile(Bnd, 0.99) + 1e-12), 0.0, 1.0).astype(np.float32)

    # ===== transport / mixing fields (BV-safe, O(2)-objective; report Section 2) =========
    #   beta_e = the SAME per-edge boundary deficit z that feeds Bnd (HARD 0 off the rim),
    #   so laminar / interior edges contribute nothing and Psi has no inflow haze.
    beta_e = np.clip(z, 0.0, 1.0)
    # (1) Psi: signed cross-boundary separation flux. beta-WEIGHTED MEAN over incident edges
    #     (a median washes out the rim minority at a mostly-interior node). + detrain / - entrain.
    #     distance is O(2)-invariant INCLUDING reflection -> Psi is a TRUE scalar (not a pseudoscalar).
    _bw = (beta_e*zeta_e)
    psi_num = np.zeros(m); psi_den = np.zeros(m)
    np.add.at(psi_num, u, _bw);    np.add.at(psi_num, v, _bw)
    np.add.at(psi_den, u, beta_e); np.add.at(psi_den, v, beta_e)
    Psi = (psi_num/np.maximum(psi_den, 1e-9)).astype(np.float32)
    # (2) Lambda residence flux = net confirmed-state displacement Delta c (== Phi_net, already computed).
    Lam = Phi_net.astype(np.float32)
    # (3) M: mixing activity = beta-weighted mean of ln-r jitter on boundary edges (non-negative).
    jit_e = np.where(jit_e_cnt > 0, jit_e_acc/np.maximum(jit_e_cnt, 1.0), 0.0)
    mix_num = np.zeros(m); _mw = (beta_e*jit_e)
    np.add.at(mix_num, u, _mw); np.add.at(mix_num, v, _mw)
    Mix = (mix_num/np.maximum(psi_den, 1e-9)).astype(np.float32)
    # (4) B: barrier strength = Bnd (deep rim) AND low net flux AND low mixing (non-negative).
    _psi0 = float(np.quantile(np.abs(Psi), 0.75)) + 1e-9
    _mix0 = float(np.quantile(Mix, 0.75)) + 1e-9
    Barr = (Bnd*(1.0 - np.tanh(np.abs(Psi)/_psi0))*np.exp(-Mix/_mix0)).astype(np.float32)

    # ---- Def-2.1 static spectral weight -> near-zero vortex count (v5 parity) -
    lockdwell = lock_acc/max(T_out, 1)
    gd = _frc_dwell_gate(lockdwell, FRC_DWELL_D0*float(np.median(lockdwell)*2.0 + 1e-12), FRC_DWELL_S)
    we_st = np.exp(-(sdist2/max(T_out, 1))/(sigma*sigma))*(locke_acc/max(T_out, 1))
    we_st = np.maximum(we_st*gd[u]*gd[v], 0.0)
    if FRC6_SKIP_EIG:
        rvort, vals = -1, np.zeros(FRC_NEVEC)
    else:
        rvort, vals = _frc_vortex_count(E, m, we_st, FRC_NEVEC, FRC_NEARZERO)

    diag = dict(n_edges=nE, n_tri=n_tri, m_nodes=m, k_graph=kg,
                vortex_count=rvort, eig=vals[:FRC_NEVEC],
                nu_mean=float(nu.mean()), posdwell_mean=float(posdwell.mean()),
                phi_net_mean=float(np.abs(Phi_net).mean()),
                act_frac=float(act_frac/max(T_out, 1)), cens_frac=cens_frac,
                T_out=T_out, win_L=Lw, delta=float(FRC6_ZX_Z0),
                zint_med=zint_med, zint_wm=zint_wm,
                keep=keep, E=E)   # S1 sample mask + edge list (diagnostics)
    diag['srot_med'] = Phi_med_signed                                    # signed objective rotation (chirality)
    diag['psi_med'] = Psi; diag['lam_med'] = Lam; diag['mix_med'] = Mix; diag['barr_med'] = Barr   # transport/mixing
    if _FRC6_CAPTURE is not None:
        _FRC6_CAPTURE['zX'] = Fv; _FRC6_CAPTURE['Vv'] = Vv; _FRC6_CAPTURE['rawv'] = _RAWv
        _FRC6_CAPTURE['we_st'] = we_st   # dwell-gated static spectral weights (SLECS near-zero machinery)
        _FRC6_CAPTURE['zXe'] = _ZXe      # edge-level standardised excess (material-line rendering)
        _FRC6_CAPTURE['srot'] = Phi_med_signed   # 带符号客观旋转 (反号不合并用)
        _FRC6_CAPTURE['psi'] = Psi; _FRC6_CAPTURE['lam'] = Lam; _FRC6_CAPTURE['mix'] = Mix; _FRC6_CAPTURE['barr'] = Barr   # 输运/混合 (局部suite用)
    diag['core_med'] = phi_core
    diag['zint_or_med'] = zint_or
    diag['zint_canon'] = zint_canon
    diag['sigstar'] = sigstar
    return (Pi.astype(np.float32), posdwell, Phi_n, Bnd, nu.astype(np.float32), diag)

def compute_candidate_frc6(STATE, layer_name='fwd'):
    """FRC6 = FRC5 + S1-S5 + R-channel.  Fields:
         phi          = Pi   composite core coherence  (P_i*(1+nu)^-1),
         phi_posdwell = P_i  dwell on the standardised curvature excess zX,
         phi_zint     = SIGNED time-median integral of zX  (threshold-free,
                        flip-free primary core map; soup ~ 0, graded rims),
         phi_objrot   = Phi_i debiased vorticity-free LAVD signature,
         phi_boundary = boundary residual with S5 dead-zone,
         phi_nu       = stabilized flip count (optional channel),
       plus diagnostics incl. cens_frac and zint_wm (winsorised-mean variant,
       SUBSAMPLE-sized; lift it yourself if wanted)."""
    layer = STATE['layers'][layer_name]
    traj = layer['traj']; T1 = traj.shape[1]
    sub_idx, _ = topo_subsample_and_lift(traj[:, 0, :], max_n=CAND_SUBSAMPLE_TOPO)
    gsel = _frc_subsample_nodes(len(sub_idx), FRC_NODECAP, seed=17)
    Pg = np.ascontiguousarray(traj[sub_idx][gsel].astype(np.float64))    # (m,T1,2)
    m = Pg.shape[0]
    snaps = np.unique(np.linspace(0, T1 - 1, min(FRC_NSNAP, T1)).round().astype(int))
    scales = FRC6_KSCALES

    acc = [np.zeros(m) for _ in range(14)]; last = None
    for kg in scales:
        Pi_s, Pd_s, Ph_s, Bd_s, Nu_s, diag = _frc6_graph_score(Pg, snaps, kg)
        for a, s in zip(acc, (Pi_s, Pd_s, Ph_s, Bd_s, Nu_s, diag['zint_med'], diag['srot_med'],
                              diag['psi_med'], diag['lam_med'], diag['mix_med'], diag['barr_med'],
                              diag['core_med'], diag['zint_or_med'], diag['zint_canon'])): a += s
        last = diag
    ns = len(scales)
    Pi, Pd, Ph, Bd, Nu, Zi, Sr, Ps, La, Mx, Br, Co, Zo, Zc = [(a / ns).astype(np.float32) for a in acc]

    traj_full = STATE['layers'][layer_name]['traj']
    src = Pg[:, 0, :]; dst = traj_full[:, 0, :]
    def _lift(val):
        x = _smooth_lift_idw(src, dst, val, k_lift=CAND_LIFT_K)
        return np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    out = dict(phi=_lift(Pi), phi_posdwell=_lift(Pd), phi_zint=_lift(Zc),
               phi_zint_raw=_lift(Zi), phi_zint_oriented=_lift(Zo), phi_core=_lift(Co),
               sigstar=float(last.get('sigstar', 1.0)),
               phi_objrot=_lift(Ph), phi_boundary=_lift(Bd), phi_nu=_lift(Nu),
               phi_objrot_signed=_lift(Sr),                                  # SIGNED rotation (chirality)
               phi_psi=_lift(Ps), phi_resflux=_lift(La), phi_mix=_lift(Mx), phi_barrier=_lift(Br),  # transport/mixing
               layer=layer_name)
    last = dict(last); last.pop('keep', None); last.pop('E', None)
    last.pop('zint_med', None)   # lifted above; zint_wm stays in diag
    for _k in ('srot_med', 'psi_med', 'lam_med', 'mix_med', 'barr_med', 'core_med', 'zint_or_med', 'zint_canon'): last.pop(_k, None)
    out.update(last); out['n_scales'] = ns
    return out

print("[cand-FRC6] S1-S5 + R-channel detector ready (censored shock samples, "
      "significance-gated locking, density-matched zX, threshold-free integral)")


# ---------------- driver: the core of compute_candidate_frc6 without subsample/lift ----------------
def afrc_fields(traj, scales=FRC6_KSCALES, nsnap=FRC_NSNAP):
    """traj (N, T1, 2) -> per-particle FRC6 fields averaged over graph scales (as in compute_candidate_frc6)."""
    Pg = np.ascontiguousarray(traj.astype(np.float64)); m, T1 = Pg.shape[0], Pg.shape[1]
    snaps = np.unique(np.linspace(0, T1 - 1, min(nsnap, T1)).round().astype(int))
    keys = ['Pi', 'posdwell', 'Phi_n', 'Bnd', 'nu', 'zint_med', 'srot_med', 'core_med', 'zint_or_med']
    acc = {k: np.zeros(m) for k in keys}
    for kg in scales:
        Pi_s, Pd_s, Ph_s, Bd_s, Nu_s, diag = _frc6_graph_score(Pg, snaps, kg)
        for k, s in zip(keys, (Pi_s, Pd_s, Ph_s, Bd_s, Nu_s, diag['zint_med'], diag['srot_med'], diag['core_med'], diag['zint_or_med'])):
            acc[k] += s
    return {k: (acc[k] / len(scales)).astype(np.float64) for k in keys}
