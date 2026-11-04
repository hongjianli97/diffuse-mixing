"""AFRC-H (hyperbolic barrier field B) core, extracted VERBATIM from part2_tc0_90.ipynb (the last definition of each
function in notebook order).  Parameters copied from that notebook.  The driver afrc_h_field() reproduces ONE graph scale of
forman_field_multiscale_lean: mutual-kNN union at the window endpoints (MKNN_N_TIME_SAMPLES = 2), coherence/jump/significance
gate, robust-slope edge weights, stretch_max node weights, augmented Forman (lambda = 3, 'auto' scale), node = -min F#(e).
Multiscale combination, despeckle and graph-TV post-processing are NOT included."""
import numpy as np, gc
from scipy.spatial import cKDTree
FORMAN_EPS = 1.0e-12; FORMAN_NODE_WEIGHT = "stretch_max"; FORMAN_REDUCE = "min"
FORMAN_EDGE_WEIGHT = "coherent"; FORMAN_COH_SLOPE = "median"; FORMAN_COH_TAU = 0.35; FORMAN_COH_TV_FLOOR = 1.0e-3
FORMAN_COH_WFLOOR_FRAC = 1.0e-3; FORMAN_COH_SEP_FLOOR_FRAC = 0.5; FORMAN_COH_JUMP_MAX = 0.5; FORMAN_COH_Z = 3.0
FORMAN_COH_CHUNK_EDGES = 4_000_000; FORMAN_AFRC_ENABLE = True; FORMAN_AFRC_LAMBDA = 3.0; FORMAN_AFRC_SCALE = "auto"
FORMAN_AFRC_CHUNK = None; FORMAN_INVSQRT_CAP_ENABLE = False; FORMAN_INVSQRT_CAP_MULT = 10.0; FORMAN_SHOCK_CENSOR = True
FORMAN_SEP_FLOOR_ENABLE = False; FORMAN_SEP_FLOOR_FRAC = 0.5; FORMAN_RATE_CLIP = None
_GRID_CELL = 1.0
def _grid_cell_size(): return _GRID_CELL            # notebook: render-grid cell (about half the seed spacing)
_FORMAN_CENSOR_CURRENT = None
FORMAN_SIGNED = False
_np_F = np                                           # alias used by some notebook cells

# ---- verbatim from part2_tc0_90.ipynb cell 29 ----
def _edge_logdist_stats_v2(traj, edges):
    """Chunked per-edge stats of log d_ij(t); v2 adds ss (z statistic).
    [v63] 若 _FORMAN_CENSOR_CURRENT 就位: 边的"任一端点事件步"置 NaN 后重算
    (net=Σ保留步 != 端点差!), 另返回 neff=有效步数 -> J/z 门按边条件生效。"""
    sep = max(float(globals().get("FORMAN_COH_SEP_FLOOR_FRAC", 0.5)) * _grid_cell_size(), 1e-12)
    M = edges.shape[0]; T = traj.shape[1]
    net = np.empty(M); TV = np.empty(M); med = np.empty(M); jmax = np.empty(M); ss = np.empty(M)
    neff = np.full(M, T - 1, np.int32)
    CM = globals().get("_FORMAN_CENSOR_CURRENT", None)
    use_c = (CM is not None and getattr(CM, "shape", None) == (traj.shape[0], T - 1)
             and bool(globals().get("FORMAN_SHOCK_CENSOR", True)))
    B = int(globals().get("FORMAN_COH_CHUNK_EDGES", 4_000_000)) or M
    for s in range(0, M, B):
        e = min(M, s + B)
        d = np.linalg.norm(traj[edges[s:e, 0]] - traj[edges[s:e, 1]], axis=2).astype(np.float64)
        logd = np.log(np.maximum(d, sep))
        step = np.diff(logd, axis=1)
        if use_c:
            bad = CM[edges[s:e, 0]] | CM[edges[s:e, 1]]
            step = np.where(bad, np.nan, step)
            neff[s:e] = (~bad).sum(axis=1).astype(np.int32)
            net[s:e] = np.nansum(step, axis=1)
            tv = np.nansum(np.abs(step), axis=1)
            with np.errstate(all="ignore"):
                med[s:e] = np.where(neff[s:e] > 0, np.nanmedian(step, axis=1), 0.0)
                jm = np.nanmax(np.abs(step), axis=1)
            jmax[s:e] = np.where(tv > 1e-30, jm / np.maximum(tv, 1e-30), 1.0)
            ss[s:e] = np.nansum(step ** 2, axis=1)
        else:
            net[s:e] = logd[:, -1] - logd[:, 0]
            tv = np.abs(step).sum(axis=1)
            med[s:e] = np.median(step, axis=1) if step.shape[1] else 0.0
            jmax[s:e] = (np.abs(step).max(axis=1) / np.maximum(tv, 1e-30)) if step.shape[1] else 1.0
            ss[s:e] = (step ** 2).sum(axis=1)
        TV[s:e] = tv
        del d, logd, step
    return net, TV, med, jmax, ss, T, neff

# ---- verbatim from part2_tc0_90.ipynb cell 29 ----
def coherent_edge_keep(traj, edges):
    """Keep-mask: TV floor + coherence C + jump-dominance J (4e) + NEW
    sign-randomization significance z (this cell)."""
    if edges.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    net, TV, _med, J, ss, T, neff = _edge_logdist_stats_v2(traj, edges)
    C = np.abs(net) / np.maximum(TV, 1e-30)
    keep = (TV >= float(globals().get("FORMAN_COH_TV_FLOOR", 1e-3))) &            (C >= float(globals().get("FORMAN_COH_TAU", 0.35)))
    jmx = float(globals().get("FORMAN_COH_JUMP_MAX", 0.5))
    ok4 = neff >= 4                                   # [v63] 剔除后有效步数逐边判
    if jmx < 1.0 and (T - 1) >= 4:
        keep &= (J <= jmx) | (~ok4)
    z = float(globals().get("FORMAN_COH_Z", 3.0))
    if z > 0 and (T - 1) >= 4:
        keep &= (np.abs(net) >= z * np.sqrt(np.maximum(ss, 1e-30))) | (~ok4)
    keep &= neff >= 2                                 # 全被挖空的边直接弃
    if FORMAN_SIGNED:
        keep &= net > 0                               # direction-aware: only pairs that SEPARATE in the integration direction
    return keep

# ---- verbatim from part2_tc0_90.ipynb cell 19 ----
def _edge_weight(traj, edges, dt):
    """[A/coherent] edge weight.
      'coherent' : robust temporal slope of log d_ij(t)  (NEW default)
      'logratio' : |log(d_N/d_0)|/dt                      (old improvement A, two-endpoint)
      'cov'      : Var_t(d)/mean_t(d)                      (oldest)
    The coherence GATE (coherent_edge_keep) is applied to the edge SET by the
    callers below, so here we only return the per-edge weight magnitude."""
    mode = str(globals().get("FORMAN_EDGE_WEIGHT", "coherent"))
    if mode == "cov":
        dij = np.linalg.norm(traj[edges[:, 0]] - traj[edges[:, 1]], axis=2)
        return np.maximum(dij.var(axis=1) / np.maximum(dij.mean(axis=1), 1e-12), FORMAN_EPS)
    if mode == "coherent":
        if edges.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        net, TV, med, _J, T = _edge_logdist_stats(traj, edges)
        if str(globals().get("FORMAN_COH_SLOPE", "median")) == "endpoint":
            rate = np.abs(net) / max(dt, 1e-30)
        else:
            dt_step = dt / max(T - 1, 1)
            rate = (np.maximum(med, 0.0) if FORMAN_SIGNED else np.abs(med)) / max(dt_step, 1e-30)
        w = np.maximum(rate, FORMAN_EPS)
        wf = float(globals().get("FORMAN_COH_WFLOOR_FRAC", 1e-3))
        if wf > 0 and w.size:
            pos = w[w > FORMAN_EPS]
            if pos.size:
                w = np.maximum(w, wf * float(np.median(pos)))
        return w
    # ---- "logratio" (old improvement A) ------------------------------------
    d0, d1 = _endpoint_distances(traj, edges)
    floor = (FORMAN_SEP_FLOOR_FRAC * _grid_cell_size()
             if globals().get("FORMAN_SEP_FLOOR_ENABLE", False) else 1e-12)
    floor = max(floor, 1e-12)
    s = np.abs(np.log(np.maximum(d1, floor) / np.maximum(d0, floor))) / dt
    if globals().get("FORMAN_RATE_CLIP", None) is not None:
        s = np.minimum(s, float(FORMAN_RATE_CLIP))
    return np.maximum(s, FORMAN_EPS)

# ---- verbatim from part2_tc0_90.ipynb cell 11 ----
def _node_weight(N, edges, w_e):
    """[A] node weight from incident-edge stretching (local, Galilean)."""
    i = edges[:, 0]; j = edges[:, 1]
    if FORMAN_NODE_WEIGHT == "unit":
        return _np_F.ones(N, dtype=_np_F.float64)
    if FORMAN_NODE_WEIGHT == "stretch_max":
        w_v = _np_F.zeros(N, dtype=_np_F.float64)
        _np_F.maximum.at(w_v, i, w_e); _np_F.maximum.at(w_v, j, w_e)
    else:  # "stretch_mean"
        acc = _np_F.zeros(N, dtype=_np_F.float64); cnt = _np_F.zeros(N, dtype=_np_F.float64)
        _np_F.add.at(acc, i, w_e); _np_F.add.at(acc, j, w_e)
        _np_F.add.at(cnt, i, 1.0); _np_F.add.at(cnt, j, 1.0)
        w_v = acc / _np_F.maximum(cnt, 1.0)
    return _np_F.maximum(w_v, FORMAN_EPS)

# ---- verbatim from part2_tc0_90.ipynb cell 21 ----
def _forman_Fe_graph(N, edges, w_e, w_v):
    """WEIGHTED graph Forman per-edge curvature F(e) (identical to the stock
    formula; honours the optional 1/sqrt(w_e) cap)."""
    i = edges[:, 0]; j = edges[:, 1]
    inv = 1.0 / np.sqrt(np.maximum(w_e, FORMAN_EPS))
    if globals().get("FORMAN_INVSQRT_CAP_ENABLE", False):
        pos = w_e[w_e > FORMAN_EPS]
        med = float(np.median(pos)) if pos.size else 1.0
        inv = np.minimum(inv, FORMAN_INVSQRT_CAP_MULT / np.sqrt(max(med, FORMAN_EPS)))
    S = np.zeros(N, dtype=np.float64)
    np.add.at(S, i, inv); np.add.at(S, j, inv)
    sum_i = (w_v[i] * inv) * (S[i] - inv)
    sum_j = (w_v[j] * inv) * (S[j] - inv)
    return w_v[i] + w_v[j] - w_e * (sum_i + sum_j)

# ---- verbatim from part2_tc0_90.ipynb cell 21 ----
def _afrc_Fe(N, edges, w_e, w_v):
    """Augmented Forman per-edge curvature = graph Forman + lambda * triangle term."""
    F = _forman_Fe_graph(N, edges, w_e, w_v)
    if (not globals().get("FORMAN_AFRC_ENABLE", True)) or edges.shape[0] == 0:
        return F
    lam = float(globals().get("FORMAN_AFRC_LAMBDA", 3.0))
    if lam == 0.0:
        return F
    m = edge_triangle_counts(N, edges).astype(np.float64)
    if str(globals().get("FORMAN_AFRC_SCALE", "raw")) == "auto":
        sF = float(np.median(np.abs(F))) if F.size else 1.0
        pos = m[m > 0]; sm = float(np.median(pos)) if pos.size else 1.0
        tri = m * (max(sF, 1e-12) / max(sm, 1e-9))
    else:
        tri = m
    return F + lam * tri

# ---- verbatim from part2_tc0_90.ipynb cell 21 ----
def _forman_node_from_weights(N, edges, w_e, w_v):
    """Node scalar from the augmented per-edge curvature, reduced as before
    (min = bottleneck; mean = smoother).  Node value = -reduce F#(e)."""
    F_e = _afrc_Fe(N, edges, w_e, w_v)
    i = edges[:, 0]; j = edges[:, 1]
    if FORMAN_REDUCE == "mean":
        Fn = np.zeros(N, dtype=np.float64); cnt = np.zeros(N, dtype=np.float64)
        np.add.at(Fn, i, F_e); np.add.at(Fn, j, F_e)
        np.add.at(cnt, i, 1.0); np.add.at(cnt, j, 1.0)
        Fn = Fn / np.maximum(cnt, 1.0)
    else:
        Fn = np.full(N, np.inf)
        np.minimum.at(Fn, i, F_e); np.minimum.at(Fn, j, F_e)
        Fn[~np.isfinite(Fn)] = 0.0
    return (-Fn).astype(np.float64)

# ---- verbatim from part2_tc0_90.ipynb cell 21 ----
def edge_triangle_counts(N, edges):
    """m_e = #triangles (3-cliques) containing edge e = |N(i) cap N(j)| on the
    CURRENT (already coherence-gated) mutual-kNN edge set.  Vectorised + chunked."""
    M = edges.shape[0]
    if M == 0:
        return np.zeros(0, dtype=np.int64)
    i = edges[:, 0].astype(np.int64); j = edges[:, 1].astype(np.int64)
    src = np.concatenate([i, j]); dst = np.concatenate([j, i])
    order = np.argsort(src, kind="stable"); src_s = src[order]; dst_s = dst[order]
    _u, first, counts = np.unique(src_s, return_index=True, return_counts=True)
    rank = np.arange(src_s.size) - np.repeat(first, counts)
    maxdeg = int(counts.max())
    NB = np.full((N, maxdeg), -1, dtype=np.int32)
    NB[src_s, rank] = dst_s.astype(np.int32)
    del src, dst, order, src_s, dst_s, rank
    m = np.empty(M, dtype=np.int64)
    B = int(globals().get("FORMAN_AFRC_CHUNK") or max(2000, 4_000_000 // max(maxdeg * maxdeg, 1)))
    for s in range(0, M, B):
        e = min(M, s + B)
        Ni = NB[i[s:e]]; Nj = NB[j[s:e]]                 # (b, maxdeg)
        match = (Ni[:, :, None] == Nj[:, None, :]) & (Ni[:, :, None] >= 0)
        m[s:e] = match.any(axis=2).sum(axis=1)
        del Ni, Nj, match
    del NB; gc.collect()
    return m

# ---- verbatim from part2_tc0_90.ipynb cell 29 ----
def _edge_logdist_stats(traj, edges):
    """[v63] 适配器: 让 cell-4e 的 _edge_weight('coherent') 也走剔除版统计
    (后定义覆盖前者; 返回 v1 的 5 元组)。"""
    net, TV, med, jmax, _ss, T, _neff = _edge_logdist_stats_v2(traj, edges)
    return net, TV, med, jmax, T

# ------------------------------------------------------------------ driver (one scale)
def _mutual_knn(P, k):
    n = len(P); _, nb = cKDTree(P).query(P, k=k + 1); nb = nb[:, 1:]
    src = np.repeat(np.arange(n), k); dst = nb.ravel(); a = np.minimum(src, dst); b = np.maximum(src, dst)
    key = a.astype(np.int64) * n + b; key.sort(); dup = np.unique(key[:-1][key[1:] == key[:-1]])
    return np.column_stack([dup // n, dup % n]).astype(np.int64)

def afrc_h_field(traj, k=24, dt=1.0, censor=None, cell=None):
    """traj (N, T, 2).  Returns (B node field, edge-input node median of w_e, kept-edge fraction)."""
    global _GRID_CELL, _FORMAN_CENSOR_CURRENT
    N, T, _ = traj.shape
    if cell is None:
        d, _ = cKDTree(traj[:, 0]).query(traj[:, 0], k=2); cell = 0.5 * float(np.median(d[:, 1]))
    _GRID_CELL = cell; _FORMAN_CENSOR_CURRENT = censor
    E = np.unique(np.vstack([_mutual_knn(traj[:, 0], k), _mutual_knn(traj[:, -1], k)]), axis=0)
    keep = coherent_edge_keep(traj, E); Ek = E[keep]
    w_e = _edge_weight(traj, Ek, dt); w_v = _node_weight(N, Ek, w_e)
    B = _forman_node_from_weights(N, Ek, w_e, w_v)
    deg = np.bincount(Ek.ravel(), minlength=N); B[deg < 3] = 0.0
    idx = np.concatenate([Ek[:, 0], Ek[:, 1]]); val = np.concatenate([w_e, w_e])
    o = np.lexsort((val, idx)); iv = idx[o]; vv = val[o]
    st = np.searchsorted(iv, np.arange(N), "left"); en = np.searchsorted(iv, np.arange(N), "right")
    win = np.array([np.median(vv[a:b]) if b > a else 0.0 for a, b in zip(st, en)])
    _FORMAN_CENSOR_CURRENT = None
    return B, win, float(keep.mean())
