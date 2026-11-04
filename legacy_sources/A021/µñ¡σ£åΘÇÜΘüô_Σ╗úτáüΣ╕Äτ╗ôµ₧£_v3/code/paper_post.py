"""verbatim from part2_tc0_90.ipynb cell 19 (graph despeckle + graph TV)."""
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix
FORMAN_TV_LAMBDA = 0.10; FORMAN_TV_ITERS = 200; FORMAN_TV_K = 16
FORMAN_DESPECKLE = "close"; FORMAN_DESPECKLE_K = 8; FORMAN_DESPECKLE_ITERS = 2; FORMAN_DESPECKLE_QERODE = 0.3
FORMAN_TV_VERBOSE = False
FORMAN_KNN_WORKERS = -1

def _vectorised_mutual_knn_at_snapshot(P, k):
    """Mutual k-NN edges among the N points P[N, 2].  Returns (M, 2)
    int64 array of edges (i < j).  Uses cKDTree + a vectorised mutual
    test (no Python inner loop).
    """
    N = P.shape[0]
    tree = cKDTree(P)
    # query k+1 because point 0 of each row is the point itself
    _, nbr = tree.query(P, k=min(k + 1, N))
    if nbr.ndim == 1:
        nbr = nbr.reshape(-1, 1)
    nbr = nbr[:, 1:]                                    # (N, k) drop self
    src = np.repeat(np.arange(N, dtype=np.int64), nbr.shape[1])
    dst = nbr.ravel().astype(np.int64)
    # remove self-edges (can happen at degenerate duplicates)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    # build a quick "is j in kNN(i)?" lookup via a sparse adjacency
    A = csr_matrix((np.ones_like(src, dtype=bool), (src, dst)), shape=(N, N))
    # mutual edges: A[i,j] AND A[j,i].  Equivalent to A.multiply(A.T).
    A_mut = A.multiply(A.T).tocoo()
    i_mut = A_mut.row.astype(np.int64)
    j_mut = A_mut.col.astype(np.int64)
    # keep canonical i<j to avoid duplicates
    mask = i_mut < j_mut
    return np.column_stack([i_mut[mask], j_mut[mask]])



def graph_tv_denoise(f, P, lam=0.10, iters=200, k=16, verbose=True):
    N = P.shape[0]
    edges = _vectorised_mutual_knn_at_snapshot(np.ascontiguousarray(P, dtype=np.float64), int(k))
    if edges.shape[0] == 0:
        return np.asarray(f, dtype=np.float64)
    M = edges.shape[0]
    rows = np.repeat(np.arange(M), 2)
    cols = edges.ravel().astype(np.int64)
    data = np.tile(np.array([-1.0, 1.0]), M)            # (D phi)_e = phi_j - phi_i
    D = csr_matrix((data, (rows, cols)), shape=(M, N))
    Dt = D.T.tocsr()
    maxdeg = max(1, int(np.bincount(cols, minlength=N).max()))
    Lnorm = np.sqrt(2.0 * maxdeg)                       # ||D||_2 <= sqrt(2*maxdeg)
    tau = 1.0 / Lnorm; sigma = 1.0 / Lnorm              # tau*sigma*L^2 < 1
    f = np.asarray(f, dtype=np.float64)
    phi = f.copy(); phibar = f.copy(); p = np.zeros(M)
    for _ in range(int(iters)):
        p = np.clip(p + sigma * (D @ phibar), -lam, lam)
        phin = (phi - tau * (Dt @ p) + tau * f) / (1.0 + tau)
        phibar = 2.0 * phin - phi
        phi = phin
    if verbose:
        print(f"  [tv] graph-TV denoise: |E|={M:,}, lambda={lam}, iters={iters}, k={k}")
    return phi


# =============================================================================
#  GRAPH SOFT-CLOSING DESPECKLE  (de-beading: dotted ridge -> continuous line)
#  -----------------------------------------------------------------------------
#  min-reduce Forman + particle-level variance leave ridges as rows of BEADS.
#  Morphological closing on the kNN graph (dilate x iters, then SOFT-quantile
#  erode) bridges the gaps WITHOUT erasing 1-particle-thin lines (a hard min
#  erode would).  Pure post-processing of the node field; positions only.
# =============================================================================

def _despeckle_nb_table(P, k):
    edges = _vectorised_mutual_knn_at_snapshot(np.ascontiguousarray(P, dtype=np.float64), int(k))
    N = P.shape[0]
    if edges.shape[0] == 0:
        return np.full((N, 1), -1, dtype=np.int64)
    src = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    dst = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)
    order = np.argsort(src, kind="stable"); src = src[order]; dst = dst[order]
    _u, first, cnt = np.unique(src, return_index=True, return_counts=True)
    rank = np.arange(src.size) - np.repeat(first, cnt)
    NB = np.full((N, int(cnt.max())), -1, dtype=np.int64)
    NB[src, rank] = dst
    return NB



def _nb_reduce(phi, NB, mode):
    """Reduce over {self} U kNN(self).  mode: 'max' | 'min' | 'median' | float q."""
    vals = np.where(NB >= 0, phi[np.maximum(NB, 0)], np.nan)
    allv = np.concatenate([phi[:, None], vals], axis=1)
    if mode == "max":
        return np.nanmax(allv, axis=1)
    if mode == "min":
        return np.nanmin(allv, axis=1)
    if isinstance(mode, float):
        return np.nanquantile(allv, mode, axis=1)
    return np.nanmedian(allv, axis=1)



def graph_despeckle(phi, P, verbose=True):
    mode = str(globals().get("FORMAN_DESPECKLE", "close"))
    if mode == "none":
        return phi
    k = int(globals().get("FORMAN_DESPECKLE_K", 8))
    iters = int(globals().get("FORMAN_DESPECKLE_ITERS", 2))
    qe = float(globals().get("FORMAN_DESPECKLE_QERODE", 0.3))
    NB = _despeckle_nb_table(P, k)
    out = np.asarray(phi, dtype=np.float64)
    if mode in ("close", "close+median"):
        for _ in range(iters):
            out = _nb_reduce(out, NB, "max")     # dilate: bridge bead gaps along ridges
        for _ in range(iters):
            out = _nb_reduce(out, NB, qe)        # soft erode: shrink halo, keep thin lines
    if mode in ("median", "close+median"):
        out = _nb_reduce(out, NB, "median")
    del NB
    if verbose:
        print(f"  [despeckle] graph soft-closing: mode={mode}, k={k}, iters={iters}, q_erode={qe}")
    return out


# =============================================================================
#  OVERRIDE the two edge-building drivers to apply the coherence gate.
#  (Both reuse the existing forman_field_one_scale / forman_edges_one_scale /
#   _node_weight / _forman_node_from_weights / degenerate_particle_mask.)
# =============================================================================