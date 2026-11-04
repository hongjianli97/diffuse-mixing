"""Elliptic structures without any rotation-strength field (no LAVD, no phi level sets).

Definition (AFRC_E-native):
  1. edge activation  = the AFRC_E dispersion gate made exact: a material edge is active iff its length, after removing
                        the isotropic part ln J/2 of the ledger, never departs from its initial value by more than
                        delta (+ its own noise allowance).  Only pair distances and the area ledger: objective, integral.
  2. cluster          = connected component of ACTIVE TRIANGLES (all three edges active) joined through shared edges.
                        This is the triangle term of the augmented Forman curvature: an active edge with no active
                        triangle has struct_act = 4 - deg - deg < 0, a bridge, and does not connect.
  3. vortex           = cluster that rotates, relative to the ensemble, significantly (vs its own noise) and
                        SUSTAINEDLY (no single step carries more than J_max of the rotation; AFRC_H's single-event gate).
  On an active cluster the motion is locally rigid, so the pair-angle winding IS the rotation angle: the strain
  contamination that sank the pair-angle driver (Spearman 0.5) is absent exactly where it is used.
"""
import numpy as np


def edges_of(tri):
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    E, inv = np.unique(e, axis=0, return_inverse=True)
    return E, inv.reshape(3, -1).T                      # E (nE,2), tri->edge ids (nT,3)


def _runmed(x, w=5):
    S = x.shape[0]; h = w // 2; out = np.empty_like(x)
    for s in range(S):
        out[s] = np.median(x[max(0, s - h):min(S, s + h + 1)], 0)
    return out


def edge_coherence(fr, E, lnJ, conformal=True):
    """D_e = max_t |l~_e(t) - l~_e(0)|,  l = ln|r_e| - (lnJ_u+lnJ_v)/4 (isotropic part removed), l~ = 5-frame running median.
    sig_l = per-edge noise of l from its own second differences."""
    r = fr[:, E[:, 1]] - fr[:, E[:, 0]]
    l = np.log(np.maximum(np.hypot(r[..., 0], r[..., 1]), 1e-300))
    if conformal:
        l = l - 0.25 * (lnJ[:, E[:, 0]] + lnJ[:, E[:, 1]])
    lt = _runmed(l); D = np.abs(lt - lt[0]).max(0)
    d2 = np.diff(l, 2, axis=0); sig = 1.4826 * np.median(np.abs(d2 - np.median(d2, 0)), 0) / np.sqrt(6.0)
    return D, sig, l


def clusters(tri, tri_e, active_e, min_tri=6):
    act_t = active_e[tri_e].all(1)
    import tcx
    adj = tcx.tri_adjacency(tri); ok = act_t[adj[:, 0]] & act_t[adj[:, 1]]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nT = len(tri); A = coo_matrix((np.ones(ok.sum()), (adj[ok, 0], adj[ok, 1])), shape=(nT, nT))
    _, lab = connected_components(A, directed=False)
    lab = np.where(act_t, lab, -1); out = []
    for c in np.unique(lab[lab >= 0]):
        T = np.where(lab == c)[0]
        if len(T) >= min_tri:
            out.append(T)
    return out


def cluster_rotation(fr, tri, tri_e, E, T, cum_ref):
    """median relative winding of the cluster's edges, its per-step series, and a noise scale."""
    ee = np.unique(tri_e[T].ravel())
    r = fr[:, E[ee, 1]] - fr[:, E[ee, 0]]
    th = np.unwrap(np.arctan2(r[..., 1], r[..., 0]), axis=0)
    w = th - th[0] - cum_ref[:, None]                    # relative to the ensemble rotation
    rho = np.median(w, 1)                                # cluster winding vs time
    step = np.diff(rho)
    d2 = np.diff(th, 2, axis=0); sig_th = 1.4826 * np.median(np.abs(d2 - np.median(d2, 0)), 0) / np.sqrt(6.0)
    nodes = np.unique(tri[T].ravel())
    sig_rho = 1.2533 * np.sqrt(2) * np.median(sig_th) / np.sqrt(len(nodes))
    return float(rho[-1]), step, float(sig_rho), nodes


def vortices(fr, tri, lnJ, cum_ref, delta=0.1, kappa=None, z=None, jmax=0.5, min_tri=6, conformal=True):
    S, N = fr.shape[:2]
    E, tri_e = edges_of(tri)
    D, sig, _ = edge_coherence(fr, E, lnJ, conformal)
    kappa = np.sqrt(2 * np.log(S)) if kappa is None else kappa
    active = D <= delta + kappa * 0.8 * sig             # 0.8 sig: noise of a difference of two 5-frame medians
    CL = clusters(tri, tri_e, active, min_tri)
    z = np.sqrt(2 * np.log(max(len(CL), 2))) + 2 if z is None else z
    out = []
    for T in CL:
        rot, step, srho, nodes = cluster_rotation(fr, tri, tri_e, E, T, cum_ref)
        tv = np.abs(step).sum() + 1e-300
        sustained = np.abs(step).max() <= jmax * tv
        signif = abs(rot) >= z * srho
        out.append(dict(tris=T, nodes=nodes, rot=rot, sig=srho, sustained=bool(sustained), significant=bool(signif),
                        single_share=float(np.abs(step).max() / tv), vortex=bool(sustained and signif)))
    return out, dict(active_frac=float(active.mean()), n_clusters=len(CL), D=D, sig=sig, E=E)


def outer_loop(P, tri, T):
    """outer boundary polygon (largest-area loop) of the triangle set T."""
    e = np.sort(np.concatenate([tri[T][:, [0, 1]], tri[T][:, [1, 2]], tri[T][:, [2, 0]]]), 1)
    u, c = np.unique(e, axis=0, return_counts=True); b = u[c == 1]
    nb = {}
    for a, bb in b:
        nb.setdefault(a, []).append(bb); nb.setdefault(bb, []).append(a)
    used = set(); loops = []
    for s0 in list(nb):
        if s0 in used:
            continue
        loop = [s0]; used.add(s0); prev = None; cur = s0
        while True:
            nxt = [x for x in nb[cur] if x != prev and (x not in used or (x == s0 and len(loop) > 2))]
            if not nxt:
                break
            x = nxt[0]
            if x == s0:
                break
            loop.append(x); used.add(x); prev, cur = cur, x
        if len(loop) >= 3:
            loops.append(np.array(loop))
    def area(L):
        x, y = P[L, 0], P[L, 1]; return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return max(loops, key=area) if loops else None
