"""lhdx.py -- benchmark runner around lhd.py.
* stars(P, cx, kind, rho_e): Hodge-star / edge-weight variants for the weight ablation
    edge  : 'cotan' (DEC) | 'bary' (dual/primal length) | 'unit' (combinatorial, Bianconi default)
            | 'gauss' (heat kernel exp(-l^2/eps), Belkin-Niyogi) | 'invlen2' (1/l^2)
    node  : barycentric star area (always; divergence is per unit area)
    rho_e : optional edge density -> mass-weighted inner product <f,g> = sum rho_e *1 f g
* run_lhd(frames, dt, ...): per-snapshot fresh Delaunay complex; node divergence, Stokes vorticity,
  Hodge energy fractions, optional pair-stretch attribution; Stokes-LAVD; chained material ln J.
* baselines: least-squares kNN velocity gradient (vorticity/divergence), LS flow-map FTLE,
  DTFE and adaptive SPH density.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import lhd as L

# ----------------------------------------------------------------------------- weights


def stars(P, cx, kind="cotan", rho_e=None, eps=1e-3):
    s0, s1c, _ = L.hodge_stars(P, cx, "cotan", eps)
    e = cx["edges"]
    le = np.linalg.norm(P[e[:, 1]] - P[e[:, 0]], axis=1) + 1e-300
    if kind == "cotan":
        s1 = s1c
    elif kind == "bary":
        s1 = L.hodge_stars(P, cx, "barycentric")[1]
    elif kind == "unit":
        s1 = np.ones(len(e))
    elif kind == "gauss":
        s1 = np.exp(-le ** 2 / np.median(le) ** 2)
    elif kind == "invlen2":
        s1 = 1.0 / le ** 2
    else:
        raise ValueError(kind)
    if rho_e is not None:
        s1 = s1 * rho_e
    return s0, s1


def divergence(f, cx, s0, s1):
    return L.node_divergence(f, cx, s0, s1)

# ----------------------------------------------------------------------------- main runner


def run_lhd(frames, dt, kinds=("cotan",), rho_frames=None, rho_kinds=(), pairs=None,
            bad=None, objective=False, max_len_fac=3.0, local_k=None, keep_fn=None,
            hodge_every=1, block=5, verbose=False, store_vec=(), regions=None, op="dec", cplx="median"):
    """frames (S,N,2).  bad (N,S) shock-stencil mask (censoring).  rho_frames (S,N) node density
    for the mass-weighted variants listed in rho_kinds.  pairs (M,2) fixed material pairs.
    Returns a dict of per-node time series and per-frame scalars."""
    S, N, _ = frames.shape
    V = np.gradient(frames, dt, axis=0)
    out = dict(div={k: np.full((S, N), np.nan) for k in kinds}, vort=np.full((S, N), np.nan),
               vort_c=np.full((S, N), np.nan), Efrac={}, wbar=np.zeros(S), t_snap=0.0)
    for k in list(kinds) + ["rho_" + k for k in rho_kinds]:
        out["Efrac"][k] = np.full((S, 3), np.nan)
    if pairs is not None:
        out["s_g"] = np.full((S, len(pairs)), np.nan, np.float32)
        out["s_c"] = np.full((S, len(pairs)), np.nan, np.float32)
        out["s_tot"] = np.full((S, len(pairs)), np.nan, np.float32)
    if regions is not None:
        out["Gam"] = {k: np.full((S, 4), np.nan) for k in regions}   # raw, clean-only, perimeter, area
    vecs = {s: {} for s in store_vec}
    t0 = time.time()
    for s in range(S):
        P = frames[s]
        Vs = L.relative_velocity(P, V[s]) if objective else V[s]
        cx = complex_peeled(P) if cplx == "peel" else L.complex_from_delaunay(P, max_len_fac, local_k=local_k, keep_fn=keep_fn)
        opn = open_star_nodes(cx)
        f = L.edge_cochain(P, Vs, cx["edges"])
        tri = cx["tri"]
        A = L.tri_signed_area(P, tri)
        wt = L.circulation(f, cx) / A
        out["wbar"][s] = np.sum(wt * A) / np.sum(A)
        out["vort"][s] = L.node_vorticity(P, cx, f)
        if regions is not None:
            circ = A * wt
            Le = L._tri_edge_lengths(P, tri); okL = Le.max(1) < max_len_fac * np.median(Le)
            clean = ~bad[tri, s].any(1) if bad is not None else np.ones(len(tri), bool)
            for kname, msk in regions.items():
                G_, per, area, _ = L.region_circulation(P, f, cx, msk, max_len_fac)
                inreg = msk[tri].all(1) & okL
                out["Gam"][kname][s] = [G_, float(circ[inreg & clean].sum()), per, area]
        if bad is not None:
            tri_ok = ~bad[tri, s].any(1)
            out["vort_c"][s] = L.node_vorticity(P, cx, f, tri_ok=tri_ok.astype(float))
        if op == "galerkin":
            dv = galerkin_div(P, cx, Vs); dv[opn] = np.nan
            out["div"][kinds[0]][s] = dv
            if s % hodge_every == 0:
                g = galerkin_hodge(P, cx, Vs)
                out["Efrac"][kinds[0]][s] = [g["Eg"], g["Ec"], g["Eh"]]
                if pairs is not None:
                    ug, uc = face_to_node(P, cx, g["ug"]), face_to_node(P, cx, g["uc"])
                    out["s_g"][s] = L.stretch_rate(P, ug, pairs); out["s_c"][s] = L.stretch_rate(P, uc, pairs)
                    out["s_tot"][s] = L.stretch_rate(P, Vs, pairs)
                if s in vecs:
                    vecs[s]["ug"] = face_to_node(P, cx, g["ug"]); vecs[s]["uc"] = face_to_node(P, cx, g["uc"])
                if rho_frames is not None:
                    r = rho_frames[s]; gr = galerkin_hodge(P, cx, Vs, rho_t=r[cx["tri"]].mean(1))
                    out["Efrac"]["rho_" + kinds[0]][s] = [gr["Eg"], gr["Ec"], gr["Eh"]]
                    if s in vecs:
                        vecs[s]["ug_rho"] = face_to_node(P, cx, gr["ug"])
        for k in (kinds if op != "galerkin" else ()):
            s0, s1 = stars(P, cx, k)
            dv = divergence(f, cx, s0, s1); dv[opn] = np.nan
            out["div"][k][s] = dv
            if s % hodge_every == 0:
                dec = L.Hodge(cx, s1)(f)
                out["Efrac"][k][s] = [dec["Eg"] / dec["E"], dec["Ec"] / dec["E"], dec["Eh"] / dec["E"]]
                if k == "cotan" and pairs is not None:
                    ug = L.node_vectors(P, cx, dec["fg"])
                    uc = L.node_vectors(P, cx, dec["fc"])
                    out["s_g"][s] = L.stretch_rate(P, ug, pairs)
                    out["s_c"][s] = L.stretch_rate(P, uc, pairs)
                    out["s_tot"][s] = L.stretch_rate(P, Vs, pairs)
                if k == "cotan" and s in vecs:
                    vecs[s]["ug"] = L.node_vectors(P, cx, dec["fg"])
                    vecs[s]["uc"] = L.node_vectors(P, cx, dec["fc"])
                    vecs[s]["uh"] = L.node_vectors(P, cx, dec["fh"])
        if rho_frames is not None and s % hodge_every == 0 and op != "galerkin":
            r = rho_frames[s]
            rho_e = 0.5 * (r[cx["edges"][:, 0]] + r[cx["edges"][:, 1]])
            for k in rho_kinds:
                s0, s1 = stars(P, cx, k, rho_e=rho_e)
                dec = L.Hodge(cx, s1)(f)
                out["Efrac"]["rho_" + k][s] = [dec["Eg"] / dec["E"], dec["Ec"] / dec["E"], dec["Eh"] / dec["E"]]
                if s in vecs:
                    vecs[s]["ug_rho"] = L.node_vectors(P, cx, dec["fg"])
        if verbose and s % 10 == 0:
            print("  frame %d/%d  %.1fs" % (s, S, time.time() - t0), flush=True)
    out["t_snap"] = (time.time() - t0) / S
    out["vecs"] = vecs
    # Stokes-LAVD (relative to the area-weighted mean), raw and censored
    w = np.nan_to_num(out["vort"] - out["wbar"][:, None])
    out["lavd"] = np.trapezoid(np.abs(w), dx=dt, axis=0)
    if bad is not None:
        wc = out["vort_c"] - out["wbar"][:, None]
        ok = np.isfinite(wc) & ~bad.T
        out["lavd_c"] = np.trapezoid(np.where(ok, np.abs(wc), 0.0), dx=dt, axis=0)
    if op == "galerkin":
        out["lnJ"] = chained_lnJ_v2(frames, bad, block=block)
    else:
        out["lnJ"] = L.chained_log_jacobian(np.transpose(frames, (1, 0, 2)), block=block,
                                            max_len_fac=max_len_fac, local_k=local_k).T  # (S,N)
    return out

# ----------------------------------------------------------------------------- baselines


def ls_gradient(P, U, k=12):
    """Least-squares velocity gradient G[i] = d U / d x (2x2) from k nearest neighbours."""
    _, nb = cKDTree(P).query(P, k=k + 1)
    nb = nb[:, 1:]
    dX = P[nb] - P[:, None, :]
    dU = U[nb] - U[:, None, :]
    A = np.einsum("nki,nkj->nij", dX, dX) + 1e-14 * np.eye(2)
    B = np.einsum("nki,nkj->nij", dX, dU)            # sum dx_i du_j
    G = np.linalg.solve(A, B)                          # G[n,i,j] = d u_j / d x_i
    return np.transpose(G, (0, 2, 1))                  # [j,i] = du_j/dx_i


def ls_vort_div(frames, dt, k=12):
    S, N, _ = frames.shape
    V = np.gradient(frames, dt, axis=0)
    om = np.zeros((S, N)); th = np.zeros((S, N))
    for s in range(S):
        G = ls_gradient(frames[s], V[s], k)
        om[s] = G[:, 1, 0] - G[:, 0, 1]
        th[s] = G[:, 0, 0] + G[:, 1, 1]
    return om, th


def ls_ftle(X0, X1, T, k=12):
    """FTLE from the least-squares flow-map gradient (standard on scattered tracers)."""
    F = ls_gradient(X0, X1, k)
    C = np.einsum("nki,nkj->nij", F, F)
    lam = np.linalg.eigvalsh(C)[:, -1]
    return np.log(np.maximum(lam, 1e-300)) / (2 * abs(T))


def star_area(P, cx=None, max_len_fac=3.0):
    if cx is None:
        cx = L.complex_from_delaunay(P, max_len_fac)
    A = np.abs(L.tri_signed_area(P, cx["tri"]))
    s = np.zeros(len(P))
    np.add.at(s, cx["tri"].ravel(), np.repeat(A, 3))
    return s


def dtfe_density(P, m):
    """Delaunay Tessellation Field Estimator (Schaap & van de Weygaert 2000): rho_i = (d+1) m_i / W_i,
    W_i = area of the Delaunay star of i."""
    W = star_area(P)
    with np.errstate(all="ignore"):
        return np.where(W > 0, 3.0 * m / W, np.nan)


def sph_density(P, m, k=24):
    """Adaptive-kernel SPH density sum (cubic spline, h_i = d_k(i)/2, gather)."""
    d, nb = cKDTree(P).query(P, k=k)
    h = d[:, -1] / 2.0
    q = d / h[:, None]
    sig = 10.0 / (7.0 * np.pi * h ** 2)
    W = np.where(q < 1, 1 - 1.5 * q ** 2 + 0.75 * q ** 3, np.where(q < 2, 0.25 * (2 - q) ** 3, 0.0))
    return (m[nb] * W).sum(1) * sig


def lavd_from_series(om, dt, A_weights=None):
    """LAVD = int |omega - <omega>| dt with the ensemble mean (equal weights unless given)."""
    if A_weights is None:
        wb = np.nanmean(om, axis=1)
    else:
        wb = np.nansum(om * A_weights, 1) / np.nansum(A_weights, 1)
    return np.trapezoid(np.abs(np.nan_to_num(om - wb[:, None])), dx=dt, axis=0)

# ----------------------------------------------------------------------------- metrics


def corr(a, b, m=None):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if m is not None:
        ok &= m
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def spearman(a, b, m=None):
    from scipy.stats import spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if m is not None:
        ok &= m
    return float(spearmanr(a[ok], b[ok])[0])


def relL2(est, ref, m=None, fit_scale=False):
    est = np.asarray(est, float); ref = np.asarray(ref, float)
    ok = np.isfinite(est) & np.isfinite(ref)
    if m is not None:
        ok &= m
    e, r = est[ok], ref[ok]
    if fit_scale:
        e = e * (e @ r) / max(e @ e, 1e-300)
    return float(np.sqrt(np.mean((e - r) ** 2)) / max(np.sqrt(np.mean(r ** 2)), 1e-300))


def jaccard_top(a, b, q=0.9, m=None):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if m is not None:
        ok &= m
    A = a[ok] >= np.quantile(a[ok], q); B = b[ok] >= np.quantile(b[ok], q)
    return float((A & B).sum() / max((A | B).sum(), 1))

# ----------------------------------------------------------------------------- Galerkin (consistent) inner product
def _grad_lambda(P, tri):
    """per-face gradients of the three barycentric coordinates (F,3,2) and areas (F,)."""
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    A = 0.5 * L._cross(b - a, c - a)
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
    return dict(u=u, ug=ug, uc=uc, uh=uh, Eg=en(ug) / E, Ec=en(uc) / E, Eh=en(uh) / E, A=A, orth=float((w * (ug * uc).sum(1)).sum()) / E,
                phi=phi, psi=psi)


def face_to_node(P, cx, Fv):
    tri = cx["tri"]; A = np.abs(L.tri_signed_area(P, tri))
    U = np.zeros((cx["N"], Fv.shape[1])); W = np.zeros(cx["N"])
    for k in range(3):
        np.add.at(U, tri[:, k], Fv * A[:, None]); np.add.at(W, tri[:, k], A)
    with np.errstate(all="ignore"):
        return np.where(W[:, None] > 0, U / W[:, None], np.nan)


def complex_peeled(P, fac=2.5, k=8):
    """Delaunay complex peeled ONLY from the outside: a triangle is removed when it touches the current
    outer boundary through an edge longer than fac x the local k-NN spacing; repeat.  Interior triangles are
    never removed, so no internal holes (open stars) appear inside stretched regions."""
    from scipy.spatial import Delaunay
    P = np.asarray(P, np.float64)
    tri = Delaunay(P).simplices.astype(np.int64)
    neg = L.tri_signed_area(P, tri) < 0
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
    return L._assemble(tri[alive], len(P))


def open_star_nodes(cx):
    cnt = np.bincount(cx["tri_e"].ravel(), minlength=len(cx["edges"]))
    m = np.zeros(cx["N"], bool); m[cx["edges"][cnt == 1].ravel()] = True
    return m


def _star(P, tri, N):
    A = np.abs(L.tri_signed_area(P, tri)); s = np.zeros(N)
    np.add.at(s, tri.ravel(), np.repeat(A, 3)); return s


def chained_lnJ_v2(frames, bad=None, block=5, max_defer=4, cplx_fn=None, diag=None):
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
        if diag is not None:                      # make the silent fallback visible (non-finite star-area ratio)
            nb_ = ~np.isfinite(inc); diag["fallback_events"] = diag.get("fallback_events", 0) + int(nb_.sum())
            diag.setdefault("fallback_mask", np.zeros(N, bool)); diag["fallback_mask"] |= nb_
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
