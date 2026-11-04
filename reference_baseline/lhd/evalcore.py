import numpy as np
from scipy.spatial import cKDTree
import lhdx as X, afrc_h_engine as AH
import stretch_est as SE


def lsq_grad(P, phi, k=12, h=None, quad=True, hess=False):
    """kNN quadratic least-squares gradient of nodal field phi at every node (current positions);
    with hess, also returns h*|Hessian| (Frobenius) as the covector error indicator numerator."""
    _, nb = cKDTree(P).query(P, k + 1); nb = nb[:, 1:]
    A = P[nb] - P[:, None]; B = (phi[nb] - phi[:, None])[..., None]
    h = np.median(np.linalg.norm(A[:, 0], axis=1)) if h is None else h
    if quad:
        x, y = A[..., 0] / h, A[..., 1] / h
        Phi = np.concatenate([A, np.stack([x * x, x * y, y * y], -1) * h], -1)
    else:
        Phi = A
    M = np.einsum("nki,nkj->nij", Phi, Phi); b = np.einsum("nki,nkj->nij", Phi, B)
    M += 1e-12 * np.trace(M, axis1=1, axis2=2)[:, None, None] * np.eye(M.shape[1])
    S = np.linalg.solve(M, b)[..., 0]
    if hess:
        return S[:, :2], np.sqrt(4 * S[:, 2] ** 2 + 2 * S[:, 3] ** 2 + 4 * S[:, 4] ** 2)
    return S[:, :2]


def evaluate(frames, h, phi0, gX, band, band_sorted, closed, lnJ, eval_frames, bad=None, block=5, Astar=3.0,
             field_frames=(), w=None, side=None, t0=None):
    """frames (S,N,2); phi0 (N,) Lagrangian label (signed distance at frame 0); gX (N,) |grad_X phi0|;
    band: interface tracers; band_sorted: same sorted by the initial curve parameter (polyline);
    lnJ (S,N) material ledger.  Returns per-frame dict of band means and per-tracer arrays."""
    S, N, _ = frames.shape
    P0 = frames[0]
    # initial tangent of the phi0 level set: rotate the LS gradient of phi0 at t0
    if t0 is None:
        g0 = lsq_grad(P0, phi0, h=h)
        t0 = g0 @ SE.R90.T; t0 /= np.linalg.norm(t0, axis=1)[:, None]
    allf = sorted(set(eval_frames) | set(field_frames))
    fidx = None if field_frames else np.asarray(band)
    FF = SE.chained_F(frames, h, bad=bad, block=block, keep=allf, idx=fidx)
    if fidx is not None:   # embed band-only results into full-size arrays
        for s in FF:
            Fb, eb = FF[s]; Ff = np.tile(np.eye(2), (N, 1, 1)); ef_ = np.zeros(N); Ff[fidx] = Fb; ef_[fidx] = eb; FF[s] = (Ff, ef_)
    E = AH._mutual_knn(P0, 12)
    pos = {v: j for j, v in enumerate(band)}
    out, fields = {}, {}
    for s in allf:
        P = frames[s]
        cx = X.complex_peeled(P); tri = cx["tri"]
        lc_p1, g = SE.covector_lnlam(P, phi0, gX, lnJ[s], tri)
        gq, hH = lsq_grad(P, phi0, h=h, hess=True)
        lc_q = lnJ[s] + np.log(np.linalg.norm(gq, axis=1)) - np.log(gX)
        lv, A = SE.vector_lnlam(FF[s][0], t0)
        e_v = A * np.sqrt(FF[s][1]); e_c = hH / np.maximum(np.linalg.norm(gq, axis=1), 1e-300)
        if s in field_frames:
            fields[s] = dict(P=P, tri=tri, cov=lc_p1, cov_q=lc_q, vec=lv, A=A, comp=-lnJ[s], e_v=e_v, e_c=e_c)
        if s not in eval_frames:
            continue
        b = band
        pair = SE.pair_lnlam(frames, band, t0[band], s, E=E)
        poly_sorted, poly_total = SE.polyline_lnlam(frames, band_sorted, s, closed)
        poly = np.empty(len(band)); poly[[pos[v] for v in band_sorted]] = poly_sorted
        comb = np.where(e_v[b] <= e_c[b], lv[b], lc_p1[b])
        ww = None if w is None else w[b]
        d = dict(pair=pair, poly=poly, vector=lv[b], cov_q=lc_q[b], covector=lc_p1[b], combined=comb, A=A[b], comp=-lnJ[s][b], e_v=e_v[b], e_c=e_c[b])
        res = {k: SE.wmean(v, ww) for k, v in d.items() if k not in ("A", "e_v", "e_c")}
        res["A_median"] = float(np.nanmedian(A[b])); res["frac_cov"] = float(np.mean(e_v[b] > e_c[b]))
        res["poly_total"] = poly_total
        for k in ("vector", "covector", "combined"):
            lam = np.exp(d[k]); res[k + "_total"] = float(np.log(SE.wmean(lam, ww)))
        if side is not None:
            for nm, sel in (("in", side[b] < 0), ("out", side[b] > 0)):
                for k in ("covector", "vector", "combined"):
                    res[k + "_" + nm] = SE.wmean(d[k][sel], None if ww is None else ww[sel])
        out[s] = dict(mean=res, per=d)
    return out, fields, t0
