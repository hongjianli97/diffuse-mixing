"""shared helpers"""
import numpy as np, paper_ell as PE, lhdx as X
def gkcorr(P, a, b, box, h):
    ZA, _ = PE.gk_grid(P, a, box, n=(230, 205), sigph=2 * h); ZB, _ = PE.gk_grid(P, b, box, n=(230, 205), sigph=2 * h)
    m = np.isfinite(ZA) & np.isfinite(ZB); return float(np.corrcoef(ZA[m], ZB[m])[0, 1])
def box_of(P): return (P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max())
def spacing(P): return float(np.sqrt((np.ptp(P[:, 0]) * np.ptp(P[:, 1])) / len(P)))
def events(fr):
    S, N = fr.shape[:2]; dl = np.zeros((S - 1, N))
    for s in range(S - 1):
        tri = X.complex_peeled(fr[s])['tri']; A0, A1 = X._star(fr[s], tri, N), X._star(fr[s + 1], tri, N)
        with np.errstate(all='ignore'): dl[s] = np.where(A0 > 0, np.log(A1 / A0), 0.0)
    det = dl < -0.2; det[:-1] |= (dl[:-1] + dl[1:]) < -0.2
    cs = det.T.copy(); cs[:, 1:] |= det.T[:, :-1]; cs[:, :-1] |= det.T[:, 1:]
    return cs                                                   # (N, S-1) censored steps
