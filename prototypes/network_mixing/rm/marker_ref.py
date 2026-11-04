import numpy as np
xi = lambda y: 0.5 + 0.1 * np.cos(2 * np.pi * y)
def ref_curve(lb):
    return np.c_[xi(lb), lb]
def marker_metrics(M, lb, dbin=0.0025, dkeep=2e-5):
    # decimate: drop points whose label is within dkeep of the last kept one (spurious insertions)
    keep = [0]
    for i in range(1, len(lb)):
        if abs(lb[i] - lb[keep[-1]]) >= dkeep or i == len(lb) - 1: keep.append(i)
    M2, l2 = M[keep], lb[keep]
    ds = np.linalg.norm(np.diff(M2, axis=0), axis=1); lm = 0.5 * (l2[1:] + l2[:-1])
    edges = np.arange(0, 0.5 + dbin / 2, dbin); idx = np.clip(np.digitize(lm, edges) - 1, 0, len(edges) - 2)
    Lb = np.bincount(idx, weights=ds, minlength=len(edges) - 1)
    r0 = ref_curve(edges); ds0 = np.linalg.norm(np.diff(r0, axis=0), axis=1)
    lam = Lb / ds0
    yy = np.linspace(0, 0.5, 20001); L0 = np.trapezoid(np.sqrt(1 + (0.2 * np.pi * np.sin(2 * np.pi * yy)) ** 2), yy)
    return dict(lnL=float(np.log(ds.sum() / L0)), lnL_raw=float(np.log(np.linalg.norm(np.diff(M, axis=0), axis=1).sum() / L0)),
                ln_lam2=float(np.log((lam ** 2 * ds0).sum() / ds0.sum())), mean_lnlam=float((np.log(np.maximum(lam, 1e-12)) * ds0).sum() / ds0.sum()), lam=lam)
