"""3D tetrahedral ledger on manufactured maps (lhd/ledger3d.py): planar shocks (exactness with deferral, failure
without), and a smooth compressible rotational map (order of convergence)."""
import sys, json, time, numpy as np
sys.path[:0] = ["lhd"]
import ledger3d as L
out = {}; t0 = time.time()
inner = lambda X, h: np.all((X > 2.5 * h) & (X < 1 - 2.5 * h), 1)
for eta, Cs in ((3.0, 1.0), (1.5, 2.0), (6.0, 1.0)):
    fr, lnJ, X = L.planar_shock3(h=1 / 16, eta=eta, Cs=Cs); m = inner(X, 1 / 16)
    l1, hot = L.ledger(fr); l0, _ = L.ledger(fr, defer=False, hot=hot)
    e1 = np.abs(np.exp(lnJ - l1) - 1)[m]; e0 = np.abs(np.exp(lnJ - l0) - 1)[m]
    out["planar eta=%g Cs=%g" % (eta, Cs)] = dict(max_err_deferral=float(e1.max()), max_err_no_deferral=float(e0.max()),
                                                   detected=float(hot[m].any(1).mean()), n=int(m.sum()), S=len(fr))
    print("planar eta=%g Cs=%g: max err deferral %.1e, without %.1f%%, detected %.2f (n=%d, S=%d)" % (eta, Cs, e1.max(), 100 * e0.max(), hot[m].any(1).mean(), m.sum(), len(fr)), flush=True)
hs, er = [], []
for h in (1 / 10, 1 / 14, 1 / 20, 1 / 28):
    fr, lnJ, X = L.smooth3(h=h); m = inner(X, h); l1, _ = L.ledger(fr)
    hs.append(h); er.append(float(np.median(np.abs(l1 - lnJ)[m]))); print("smooth h=%.3f median |dlnJ| %.2e" % (h, er[-1]), flush=True)
order = float(np.polyfit(np.log(hs), np.log(er), 1)[0])
out["smooth"] = dict(h=hs, median_abs_dlnJ=er, order=order); print("smooth observed order %.2f   %.0fs" % (order, time.time() - t0))
json.dump(out, open("results/res_v12_ledger3d.json", "w"), indent=1)
