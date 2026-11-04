import numpy as np, cases
from bcom import *
import stream as ST
c = cases.case_sod(h=0.004); fr = c["frames"]; RHO = c["rho"]; S, N, _ = fr.shape
cross = np.full(N, -1); cross[c["ev_pidx"]] = c["ev_step"]; rng = np.random.default_rng(7)
forced = np.zeros((N, S), bool); sel = (cross >= 0) & (rng.random(N) < 0.3); forced[np.where(sel)[0], cross[sel]] = True
P, own, st, en = ST.tracks_from_particles(fr, 0.0, seed=3, forced=forced)
rinit = np.where(st == 0, RHO[0][own], np.nan)
r, dg = ST.ledger_stream(P, rinit, calib="deferred")
fin = ~np.isnan(P[-1, :, 0]); tr = RHO[-1][own]; m = c["interior"](fr[0])[own]
e = np.abs(r[-1]/tr-1)
for nm, g in [("orig", fin & (st == 0) & m), ("newborn", fin & (st > 0) & m)]:
    print(nm, g.sum(), np.nanmedian(e[g]), np.nanquantile(e[g], .95))
print("forced in shock", dg["n_forced_shock"], "shrunk", dg["n_shrunk"])
# newborn: where is the error? at birth frame vs truth
nb = np.where(fin & (st > 0) & m)[0][:2000]
eb = np.array([abs(r[st[j], j]/RHO[st[j], own[j]]-1) for j in nb]); print("newborn error at birth: med %.4f p95 %.4f" % (np.median(eb), np.quantile(eb, .95)))
# is the error in ln J after birth (ledger) or in calibration?
lj = dg["lnJ"]; ex = np.array([np.log(RHO[st[j], own[j]]/RHO[-1, own[j]]) for j in nb]); est_ = np.array([lj[-1, j]-lj[st[j], j] for j in nb])
print("newborn lnJ err med %.4f p95 %.4f" % (np.median(np.abs(est_-ex)), np.quantile(np.abs(est_-ex), .95)))
x0 = c["x0"][own[nb]]; print("birth frames", np.bincount(st[nb])[:10], "x0 range", x0.min(), x0.max())
bad = eb > 0.1; print("bad births:", bad.sum(), "their birth frames", np.bincount(st[nb][bad], minlength=S)[:S])
