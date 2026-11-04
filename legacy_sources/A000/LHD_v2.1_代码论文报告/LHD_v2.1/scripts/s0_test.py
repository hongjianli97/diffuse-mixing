import numpy as np, cases, time
from bcom import *
import stream as ST
c = cases.case_sod(h=0.004); fr = c["frames"]; ref = c["rho"][-1]; mask = c["interior"](fr[0]); rho0 = c["rho"][0]
t0 = time.time(); rho, dg = ST.ledger_stream(fr.copy(), rho0, calib="immediate"); t1 = time.time()
print("stream (no breaks)", errstats(rho[-1], ref, mask), {k: v for k, v in dg.items() if k.startswith("n_")}, "%.1fs" % (t1-t0))
lnJ, hot, bad, tau = ledger(fr); print("chained_v2      ", errstats(rho0*np.exp(-lnJ[-1]), ref, mask))
P, own, st, en = ST.tracks_from_particles(fr, 0.03, seed=1)
print("tracks", P.shape, "breaks/particle", (len(own)-fr.shape[1])/fr.shape[1])
rho, dg = ST.ledger_stream(P, np.where(~np.isnan(P[0, :, 0]), rho0[own], np.nan))
fin = ~np.isnan(P[-1, :, 0]); e = errstats(rho[-1][fin], c["rho"][-1][own[fin]], mask[own[fin]]); print("stream p=0.03", e, {k: v for k, v in dg.items() if k.startswith("n_")})
rw = ST.ledger_windowed(P, np.where(~np.isnan(P[0, :, 0]), rho0[own], np.nan))
okw = fin & np.isfinite(rw[-1]); print("windowed p=0.03", errstats(rw[-1][okw], c["rho"][-1][own[okw]], mask[own[okw]]), "usable", okw.sum()/fin.sum())
