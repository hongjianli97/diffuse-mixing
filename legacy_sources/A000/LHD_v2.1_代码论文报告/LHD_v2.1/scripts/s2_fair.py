import numpy as np, json, pickle, cases, sys
from bcom import *
import stream as ST
which = sys.argv[1]; p = float(sys.argv[2]); out = {}
if which == "sv":
    D = pickle.load(open("sv_dns.pkl", "rb")); fr = D["fr_in"]; RHO = D["Ev"]["rho"].astype(float); x = fr[-1]
    mask0 = (x[:, 0] > x[:, 0].min()+.02) & (x[:, 0] < x[:, 0].max()-.02) & (x[:, 1] > x[:, 1].min()+.02) & (x[:, 1] < x[:, 1].max()-.02)
    core = np.hypot(*(fr[0]-np.array([D["xv"], D["yv"]])).T) < 2*D["Rv"]
else:
    c = cases.case_sod(h=0.004); fr = c["frames"]; RHO = c["rho"]; mask0 = c["interior"](fr[0]); core = np.zeros(fr.shape[1], bool)
P, own, st, en = ST.tracks_from_particles(fr, p, seed=3); rinit = np.where(st == 0, RHO[0][own], np.nan)
rs, _ = ST.ledger_stream(P, rinit, calib="deferred"); rw = ST.ledger_windowed(P, rinit)
fin = ~np.isnan(P[-1, :, 0]) & mask0[own]; tr = RHO[-1][own]
es = np.abs(rs[-1]/tr-1); ew = np.abs(rw[-1]/tr-1)
com = fin & np.isfinite(ew) & np.isfinite(es); only = fin & ~np.isfinite(ew) & np.isfinite(es)
q = lambda e, m: [float(np.median(e[m])), float(np.quantile(e[m], .95)), int(m.sum())]
out = dict(common_stream=q(es, com), common_windowed=q(ew, com), stream_only_tracks=q(es, only),
           stream_orig=q(es, fin & (st == 0)), stream_newborn=q(es, fin & (st > 0)),
           stream_core=q(es, fin & core[own]) if core.any() else None, stream_outside_core=q(es, fin & ~core[own]) if core.any() else None,
           windowed_core=q(ew, com & core[own]) if core.any() else None)
print(which, p, json.dumps(out)); json.dump(out, open("res_s2_%s_%g.json" % (which, p), "w"), indent=1)
