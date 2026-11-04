import numpy as np, json, pickle, cases, sys, time
from bcom import *
import stream as ST
which = sys.argv[1]; out = {}
if which == "sod":
    c = cases.case_sod(h=0.004); fr = c["frames"]; RHO = c["rho"]; mask0 = c["interior"](fr[0])
    cross = np.full(fr.shape[1], -1); cross[c["ev_pidx"]] = c["ev_step"]
elif which == "blast":
    c = cases.case_blast(h=0.022); fr = c["frames"]; r0 = np.where(c["rs"] <= 0.2, 6.0, 1.0); RHO = r0[None, :]*np.exp(-c["lnJ_ex"]); mask0 = c["interior"](fr[0])
    cross = np.full(fr.shape[1], -1); cross[c["ev_pidx"]] = c["ev_step"]
elif which == "sv":
    D = pickle.load(open("sv_dns.pkl", "rb")); fr = D["fr_in"]; RHO = D["Ev"]["rho"].astype(float); x = fr[-1]
    mask0 = (x[:, 0] > x[:, 0].min()+.02) & (x[:, 0] < x[:, 0].max()-.02) & (x[:, 1] > x[:, 1].min()+.02) & (x[:, 1] < x[:, 1].max()-.02)
    lr = np.abs(np.log(RHO[1:]/RHO[:-1])).T; cross = np.where(lr.max(1) > np.log(1.15), lr.argmax(1), -1)
S, N, _ = fr.shape; rng = np.random.default_rng(7)
def evaluate(P, own, rho_est, newborn):
    fin = ~np.isnan(P[-1, :, 0]); est = np.isfinite(rho_est[-1]) & fin
    tr = RHO[-1][own]; m = mask0[own]
    o = dict(coverage=float(est.sum()/fin.sum()), all=errstats(rho_est[-1][est], tr[est], m[est]))
    nb = est & newborn
    if nb.sum() > 20: o["newborn"] = errstats(rho_est[-1][nb], tr[nb], m[nb]); o["n_newborn"] = int(nb.sum())
    # pooled over all observations (track, frame)
    pr = ~np.isnan(P[..., 0]) & np.isfinite(rho_est); T = RHO[:, own]
    e = np.abs(rho_est[pr]/T[pr]-1); o["pooled_med"] = float(np.median(e)); o["pooled_p95"] = float(np.quantile(e, .95))
    return o
lnJ0 = ledger(fr)[0]; out["no_breaks"] = errstats(RHO[0]*np.exp(-lnJ0[-1]), RHO[-1], mask0)
scen = [("p0.01", 0.01, None, fr), ("p0.03", 0.03, None, fr), ("p0.05", 0.05, None, fr)]
forced = np.zeros((N, S), bool); sel = (cross >= 0) & (rng.random(N) < 0.3); forced[np.where(sel)[0], cross[sel]] = True
scen.append(("shock_forced30%+p0.01", 0.01, forced, fr))
d, _ = cKDTree(fr[0]).query(fr[0], k=2); h = np.median(d[:, 1])
sub = np.sort(rng.choice(N, N//4, replace=False)); frn = fr.copy(); frn[:, sub] += 0.02*(2*h)*rng.normal(size=frn[:, sub].shape)
for nm, p, fo, F in scen+[("sparse1/4+0.02h+p0.03", 0.03, None, frn[:, sub])]:
    t0 = time.time()
    own_map = np.arange(F.shape[1]) if nm[:6] != "sparse" else sub
    P, own, st, en = ST.tracks_from_particles(F, p, seed=3, forced=(fo if fo is None else fo))
    own = own_map[own]; newborn = st > 0
    rinit = np.where(st == 0, RHO[0][own], np.nan)
    o = dict(breaks_per_particle=float(len(st)/F.shape[1]-1), tracks=int(len(st)))
    for meth in ["deferred"]:
        r, dg = ST.ledger_stream(P, rinit, calib=meth); o["stream_"+meth] = evaluate(P, own, r, newborn)
        o["stream_"+meth]["forced_reanchor_in_shock"] = dg["n_forced_shock"]
    rw = ST.ledger_windowed(P, rinit); o["windowed"] = evaluate(P, own, rw, newborn)
    out[nm] = o; o["time"] = time.time()-t0
    print(nm, json.dumps({k: (v["coverage"], round(v["all"]["med"], 4), round(v["all"]["p95"], 4), round(v.get("newborn", {}).get("p95", -1), 4)) if isinstance(v, dict) and "all" in v else v for k, v in o.items()}), flush=True)
    json.dump(out, open("res_s1_%s.json" % which, "w"), indent=1)
