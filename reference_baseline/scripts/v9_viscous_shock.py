"""Resolved viscous planar shock (1D Navier-Stokes DNS embedded in 2D, lhd/ns1d.py): does the shock detection find it,
and does the ledger telescope across it?  v2 (single-step C channel), v3 (multi-window w<=6), v4 (geometric windows).
Evaluated (a) at the final frame, when every tracer has crossed the shock, against the DNS density at the tracers,
and (b) mid-crossing, with the ledger re-run on the truncated record (no future information).
usage: python scripts/v9_viscous_shock.py [part=all|v2|v3|v4] Ms:delta_h:Cs [...]     e.g. 2.24:2:0.25
The DNS record is cached in runs/ns_<key>.npz; part=v2/v3/v4 evaluates one estimator only (300 s command limit)."""
import sys, json, time, numpy as np
sys.path[:0] = ["lhd", "scripts"]
import ns1d, classify as C
from vcases import st
from ledger_ub import chained_lnJ_ind, bad_from_hot, ledger_v3
from ledger_v4 import ledger_v4
RES = "results/res_v9_viscous_shock.json"
try: out = json.load(open(RES))
except Exception: out = {}
MPD = {1.5: 0.161, 2.24: 0.443}                     # mu / delta (fine-grid calibration, ns1d.calib)


PART = "all"
for a in sys.argv[1:]:
    if a.startswith("part="): PART = a.split("=")[1]


def evaluate(fr, rho0, ref, m, crossed, prev=None):
    o = dict(prev or {}); E = lambda lj: rho0 * np.exp(-lj) / ref - 1; arr = {}
    def put(k, lj, hot):
        o[k] = st(E(lj), m); o[k]["detected"] = float(hot[crossed & m].any(1).mean()) if (crossed & m).any() else None
    if PART in ("all", "v2"):
        hot2 = C.hot_steps(fr)[0]; l2, _, _ = chained_lnJ_ind(fr, bad_from_hot(hot2)); put("v2", l2[-1], hot2); arr["v2"] = l2[-1]
    if PART in ("all", "v3"):
        r3 = ledger_v3(fr); put("v3_polygon", r3["lnJ"][-1], r3["hot"]); put("v3_hybrid", r3["lnJ_hybrid"], r3["hot"]); o["v3_max_defer"] = r3["diag"]["max_defer"]
    if PART in ("all", "v4"):
        r4 = ledger_v4(fr, rho0=rho0); put("v4_polygon", r4["lnJ_poly"], r4["hot"]); put("v4", r4["lnJ_v4"], r4["hot"])
        o["v4_diag"] = {k: r4["diag"][k] for k in ("max_defer", "hot_frac", "tensor_share", "arc_repairs")}; arr["v4"] = r4["lnJ_v4"]; arr["v4p"] = r4["lnJ_poly"]
    return o, arr


def evaluate_old(fr, rho0, ref, m, crossed):
    o = {}
    hot2 = C.hot_steps(fr)[0]; l2, _, _ = chained_lnJ_ind(fr, bad_from_hot(hot2))
    r3 = ledger_v3(fr); r4 = ledger_v4(fr, rho0=rho0)
    E = lambda lj: rho0 * np.exp(-lj) / ref - 1
    for k, lj, hot in (("v2", l2[-1], hot2), ("v3_polygon", r3["lnJ"][-1], r3["hot"]), ("v3_hybrid", r3["lnJ_hybrid"], r3["hot"]),
                       ("v4_polygon", r4["lnJ_poly"], r4["hot"]), ("v4", r4["lnJ_v4"], r4["hot"])):
        o[k] = st(E(lj), m)
        o[k]["detected"] = float(hot[crossed & m].any(1).mean()) if (crossed & m).any() else None
    o["v4_diag"] = {k: r4["diag"][k] for k in ("max_defer", "hot_frac", "tensor_share", "arc_repairs")}
    o["v3_max_defer"] = r3["diag"]["max_defer"]
    return o, dict(v2=l2[-1], v4=r4["lnJ_v4"], v4p=r4["lnJ_poly"])


import os, pickle
for spec in [a for a in sys.argv[1:] if ":" in a]:
    Ms, dh, Cs = (float(x) for x in spec.split(":")); key = "Ms%g_d%g_Cs%g" % (Ms, dh, Cs)
    t0 = time.time(); cf = "runs/ns_%s.pkl" % key
    if os.path.exists(cf): D = pickle.load(open(cf, "rb"))
    else: D = ns1d.run(Ms=Ms, delta_h=dh, Cs=Cs, mu_per_delta=MPD); pickle.dump(D, open(cf, "wb"))
    fr, rho, h = D["frames"], D["rho"], D["h"]; X = fr[0]
    m = (X[:, 0] > 3 * h) & (X[:, 0] < 1 - 3 * h) & (X[:, 1] > 3 * h) & (X[:, 1] < 1 - 3 * h)
    prev = out.get(key, {})
    o = dict(prev); o.update(Ms=Ms, delta_h_target=dh, Cs=Cs, mu=D["mu"], dx=D["dx"], eta_RH=D["eta"], S=len(D["t"]),
             delta_measured_over_h=float(np.median(D["delta"][len(D["t"]) // 3:]) / h),
             dns_secs=time.time() - t0, nstep=D["nstep"])
    o["dns_post_density_vs_RH"] = float(np.median(rho[-1][m]) / D["eta"] - 1)
    xi = (fr - X[None] - D["slip"] * D["t"][:, None, None] * np.array([-D["n"][1], D["n"][0]])) @ D["n"] + D["xi0"][None]
    cr_all = xi < D["xs"][:, None] - 6 * D["delta"][:, None]
    o["final"], arrs = evaluate(fr, rho[0], rho[-1], m, cr_all[-1], prev.get("final"))
    o["final"]["crossed_frac"] = float(cr_all[-1][m].mean())
    if Cs >= 0.5:
        sm = int(np.argmin(np.abs(cr_all[:, m].mean(1) - 0.5)))
        o["mid"], am = evaluate(fr[:sm + 1], rho[0], rho[sm], m, cr_all[sm], prev.get("mid")); o["mid"]["frame"] = sm
        if "v4" not in am:
            o["secs"] = time.time() - t0; out[key] = o; json.dump(out, open(RES, "w"), indent=1); continue
        inlayer = np.abs(xi[sm] - D["xs"][sm]) < 2 * D["delta"][sm]
        o["mid"]["v4_outside_layer"] = st(rho[0] * np.exp(-am["v4"]) / rho[sm] - 1, m & ~inlayer)
        if (m & inlayer).sum() > 5: o["mid"]["v4_in_layer"] = st(rho[0] * np.exp(-am["v4"]) / rho[sm] - 1, m & inlayer)
        o["mid"]["n_in_layer"] = int((m & inlayer).sum())
        np.savez_compressed("runs/visc_%s_mid.npz" % key, xi=xi[sm], rho_dns=rho[sm], rho0=rho[0], v4=am["v4"], v4p=am["v4p"], m=m, xs=D["xs"][sm], delta=D["delta"][sm])
    o["secs"] = time.time() - t0; out[key] = o
    f = lambda d: "-" if d is None else "%.3f%%/%.2f%% det %s" % (100 * d["med"], 100 * d["p95"], "-" if d["detected"] is None else "%.2f" % d["detected"])
    print(key, "delta/h=%.2f S=%d dns %.0fs" % (o["delta_measured_over_h"], o["S"], o["dns_secs"]), "| final v2", f(o["final"].get("v2")), "v3", f(o["final"].get("v3_hybrid")), "v4", f(o["final"].get("v4")), "v4poly", f(o["final"].get("v4_polygon")), flush=True)
    if "mid" in o and "v4_outside_layer" in o["mid"]: print("    mid v2", f(o["mid"].get("v2")), "v3", f(o["mid"].get("v3_hybrid")), "v4", f(o["mid"].get("v4")), "| v4 outside layer %.3f%%/%.2f%%" % (100 * o["mid"]["v4_outside_layer"]["med"], 100 * o["mid"]["v4_outside_layer"]["p95"]), flush=True)
    json.dump(out, open(RES, "w"), indent=1)
