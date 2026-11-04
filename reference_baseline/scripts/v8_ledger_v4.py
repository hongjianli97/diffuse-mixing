"""Ledger v4 (geometric-window detection, segment-wise guarded tensor increments, one-sided arc repair) against the
v2 ledger and the v3 hybrid on all cases.  Reports all tracers and, for the DNS cases, the 'clean' tracers whose DNS
passive scalar did not change (|Y1(T)-Y1(0)| <= 0.01): where the DNS has numerically mixed the tracer's fluid with the
other gas the DNS density is a mixed value and is not a valid reference.
usage: python scripts/v8_ledger_v4.py [key=value ...] case1 case2 ...   (tag=... names the result entry)"""
import sys, json, pickle, time, numpy as np
sys.path[:0] = ["lhd", "scripts"]
import classify as C, mfm
from vcases import load, st
from ledger_ub import chained_lnJ_ind, bad_from_hot, ledger_v3
from ledger_v4 import ledger_v4

RES = "results/res_v8_ledger_v4.json"
try: out = json.load(open(RES))
except Exception: out = {}
kw = dict(a.split("=") for a in sys.argv[1:] if "=" in a); tag = kw.pop("tag", "default")
noise = kw.pop("noise", None); skipv3 = kw.pop("skipv3", "0") == "1"
kw = {k: (tuple(int(x) for x in v.split(",")) if "," in v else (bool(int(v)) if k in ("geo", "arc_shock", "arc_iface") else float(v))) for k, v in kw.items()}
if "defer_cap" in kw: kw["defer_cap"] = int(kw["defer_cap"])
FILES = {"sb": "runs/sb_dns_dx0.005.pkl", "rm": "runs/rm_dns_dx0.005.pkl", "sbv": "runs/sbv_dns_dx0.005.pkl"}


def load2(nm):
    if nm == "planar_mid":
        c = mfm.planar_shock(h=0.02, eta=3.0, theta=0.5, u1=(-0.14, 0.26), Cs=1.0); S = len(c["t"]) // 2
        fr = c["frames"][:S]; return fr, np.ones(fr.shape[1]), np.exp(-c["lnJ_ex"][S - 1]), c["interior"](fr[0]), None
    if nm.startswith("visc"):                         # visc_<delta/h>_<Cs>[_mid]
        p = nm.split("_"); dh, cs = float(p[1]), float(p[2]); stop = 0.5 if nm.endswith("mid") else 1.0
        c = mfm.viscous_shock(h=0.02, eta=3.0, delta=dh * 0.02, Cs=cs, stop=stop); fr = c["frames"]
        return fr, c["rho0"], c["rho0"] * np.exp(-c["lnJ_ex"][-1]), c["interior"](fr[0]), None
    fr, rho0, ref, m = load(nm); clean = None
    if nm in FILES:
        Y = pickle.load(open(FILES[nm], "rb"))["E"]["Y1"]; clean = np.abs(Y[-1] - Y[0]) <= 0.01
    return fr, rho0, ref, m, clean


for nm in [a for a in sys.argv[1:] if "=" not in a]:
    t0 = time.time(); fr, rho0, ref, m, clean = load2(nm)
    if noise:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(fr[0]).query(fr[0], k=2); fr = fr + float(noise) * np.median(d[:, 1]) * np.random.default_rng(7).normal(size=fr.shape)
    E = lambda lj: rho0 * np.exp(-lj) / ref - 1
    o = out.get(nm + ("_noise" + noise if noise else ""), {})
    if "old_v2" not in o:
        lnJ2, _, _ = chained_lnJ_ind(fr, bad_from_hot(C.hot_steps(fr)[0])); o["old_v2"] = st(E(lnJ2[-1]), m)
        if clean is not None: o["old_v2_clean"] = st(E(lnJ2[-1]), m & clean)
    if "v3_hybrid" not in o and not skipv3:
        r3 = ledger_v3(fr); o["v3_hybrid"] = st(E(r3["lnJ_hybrid"]), m)
        if clean is not None: o["v3_hybrid_clean"] = st(E(r3["lnJ_hybrid"]), m & clean)
    import os, ledger_v4 as L4
    Ws = kw.get("Ws", L4.GEO_W); hf = "runs/hot_%s%s_%s_ad.npy" % (nm, "_noise" + noise if noise else "", "-".join(map(str, Ws)))
    if kw.get("geo", True) and "hot" not in kw:
        if os.path.exists(hf): hot = np.load(hf)
        else: hot = L4.hot_steps_geo(fr, Ws=Ws)[0]; np.save(hf, hot)
        kw2 = dict(kw, hot=hot)
    else: kw2 = kw
    t1 = time.time(); r = ledger_v4(fr, rho0=rho0, **kw2)
    res = dict(poly=st(E(r["lnJ_poly"]), m), v4=st(E(r["lnJ_v4"]), m), diag=r["diag"], secs=time.time() - t1)
    if clean is not None:
        res["poly_clean"] = st(E(r["lnJ_poly"]), m & clean); res["v4_clean"] = st(E(r["lnJ_v4"]), m & clean)
        res["clean_frac"] = float(clean[m].mean())
    o[tag] = res; o["n"] = int(m.sum()); out[nm + ("_noise" + noise if noise else "")] = o
    f = lambda d: "%.3f%%/%.2f%%" % (100 * d["med"], 100 * d["p95"])
    print("%-14s %-10s v2 %s | v3 %s | v4poly %s | v4 %s | tens %.2f arc %d defer %d  %.0fs" % (
        nm, tag, f(o["old_v2"]), f(o["v3_hybrid"]) if "v3_hybrid" in o else "-", f(res["poly"]), f(res["v4"]),
        res["diag"]["tensor_share"], res["diag"]["arc_repairs"], res["diag"]["max_defer"], time.time() - t0), flush=True)
    if clean is not None:
        print("%-14s clean %.2f  v2 %s | v3 %s | v4 %s" % ("", res["clean_frac"], f(o["old_v2_clean"]), f(o["v3_hybrid_clean"]) if "v3_hybrid_clean" in o else "-", f(res["v4_clean"])), flush=True)
    json.dump(out, open(RES, "w"), indent=1)
