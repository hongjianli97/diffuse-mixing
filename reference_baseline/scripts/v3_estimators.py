"""Polygon ledger vs point-wise tensor ledger; single-step vs multi-window shock detection; indicator AUC.
usage: python scripts/v3_estimators.py smooth cyl weak sod blast dg sb  [noise]"""
import sys, json, pickle, numpy as np, time
sys.path[:0] = ["lhd", "scripts"]
import mfm, cases, classify as C, lhdx as X
from ledger_ub import chained_lnJ_ind, hot_steps_multi, tensor_lnJ, bad_from_hot, auc

RES = "results/res_v3_estimators.json"
try: out = json.load(open(RES))
except Exception: out = {}
noise = "noise" in sys.argv


from vcases import load, st


for nm in [a for a in sys.argv[1:] if a != "noise"]:
    t0 = time.time(); fr, rho0, ref, m = load(nm)
    if noise:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(fr[0]).query(fr[0], k=2); h = np.median(d[:, 1])
        fr = fr + 0.02 * h * np.random.default_rng(7).normal(size=fr.shape)
    S, N, _ = fr.shape
    hot1, R, tau = C.hot_steps(fr); hotm, _, _ = hot_steps_multi(fr)
    o = dict(hot_frac_single=float(hot1.mean()), hot_frac_multi=float(hotm.mean()))
    for lab, hot in (("single", hot1), ("multi", hotm)):
        bad = bad_from_hot(hot); lnJ, I, dg = chained_lnJ_ind(fr, bad)
        e = rho0 * np.exp(-lnJ[-1]) / ref - 1
        o["polygon_" + lab] = st(e, m); o["fallbacks_" + lab] = dg["fallback_events"]
        top = np.abs(e) >= np.quantile(np.abs(e[m]), .95)
        o["indicator_auc_top5_" + lab] = auc(I[-1][m], top[m])
        lt = tensor_lnJ(fr, bad)[S - 1]; et = rho0 * np.exp(-lt) / ref - 1
        o["tensor_" + lab] = st(et, m)
    o["secs"] = time.time() - t0
    key = nm + ("_noise0.02h" if noise else ""); out[key] = o
    print(key, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in o.items()}), flush=True)
    json.dump(out, open(RES, "w"), indent=1)
