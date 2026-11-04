"""Old ledger (v2: single-step C channel, fixed deferral 4 x 5 frames) vs recommended ledger_v3 (multi-window detection
with per-tracer background, deferral cap from measured hot-run length, polygon / tensor / hybrid estimators).
usage: python scripts/v4_ledger_v3.py smooth cyl weak slow sod blast dg sb [noise]"""
import sys, json, numpy as np, time
sys.path[:0] = ["lhd", "scripts"]
import classify as C
from vcases import load, st
from ledger_ub import chained_lnJ_ind, ledger_v3, bad_from_hot, auc

RES = "results/res_v4_ledger_v3.json"
try: out = json.load(open(RES))
except Exception: out = {}
noise = "noise" in sys.argv
for nm in [a for a in sys.argv[1:] if a != "noise"]:
    t0 = time.time(); fr, rho0, ref, m = load(nm)
    if noise:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(fr[0]).query(fr[0], k=2); h = np.median(d[:, 1])
        fr = fr + 0.02 * h * np.random.default_rng(7).normal(size=fr.shape)
    hot1 = C.hot_steps(fr)[0]; lnJ2, I2, dg2 = chained_lnJ_ind(fr, bad_from_hot(hot1))
    r = ledger_v3(fr)
    E = lambda lj: rho0 * np.exp(-lj) / ref - 1
    o = dict(old_v2=st(E(lnJ2[-1]), m), v3_polygon=st(E(r["lnJ"][-1]), m), v3_tensor=st(E(r["lnJ_tensor"]), m),
             v3_hybrid=st(E(r["lnJ_hybrid"]), m), hybrid_tensor_share=r["hybrid_used"], max_defer=r["diag"]["max_defer"],
             hot_frac=r["diag"]["hot_frac"], fallbacks=r["diag"]["fallback_events"])
    e = np.abs(E(r["lnJ"][-1])); top = e >= np.quantile(e[m], .95); o["indicator_auc_top5"] = auc(r["ind"][-1][m], top[m])
    o["secs"] = time.time() - t0
    key = nm + ("_noise0.02h" if noise else ""); out[key] = o
    print(key, json.dumps({k: (round(v, 5) if isinstance(v, float) else ({kk: round(vv, 5) for kk, vv in v.items()} if isinstance(v, dict) else v)) for k, v in o.items()}), flush=True)
    json.dump(out, open(RES, "w"), indent=1)
