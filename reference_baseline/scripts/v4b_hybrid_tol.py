"""hybrid tolerance sweep and position-noise robustness of polygon / tensor / hybrid ledgers.
usage: python scripts/v4b_hybrid_tol.py cyl smooth sod blast dg sb [noise]"""
import sys, json, numpy as np
sys.path[:0] = ["lhd", "scripts"]
from vcases import load, st
from ledger_ub import ledger_v3
RES = "results/res_v4b_hybrid.json"
try: out = json.load(open(RES))
except Exception: out = {}
noise = "noise" in sys.argv
for nm in [a for a in sys.argv[1:] if a != "noise"]:
    fr, rho0, ref, m = load(nm)
    if noise:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(fr[0]).query(fr[0], k=2); fr = fr + 0.02 * np.median(d[:, 1]) * np.random.default_rng(7).normal(size=fr.shape)
    r = ledger_v3(fr); lp, lt = r["lnJ"][-1], r["lnJ_tensor"]; E = lambda lj: rho0 * np.exp(-lj) / ref - 1
    o = dict(polygon=st(E(lp), m), tensor=st(E(lt), m))
    for tol in (0.05, 0.1, 0.2, 0.5):
        ok = np.isfinite(lt) & (np.abs(lt - lp) <= tol); o["hybrid_tol%g" % tol] = dict(st(E(np.where(ok, lt, lp)), m), share=float(ok[m].mean()))
    key = nm + ("_noise0.02h" if noise else ""); out[key] = o
    print(key, " | ".join("%s %.2e/%.2e" % (k, v["med"], v["p95"]) for k, v in o.items()), flush=True)
    json.dump(out, open(RES, "w"), indent=1)
