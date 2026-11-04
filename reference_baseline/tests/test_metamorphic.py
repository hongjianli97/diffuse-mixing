"""Metamorphic tests of the ledger pipeline (bcom.ledger = C channel + chained_lnJ_v2) on the exact Sod case:
rigid motion with time-dependent rotation, tracer permutation, uniform scaling must leave ln J unchanged; time
reversal must recover the initial density.  Run: python tests/test_metamorphic.py"""
import sys, json, numpy as np
sys.path[:0] = ["lhd", "scripts"]
import cases
from bcom import ledger

c = cases.case_sod(h=0.004); fr = c["frames"]; S, N, _ = fr.shape; m = c["interior"](fr[0])
L0 = ledger(fr)[0]; res = {}
rot = lambda b: np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
fr_r = np.stack([fr[s] @ rot(0.3 + 0.05 * s).T + np.array([1.0 - 0.2 * s, 2.0 + 0.1 * s]) for s in range(S)])
res["rigid_motion_max_dlnJ"] = float(np.max(np.abs(ledger(fr_r)[0] - L0)))
perm = np.random.default_rng(0).permutation(N)
res["permutation_max_dlnJ"] = float(np.max(np.abs(ledger(fr[:, perm])[0][:, np.argsort(perm)] - L0)))
res["scaling_max_dlnJ"] = float(np.max(np.abs(ledger(7.3 * fr)[0] - L0)))
Lr = ledger(fr[::-1].copy())[0]                       # backwards from the final frame
rec = c["rho"][-1] * np.exp(-Lr[-1]); e = np.abs(rec / c["rho"][0] - 1)[m]
res["time_reversal_rho0_med"] = float(np.median(e)); res["time_reversal_rho0_p95"] = float(np.quantile(e, .95))
res["PASS"] = bool(res["rigid_motion_max_dlnJ"] < 1e-9 and res["permutation_max_dlnJ"] < 1e-9 and res["scaling_max_dlnJ"] < 1e-9
                   and res["time_reversal_rho0_p95"] < 0.02)
print(json.dumps(res, indent=1)); json.dump(res, open("results/res_t_metamorphic.json", "w"), indent=1)
