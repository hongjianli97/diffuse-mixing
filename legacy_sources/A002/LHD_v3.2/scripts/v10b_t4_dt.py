"""Snapshot-interval dependence of the T4 split: position form vs DNS velocity circulation of the same material
triangles (hat kernel), both on the same record subsampled by `step` (step=1: dt=0.02; step=2: dt=0.04), detection
re-run on the subsampled record.  usage: python scripts/v10b_t4_dt.py nm step [pkl]"""
import sys, json, pickle, numpy as np
sys.path[:0] = ["lhd", "scripts"]
import lhdx as X, t4split as T4, ledger_v4 as L4
from lhdp import tri_area
nm, step = sys.argv[1], int(sys.argv[2]); mg = int(sys.argv[3]) if len(sys.argv) > 3 else 0; pk = {"sb": "runs/sb_dns_dx0.005.pkl", "sbv": "runs/sbv_dns_dx0.005.pkl"}[nm]
RES = "results/res_v10b_t4_dt.json"
try: out = json.load(open(RES))
except Exception: out = {}
D = pickle.load(open(pk, "rb")); fr, t, E = D["frames"][::step], D["t"][::step], {k: v[::step] for k, v in D["E"].items()}
hot = L4.hot_steps_geo(fr)[0]; P0 = fr[0]
tri0 = X.complex_peeled(P0)["tri"]; a0 = tri_area(P0, tri0); tri0 = np.where((a0 < 0)[:, None], tri0[:, [0, 2, 1]], tri0)
phi = np.hypot(P0[:, 0] - D["xb"], P0[:, 1]) - D["R"]; o = {}
for w in (0.15,):
    r = T4.split(fr, t, E["rho"], E["p"], hot, np.abs(phi) < w, tri0, margin=mg); triR = r["triR"]; Gf = np.zeros((len(t), len(triR)))
    for n in range(len(t)):
        Uv = np.c_[E["u"][n], E["v"][n]]
        for j in range(3):
            i0, i1 = triR[:, j], triR[:, (j + 1) % 3]; Gf[n] += (0.5 * (Uv[i0] + Uv[i1]) * (fr[n][i1] - fr[n][i0])).sum(1)
    dGv = 0.5 * (Gf[2:] - Gf[:-2]); th = r["th"]; i6 = int(np.argmin(np.abs(r["t"] - 0.6)))
    f = lambda x, i=None: float(-np.cumsum(x)[-1 if i is None else i])
    o["band %g" % w] = dict(dt=float(t[1] - t[0]), pos_imp=f(r["imp"]), pos_smo=f(r["smo"]), vel_imp=f((dGv * th).sum(1)), vel_smo=f((dGv * ~th).sum(1)),
                            pos_imp_06=f(r["imp"], i6), pos_smo_06=f(r["smo"], i6), vel_imp_06=f((dGv * th).sum(1), i6), vel_smo_06=f((dGv * ~th).sum(1), i6),
                            pos_total=f(r["dG"]), vel_total=f(dGv.sum(1)))
    q = o["band %g" % w]; print("%s margin=%d dt=%.3f: pos imp %.4f smo %.4f | vel imp %.4f smo %.4f | gap imp %.4f (t=0.6: pos %.4f/%.4f vel %.4f/%.4f)" % (
        nm, mg, q["dt"], q["pos_imp"], q["pos_smo"], q["vel_imp"], q["vel_smo"], q["pos_imp"] - q["vel_imp"], q["pos_imp_06"], q["pos_smo_06"], q["vel_imp_06"], q["vel_smo_06"]), flush=True)
out["%s_dt%.2f_margin%d" % (nm, t[1] - t[0], mg)] = o; json.dump(out, open(RES, "w"), indent=1)
