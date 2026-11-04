"""The three DNS cases not rerun in v2 (shock-vortex, single-mode RM, viscous shock-bubble dx=0.005):
old ledger vs ledger_v3 (polygon / tensor / hybrid), indicator AUC, optional 0.02h noise, and like-for-like circulation
(position form vs DNS velocity line integral along the same material boundary vs vertex-vorticity area quadrature).
usage: python scripts/v7_dns3.py sv rm sbv [noise]"""
import sys, json, pickle, numpy as np, time
sys.path[:0] = ["lhd", "scripts"]
import classify as C, lhdx as X
from vcases import load, st
from ledger_ub import chained_lnJ_ind, ledger_v3, bad_from_hot, auc
from lhdp import disp_cochain_tri, tri_area
RES = "results/res_v7_dns3.json"
try: out = json.load(open(RES))
except Exception: out = {}
noise = "noise" in sys.argv
FILES = {"sv": "runs/sv_dns.pkl", "rm": "runs/rm_dns_dx0.005.pkl", "sbv": "runs/sbv_dns_dx0.005.pkl"}


def circulation(nm):
    D = pickle.load(open(FILES[nm], "rb")); fr = D["fr_in"] if nm == "sv" else D["frames"]; E = D["Ev"] if nm == "sv" else D["E"]; t = D["t"]
    if nm == "sv": fr, t = fr[:31], t[:31]
    P0 = fr[0]; tri0 = X.complex_peeled(P0)["tri"]; A0 = tri_area(P0, tri0); tri0 = np.where((A0 < 0)[:, None], tri0[:, [0, 2, 1]], tri0)
    if nm == "sv": regs = {"core r0<2Rv": np.hypot(P0[:, 0] - 0.7, P0[:, 1] - 0.5) < 0.15, "core r0<Rv": np.hypot(P0[:, 0] - 0.7, P0[:, 1] - 0.5) < 0.075}
    elif nm == "rm":
        a0 = D["a0"]; phi = (P0[:, 0] - 0.5 - a0 * np.cos(2 * np.pi * P0[:, 1])) / np.sqrt(1 + (2 * np.pi * a0 * np.sin(2 * np.pi * P0[:, 1])) ** 2)
        regs = {"band 0.08": np.abs(phi) < 0.08, "band 0.03": np.abs(phi) < 0.03}
    else:
        phi = np.hypot(P0[:, 0] - D["xb"], P0[:, 1]) - D["R"]; regs = {"band %g" % w: np.abs(phi) < w for w in (0.04, 0.08, 0.15)}
    tm = 0.5 * (t[1:] + t[:-1]); o = {}
    for lab, reg in regs.items():
        triR = tri0[reg[tri0].all(1)]; e = np.vstack([triR[:, [0, 1]], triR[:, [1, 2]], triR[:, [2, 0]]])
        _, inv, cnt = np.unique(np.sort(e, 1), axis=0, return_inverse=True, return_counts=True); be = e[cnt[inv] == 1]
        G = []
        for s in range(len(t) - 1):
            c, _ = disp_cochain_tri(fr[s], fr[s + 1], triR); Pm = 0.5 * (fr[s] + fr[s + 1])
            U = 0.5 * np.c_[E["u"][s] + E["u"][s + 1], E["v"][s] + E["v"][s + 1]]; w = 0.5 * (E["om"][s] + E["om"][s + 1])
            G.append((c.sum() / (t[s + 1] - t[s]), (0.5 * (U[be[:, 0]] + U[be[:, 1]]) * (Pm[be[:, 1]] - Pm[be[:, 0]])).sum(), (w[triR].mean(1) * tri_area(Pm, triR)).sum()))
        G = np.array(G); sc = np.abs(G[:, 1]).max(); late = np.abs(G[:, 1]) > 0.2 * sc
        o[lab] = dict(med_rel_pos_vs_line=float(np.median(np.abs(G[late, 0] - G[late, 1]) / np.abs(G[late, 1]))),
                      max_rel_pos_vs_line=float(np.max(np.abs(G[late, 0] - G[late, 1]) / np.abs(G[late, 1]))),
                      med_rel_quad_vs_line=float(np.median(np.abs(G[late, 2] - G[late, 1]) / np.abs(G[late, 1]))),
                      final=[float(x) for x in G[-1]])
    return o


for nm in [a for a in sys.argv[1:] if a != "noise"]:
    t0 = time.time(); fr, rho0, ref, m = load(nm)
    if noise:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(fr[0]).query(fr[0], k=2); fr = fr + 0.02 * np.median(d[:, 1]) * np.random.default_rng(7).normal(size=fr.shape)
    lnJ2, I2, dg2 = chained_lnJ_ind(fr, bad_from_hot(C.hot_steps(fr)[0])); r = ledger_v3(fr)
    E_ = lambda lj: rho0 * np.exp(-lj) / ref - 1
    o = dict(old_v2=st(E_(lnJ2[-1]), m), v3_polygon=st(E_(r["lnJ"][-1]), m), v3_tensor=st(E_(r["lnJ_tensor"]), m), v3_hybrid=st(E_(r["lnJ_hybrid"]), m),
             hybrid_share=r["hybrid_used"], max_defer=r["diag"]["max_defer"], hot_frac=r["diag"]["hot_frac"], fallbacks=r["diag"]["fallback_events"] + dg2["fallback_events"])
    e = np.abs(E_(r["lnJ"][-1])); o["indicator_auc_top5"] = auc(r["ind"][-1][m], (e >= np.quantile(e[m], .95))[m])
    if not noise: o["circulation"] = circulation(nm)
    o["secs"] = time.time() - t0; key = nm + ("_noise0.02h" if noise else ""); out[key] = o
    print(key, json.dumps(o, default=lambda x: round(x, 5))[:1500], flush=True); json.dump(out, open(RES, "w"), indent=1)
