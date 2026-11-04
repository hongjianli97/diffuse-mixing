"""Like-for-like reference for the shock-bubble circulation.

Position form  : sum over the material band triangles of Re(d1 c) / dt  (= boundary sum, Theorem 1)
DNS line       : trapezoidal line integral of the DNS velocity (sampled at the tracers) along the SAME material
                 polygon boundary at mid-step                                    -> same region, same boundary
DNS area quad  : vertex-averaged DNS vorticity x mid-step triangle area (old reference in g6_sz.py)
DNS domain     : grid sum of DNS vorticity over the whole upper half plane       (old 'agrees to 1.5%' comparison)
"""
import sys, json, pickle, numpy as np
sys.path[:0] = ["lhd", "scripts"]
import lhdx as X
from lhdp import disp_cochain_tri, tri_area

f = sys.argv[1] if len(sys.argv) > 1 else "runs/sb_dns_dx0.005.pkl"
D = pickle.load(open(f, "rb")); fr = D["frames"]; t = D["t"]; E = D["E"]; S = len(t)
P0 = fr[0]; phi = np.hypot(P0[:, 0] - D["xb"], P0[:, 1]) - D["R"]; tri0 = X.complex_peeled(P0)["tri"]
# orient triangles counter-clockwise at t0
A0 = tri_area(P0, tri0); tri0 = np.where((A0 < 0)[:, None], tri0[:, [0, 2, 1]], tri0)


def boundary_edges(tri):
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    key = np.sort(e, 1); u, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return e[cnt[inv] == 1]                               # oriented boundary edges (interior edges cancel)


out = dict(file=f)
tm = 0.5 * (t[1:] + t[:-1])
for wb in [0.04, 0.08, 0.15]:
    reg = np.abs(phi) < wb; triR = tri0[reg[tri0].all(1)]; be = boundary_edges(triR)
    Gp, Gl, Gq = [], [], []
    for s in range(S - 1):
        c, _ = disp_cochain_tri(fr[s], fr[s + 1], triR); Gp.append(c.sum() / (t[s + 1] - t[s]))
        Pm = 0.5 * (fr[s] + fr[s + 1])
        U = 0.5 * np.c_[E["u"][s] + E["u"][s + 1], E["v"][s] + E["v"][s + 1]]
        a, b = be[:, 0], be[:, 1]
        Gl.append(float((0.5 * (U[a] + U[b]) * (Pm[b] - Pm[a])).sum()))
        w = 0.5 * (E["om"][s] + E["om"][s + 1]); Gq.append(float((w[triR].mean(1) * tri_area(Pm, triR)).sum()))
    Gp, Gl, Gq = map(np.array, (Gp, Gl, Gq)); late = tm > 0.4
    rel = lambda a, b: float(np.median(np.abs(a[late] - b[late]) / np.abs(b[late])))
    out["band%g" % wb] = dict(
        at_t=[0.6, 1.0, 1.4],
        G_position=[float(np.interp(x, tm, Gp)) for x in (0.6, 1.0, 1.4)],
        G_dns_line=[float(np.interp(x, tm, Gl)) for x in (0.6, 1.0, 1.4)],
        G_dns_areaquad=[float(np.interp(x, tm, Gq)) for x in (0.6, 1.0, 1.4)],
        relerr_med_position_vs_line=rel(Gp, Gl), relerr_med_areaquad_vs_line=rel(Gq, Gl),
        relerr_max_position_vs_line=float(np.max(np.abs(Gp[late] - Gl[late]) / np.abs(Gl[late]))),
        n_boundary_edges=int(len(be)))
    print("band %.2f" % wb, json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else v) for k, v in out["band%g" % wb].items()}), flush=True)
dx = D["dx"]; out["dns_domain_upper_half"] = {str(k): float(v["om"].sum() * dx * dx) for k, v in D["grids"].items()}
print("domain", out["dns_domain_upper_half"])
json.dump(out, open("results/res_v1_circ_line.json", "w"), indent=1)
