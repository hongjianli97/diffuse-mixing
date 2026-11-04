"""Quadratic (P2) image of the boundary: least-squares quadratic map on the k nearest t0 tracers of each vertex."""
import json, numpy as np, cases, cases_mix as CM, tcx
from scipy.spatial import cKDTree
from verify_perim import dens, adv
def quad_image(P0, PT, Q, k=12):
    tr = cKDTree(P0); _, idx = tr.query(Q, k); out = np.empty_like(Q)
    for i, (q, nb) in enumerate(zip(Q, idx)):
        d = P0[nb] - q; A = np.c_[np.ones(k), d, d[:, 0]**2, d[:, 0]*d[:, 1], d[:, 1]**2]
        cf, *_ = np.linalg.lstsq(A, PT[nb], rcond=None); out[i] = cf[0]
    return out
def perim(Q): return float(np.linalg.norm(np.diff(np.vstack([Q, Q[:1]]), axis=0), axis=1).sum())
res = {}
for nm, N in [("LO", 8000), ("LO", 32000), ("DG", None)]:
    c = cases.case_lo(a=1.0, Rd=6.5, N=N) if nm == "LO" else cases.case_dg(); vel = CM.lo_vel if nm == "LO" else cases.dg_fields()[0]
    fr = c["frames"]; S = len(fr); t = c["t"]; mus, _ = tcx.mu_threshold(fr); Ks = [tcx.cplx(fr[s])["tri"] for s in range(S)]
    phi = tcx.rotation_fields(tcx.run_chain(fr, mus, Ks=Ks[:-1]))[0]; tri0 = Ks[0]; E0 = tcx.graph_edges(tri0); n = len(phi)
    deg = np.bincount(E0.ravel(), minlength=n).astype(float); g = phi.copy()
    for _ in range(2):
        s_ = np.zeros(n); np.add.at(s_, E0[:, 0], g[E0[:, 1]]); np.add.at(s_, E0[:, 1], g[E0[:, 0]]); g = .5*g + .5*s_/np.maximum(deg, 1)
    P0 = fr[0]
    if nm == "LO":
        core = int(np.argmax(np.where(P0[:, 0] < 0, g, -np.inf))); oth = [int(np.argmax(np.where(P0[:, 0] > 0, g, -np.inf)))]
    else:
        core = int(np.argmax(np.where(P0[:, 0] > 1, g, -np.inf))); oth = [int(np.argmax(np.where(P0[:, 0] < 1, -g, -np.inf)))]
    frames = np.unique(np.linspace(0, S-1, 11).round().astype(int))
    rec = [r for r in tcx.scan_levels(fr, tri0, np.sign(g[core])*g, core, frames=frames, others=oth) if r["found"] and not r["merged"]][::4]
    rows = []
    for r in rec:
        Q0 = r["poly"]; L0 = perim(Q0)
        q2 = max(np.log(perim(quad_image(P0, fr[s], Q0)) / L0) for s in frames)
        q2d = max(np.log(perim(quad_image(P0, fr[s], dens(Q0, 4*len(Q0)))) / perim(dens(Q0, 4*len(Q0)))) for s in frames)
        ex = adv(vel, dens(Q0, 4000), t, set(frames.tolist()), 10)
        rows.append((round(float(np.sqrt(r["area"]/np.pi)), 3), round(r["lnLd"], 3), round(q2, 3), round(q2d, 3), round(ex, 3)))
    key = f"{nm} N={N or 12800}"; res[key] = rows; a = np.array(rows)
    print(key, "median |err|  P1 %.3f  P2 %.3f  P2+4x densify %.3f" % tuple(np.median(np.abs(a[:, j] - a[:, 4])) for j in (1, 2, 3)), flush=True)
    for x in rows: print("   r_eq, P1, P2, P2dense, exact:", x)
json.dump(res, open("res_iss_perim2.json", "w"), indent=1)
