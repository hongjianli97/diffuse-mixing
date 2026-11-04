import json, numpy as np, cases, cases_mix as CM, tcx
from tcx_cases import _lo_omega
from verify_perim import dens, adv
out = {}
for N in (8000, 32000):
    c = cases.case_lo(a=1.0, Rd=6.5, N=N); fr = c["frames"]; S = len(fr); t = c["t"]
    mus, _ = tcx.mu_threshold(fr); ch = tcx.run_chain(fr, mus, Ks=[tcx.cplx(fr[s])["tri"] for s in range(S-1)])
    phi = tcx.rotation_fields(ch)[0]; tri0 = tcx.cplx(fr[0])["tri"]; E0 = tcx.graph_edges(tri0); n = len(phi)
    deg = np.bincount(E0.ravel(), minlength=n).astype(float); g = phi.copy()
    for _ in range(2):
        s_ = np.zeros(n); np.add.at(s_, E0[:, 0], g[E0[:, 1]]); np.add.at(s_, E0[:, 1], g[E0[:, 0]]); g = .5*g + .5*s_/np.maximum(deg, 1)
    core = int(np.argmax(np.where(fr[0][:, 0] < 0, g, -np.inf)))
    frames = np.unique(np.linspace(0, S-1, 11).round().astype(int))
    rec = tcx.scan_levels(fr, tri0, g, core, frames=frames, others=[int(np.argmax(np.where(fr[0][:, 0] > 0, g, -np.inf)))])
    rows = []
    for r in [r for r in rec if r["found"]][::5]:
        ex = adv(CM.lo_vel, dens(r["poly"], 4000), t, set(frames.tolist()), 10)
        rows.append((round(float(np.sqrt(r["area"]/np.pi)), 3), round(r["lnLd"], 3), round(ex, 3)))
    out[N] = dict(h=c["h"], rows=rows); print(N, c["h"], rows, flush=True)
json.dump(out, open("res_verify_res.json", "w"))
