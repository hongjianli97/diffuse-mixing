"""Issue 1: network perimeter bias in shear.  P1 image of t0 triangles vs level set on current K(t) vs exact advection."""
import json, numpy as np, cases, cases_mix as CM, tcx, afe_cases as AC
from verify_perim import dens, adv
out = {}
for nm, vel, Nlo in [("LO", CM.lo_vel, 8000), ("DG", cases.dg_fields()[0], None)]:
    for N in ((8000, 32000) if nm == "LO" else (None,)):
        c = cases.case_lo(a=1.0, Rd=6.5, N=N) if nm == "LO" else cases.case_dg()
        fr = c["frames"]; S = len(fr); t = c["t"]
        mus, _ = tcx.mu_threshold(fr); Ks = [tcx.cplx(fr[s])["tri"] for s in range(S)]
        phi = tcx.rotation_fields(tcx.run_chain(fr, mus, Ks=Ks[:-1]))[0]; tri0 = Ks[0]; E0 = tcx.graph_edges(tri0); n = len(phi)
        deg = np.bincount(E0.ravel(), minlength=n).astype(float); g = phi.copy()
        for _ in range(2):
            s_ = np.zeros(n); np.add.at(s_, E0[:, 0], g[E0[:, 1]]); np.add.at(s_, E0[:, 1], g[E0[:, 0]]); g = .5*g + .5*s_/np.maximum(deg, 1)
        P0 = fr[0]
        if nm == "LO":
            core = int(np.argmax(np.where(P0[:, 0] < 0, g, -np.inf))); oth = [int(np.argmax(np.where(P0[:, 0] > 0, g, -np.inf)))]
        else:
            core = int(np.argmax(np.where(P0[:, 0] > 1, g, -np.inf))); oth = [int(np.argmax(np.where(P0[:, 0] < 1, -g, -np.inf)))]
        sg = np.sign(g[core]); frames = np.unique(np.linspace(0, S-1, 11).round().astype(int))
        rec = tcx.scan_levels_K(fr, tri0, Ks, sg*g, core, frames=frames, others=oth)
        rows = []
        for r in [r for r in rec if r["found"] and not r["merged"]][::4]:
            ex = adv(vel, dens(r["poly"], 4000), t, set(frames.tolist()), 10)
            rows.append(dict(r_eq=round(float(np.sqrt(r["area"]/np.pi)), 3), P1=round(r["lnLd"], 3), Kt=round(r["lnLK"], 3) if np.isfinite(r["lnLK"]) else None, exact=round(ex, 3)))
        key = f"{nm} N={N or 12800}"; out[key] = rows
        e1 = [abs(x["P1"]-x["exact"]) for x in rows]; e2 = [abs(x["Kt"]-x["exact"]) for x in rows if x["Kt"] is not None]
        print(key, "median |err| P1", round(float(np.median(e1)), 3), " Kt", round(float(np.median(e2)), 3), " Kt undefined", sum(x["Kt"] is None for x in rows), "/", len(rows), flush=True)
        for x in rows: print("   ", x)
json.dump(out, open("res_iss_perim.json", "w"), indent=1)
