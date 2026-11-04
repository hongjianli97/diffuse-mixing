"""Issue 5: is the SB main roll-up a coherent vortex on shorter, later windows?"""
import json, pickle, numpy as np, tcx, tcx_cases as TC
D = pickle.load(open("sb_dns.pkl", "rb")); out = {}
for t0, L in [(0.0, 1.4), (0.6, 0.3), (0.8, 0.3), (1.0, 0.3), (1.1, 0.3), (1.2, 0.2)]:
    a = int(round(t0/D["dt"])); b = a + int(round(L/D["dt"])) + 1
    fr = D["frames"][a:b]; om = D["E"]["om"][a:b].astype(float); S, N = fr.shape[:2]; P0 = fr[0]
    m = (P0[:, 0] > 0.3) & (P0[:, 0] < 1.8) & (P0[:, 1] < 0.45)
    phi_ref, _ = TC.phi_reference(om, fr, D["dt"])
    mus, _ = tcx.mu_threshold(fr); Ks = [tcx.cplx(fr[s])["tri"] for s in range(S-1)]
    ch = tcx.run_chain(fr, mus, Ks=Ks); phi, _, nseg, psib = tcx.rotation_fields(ch)
    tri0 = Ks[0]; E0 = tcx.graph_edges(tri0); deg = np.bincount(E0.ravel(), minlength=N).astype(float)
    def smooth(fv, n=2):
        g = fv.copy()
        for _ in range(n):
            s_ = np.zeros(N); np.add.at(s_, E0[:, 0], g[E0[:, 1]]); np.add.at(s_, E0[:, 1], g[E0[:, 0]]); g = .5*g + .5*s_/np.maximum(deg, 1)
        return g
    g = smooth(phi); frames = np.unique(np.linspace(0, S-1, 11).round().astype(int))
    cc = tcx.cores(np.abs(phi), E0, m, 0.3)[:6]; ids = [z[2] for z in cc]; rows = []
    for bb, dd, r_ in cc:
        rec = tcx.scan_levels(fr, tri0, (np.sign(phi[r_]) or 1)*g, r_, frames=frames, others=[o for o in ids if o != r_])
        f = [x for x in rec if x["found"]]
        gp = tcx.pick(rec, "lnLd", 0.1)
        rows.append(dict(pos=P0[r_].round(3).tolist(), pers=round(float(bb-(dd if np.isfinite(dd) else 0)), 2),
                         innermost_lnL=round(f[0]["lnLd"], 3) if f else None,
                         coherent_area=round(gp["area"], 5) if gp else None, coherent_req_over_h=round(float(np.sqrt(gp["area"]/np.pi)/D["h"]), 1) if gp else None))
    key = f"[{t0}, {t0+L}]"; out[key] = rows; print(key, rows[:4], flush=True)
json.dump(out, open("res_iss_sb.json", "w"), indent=1)
