"""Strength relative to the vortex's own coherent boundary: S_i = phi_i - L_delta inside, 0 outside."""
import json, numpy as np, tcx, cases, afe_cases as AC
from sync_lib import chain_phi, smooth
out = {}
for name, cut in [("LO", lambda P: np.hypot(P[:, 0], P[:, 1]) < 4.5), ("DG", lambda P: P[:, 0] < 1.85)]:
    c = AC.load(name); fr = c["fr"]; S = len(fr); P0 = fr[0]; frames = np.unique(np.linspace(0, S-1, 11).round().astype(int))
    res = {}
    for lab, idx in [("full", np.arange(len(P0))), ("sub", np.where(cut(P0))[0])]:
        frx = fr[:, idx]; phi = chain_phi(frx); tri = tcx.cplx(frx[0])["tri"]; E = tcx.graph_edges(tri); g = smooth(phi, E, len(idx))
        Px = frx[0]; rows = []
        for centre in ([(-3, 0), (3, 0)] if name == "LO" else [(1.5, 0.37), (0.75, 0.73)]):
            near = np.linalg.norm(Px - centre, axis=1) < 0.5*(1 if name == "LO" else 0.4)
            sg = np.sign(np.median(g[near])); core = int(np.where(near)[0][np.argmax(sg*g[near])])
            other = [int(np.argmax(np.where(np.linalg.norm(Px - [-centre[0], centre[1]] if name == "LO" else Px - [0.75 if centre[0] > 1 else 1.5, 0.73 if centre[0] > 1 else 0.37], axis=1) < 0.4, sg*g, -np.inf)))]
            rec = tcx.scan_levels(frx, tri, sg*g, core, frames=frames, others=other)
            row = dict(centre=centre, phi_core=float(phi[core]))
            for d in (0.1, 0.2):
                b = tcx.pick(rec, "lnLd", d)
                row[f"S_core(delta={d})"] = float(sg*g[core] - b["level"]) if b else None
            rows.append(row)
        res[lab] = rows
    out[name] = res
    for k in range(2):
        f, s_ = res["full"][k], res["sub"][k]
        print(name, f["centre"], "phi_core full/sub %.3f / %.3f" % (f["phi_core"], s_["phi_core"]),
              " S(0.1) full/sub", f["S_core(delta=0.1)"], s_["S_core(delta=0.1)"], " S(0.2)", f["S_core(delta=0.2)"], s_["S_core(delta=0.2)"], flush=True)
json.dump(out, open("sync2.json", "w"), indent=1)
