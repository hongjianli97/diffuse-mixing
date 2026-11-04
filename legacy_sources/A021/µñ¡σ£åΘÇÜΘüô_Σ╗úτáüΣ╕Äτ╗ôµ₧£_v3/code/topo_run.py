"""LAVD-free elliptic structures on all cases.  python topo_run.py CASE"""
import sys, json, pickle, time, numpy as np, tcx, topo, afe_cases as AC
from matplotlib.path import Path
name = sys.argv[1]; T0 = time.time(); log = lambda *a: print(name, f"{time.time()-T0:5.0f}s", *a, flush=True)
c = AC.load(name); fr = c["fr"]; S, N = fr.shape[:2]; h = c["h"]; m = c["inner"]
def chain(frx):
    mus, _ = tcx.mu_threshold(frx); ch = tcx.run_chain(frx, mus, Ks=[tcx.cplx(frx[s])["tri"] for s in range(S-1)])
    phi, lavd, nseg, psib = tcx.rotation_fields(ch); return ch["lnJ"], np.r_[0, np.cumsum(psib)]
lnJ, cum = chain(fr)
rng = np.random.default_rng(11); frn = fr + rng.normal(0, 0.1*h, fr.shape); lnJn, cumn = chain(frn); log("chains")
tri0 = tcx.cplx(fr[0])["tri"]
# phi-based references from the tcx run (if present)
ref_masks = {}
try:
    Fp = pickle.load(open(f"tcx_fields_{name}.pkl", "rb"))
    for k, pol in enumerate(Fp["P11_polys"]):
        for q in ("lnLd<=0.1", "cdef<=0.02"):
            if "ref" in pol[q]:
                ref_masks[f"core{k} {q} ref"] = Path(pol[q]["ref"]).contains_points(fr[0]) & m
except FileNotFoundError:
    pass
def summarize(V, label):
    vs = [v for v in V if v["vortex"]]; rows = []
    for v in sorted(vs, key=lambda v: -len(v["tris"])):
        mask = np.zeros(N, bool); mask[v["nodes"]] = True; mask &= m
        best = max(((k, float((mask & r).sum()/max((mask | r).sum(), 1))) for k, r in ref_masks.items()), key=lambda z: z[1], default=(None, 0))
        rows.append(dict(n_tri=len(v["tris"]), rot=round(v["rot"], 3), share=round(v["single_share"], 3),
                         centre=fr[0][v["nodes"]].mean(0).round(3).tolist(), area=float(np.abs(tcx.tri_area(fr[0], tri0[v["tris"]])).sum()),
                         best_ref=best[0], iou=round(best[1], 3)))
    rej = dict(not_sustained=sum(1 for v in V if v["significant"] and not v["sustained"]),
               not_significant=sum(1 for v in V if not v["significant"]))
    return dict(n_vortex=len(vs), n_clusters=len(V), rejected=rej, vortices=rows)
R = dict(case=name)
for delta in ((0.1,) if len(sys.argv) > 2 else (0.05, 0.1, 0.2)):
    for conf in ((True,) if len(sys.argv) > 2 else (True, False)):
        Vc, dc = topo.vortices(fr, tri0, lnJ, cum, delta=delta, conformal=conf)
        Vn, dn = topo.vortices(frn, tri0, lnJn, cumn, delta=delta, conformal=conf)
        key = f"delta={delta} conformal={conf}"
        R[key] = dict(clean=summarize(Vc, "clean"), noise=summarize(Vn, "noise"), active_clean=dc["active_frac"], active_noise=dn["active_frac"])
        # clean-vs-noise IoU of matched vortices
        ious = []
        for a in [v for v in Vc if v["vortex"]]:
            ma = np.zeros(N, bool); ma[a["nodes"]] = True
            b = max([v for v in Vn if v["vortex"]], key=lambda v: len(np.intersect1d(v["nodes"], a["nodes"])), default=None)
            if b is not None:
                mb = np.zeros(N, bool); mb[b["nodes"]] = True; ious.append(round(float((ma & mb).sum()/(ma | mb).sum()), 3))
        R[key]["iou_clean_noise"] = ious
        if delta == 0.1 and conf:
            pickle.dump(dict(Vc=Vc, Vn=Vn, tri0=tri0, P0=fr[0]), open(f"topo_F_{name}.pkl", "wb"))
        log(key, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "vortices"}) for k, v in R[key].items()}))
        log("   clean vortices", R[key]["clean"]["vortices"][:4]); log("   noise vortices", R[key]["noise"]["vortices"][:4])
json.dump(R, open(f"topo_res_{name}.json", "w"), indent=1)
