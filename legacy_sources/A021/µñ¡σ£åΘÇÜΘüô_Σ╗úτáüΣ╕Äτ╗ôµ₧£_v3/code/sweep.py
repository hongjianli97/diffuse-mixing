"""Smoothing size cap, delta and J_max sweeps.  python sweep.py CASE seeds"""
import sys, json, numpy as np, tcx, afe_cases as AC, tcx_cases as TC
name = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(",")]
c = AC.load(name); fr = c["fr"]; S, N = fr.shape[:2]; h = c["h"]; m = c["inner"]
phi_ref, _ = TC.phi_reference(c["om"], fr, c["dt"])
tri0 = tcx.cplx(fr[0])["tri"]; E0 = tcx.graph_edges(tri0); deg = np.bincount(E0.ravel(), minlength=N).astype(float)
def smooth(fv, n):
    g = fv.copy()
    for _ in range(n):
        s_ = np.zeros(N); np.add.at(s_, E0[:, 0], g[E0[:, 1]]); np.add.at(s_, E0[:, 1], g[E0[:, 0]]); g = .5*g + .5*s_/np.maximum(deg, 1)
    return g
# kernel width: smooth deltas at 20 interior nodes, rms radius of the kernel
P0 = fr[0]; rng0 = np.random.default_rng(0); probes = rng0.choice(np.where(m)[0], 20, replace=False); NS = (2, 3, 4, 6, 8, 12, 16)
kw = {}
for n in NS:
    w = []
    for p in probes:
        dlt = np.zeros(N); dlt[p] = 1.0; k = smooth(dlt, n); w.append(np.sqrt((k*((P0 - P0[p])**2).sum(1)).sum()/k.sum()/2))
    kw[n] = float(np.median(w)/h)
print(name, "kernel rms width / h:", {n: round(v, 2) for n, v in kw.items()}, flush=True)
def chain(frx):
    mus, _ = tcx.mu_threshold(frx); ch = tcx.run_chain(frx, mus, Ks=[tcx.cplx(frx[s])["tri"] for s in range(S-1)])
    phi, _, nseg, psib = tcx.rotation_fields(ch); return ch, phi, nseg, psib
ch, phi, nseg, psib = chain(fr)
gN = np.sqrt(2*np.log(max(m.sum(), 2))); thr0 = 0.0
def rough_s(fv): r_ = (fv - smooth(fv, 6))[m]; return float(1.4826*np.median(np.abs(r_ - np.median(r_))))
thr = 2*gN*max(rough_s(phi), rough_s(phi_ref)) + 1e-12
cr = tcx.cores(np.abs(phi_ref), E0, m, thr)[:6]; ids = [z[2] for z in cr]
frames = np.unique(np.linspace(0, S-1, 11).round().astype(int)); Atri = float(np.median(np.abs(tcx.tri_area(P0, tri0))))
DEL = (0.05, 0.1, 0.15, 0.2, 0.3)
def scan(frx, fv, r_, sig=0.0):
    return tcx.scan_levels(frx, tri0, (np.sign(phi_ref[r_]) or 1)*fv, r_, frames=frames, others=[o for o in ids if o != r_], sig=sig)
def mask(g): return tcx.inside(P0, g["poly"]) & m if g is not None else None
def iou(a, b): return float((a & b).sum()/max((a | b).sum(), 1)) if a is not None and b is not None else 0.0
R = dict(case=name, h=h, kernel_width=kw, vortices=[])
refsc = {r_: scan(fr, smooth(phi_ref, 2), r_) for r_ in ids}
netsc = {r_: scan(fr, smooth(phi, 2), r_) for r_ in ids}
noisy = []
for sd in seeds:
    rng = np.random.default_rng(sd); frn = fr + rng.normal(0, 0.1*h, fr.shape); chn, phin, nsegn, psibn = chain(frn)
    noisy.append((frn, phin, float(tcx.sigma_hat(frn)), chn, psibn))
for r_ in ids:
    v = dict(pos=P0[r_].round(3).tolist(), by_delta={})
    for d in DEL:
        gr, gn = tcx.pick(refsc[r_], "lnLd", d), tcx.pick(netsc[r_], "lnLd", d)
        if gr is None or gr["area"] < 6*Atri:
            continue
        v["by_delta"][d] = dict(area_ref=gr["area"], req_over_h=float(np.sqrt(gr["area"]/np.pi)/h), iou_clean=iou(mask(gn), mask(gr)), iou_noise={})
    if not v["by_delta"]:
        continue
    for n in NS:
        per = {d: [] for d in v["by_delta"]}
        for frn, phin, sigp, _, _ in noisy:
            rn = int(np.argmax(np.where(np.linalg.norm(P0 - P0[r_], axis=1) < 4*h, (np.sign(phi_ref[r_]) or 1)*smooth(phin, n), -np.inf)))
            sc = tcx.scan_levels(frn, tri0, (np.sign(phi_ref[r_]) or 1)*smooth(phin, n), rn, frames=frames, others=[o for o in ids if o != r_], sig=sigp)
            for d in per:
                per[d].append(iou(mask(tcx.pick(sc, "lnLd", d)), mask(tcx.pick(refsc[r_], "lnLd", d))))
        for d in per:
            v["by_delta"][d]["iou_noise"][n] = per[d]
    R["vortices"].append(v)
    print(name, v["pos"], {d: (round(x["req_over_h"], 1), round(x["iou_clean"], 2), {n: round(float(np.mean(y)), 2) for n, y in x["iou_noise"].items()}) for d, x in v["by_delta"].items()}, flush=True)
# J_max: single-step shares of all significant cores, clean and noisy
sg = tcx.node_noise(ch, psib); z = gN*(1 + 1.67/np.sqrt(S-2))
phis, _ = tcx.phi_shrink(phi, sg, nseg, z); cc = tcx.cores(np.abs(phis), E0, m, thr)
R["shares_clean"] = [round(tcx.single_step_share(ch, psib, q[2], E0), 3) for q in cc]
R["shares_noise"] = []
for frn, phin, sigp, chn, psibn in noisy:
    sgn_ = tcx.node_noise(chn, psibn); nsg = tcx.rotation_fields(chn)[2]; phins, _ = tcx.phi_shrink(phin, sgn_, nsg, z)
    thrn = 2*gN*max(rough_s(phin), rough_s(phi)) + 1e-12
    R["shares_noise"].append([round(tcx.single_step_share(chn, psibn, q[2], E0), 3) for q in tcx.cores(np.abs(phins), E0, m, thrn)])
print(name, "shares clean", R["shares_clean"][:10], " noise", [s_[:8] for s_ in R["shares_noise"]], flush=True)
json.dump(R, open(f"sweep_{name}.json", "w"), indent=1)
