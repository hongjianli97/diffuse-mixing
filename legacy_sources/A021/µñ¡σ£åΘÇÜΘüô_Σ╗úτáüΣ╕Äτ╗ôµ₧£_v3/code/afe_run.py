"""LAVD-path defects vs AFRC_E-derived fixes on one case.  python afe_run.py CASE"""
import sys, json, time, pickle, numpy as np
from scipy.stats import spearmanr
import tcx, afe, lhdx as X
import afe_cases as AC
name = sys.argv[1]; T0 = time.time()
log = lambda *a: print(name, f"{time.time()-T0:6.1f}s", *a, flush=True)
c = AC.load(name); fr = c["fr"]; S, N = fr.shape[:2]; dt = c["dt"]; h = c["h"]; m = c["inner"]
R = dict(case=name, S=int(S), N=int(N)); F = dict(P0=fr[0], inner=m)
def stat(est, ref, mask):
    ok = mask & np.isfinite(est) & np.isfinite(ref)
    if np.std(ref[ok]) < 1e-12:
        return dict(p99=float(np.quantile(np.abs(est[ok]), .99)), max=float(np.abs(est[ok]).max()))
    return dict(pearson=float(np.corrcoef(est[ok], ref[ok])[0, 1]), spearman=float(spearmanr(est[ok], ref[ok])[0]),
                rel_rms=float(np.sqrt(np.mean((est[ok]-ref[ok])**2))/np.std(ref[ok])), bias=float(np.median((est-ref)[ok])))
phi_ref, lavd_ref = AC.phi_reference(c["om"], fr, dt)
# ---------------- chains
def chains(frx):
    mus, _ = tcx.mu_threshold(frx); Ks = [tcx.cplx(frx[s])["tri"] for s in range(S-1)]
    ch = tcx.run_chain(frx, mus, Ks=Ks); phi, lavd, nseg, psib = tcx.rotation_fields(ch)
    return ch, phi, lavd, nseg, psib
ch, phi, lavd, nseg, psib = chains(fr); log("clean chain")
ch0 = tcx.run_chain(fr, np.inf, cplx_cache=ch["cache"]); phi0, _, _, _ = tcx.rotation_fields(ch0)
rng = np.random.default_rng(5); frn = fr + rng.normal(0, 0.1*h, fr.shape); sig_noise = float(tcx.sigma_hat(frn))
chn, phin, lavdn, nsegn, psibn = chains(frn); log("noisy chain")
tri0 = tcx.cplx(fr[0])["tri"]; E0 = tcx.graph_edges(tri0); ell = tcx.local_spacing(fr[0], tri0, N)
# ---------------- strength: LAVD noise floor
sg = afe.node_noise(ch, psib); sgn_ = afe.node_noise(chn, psibn)
lavd_e, _ = afe.lavd_energy(ch, psib, sg); lavdn_e, _ = afe.lavd_energy(chn, psibn, sgn_)
cnode = 0.59; eps_i = cnode*tcx.sigma_hat(frn)/ell
_, lavdn_g, _, _ = tcx.rotation_fields(chn, floor_eps=eps_i)
R["strength"] = dict(clean_raw=stat(lavd, lavd_ref, m), clean_energy=stat(lavd_e, lavd_ref, m),
                     noise_raw=stat(lavdn, lavd_ref, m), noise_globalfloor=stat(lavdn_g, lavd_ref, m), noise_energy=stat(lavdn_e, lavd_ref, m),
                     sig_seg_med_clean=float(np.median(sg[m])), sig_seg_med_noise=float(np.median(sgn_[m])))
log("strength", json.dumps(R["strength"]))
F.update(phi=phi, phin=phin, phi_ref=phi_ref, lavd_ref=lavd_ref, lavd=lavd, lavdn=lavdn, lavdn_e=lavdn_e, lavdn_g=lavdn_g)
# ---------------- cores (persistence on |phi|), old threshold vs significance-shrunk phi
gN = np.sqrt(2*np.log(max(m.sum(), 2))); deg = np.bincount(E0.ravel(), minlength=N).astype(float)
def smooth(fv, n):
    g = fv.copy()
    for _ in range(n):
        s_ = np.zeros(N); np.add.at(s_, E0[:, 0], g[E0[:, 1]]); np.add.at(s_, E0[:, 1], g[E0[:, 0]]); g = 0.5*g + 0.5*s_/np.maximum(deg, 1)
    return g
def rough_s(fv):
    r_ = (fv - smooth(fv, 6))[m]; return float(1.4826*np.median(np.abs(r_ - np.median(r_))))
eps_sys = float(np.quantile(np.abs(phi - phi0)[m], 0.999))
def thr_of(*fs): return 2*max(gN*max(rough_s(f) for f in fs), eps_sys) + 1e-12
def cores(fv, thr):
    cc = tcx.cores(np.abs(fv), E0, m, thr); return sorted(cc, key=lambda z: -(z[0] - (z[1] if np.isfinite(z[1]) else 0)))
radm = 0.1*float(np.max(np.ptp(fr[0][m], axis=0)))
def matched(ca, cb): return sum(1 for (_, _, ra) in ca if any(np.linalg.norm(fr[0][ra]-fr[0][rb]) < radm for (_, _, rb) in cb))
thr = thr_of(phi, phi_ref); cr = cores(phi_ref, thr)
zst = gN * (1 + 1.67 / np.sqrt(S - 2))   # multiple testing x uncertainty of each node's own MAD (n = S-2 steps)
phis, env = afe.phi_shrink(phi, sg, nseg, zst); phins, envn = afe.phi_shrink(phin, sgn_, nsegn, zst)
out = {}
thrn = thr_of(phin, phi)
for lab, fv, th in [("old_clean", phi, thr), ("old_noise", phin, thrn), ("shrink_clean", phis, thr), ("shrink_noise", phins, thrn)]:
    cc = cores(fv, th); out[lab] = dict(n=len(cc), matched=matched(cc, cr), thr=th,
                                        pers=[round(float(b-(d if np.isfinite(d) else 0)), 3) for b, d, _ in cc[:6]])
out["ref"] = dict(n=len(cr), pers=[round(float(b-(d if np.isfinite(d) else 0)), 3) for b, d, _ in cr[:6]])
out["env_med_noise"] = float(np.median(envn[m])); out["z"] = float(zst)
# vortex = significant core that owns a closed material curve whose perimeter does not grow (lnL<=0.1)
Atri = float(np.median(np.abs(tcx.tri_area(fr[0], tri0))))
frames = np.unique(np.linspace(0, S-1, 11).round().astype(int))
def vortices(fv, frx, cc, delta=0.1, amin=6, sig=0.0):
    keep = []; ids = [z[2] for z in cc]; fs = smooth(fv, 2)
    for r_ in ids:
        sg_ = np.sign(fv[r_]) or 1.0
        rec = afe.scan_levels(frx, tri0, sg_*fs, r_, frames=frames, others=[o for o in ids if o != r_], sig=sig)
        g = afe.pick(rec, "lnLd", delta)
        if g is not None and g["area"] >= amin*Atri:
            keep.append(r_)
    return keep
vref = vortices(phi_ref, fr, cr)
for lab, fv, frx, th in [("vort_old_clean", phi, fr, thr), ("vort_new_clean", phis, fr, thr), ("vort_old_noise", phin, frn, thrn), ("vort_new_noise", phins, frn, thrn)]:
    cc = cores(fv, th); vk = vortices(fv, frx, cc, sig=(sig_noise if frx is frn else 0.0))
    out[lab] = dict(n=len(vk), matched=sum(1 for a_ in vk if any(np.linalg.norm(fr[0][a_]-fr[0][b_]) < radm for b_ in vref)))
out["vort_ref"] = dict(n=len(vref))
R["cores"] = out; log("cores", json.dumps(out))
# ---------------- boundaries: convexity deficiency (Haller) vs material-perimeter coherence
cn = cores(phis, thr)[:6]; ccore = [r_ for _, _, r_ in cn]
phs, prs, phns = smooth(phi, 2), smooth(phi_ref, 2), smooth(phin, 2)
B = []
for i, r_ in enumerate(ccore):
    sgnc = np.sign(phi[r_]) or 1.0; oth = [o for o in ccore if o != r_]
    rr_ = min([z[2] for z in cr], key=lambda z: np.linalg.norm(fr[0][z]-fr[0][r_])) if cr else r_
    recs = dict(net=afe.scan_levels(fr, tri0, sgnc*phs, r_, frames=frames, others=oth),
                ref=afe.scan_levels(fr, tri0, sgnc*prs, rr_, frames=frames, others=oth),
                noise=afe.scan_levels(frn, tri0, sgnc*phns, r_, frames=frames, others=oth, sig=sig_noise))
    rec = dict(core=int(r_), pos=fr[0][r_].tolist())
    for crit, th in [("cdef", 2e-2), ("lnL", 0.05), ("lnL", 0.1), ("lnL", 0.2), ("lnLd", 0.1), ("lnLd", 0.2)]:
        key = f"{crit}<={th}"; got = {k: afe.pick(v, crit, th) for k, v in recs.items()}
        d = {k: (g is not None) for k, g in got.items()}
        ins = {k: afe.inside(fr[0], g["poly"]) & m for k, g in got.items() if g is not None}
        if "net" in ins and "ref" in ins:
            d["iou_ref"] = float((ins["net"] & ins["ref"]).sum()/max((ins["net"] | ins["ref"]).sum(), 1))
        if "net" in ins and "noise" in ins:
            d["iou_noise"] = float((ins["net"] & ins["noise"]).sum()/max((ins["net"] | ins["noise"]).sum(), 1))
        for k, g in got.items():
            if g is not None:
                d[k+"_lnL"] = round(g["lnL"], 3); d[k+"_cdef"] = round(float(g["cdef"]), 4); d[k+"_area"] = round(g["area"], 4)
        rec[key] = d
        if crit == "lnLd" and th == 0.1:
            F[f"bnd{i}"] = {k: g["poly"] for k, g in got.items() if g is not None}
        if crit == "cdef":
            F[f"bndc{i}"] = {k: g["poly"] for k, g in got.items() if g is not None}
    rec["profile_net"] = [(round(float(r["level"]), 3), round(r["lnL"], 3), round(float(r["cdef"]), 4)) for r in recs["net"] if r["found"]][::6]
    B.append(rec); log("bnd", i, json.dumps({k: v for k, v in rec.items() if k != "profile_net"}))
R["boundary"] = B
json.dump(R, open(f"afe_res_{name}.json", "w"), indent=1); pickle.dump(F, open(f"afe_F_{name}.pkl", "wb"))
log("done")
