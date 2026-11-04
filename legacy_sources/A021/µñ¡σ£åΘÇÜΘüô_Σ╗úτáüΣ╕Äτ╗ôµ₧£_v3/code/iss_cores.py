"""Issues 2-4: sustained-rotation gate (Sod false core), noise-adaptive smoothing of boundaries, SV noise diagnosis."""
import sys, json, numpy as np, tcx, afe_cases as AC, tcx_cases as TC
name = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [5]
c = AC.load(name); fr = c["fr"]; S, N = fr.shape[:2]; h = c["h"]; m = c["inner"]; dt = c["dt"]
phi_ref, _ = TC.phi_reference(c["om"], fr, dt)
tri0 = tcx.cplx(fr[0])["tri"]; E0 = tcx.graph_edges(tri0); deg = np.bincount(E0.ravel(), minlength=N).astype(float)
def smooth(fv, n):
    g = fv.copy()
    for _ in range(n):
        s_ = np.zeros(N); np.add.at(s_, E0[:, 0], g[E0[:, 1]]); np.add.at(s_, E0[:, 1], g[E0[:, 0]]); g = .5*g + .5*s_/np.maximum(deg, 1)
    return g
gN = np.sqrt(2*np.log(max(m.sum(), 2))); z = gN*(1 + 1.67/np.sqrt(S-2))
def rough_s(fv): r_ = (fv - smooth(fv, 6))[m]; return float(1.4826*np.median(np.abs(r_ - np.median(r_))))
def chain(frx):
    mus, _ = tcx.mu_threshold(frx); Ks = [tcx.cplx(frx[s])["tri"] for s in range(S-1)]
    ch = tcx.run_chain(frx, mus, Ks=Ks); phi, lavd, nseg, psib = tcx.rotation_fields(ch)
    return ch, phi, nseg, psib, Ks
ch, phi, nseg, psib, Ks = chain(fr); ch0 = tcx.run_chain(fr, np.inf, cplx_cache=ch["cache"]); phi0 = tcx.rotation_fields(ch0)[0]
eps_sys = float(np.quantile(np.abs(phi - phi0)[m], 0.999))
thr = 2*max(gN*max(rough_s(phi), rough_s(phi_ref)), eps_sys) + 1e-12
d = ch["inc"] - psib[:, None]
def share(dd, r_):                                     # single-step share of the net rotation, 1-ring averaged
    nb = np.r_[r_, E0[E0[:, 0] == r_, 1], E0[E0[:, 1] == r_, 0]]; x = dd[:, nb].mean(1)
    return float(np.abs(x).max() / max(abs(x.sum()), 1e-300))
sg = tcx.node_noise(ch, psib); phis, env = tcx.phi_shrink(phi, sg, nseg, z)
cs = tcx.cores(np.abs(phis), E0, m, thr); cr = tcx.cores(np.abs(phi_ref), E0, m, thr)
R = dict(case=name, thr=thr, z=z, cores_clean=[dict(pos=fr[0][r].round(3).tolist(), pers=round(float(b-(dd_ if np.isfinite(dd_) else 0)), 4),
                                                   share=round(share(d, r), 3)) for b, dd_, r in cs[:8]])
print(name, "clean cores (pers, single-step share):", [(x["pers"], x["share"]) for x in R["cores_clean"]], flush=True)
# reference coherent boundaries (clean ref field, 2 passes) for the significant ref cores
frames = np.unique(np.linspace(0, S-1, 11).round().astype(int))
ids_r = [z_[2] for z_ in cr[:6]]; refB = {}
for r_ in ids_r:
    g = tcx.pick(tcx.scan_levels(fr, tri0, (np.sign(phi_ref[r_]) or 1)*smooth(phi_ref, 2), r_, frames=frames, others=[o for o in ids_r if o != r_]), "lnLd", 0.1)
    if g is not None and g["area"] >= 6*np.median(np.abs(tcx.tri_area(fr[0], tri0))):
        refB[r_] = tcx.inside(fr[0], g["poly"]) & m
R["ref_vortices"] = [fr[0][r_].round(3).tolist() for r_ in refB]
R["noise"] = []
for sd in seeds:
    rng = np.random.default_rng(sd); frn = fr + rng.normal(0, 0.1*h, fr.shape); sigp = float(tcx.sigma_hat(frn))
    chn, phin, nsegn, psibn, _ = chain(frn); sgn_ = tcx.node_noise(chn, psibn); phins, envn = tcx.phi_shrink(phin, sgn_, nsegn, z)
    dn = chn["inc"] - psibn[:, None]
    thrn = 2*max(gN*max(rough_s(phin), rough_s(phi)), eps_sys) + 1e-12
    csn = tcx.cores(np.abs(phins), E0, m, thrn)
    rec = dict(seed=sd, thr_noise=thrn, n_cores=len(csn),
               cores=[dict(pos=fr[0][r].round(3).tolist(), pers=round(float(b-(dd_ if np.isfinite(dd_) else 0)), 3), share=round(share(dn, r), 3),
                          snr=round(float((b-(dd_ if np.isfinite(dd_) else 0))/(np.sqrt(2)*envn[r]+1e-12)), 2)) for b, dd_, r in csn[:8]])
    # SV-type diagnosis: for every reference vortex, signal vs noise envelope at its core
    diag = []
    for r_, mk in refB.items():
        pr = [b - (dd_ if np.isfinite(dd_) else 0) for b, dd_, rr in cr if rr == r_][0]
        diag.append(dict(pos=fr[0][r_].round(3).tolist(), ref_pers=round(float(pr), 3), noise_env=round(float(envn[r_]), 3),
                         pers_over_noise=round(float(pr/(np.sqrt(2)*envn[r_] + 1e-12)), 2), thr_noise=round(thrn, 3)))
    rec["ref_vortex_snr"] = diag
    # boundary IoU vs reference coherent boundary: smoothing passes on the noisy field
    bi = {}
    for npass in (2, 4, 8, 16):
        fs = smooth(phin, npass); io = []
        for r_, mk in refB.items():
            rn = int(np.argmax(np.where(np.linalg.norm(fr[0] - fr[0][r_], axis=1) < 4*h, (np.sign(phi_ref[r_]) or 1)*fs, -np.inf)))
            g = tcx.pick(tcx.scan_levels(frn, tri0, (np.sign(phi_ref[r_]) or 1)*fs, rn, frames=frames, others=[o for o in refB if o != r_], sig=sigp), "lnLd", 0.1)
            io.append(round(float((tcx.inside(fr[0], g["poly"]) & m & mk).sum()/max(((tcx.inside(fr[0], g["poly"]) & m) | mk).sum(), 1)), 3) if g is not None else 0.0)
        bi[npass] = io
    rec["iou_vs_ref_by_smoothing"] = bi
    R["noise"].append(rec)
    print(name, "seed", sd, "noise cores (pers, share, snr):", [(x["pers"], x["share"], x["snr"]) for x in rec["cores"]], "thr", round(thrn, 3), flush=True)
    print(name, "   ref vortex SNR:", diag, flush=True); print(name, "   IoU by smoothing passes:", bi, flush=True)
json.dump(R, open(f"iss_cores_{name}.json", "w"), indent=1)
