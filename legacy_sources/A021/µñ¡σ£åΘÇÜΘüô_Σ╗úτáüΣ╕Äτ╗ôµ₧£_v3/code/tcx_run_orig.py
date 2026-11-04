"""Run every part of the completed theory on one case.  Usage: python tcx_run.py DG|LO|SOD|BLAST|SB|SV"""
import sys, json, time, pickle, numpy as np
from scipy.stats import spearmanr
import tcx, tcx_cases as TC, lhdx as X
from trajnet import TrajectoryNetwork as TN

name = sys.argv[1]; T0 = time.time()
c = TC.load(name); fr = c["fr"]; S, N = fr.shape[:2]; dt = c["dt"]; h = c["h"]; m = c["inner"]
R = dict(case=name, S=int(S), N=int(N), kind=c["kind"]); F = {}
rng = np.random.default_rng(5)
log = lambda *a: print(name, f"{time.time() - T0:6.1f}s", *a, flush=True)


def stat(est, ref, mask):
    ok = mask & np.isfinite(est) & np.isfinite(ref)
    if np.std(ref[ok]) < 1e-12:
        return dict(p99=float(np.quantile(np.abs(est[ok]), .99)), max=float(np.abs(est[ok]).max()))
    return dict(pearson=float(np.corrcoef(est[ok], ref[ok])[0, 1]), spearman=float(spearmanr(est[ok], ref[ok])[0]),
                rel_rms=float(np.sqrt(np.mean((est[ok] - ref[ok]) ** 2)) / np.std(ref[ok])))


phi_ref, lavd_ref = TC.phi_reference(c["om"], fr, dt); F["phi_ref"] = phi_ref; F["lavd_ref"] = lavd_ref; F["P0"] = fr[0]
F["inner"] = m

# ---------------------------------------------------------------- P1 identities on the data
k = S // 2; tri = tcx.cplx(fr[k])["tri"]; sel = rng.choice(len(tri), min(3000, len(tri)), replace=False); trs = tri[sel]
Fm = tcx.map_F(fr[k], fr[k + 1], trs); p, q = tcx.pq_of(Fm)
det = Fm[:, 0, 0] * Fm[:, 1, 1] - Fm[:, 0, 1] * Fm[:, 1, 0]
circ = tcx.tri_disp_circ(fr, trs, k, 1.0) / tcx.tri_area(fr[k], trs)
R["P1_identities"] = dict(det_err=float(np.max(np.abs(det - (np.abs(p) ** 2 - np.abs(q) ** 2)))),
                          circ_err=float(np.max(np.abs(circ - 2 * p.imag))))
log("P1", R["P1_identities"])

# ---------------------------------------------------------------- chains (clean, no-merge)
mus, med = tcx.mu_threshold(fr); sig_hat = tcx.sigma_hat(fr)
Ks = [tcx.cplx(fr[s])["tri"] for s in range(S - 1)]
ch = tcx.run_chain(fr, mus, record_sig=True, Ks=Ks)
phi, lavd, nseg, psibar = tcx.rotation_fields(ch)
ch0 = tcx.run_chain(fr, np.inf, cplx_cache=ch["cache"]); phi0, lavd0, _, _ = tcx.rotation_fields(ch0)
phi_circ = tcx.phi_circ_retri(fr, Ks); phi_argp = tcx.phi_argp_retri(fr, Ks)
F.update(phi=phi, lavd=lavd, phi_nomerge=phi0, phi_circ=phi_circ)
R["chain"] = dict(mu_star=mus, mu_median=med, hot_frac=ch["hot_frac"], nseg_mean=ch["nseg_mean"], sigma_hat_over_h=sig_hat / h)
log("chain", R["chain"])

# ---------------------------------------------------------------- P3 relative rotation
R["P3_phi"] = dict(new=stat(phi, phi_ref, m), new_nomerge=stat(phi0, phi_ref, m), old_circ_retri=stat(phi_circ, phi_ref, m),
                   argp_retri=stat(phi_argp, phi_ref, m))
log("P3", R["P3_phi"])

# ---------------------------------------------------------------- P2 hot steps and shock signature
sg_all = []; refeta = []
for trix, sg in ch["sigs"]:
    comp = (sg[:, 3] < -0.05) & (sg[:, 3] > -50) & (sg[:, 4] < 1)
    if not comp.any():
        continue
    s2 = sg[comp]; tt = trix[s2[:, 0].astype(int)]; a0 = s2[:, 1].astype(int); a1 = s2[:, 2].astype(int)
    lr = c["lnJ_ref"]
    ln_eta_ref = -(lr[a1[:, None], tt] - lr[a0[:, None], tt]).mean(1)
    sg_all.append(s2); refeta.append(ln_eta_ref)
nblk = sum(len(sg_) for _, sg_ in ch["sigs"])
if sg_all:
    sg = np.concatenate(sg_all); re = np.concatenate(refeta)
    dev = np.abs(sg[:, 4] - np.tanh(0.5 * np.abs(sg[:, 3])))
    true_c = re > 0.05
    R["P2_signature"] = dict(n_blocks=int(nblk), n_comp_blocks=int(len(sg)), n_true_compression=int(true_c.sum()),
                             frac_on_curve_all=float(np.mean(dev < 0.01)))
    if true_c.sum() > 10:
        d2 = dev[true_c]; e2 = np.abs(-sg[true_c, 3] - re[true_c])
        R["P2_signature"].update(dev_median=float(np.median(d2)), dev_p90=float(np.quantile(d2, .9)),
                                 argP_median=float(np.median(np.abs(sg[true_c, 5]))), ln_eta_ref_median=float(np.median(re[true_c])),
                                 ln_eta_err_median=float(np.median(e2)), ln_eta_err_rel=float(np.median(e2 / re[true_c])),
                                 frac_on_curve=float(np.mean(d2 < 0.01)))
    fp = ~true_c
    if fp.sum():
        R["P2_signature"]["false_shock_rate"] = float(np.mean(dev[fp] < 0.01) * fp.sum() / max(nblk, 1))
    F["sig"] = np.c_[sg[:, 3:6], re][rng.choice(len(sg), min(4000, len(sg)), replace=False)]
else:
    R["P2_signature"] = dict(n_blocks=int(nblk), n_comp_blocks=0)
R["P2_signature"]["hot_frac"] = ch["hot_frac"]
log("P2", R["P2_signature"])

# ---------------------------------------------------------------- P5 LAVD
cnode = tcx.calib_node_noise(); tri0 = tcx.cplx(fr[0])["tri"]; ell = tcx.local_spacing(fr[0], tri0, N)
R["P5_lavd"] = dict(new=stat(lavd, lavd_ref, m), nomerge=stat(lavd0, lavd_ref, m), c_node=cnode)
log("P5", R["P5_lavd"])

# ---------------------------------------------------------------- P6 ledger
lnJ_new = ch["lnJ"][-1]
ch5 = tcx.run_chain(fr, mus, Rmax=5, cplx_cache=ch["cache"], Ks=Ks); lnJ_new5 = ch5["lnJ"][-1]
net = TN(fr, dt); net.events(); lnJ_old = X.chained_lnJ_v2(fr, net.bad)[-1]
eN = np.abs(lnJ_new - c["lnJ_ref"][-1]); eO = np.abs(lnJ_old - c["lnJ_ref"][-1]); e5 = np.abs(lnJ_new5 - c["lnJ_ref"][-1])
R["P6_ledger"] = dict(new_med=float(np.median(eN[m])), new_p95=float(np.quantile(eN[m], .95)),
                      new_R5_med=float(np.median(e5[m])), new_R5_p95=float(np.quantile(e5[m], .95)),
                      old_Cchannel_med=float(np.median(eO[m])), old_Cchannel_p95=float(np.quantile(eO[m], .95)))
log("P6", R["P6_ledger"])

# ---------------------------------------------------------------- P1b objectivity (rotating + translating frame)
tt = c["t"]; beta = 0.6 * (tt - tt[0]) / (tt[-1] - tt[0]); Lx = np.ptp(fr[0], axis=0)
bvec = np.outer((tt - tt[0]) / (tt[-1] - tt[0]), [0.4 * Lx[0], -0.3 * Lx[1]])
frr = np.empty_like(fr)
for s in range(S):
    cb, sb = np.cos(beta[s]), np.sin(beta[s]); y = fr[s] - bvec[s]
    frr[s] = np.c_[cb * y[:, 0] + sb * y[:, 1], -sb * y[:, 0] + cb * y[:, 1]]
Ksr = [tcx.cplx(frr[s])["tri"] for s in range(S - 1)]
chr_ = tcx.run_chain(frr, mus, Ks=Ksr); phir, lavdr, _, _ = tcx.rotation_fields(chr_); phi_circ_r = tcx.phi_circ_retri(frr, Ksr)
sc = max(np.abs(phi[m]).max(), 1e-12)
R["P1_objectivity"] = dict(phi_new=float(np.abs(phir - phi)[m].max() / sc), lavd_new=float(np.abs(lavdr - lavd)[m].max() / max(np.abs(lavd[m]).max(), 1e-12)),
                           lnJ_new=float(np.abs(chr_["lnJ"][-1] - lnJ_new)[m].max()),
                           phi_old_circ=float(np.abs(phi_circ_r - phi_circ)[m].max() / max(np.abs(phi_circ[m]).max(), 1e-12)),
                           frame_rotation_rad=0.6)
log("P1b", R["P1_objectivity"])

# ---------------------------------------------------------------- P4 noise 0.1h
frn = fr + rng.normal(0, 0.1 * h, fr.shape); sig_n = tcx.sigma_hat(frn)
musn, medn = tcx.mu_threshold(frn); Ksn = [tcx.cplx(frn[s])["tri"] for s in range(S - 1)]; chn = tcx.run_chain(frn, musn, Ks=Ksn)
eps_i = cnode * sig_n / ell
phin, lavdn_raw, nsegn, _ = tcx.rotation_fields(chn); _, lavdn, _, _ = tcx.rotation_fields(chn, floor_eps=eps_i)
phin_circ = tcx.phi_circ_retri(frn, Ksn); phin_argp = tcx.phi_argp_retri(frn, Ksn)
en = np.abs(chn["lnJ"][-1] - c["lnJ_ref"][-1])
R["P4_noise"] = dict(sigma_hat_over_h=sig_n / h, mu_star=musn, hot_frac=chn["hot_frac"], nseg_mean=chn["nseg_mean"],
                     phi_new_anchored=stat(phin, phi_ref, m), phi_old_circ_retri=stat(phin_circ, phi_ref, m),
                     phi_argp_retri=stat(phin_argp, phi_ref, m),
                     lavd_floor=stat(lavdn, lavd_ref, m), lavd_nofloor=stat(lavdn_raw, lavd_ref, m),
                     lavd_bias_floor=float(np.median((lavdn - lavd_ref)[m])), lavd_bias_nofloor=float(np.median((lavdn_raw - lavd_ref)[m])),
                     lnJ_med=float(np.median(en[m])), lnJ_p95=float(np.quantile(en[m], .95)))
F.update(phi_noise=phin, phi_noise_old=phin_circ)
log("P4", R["P4_noise"])

# ---------------------------------------------------------------- P7 circulation jump on cells
a, b = c["circ"]; P7 = {}
for lab_, frx in (("clean", fr), ("noise", frn)):
    for blk in (None, 5):
        for mm in (1, 2, 4):
            netd, refd = tcx.cell_deposit(frx, c["om"], dt, a, b, m=mm, block=blk)
            if lab_ == "noise":
                _, refd = tcx.cell_deposit(fr, c["om"], dt, a, b, m=mm, block=blk)
            P7[f"{lab_}_{'single' if blk is None else 'chain5'}_m{mm}"] = stat(netd, refd, m)
            if lab_ == "clean" and blk == 5 and mm == 1:
                F["dep_net"], F["dep_ref"] = netd, refd
R["P7_circulation"] = P7
log("P7", {k_: v for k_, v in P7.items() if "chain5" in k_})

# ---------------------------------------------------------------- P8 cells: non-affinity, hollow cells, tearing frequency
epsn = np.zeros(N); cnt = np.zeros(N); fr_thr = {0.002: [], 0.005: [], 0.01: [], 0.02: []}
for k0 in range(0, S - 5, 5):
    na = tcx.cell_nonaffinity(fr, k0, k0 + 5, m=2); ok = np.isfinite(na["eps"])
    for th_ in fr_thr:
        fr_thr[th_].append(float(np.mean(na["eps"][ok] > th_)))
    ev = np.where(ok, na["eps"], 0.0)[na["lab"]]
    for j in range(3):
        np.add.at(epsn, na["tri"][:, j], ev); np.add.at(cnt, na["tri"][:, j], ok[na["lab"]].astype(float))
epsn = epsn / np.maximum(cnt, 1); F["eps_node"] = epsn
P8 = dict(hollow_frac={str(k_): float(np.mean(v)) for k_, v in fr_thr.items()}, eps_median=float(np.median(epsn[m])))
if np.std(phi_ref[m]) > 1e-12:
    core = m & (np.abs(phi_ref) >= np.quantile(np.abs(phi_ref[m]), 0.8)); out = m & ~core
    P8.update(eps_in_cores=float(np.median(epsn[core])), eps_outside=float(np.median(epsn[out])),
              enrichment=float(np.median(epsn[out]) / max(np.median(epsn[core]), 1e-12)))
if "ftle" in c:
    P8["spearman_eps_ftle"] = float(spearmanr(epsn[m], c["ftle"][m])[0])
if np.std(phi_ref[m]) > 1e-12:
    from lhdx import _grad_lambda
    Gt, At_ = _grad_lambda(fr[0], tri0); gr = np.einsum("tk,tkd->td", phi_ref[tri0], Gt); gm = np.hypot(*gr.T)
    gnode = np.zeros(N); wn = np.zeros(N)
    for j in range(3):
        np.add.at(gnode, tri0[:, j], At_ * gm); np.add.at(wn, tri0[:, j], At_)
    gnode /= np.maximum(wn, 1e-300); P8["spearman_eps_gradphi"] = float(spearmanr(epsn[m], gnode[m])[0])
R["P8_cells"] = P8
log("P8", P8)

# ---------------------------------------------------------------- P9 Hodge: general SPD metrics on a subdomain + harmonic dim
kk = min(S // 2, S - 6); Pk = fr[kk]; cen = np.median(Pk[m], 0)
dist = np.hypot(*(Pk - cen).T); rad = np.sort(dist)[min(350, N - 1)]; sub = np.where(dist <= rad)[0]
Ps = Pk[sub]; trs = tcx.cplx(Ps)["tri"]; Ns = len(sub)
bnd_v = np.zeros(Ns, bool)
eds = np.sort(np.concatenate([trs[:, [0, 1]], trs[:, [1, 2]], trs[:, [2, 0]]]), 1); ue, cnt = np.unique(eds, axis=0, return_counts=True)
bnd_v[ue[cnt == 1].ravel()] = True
interior_t = np.where(~bnd_v[trs].any(1))[0]; kh = 3
hollow = rng.choice(interior_t, kh, replace=False)
filled = np.setdiff1d(np.arange(len(trs)), hollow)
edges, B1, B2all = tcx.boundary_mats(trs, Ns); B2 = B2all[:, filled]
At = np.abs(tcx.tri_area(Ps, trs)); M0 = np.diag(np.maximum(X._star(Ps, trs, Ns) / 3, 1e-300)); M2 = np.diag(1 / At[filled])
# diagonal cotan star (clipped), Whitney mass, random SPD
el = np.linalg.norm(Ps[edges[:, 1]] - Ps[edges[:, 0]], axis=1)
from lhdx import _grad_lambda
Gl, Al = _grad_lambda(Ps, trs); cot = np.zeros(len(edges)); key = {(int(u), int(v)): i for i, (u, v) in enumerate(edges)}
for t_ in range(len(trs)):
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        u, v = trs[t_, i], trs[t_, j]; cot[key[(min(u, v), max(u, v))]] += -Al[t_] * float(Gl[t_, i] @ Gl[t_, j])
Md = np.diag(np.maximum(cot, 1e-6 * np.abs(cot).mean())); Mw = tcx.whitney_mass(Ps, trs, edges)
Ar = rng.normal(size=(len(edges), len(edges))) / np.sqrt(len(edges)); Mr = np.eye(len(edges)) * np.mean(np.diag(Mw)) + 0.3 * np.mean(np.diag(Mw)) * (Ar @ Ar.T) / len(edges)
fd = (fr[kk + 5] - fr[kk])[sub]; f = 0.5 * ((fd[edges[:, 0]] + fd[edges[:, 1]]) * (Ps[edges[:, 1]] - Ps[edges[:, 0]])).sum(1)
P9 = dict(n_nodes=int(Ns), n_edges=int(len(edges)), n_hollow=kh)
for lab_, M1 in (("diag_cotan", Md), ("whitney", Mw), ("random_spd", Mr)):
    P9[lab_] = dict(harmonic_dim=tcx.harmonic_dim(B1, B2, M1), **tcx.hodge_dense(f, B1, B2, M0, M1, M2))
# full-domain Galerkin shares on position increments
for lab_, B_ in (("galerkin_step1", 1), ("galerkin_block5", 5)):
    Eg = []; Ec = []; Eh = []; orth = []
    for k0 in range(0, S - B_, max(B_, 2)):
        cx = tcx.cplx(fr[k0]); V = (fr[k0 + B_] - fr[k0]) / (B_ * dt)
        gh = X.galerkin_hodge(fr[k0], cx, V); Eg.append(gh["Eg"]); Ec.append(gh["Ec"]); Eh.append(gh["Eh"]); orth.append(abs(gh["orth"]))
    P9[lab_] = dict(Eg=float(np.mean(Eg)), Ec=float(np.mean(Ec)), Eh=float(np.mean(Eh)), orth_max=float(np.max(orth)))
R["P9_hodge"] = P9
log("P9", {k_: (v if not isinstance(v, dict) else {kk_: (round(vv, 6) if isinstance(vv, float) else vv) for kk_, vv in v.items()}) for k_, v in P9.items()})

# ---------------------------------------------------------------- P10/P11 persistence and convex boundaries
E0 = tcx.graph_edges(tri0)
def core_count(fv, fref, thr):
    cn = tcx.cores(np.abs(fv), E0, m, thr); cr = tcx.cores(np.abs(fref), E0, m, thr); return cn, cr
gN = np.sqrt(2 * np.log(max(m.sum(), 2)))
def rough_s(fv, n=6):
    g = fv.copy(); deg = np.bincount(E0.ravel(), minlength=N).astype(float)
    for _ in range(n):
        s_ = np.zeros(N); np.add.at(s_, E0[:, 0], g[E0[:, 1]]); np.add.at(s_, E0[:, 1], g[E0[:, 0]])
        g = 0.5 * g + 0.5 * s_ / np.maximum(deg, 1)
    r_ = (fv - g)[m]; return float(1.4826 * np.median(np.abs(r_ - np.median(r_))))
eps_sys = float(np.quantile(np.abs(phi - phi0)[m], 0.999))
rough = max(rough_s(phi), rough_s(phi_ref)); thr = 2 * max(gN * rough, eps_sys) + 1e-12
cn, cr = core_count(phi, phi_ref, thr)
roughn = rough_s(phin); thrn = 2 * max(gN * max(roughn, rough), eps_sys) + 1e-12
cnn, _ = core_count(phin, phi_ref, thrn)
def matched(ca, cb, rad):
    return sum(1 for (_, _, ra) in ca if any(np.linalg.norm(fr[0][ra] - fr[0][rb]) < rad for (_, _, rb) in cb))
radm = 0.1 * float(np.max(np.ptp(fr[0][m], axis=0)))
P10 = dict(thr=thr, eps_sys=eps_sys, rough=rough, cores_net=len(cn), cores_ref=len(cr), matched_net_in_ref=matched(cn, cr, radm), thr_noise=thrn,
           cores_net_noise=len(cnn), matched_noise_in_ref=matched(cnn, cr, radm),
           top_persistence_net=[float(bb - (dd if np.isfinite(dd) else 0)) for bb, dd, _ in sorted(cn, key=lambda x: -(x[0] - (x[1] if np.isfinite(x[1]) else 0)))[:4]])
R["P10_persistence"] = P10
log("P10", P10)
def smooth(fv, n=2):
    g = fv.copy(); deg = np.bincount(E0.ravel(), minlength=N).astype(float)
    for _ in range(n):
        s_ = np.zeros(N); np.add.at(s_, E0[:, 0], g[E0[:, 1]]); np.add.at(s_, E0[:, 1], g[E0[:, 0]])
        g = 0.5 * g + 0.5 * s_ / np.maximum(deg, 1)
    return g
phs, prs = smooth(phi), smooth(phi_ref)
P11 = []
for bb, dd, r_ in cn[:6]:
    sgn = np.sign(phi[r_]) or 1.0
    bn = tcx.convex_boundary(fr[0], tri0, sgn * phs, r_, 2e-2)
    # matched reference core
    if cr:
        rr_ = min(cr, key=lambda z: np.linalg.norm(fr[0][z[2]] - fr[0][r_]))[2]
        br = tcx.convex_boundary(fr[0], tri0, (np.sign(phi_ref[rr_]) or 1.0) * prs, rr_, 2e-2)
    else:
        br = None
    rec = dict(found=bn is not None, ref_found=br is not None)
    if bn is not None and br is not None:
        A_, B_ = bn["inside"] & m, br["inside"] & m
        rec.update(iou=float((A_ & B_).sum() / max((A_ | B_).sum(), 1)), area_ratio=float(bn["area"] / br["area"]))
    P11.append(rec)
R["P11_boundary"] = P11
log("P11", P11)

# ---------------------------------------------------------------- P12 three-point chained tangential stretch
frs = c.get("stretch_fr", fr); band = c["band"]
chS = ch if frs is fr else tcx.run_chain(frs, tcx.mu_threshold(frs)[0])
lam = tcx.chain_stretch(frs, chS, band, c["T0"])
P12 = dict(n_band=int(len(band)), net_mean=float(lam.mean()), ref_mean=c["lam_ref"]["mean"])
if c["lam_ref"]["point"] is not None:
    P12["point_err_median"] = float(np.median(np.abs(lam - c["lam_ref"]["point"])))
if "side" in c["lam_ref"]:
    P12["ref_side_range"] = [min(c["lam_ref"]["side"]), max(c["lam_ref"]["side"])]
R["P12_stretch"] = P12
log("P12", P12)

R["runtime_s"] = time.time() - T0
json.dump(R, open(f"/home/claude/bench/tcx_res_{name}.json", "w"), indent=1)
pickle.dump(F, open(f"/home/claude/bench/tcx_fields_{name}.pkl", "wb"))
log("done")
