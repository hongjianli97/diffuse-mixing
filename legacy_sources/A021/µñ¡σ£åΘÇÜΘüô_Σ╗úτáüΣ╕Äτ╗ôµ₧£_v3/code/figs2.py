import sys, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figs import tri_of, pf, norm01

C_OLD, C_DEC, C_GAL, C_LS = "C7", "C1", "C3", "C2"


def fig_dg():
    d = np.load("res_dg.npz"); R = json.load(open("res_dg.json"))
    Pm, P0 = d["Pm"], d["P0"]; trm, tr0 = tri_of(Pm), tri_of(P0)
    fig, ax = plt.subplots(3, 4, figsize=(11.5, 5.4))
    a = np.nanquantile(np.abs(d["om_ex"]), 0.99)
    pf(ax[0, 0], Pm, d["om_ex"], sym=True, vmax=a, tr=trm, title=r"(a) exact $\omega$, $t=5$")
    pf(ax[0, 1], Pm, d["om_st"], sym=True, vmax=a, tr=trm, title=r"(b) LHD Stokes $\omega$   r=%.4f" % R["vort"]["stokes"]["r"])
    pf(ax[0, 2], Pm, d["om_ls"], sym=True, vmax=a, tr=trm, title=r"(c) LS-kNN $\omega$   r=%.4f" % R["vort"]["ls"]["r"])
    pf(ax[0, 3], Pm, d["div_gal"], sym=True, vmax=a, tr=trm,
       title="(d) LHD $\\nabla\\!\\cdot u$ (Galerkin), same scale\nrms/rms$\\,\\omega$ = %.3f" % R["v2"]["div_noise"])
    for j, (k, ttl) in enumerate([("lavd_ex", "(e) exact LAVD"), ("lavd_st", "(f) Stokes-LAVD   sp=%.4f" % R["lavd"]["stokes"]["sp"]),
                                  ("afrcE", "(g) AFRC$_E$, paper input   sp=%.2f" % R["afrcE"]["AFRC_E"]["sp"]),
                                  ("afrcE2", "(h) AFRC$_E$, Stokes input   sp=%.2f" % R["afrcE"]["AFRC_E_stokes"]["sp"])]):
        pf(ax[1, j], P0, norm01(d[k]), cmap="magma", vmin=0, vmax=1, tr=tr0, title=ttl)
    for j, (k, ttl) in enumerate([("ftle_ex", "(i) exact FTLE"), ("ftle_ls", "(j) LS-kNN FTLE   sp=%.3f" % R["afrcH"]["FTLE_ls"]["sp"]),
                                  ("afrcH", "(k) AFRC$_H$, paper weights   sp=%.2f" % R["afrcH"]["AFRC_H"]["sp"]),
                                  ("afrcH_rb", "(l) AFRC$_H$, coexact weights   sp=%.2f" % R["afrcH"]["AFRC_H_rebuilt_v2"]["sp"])]):
        pf(ax[2, j], P0, norm01(d[k]), cmap="Blues", vmin=0, vmax=1, tr=tr0, title=ttl)
    fig.tight_layout(); fig.savefig("fig_dg.png"); plt.close(fig)


def fig_weights():
    d = np.load("res_dg.npz"); R = json.load(open("res_dg.json"))
    Rs = json.load(open("res_sod.json")); Rb = json.load(open("res_blast.json"))
    Pm = d["Pm"]; tr = tri_of(Pm); a = 0.3 * np.sqrt(np.nanmean(d["om_ex"] ** 2))
    fig = plt.figure(figsize=(11.5, 5.8)); g = fig.add_gridspec(2, 4)
    for j, (k, ttl) in enumerate([("div_cot", "(a) DEC cotan, median-edge filter\n(old default; internal holes)"),
                                  ("div_cot_peel", "(b) DEC cotan,\noutside-peeled complex"),
                                  ("div_ls", "(c) least-squares kNN\n(12 neighbours)"),
                                  ("div_gal", "(d) Galerkin P1/CR,\npeeled complex")]):
        ax = fig.add_subplot(g[0, j]); pf(ax, Pm, d[k], sym=True, vmax=a, tr=tr, title=ttl)
    ax = fig.add_subplot(g[1, 0])
    labs = ["cotan\n(old)", "cotan", "bary", "unit", "gauss", "1/l$^2$", "Galerkin", "Galerkin\n$\\rho$-w."]
    vals = [R["Efrac_grad_med"]["cotan"]] + [R["ablation_peel"][k]["Egrad_med"] for k in ("cotan", "bary", "unit", "gauss", "invlen2")] + \
           [R["v2"]["Egrad_med"], R["v2"]["Egrad_rho_med"]]
    cols = [C_OLD] + [C_DEC] * 5 + [C_GAL, "C5"]
    ax.bar(range(len(vals)), vals, color=cols); ax.set_yscale("log"); ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, fontsize=6)
    ax.set_title("(e) DG: gradient-energy leakage\n(incompressible; truth 0, median over time)"); ax.set_ylim(3e-4, 0.5)
    ax = fig.add_subplot(g[1, 1])
    ops = ["cotan", "bary", "unit", "gauss", "galerkin"]
    dg = [R["ablation_peel"][k]["div_noise"] if k != "galerkin" else R["v2"]["div_noise"] for k in ops] + [R["div_noise_over_rms_vort"]["ls"]]
    sod = [Rs["div_smooth"][k]["relL2_fit"] for k in ops] + [Rs["div_smooth"]["LS"]["relL2"]]
    bl = [Rb["div_smooth"][k]["relL2_fit"] for k in ops] + [Rb["div_smooth"]["LS"]["relL2"]]
    x = np.arange(len(ops) + 1); w = 0.27
    ax.bar(x - w, np.maximum(dg, 1e-4), w, label="DG (rms / rms $\\omega$)", color="C0")
    ax.bar(x, np.maximum(sod, 1e-4), w, label="Sod fan (rel. L2, best scale)", color="C1")
    ax.bar(x + w, np.maximum(bl, 1e-4), w, label="blast post-shock (rel. L2)", color="C3")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(ops + ["LS-kNN"], fontsize=6.5); ax.legend(fontsize=5.5, loc="upper right")
    for xx in (4, 5):
        ax.text(xx + w, 1.3e-4, "exact", rotation=90, fontsize=5.5, ha="center", va="bottom", color="C3")
    ax.set_title("(f) node-divergence error by edge weight"); ax.set_ylim(1e-4, 100)
    ax = fig.add_subplot(g[1, 2])
    labs = ["Sod\nmedian", "Sod\np95", "blast\nmedian", "blast\np95"]
    old = [Rs["lnJ_err"]["old_chained_median"], Rs["lnJ_err"]["old_chained_p95"], Rb["lnJ_err"]["old_chained_median"], Rb["lnJ_err"]["old_chained_p95"]]
    fix = [Rs["lnJ_err"]["fixed_median"], Rs["lnJ_err"]["fixed_p95"], Rb["lnJ_err"]["fixed_median"], Rb["lnJ_err"]["fixed_p95"]]
    new = [Rs["lnJ_err"]["chained_median"], Rs["lnJ_err"]["chained_p95"], Rb["lnJ_err"]["chained_median"], Rb["lnJ_err"]["chained_p95"]]
    x = np.arange(4)
    ax.bar(x - w, np.maximum(old, 1e-6), w, color=C_OLD, label="re-anchor every 5 frames (old)")
    ax.bar(x, np.maximum(fix, 1e-6), w, color="C0", label="fixed $t_0$ complex")
    ax.bar(x + w, np.maximum(new, 1e-6), w, color=C_GAL, label="per-tracer, shock-deferred (new)")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.5); ax.legend(fontsize=5.5, loc="upper left")
    ax.text(0, 1.5e-6, "all three at\nmachine precision", fontsize=5.5, ha="center", va="bottom")
    ax.set_title(r"(g) material $\ln J$ error (point weight)"); ax.set_ylim(1e-6, 2)
    ax = fig.add_subplot(g[1, 3])
    try:
        Rsb = json.load(open("res_sb.json")); fr = Rsb["frames"]
        ks = ["cotan", "unit", "gauss", "galerkin", "ls"]
        for j, s in enumerate(["15", "30", "50", "70"]):
            ax.plot(range(len(ks)), [fr[s]["div_r"][k] for k in ks], "-o", ms=3, label="t=%.1f" % fr[s]["t"])
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(["cotan", "unit", "gauss", "Galerkin", "LS-kNN"], fontsize=6.5)
        ax.set_ylabel("corr. with DNS $\\nabla\\cdot u$"); ax.legend(fontsize=6); ax.set_ylim(-0.1, 1)
        ax.set_title("(h) shock-bubble: divergence vs DNS\n(clean tracers)")
    except Exception as e:
        ax.set_title("(h) shock-bubble pending")
    fig.tight_layout(); fig.savefig("fig_weights.png"); plt.close(fig)


def fig_sod():
    d = np.load("res_sod.npz"); R = json.load(open("res_sod.json"))
    P, inner = d["P"], d["inner"]; x = P[:, 0]; o = np.argsort(x); m = inner[o]; xs = x[o][m]
    fig = plt.figure(figsize=(11.5, 6.2)); g = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(g[0, 0])
    ax.plot(xs, d["rho_ex"][o][m], "k-", lw=1.2, label="exact")
    for k, c, mk, lb in [("rho_dtfe", "C0", "x", "DTFE  (%.3f)" % R["rho_relerr"]["DTFE"]["all"]),
                         ("rho_sph", C_LS, "+", "SPH  (%.3f)" % R["rho_relerr"]["SPH"]["all"]),
                         ("rho_lhd", C_GAL, "o", r"LHD $\rho(t_0)/J$  (%.1e)" % R["rho_relerr"]["LHD_chained"]["all"])]:
        ax.plot(xs[::3], d[k][o][m][::3], mk, ms=(1.6 if mk == "o" else 2.2), mfc="none" if mk == "o" else None, color=c, label=lb)
    ax.set_xlabel("x"); ax.set_ylabel(r"$\rho$"); ax.set_title(r"(a) density at $t=0.24$ (median rel. error)"); ax.legend(fontsize=6)
    ax.set_xlim(-0.45, 0.48)
    ax = fig.add_subplot(g[0, 1])
    Pm = d["Pm"]; xm = Pm[:, 0]; om_ = np.argsort(xm); mm = inner[om_]
    ax.plot(xm[om_][mm], d["div_ex"][om_][mm], "k-", lw=1.2, label="exact (shock: $\\delta$)")
    ax.plot(xm[om_][mm][::2], d["div_cot"][om_][mm][::2], ".", ms=2, color=C_DEC, label="DEC cotan  (fan %.2f)" % R["div_smooth"]["cotan"]["relL2"])
    ax.plot(xm[om_][mm][::2], d["div_ls"][om_][mm][::2], "+", ms=2, color=C_LS, label="LS-kNN  (%.3f)" % R["div_smooth"]["LS"]["relL2"])
    ax.plot(xm[om_][mm][::2], d["div_gal"][om_][mm][::2], ".", ms=2, color=C_GAL, label="Galerkin  (%.3f)" % R["div_smooth"]["galerkin"]["relL2"])
    ax.set_ylim(-1.5, 6); ax.set_xlim(-0.45, 0.48); ax.set_xlabel("x"); ax.set_title(r"(b) $\nabla\cdot u$ at $t=0.235$ (fan rel. L2)"); ax.legend(fontsize=6)
    ax = fig.add_subplot(g[0, 2])
    tt = d["t"]; Fr = d["frames_sub"]; lnJ = d["lnJ"][:, ::7]
    for i in range(0, Fr.shape[1], 9):
        sc = ax.scatter(Fr[:, i, 0], tt, c=lnJ[:, i], s=0.6, cmap="viridis", vmin=-1.3, vmax=1.0, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.04).set_label(r"$\ln J$")
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_title(r"(c) tracer paths coloured by LHD $\ln J$")
    ax = fig.add_subplot(g[1, 0])
    ax.semilogy(x[o][m], np.abs(d["lnJ_old"] - d["lnJ_ex"][-1])[o][m] + 1e-16, ".", ms=1.5, color=C_OLD, label="re-anchor every 5 frames (old)")
    ax.semilogy(x[o][m], np.abs(d["lnJ"][-1] - d["lnJ_ex"][-1])[o][m] + 1e-16, ".", ms=1.5, color=C_GAL, label="per-tracer, shock-deferred")
    ax.set_xlabel("x"); ax.set_ylabel(r"$|\ln J-\ln J_{exact}|$"); ax.set_title(r"(d) $\ln J$ error at $t=0.24$"); ax.legend(fontsize=6)
    ax.set_xlim(-0.45, 0.48); ax.set_ylim(1e-16, 1)
    ax = fig.add_subplot(g[1, 1])
    ax.plot(xm[om_][mm], d["vort_raw"][om_][mm], ".", ms=1.5, color=C_DEC, label="Stokes, raw")
    ax.plot(xm[om_][mm], d["om_ls"][om_][mm], "+", ms=1.5, color=C_LS, label="LS-kNN")
    ax.plot(xm[om_][mm], d["vort_c"][om_][mm], ".", ms=1.5, color=C_GAL, label="Stokes, censored")
    ax.set_xlim(-0.45, 0.48); ax.set_ylim(-3, 3); ax.set_xlabel("x"); ax.set_title(r"(e) spurious $\omega$ (exact 0), $t=0.235$"); ax.legend(fontsize=6)
    ax = fig.add_subplot(g[1, 2])
    for k, c in [("cotan", C_DEC), ("bary", "C4"), ("unit", "C8"), ("invlen2", "C5"), ("galerkin", C_GAL)]:
        E = d["Ef_" + k]; ok = np.isfinite(E[:, 1])
        ax.semilogy(tt[ok], E[ok, 1] + 1e-12, "-o", ms=2, color=c, label=k)
    ax.set_xlabel("t"); ax.set_ylabel("coexact energy fraction"); ax.set_title("(f) Hodge leakage (exact: pure gradient)"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig("fig_sod.png"); plt.close(fig)


def fig_blast():
    d = np.load("res_blast.npz"); R = json.load(open("res_blast.json"))
    P, inner = d["P"], d["inner"]; tr = tri_of(P)
    fig, ax = plt.subplots(2, 4, figsize=(11.5, 5.6))
    pf(ax[0, 0], P, d["lnJ"][-1], cmap="viridis", vmin=-1.8, vmax=1.2, tr=tr, title=r"(a) LHD $\ln J$, $t=25$")
    pf(ax[0, 1], P, np.abs(d["lnJ_old"] - d["lnJ_ex"][-1]), cmap="magma", vmin=0, vmax=0.3, tr=tr, title=r"(b) $|\ln J$ error$|$, old re-anchoring")
    pf(ax[0, 2], P, np.abs(d["lnJ"][-1] - d["lnJ_ex"][-1]), cmap="magma", vmin=0, vmax=0.3, tr=tr, title=r"(c) $|\ln J$ error$|$, shock-deferred")
    vm = np.quantile(d["lavd"], 0.99)
    pf(ax[0, 3], P, d["lavd"], cmap="magma", vmin=0, vmax=vm, tr=tr, title="(d) Stokes-LAVD raw (exact 0)\np99 = %.2f rad" % R["spurious_lavd_p99_rad"]["raw"])
    pf(ax[1, 3], P, d["lavd_c"], cmap="magma", vmin=0, vmax=vm, tr=tr, title="(h) Stokes-LAVD censored\np99 = %.2f rad (LS-kNN %.2f)" % (
        R["spurious_lavd_p99_rad"]["censored"], R["spurious_lavd_p99_rad"]["LS"]))
    r = np.hypot(P[:, 0], P[:, 1]); o = np.argsort(r); m = inner[o]
    ax[1, 0].remove(); ax[1, 1].remove()
    a = fig.add_subplot(2, 4, (5, 6))
    a.plot(r[o][m], d["rho_ex"][o][m], "k-", lw=1.2, label="exact")
    for k, c, mk, lb in [("rho_dtfe", "C0", "x", "DTFE  (%.3f)" % R["rho_relerr"]["DTFE"]["post"]),
                         ("rho_sph", C_LS, "+", "SPH  (%.3f)" % R["rho_relerr"]["SPH"]["post"]),
                         ("rho_lhd", C_GAL, "o", r"LHD $\rho(t_0)/J$  (post %.3f)" % R["rho_relerr"]["LHD_chained"]["post"])]:
        a.plot(r[o][m][::3], d[k][o][m][::3], mk, ms=(1.6 if mk == "o" else 2), mfc="none" if mk == "o" else None, color=c, label=lb)
    a.set_ylim(0, 8); a.set_xlim(0.6, 1.15); a.set_xlabel("r"); a.set_ylabel(r"$\rho$"); a.set_title(r"(e) density, radial ($\eta=6$; median rel. error behind the shock)"); a.legend(fontsize=6)
    a = ax[1, 2]
    for k, c in [("cotan", C_DEC), ("bary", "C4"), ("unit", "C8"), ("invlen2", "C5"), ("galerkin", C_GAL)]:
        E = d["Ef_" + k]; ok = np.isfinite(E[:, 1])
        a.semilogy(d["t"][ok], E[ok, 1] + 1e-12, "-", color=c, label=k)
    a.set_xlabel("t"); a.set_title("(g) coexact energy fraction\n(exact: pure gradient)"); a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig("fig_blast.png"); plt.close(fig)


def merge_sb():
    """merge AFRC stage into res_sb.json window metrics."""
    import lhdx as X
    R = json.load(open("res_sb.json")); S = pickle.load(open("res_sb.pkl", "rb")); A = np.load("sb_afrc.npz")
    Z = np.load("sb_stage1.npz"); inner0, lavd, ftle = Z["inner0"], Z["lavd_dns"], Z["ftle_ls"]
    r0 = S["r0"]; band = inner0 & (r0 > 0.37) & (lavd < np.median(lavd[inner0])); core = inner0 & (lavd >= np.quantile(lavd[inner0], 0.95))
    sal = lambda F: float(np.nanquantile(F[band], 0.95) / max(np.nanquantile(F[core], 0.95), 1e-30))
    li = lavd_interp_censor(); arrs = {k: A[k] for k in A.files}; arrs["Stokes_LAVD_interp"] = li
    for k, v in arrs.items():
        S["win"][k] = v
        R["window"][k] = dict(sp_vs_DNSLAVD=X.spearman(v, lavd, inner0), J_vs_DNSLAVD=X.jaccard_top(v, lavd, 0.9, inner0),
                              sp_vs_FTLE=X.spearman(v, ftle, inner0), J_vs_FTLE=X.jaccard_top(v, ftle, 0.9, inner0), salience=sal(v))
    R["afrc_times"] = json.load(open("sb_afrc_times.json"))
    json.dump(R, open("res_sb.json", "w"), indent=1, default=float); pickle.dump(S, open("res_sb.pkl", "wb"))




def lavd_interp_censor():
    """Stokes-LAVD with censored frames FILLED by linear interpolation of |omega - <omega>| in time."""
    import lhdx as X, pickle as pk
    D = pk.load(open("sb_dns.pkl", "rb")); fr = D["frames"]; dt = D["dt"]; del D
    Z = np.load("sb_stage1.npz"); vc = Z["vort_c"]; S, N = vc.shape
    W = np.array([X.star_area(fr[s]) for s in range(S)])
    wb = np.array([np.nansum(vc[s] * W[s]) / np.nansum(W[s] * np.isfinite(vc[s])) for s in range(S)])
    g = np.abs(vc - wb[:, None]); t = np.arange(S)
    for i in range(N):
        ok = np.isfinite(g[:, i])
        if ok.sum() >= 2 and (~ok).any():
            g[~ok, i] = np.interp(t[~ok], t[ok], g[ok, i])
    return np.trapezoid(np.nan_to_num(g), dx=dt, axis=0)


def fig_sb():
    S = pickle.load(open("res_sb.pkl", "rb")); R = json.load(open("res_sb.json"))
    grids, xc, yc = S["grids"], S["xc"], S["yc"]
    fig, ax = plt.subplots(1, 4, figsize=(11.5, 2.3))
    sh = S["r0"] < 0.30; frames = {0.3: 15, 0.6: 30, 1.0: 50, 1.4: 70}
    for j, tk in enumerate(sorted(grids)):
        Wg = grids[tk]["W"]; gy, gx = np.gradient(Wg[0], yc, xc); G = np.hypot(gx, gy)
        sch = np.exp(-12 * G / G.max())
        ax[j].imshow(np.vstack([sch[::-1], sch]), extent=[xc[0], xc[-1], -yc[-1], yc[-1]], cmap="gray", vmin=0, vmax=1, aspect="equal")
        P = S["keep"][frames[tk]]["P"]
        ax[j].plot(P[::4, 0], -P[::4, 1], ",", color="C0", alpha=0.35)
        ax[j].plot(P[sh][::2, 0], -P[sh][::2, 1], ",", color="C3", alpha=0.7)
        ax[j].set_xlim(0.2, 2.0); ax[j].set_ylim(-0.6, 0.6); ax[j].set_title("t = %.1f  (top: DNS schlieren; bottom: tracers)" % tk, fontsize=7)
        ax[j].set_xticks([]); ax[j].set_yticks([])
    fig.tight_layout(); fig.savefig("fig_sb_dns.png"); plt.close(fig)
    for s in (50, 15):
        K = S["keep"][s]; P = K["P"]; tr = tri_of(P); f_ = R["frames"][str(s)]
        fig, ax = plt.subplots(3, 4, figsize=(11.5, 6.2))
        a = np.nanquantile(np.abs(K["om"]), 0.99)
        pf(ax[0, 0], P, K["om"], sym=True, vmax=a, tr=tr, title=r"(a) DNS $\omega$ at tracers")
        pf(ax[0, 1], P, K["om_st"], sym=True, vmax=a, tr=tr, title="(b) LHD Stokes $\\omega$, censored\nr = %.3f" % f_["vort_r"]["stokes_cens"])
        pf(ax[0, 2], P, K["om_raw"], sym=True, vmax=a, tr=tr, title="(c) LHD Stokes $\\omega$, raw\nr = %.3f" % f_["vort_r"]["stokes_raw"])
        pf(ax[0, 3], P, K["om_ls"], sym=True, vmax=a, tr=tr, title="(d) LS-kNN $\\omega$\nr = %.3f" % f_["vort_r"]["ls"])
        b = np.nanquantile(np.abs(K["th"]), 0.99)
        pf(ax[1, 0], P, K["th"], sym=True, vmax=b, tr=tr, title=r"(e) DNS $\nabla\cdot u$")
        pf(ax[1, 1], P, K["div"], sym=True, vmax=b, tr=tr, title="(f) LHD div, Galerkin\nr = %.3f" % f_["div_r"]["galerkin"])
        pf(ax[1, 2], P, K["div_cot"], sym=True, vmax=b, tr=tr, title="(g) LHD div, DEC cotan\nr = %.3f" % f_["div_r"]["cotan"])
        pf(ax[1, 3], P, K["div_ls"], sym=True, vmax=b, tr=tr, title="(h) LS-kNN div\nr = %.3f" % f_["div_r"]["ls"])
        lo, hi = np.nanquantile(K["rho"], 0.005), np.nanquantile(K["rho"], 0.995)
        pf(ax[2, 0], P, K["rho"], cmap="viridis", vmin=lo, vmax=hi, tr=tr, title=r"(i) DNS $\rho$ at tracers")
        for j, (k, nm) in enumerate([("rho_lhd", "LHD"), ("rho_dt", "DTFE"), ("rho_sph", "SPH")]):
            e = f_["rho_mederr"][nm]
            pf(ax[2, j + 1], P, K[k], cmap="viridis", vmin=lo, vmax=hi, tr=tr, title="(%s) %s   median err %.3f" % ("jkl"[j], nm, e["all"]))
        for a_ in ax.ravel():
            a_.set_xlim(np.nanquantile(P[:, 0], 0.002), np.nanquantile(P[:, 0], 0.998))
        fig.tight_layout(); fig.savefig("fig_sb_t%02d.png" % s); plt.close(fig)
    P = S["Pend"]; tr = tri_of(P); w = S["win"]; Rw = R["window"]
    fig, ax = plt.subplots(2, 5, figsize=(12.5, 4.3))
    for j, (k, nm) in enumerate([("DNS_LAVD", "DNS LAVD (reference)"), ("Stokes_LAVD_interp", "Stokes-LAVD"), ("LS_LAVD", "LS-kNN LAVD"),
                                 ("AFRC_E", "AFRC$_E$, paper input"), ("AFRC_E_stokes", "AFRC$_E$, Stokes input")]):
        ttl = nm if k == "DNS_LAVD" else nm + "\nsp=%.2f  J=%.2f  sal=%.2f" % (Rw[k]["sp_vs_DNSLAVD"], Rw[k]["J_vs_DNSLAVD"], Rw[k]["salience"])
        pf(ax[0, j], P, norm01(w[k]), cmap="magma", vmin=0, vmax=1, tr=tr, title=ttl)
    for j, (k, nm) in enumerate([("FTLE_LS", "LS-kNN FTLE"), ("AFRC_H", "AFRC$_H$, paper weights"), ("AFRC_H_rebuilt", "AFRC$_H$, coexact weights")]):
        ttl = nm + ("\nsal=%.2f" % Rw[k]["salience"] if k == "FTLE_LS" else "\nsp(FTLE)=%.2f  sal=%.2f" % (Rw[k]["sp_vs_FTLE"], Rw[k]["salience"]))
        pf(ax[1, j], P, norm01(w[k]), cmap="Blues", vmin=0, vmax=1, tr=tr, title=ttl)
    pf(ax[1, 3], P, S["node_share"], cmap="coolwarm", vmin=0, vmax=1, tr=tr, title="gradient share of pair stretch\n(clean frames)")
    pf(ax[1, 4], P, S["node_share_raw"], cmap="coolwarm", vmin=0, vmax=1, tr=tr, title="gradient share\n(shock frames included)")
    for a_ in ax.ravel():
        a_.set_xlim(np.quantile(P[:, 0], 0.002), np.quantile(P[:, 0], 0.998))
    fig.tight_layout(); fig.savefig("fig_sb_window.png"); plt.close(fig)
    t = S["t"]; GE = S["GamE"]
    fig, ax = plt.subplots(1, 4, figsize=(12.5, 2.9))
    a = ax[0]
    a.plot(GE["t"], GE["Y2"], "k-", lw=1.2, label=r"Eulerian $\int_{Y_2>1/2}\omega\,dA$")
    a.plot(GE["t"], GE["bar"], "k--", lw=1, label=r"Eulerian $\int_0^t\!\int_{Y_2>1/2}\nabla\rho\times\nabla p/\rho^2$")
    a.plot(t, S["Gam"]["sheet"][:, 0], "o", ms=2.2, color=C_GAL, label="LHD material region, all triangles")
    a.plot(t, S["Gam"]["sheet"][:, 1], "s", ms=1.6, color="C0", label="LHD, clean triangles only")
    a.set_xlabel("t"); a.set_ylabel(r"$\Gamma$"); a.set_title("(a) Kelvin budget: bubble + 0.05 margin"); a.legend(fontsize=5.3)
    a = ax[1]
    a.plot(GE["t"], GE["Y1"], "k-", lw=1.2, label=r"Eulerian $Y_1>1/2$")
    a.plot(t, S["Gam"]["bubble"][:, 0], "o", ms=2.2, color=C_GAL, label="LHD, bubble tracers only")
    a.plot(t, S["Gam"]["sheet"][:, 2] / S["Gam"]["sheet"][0, 2] * 0 - 0.0, alpha=0)
    a2 = a.twinx(); a2.plot(t, S["Gam"]["sheet"][:, 2] / S["Gam"]["sheet"][0, 2], "-", color="C2", lw=0.9)
    a2.set_ylabel("perimeter ratio (margin region)", color="C2", fontsize=6.5); a2.tick_params(labelsize=6)
    a.set_xlabel("t"); a.set_title("(b) bubble-only region (sheet on its edge)"); a.legend(fontsize=5.5, loc="lower left")
    a = ax[2]
    for k, c, lb in [("galerkin", C_GAL, "Galerkin"), ("rho_galerkin", "C5", r"Galerkin, $\rho$-weighted"), ("cotan", C_DEC, "DEC cotan")]:
        E = np.asarray(S["Efrac"][k]); ok = np.isfinite(E[:, 0]); a.plot(t[ok], E[ok, 0], "-", color=c, label=lb)
    a.set_xlabel("t"); a.set_ylabel("gradient energy fraction"); a.set_title("(c) Hodge split over the whole cloud\n(inertial frame)"); a.legend(fontsize=6)
    a = ax[3]
    EH, Ps, inb = S["EH"], S["Psub"], S["inb"]
    i = np.clip(np.round((Ps[:, 0] - EH["xc"][0]) / (EH["xc"][1] - EH["xc"][0])).astype(int), 0, len(EH["xc"]) - 1)
    j = np.clip(np.round((Ps[:, 1] - EH["yc"][0]) / (EH["yc"][1] - EH["yc"][0])).astype(int), 0, len(EH["yc"]) - 1)
    udp = EH["ud"][:, j, i].T
    a.plot(udp[inb].ravel(), S["ugr"][inb].ravel(), ".", ms=0.6, color="C5", alpha=0.5, label=r"Galerkin, $\rho$-weighted")
    a.plot(udp[inb].ravel(), S["ugb"][inb].ravel(), ".", ms=0.6, color=C_GAL, alpha=0.6, label="Galerkin, unweighted")
    lim = np.nanquantile(np.abs(udp[inb]), 0.998); a.plot([-lim, lim], [-lim, lim], "k-", lw=0.6)
    a.set_xlabel("Eulerian dilatational part $u_d$ (box, t=1.0)"); a.set_ylabel("LHD gradient part")
    a.set_title("(d) Hodge vs Helmholtz, box at t=1.0\nrel. L2: %.3f unw., %.3f $\\rho$-w., %.3f DEC" % (R["box"]["ug_vs_ud_relL2"], R["box"]["ugrho_vs_ud_relL2"], R["box"]["ugDEC_vs_ud_relL2"]))
    a.legend(fontsize=5.5, markerscale=8)
    fig.tight_layout(); fig.savefig("fig_sb_budget.png"); plt.close(fig)


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        globals()[nm if nm == "merge_sb" else "fig_" + nm]()
        print("ok", nm)


def ridge(v, q=0.995):
    """display stretch for ridge-type fields: remove the structural/median background, normalise."""
    v = np.asarray(v, float); b = np.maximum(v - np.nanmedian(v), 0)
    return np.clip(b / max(np.nanquantile(b, q), 1e-30), 0, 1)


def fig_dg_phys():
    d = np.load("res_dg.npz"); R = json.load(open("res_dg.json"))
    Pm, P0 = d["Pm"], d["P0"]; trm, tr0 = tri_of(Pm), tri_of(P0)
    fig, ax = plt.subplots(3, 4, figsize=(11.5, 5.4))
    a = np.nanquantile(np.abs(d["om_ex"]), 0.99)
    pf(ax[0, 0], Pm, d["om_ex"], sym=True, vmax=a, tr=trm, title=r"(a) exact $\omega$, $t=5$")
    pf(ax[0, 1], Pm, d["om_st"], sym=True, vmax=a, tr=trm, title=r"(b) LHD Stokes $\omega$ (positions only)")
    pf(ax[0, 2], Pm, d["om_ls"], sym=True, vmax=a, tr=trm, title=r"(c) LS-kNN $\omega$")
    b = np.nanquantile(np.abs(d["psi_ex"]), 0.995)
    pf(ax[0, 3], Pm, d["psi_lhd"], sym=True, vmax=b, tr=trm, cmap="PuOr_r",
       title="(d) LHD stream function (CR part)\nlines: exact $\\psi$; r = %.4f" % R["psi"]["r"])
    ax[0, 3].tricontour(trm, d["psi_ex"], levels=np.linspace(-0.9 * b, 0.9 * b, 9), colors="k", linewidths=0.5)
    for j, (k, ttl) in enumerate([("lavd_ex", "(e) exact LAVD, $[0,10]$"), ("lavd_st", "(f) Stokes-LAVD"),
                                  ("afrcE", "(g) AFRC$_E$, paper input"), ("afrcE2", "(h) AFRC$_E$, Stokes input")]):
        pf(ax[1, j], P0, norm01(d[k]), cmap="magma", vmin=0, vmax=1, tr=tr0, title=ttl)
    hm = np.mean([ridge(d["afrcH_k%d" % k]) for k in (12, 16, 24)], 0)
    for j, (v, ttl) in enumerate([(d["ftle_ex"], "(i) exact FTLE (forward)"), (d["ftle_ls"], "(j) LS-kNN FTLE"),
                                  (hm, "(k) AFRC$_H$, paper weights, 3 scales"), (d["afrcH_rb"], "(l) AFRC$_H$, coexact weights")]):
        pf(ax[2, j], P0, ridge(v), cmap="Blues", vmin=0, vmax=1, tr=tr0, title=ttl)
    fig.tight_layout(); fig.savefig("fig_dg.png"); plt.close(fig)


def fig_blast_phys():
    d = np.load("res_blast.npz"); R = json.load(open("res_blast.json"))
    P, P0, inner = d["P"], d["P0"], d["inner"]; tr = tri_of(P); tr0 = tri_of(P0); t = d["t"]
    fig, ax = plt.subplots(2, 4, figsize=(11.5, 5.6))
    pf(ax[0, 0], P, d["lnJ"][-1], cmap="viridis", vmin=-1.8, vmax=1.2, tr=tr, title=r"(a) LHD $\ln J$, $t=25$")
    pf(ax[0, 1], P, np.abs(d["lnJ_old"] - d["lnJ_ex"][-1]), cmap="magma", vmin=0, vmax=0.3, tr=tr, title=r"(b) $|\ln J$ error$|$, old re-anchoring")
    pf(ax[0, 2], P, np.abs(d["lnJ"][-1] - d["lnJ_ex"][-1]), cmap="magma", vmin=0, vmax=0.3, tr=tr, title=r"(c) $|\ln J$ error$|$, shock-deferred")
    vm = np.quantile(d["lavd"], 0.99)
    pf(ax[0, 3], P, d["lavd"], cmap="magma", vmin=0, vmax=vm, tr=tr,
       title="(d) Stokes-LAVD, raw (flow is irrotational)\np99: raw %.2f, censored %.2f, LS %.2f rad" % (
           R["spurious_lavd_p99_rad"]["raw"], R["spurious_lavd_p99_rad"]["censored"], R["spurious_lavd_p99_rad"]["LS"]))
    r = np.hypot(P[:, 0], P[:, 1]); o = np.argsort(r); m = inner[o]
    ax[1, 0].remove(); ax[1, 1].remove()
    a = fig.add_subplot(2, 4, (5, 6))
    a.plot(r[o][m], d["rho_ex"][o][m], "k-", lw=1.2, label="exact")
    for k, c, mk, lb in [("rho_dtfe", "C0", "x", "DTFE"), ("rho_sph", C_LS, "+", "SPH"), ("rho_lhd", C_GAL, "o", r"LHD $\rho(t_0)/J$")]:
        a.plot(r[o][m][::3], d[k][o][m][::3], mk, ms=(1.6 if mk == "o" else 2), mfc="none" if mk == "o" else None, color=c, label=lb)
    a.set_ylim(0, 8); a.set_xlim(0.6, 1.15); a.set_xlabel("r"); a.set_ylabel(r"$\rho$"); a.set_title(r"(e) density at $t=25$: thin Sedov shell, $\eta=6$"); a.legend(fontsize=6)
    a = ax[1, 2]
    for k, c in [("cotan", C_DEC), ("bary", "C4"), ("unit", "C8"), ("invlen2", "C5"), ("galerkin", C_GAL)]:
        E = d["Ef_" + k]; ok = np.isfinite(E[:, 1]); a.semilogy(t[ok], E[ok, 1] + 1e-12, "-", color=c, label=k)
    a.set_xlabel("t"); a.set_title("(g) coexact energy fraction\n(irrotational flow: should be 0)"); a.legend(fontsize=6)
    lnJ = d["lnJ"]; dl = np.diff(lnJ, axis=0); d2 = dl[:-1] + dl[1:]
    det = (dl < -0.25); det[:-1] |= d2 < -0.25
    first = np.where(det.any(0), det.argmax(0) + 1, -1)
    rs = np.hypot(P0[:, 0], P0[:, 1]); tc = (rs / 0.2) ** 2
    ok = (first > 0) & (rs > 0.25) & (tc < t[-1] - 1)
    tdet = np.where(ok, t[np.clip(first, 0, len(t) - 1)], np.nan)
    err = np.nanmedian(np.abs(tdet[ok] - tc[ok])) / (t[1] - t[0])
    a = ax[1, 3]
    a.scatter(rs[ok] ** 2 / 0.04, tdet[ok], s=0.3, c="C3", rasterized=True)
    xx = np.linspace(1, t[-1], 50); a.plot(xx, xx, "k-", lw=0.8)
    a.set_xlabel(r"exact arrival $t_0(r_0/R_0)^2$"); a.set_ylabel("detected (positions only)")
    a.set_title("(h) shock arrival from the C channel\nmedian error %.2f frames (Sedov $R_s\\propto t^{1/2}$)" % err)
    fig.tight_layout(); fig.savefig("fig_blast.png"); plt.close(fig)
    return err


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "phys":
    fig_dg_phys(); print("blast arrival err (frames):", fig_blast_phys())


def fig_sb_phys():
    d = np.load("res_sb_phys.npz"); S = pickle.load(open("res_sb.pkl", "rb")); G = pickle.load(open("sb_grid_tc.pkl", "rb"))
    P, P0, r0, q, db = d["P"], d["P0"], d["r0"], d["quiet"], d["dose_b"]; tr = tri_of(P)
    x0 = P0[:, 0]; edge = (db == 0) & (x0 > 0.80) & (x0 < 0.86) | (db >= 1) & (x0 > 0.84) & (x0 < 0.90)
    line = (np.abs(x0 - 0.85) < 0.004) & (P0[:, 1] < 0.40)
    fig, ax = plt.subplots(3, 4, figsize=(12, 7.6))
    a = np.nanquantile(np.abs(d["om_tc"]), 0.99)
    pf(ax[0, 0], P, d["om_tc"], sym=True, vmax=a, tr=tr, title=r"(a) DNS $\omega$ at $t_c=1.0$ (context)")
    for j, (k, ttl) in enumerate([("DNS_LAVD", "(b) DNS LAVD, $[0.6,1.4]$"), ("Stokes_LAVD", "(c) Stokes-LAVD (positions only)"),
                                  ("AFRC_E", "(d) AFRC$_E$, $[0.6,1.4]$")]):
        pf(ax[0, j + 1], P, norm01(d[k]), cmap="magma", vmin=0, vmax=1, tr=tr, title=ttl)
    for j, (k, ttl) in enumerate([("FTLE_fwd", "(e) FTLE forward $[1.0\\to1.4]$\n(repelling)"), ("AFRC_H_fwd", "(f) AFRC$_H$ forward, 3 scales"),
                                  ("FTLE_bwd", "(g) FTLE backward $[1.0\\to0.4]$\n(attracting)"), ("AFRC_H_bwd", "(h) AFRC$_H$ backward, censored")]):
        pf(ax[1, j], P, ridge(d[k]), cmap="Blues", vmin=0, vmax=1, tr=tr, title=ttl)
        if j >= 2:
            ax[1, j].plot(P[line, 0], P[line, 1], ".", ms=1.2, color="C1")
    dd = np.minimum(db, 2).astype(float)
    pf(ax[2, 0], P, dd, cmap="YlOrRd", vmin=0, vmax=2, tr=tr, title="(i) shock crossings inside the\nbackward window (orange: window-edge line)")
    ax[2, 0].plot(P[line, 0], P[line, 1], ".", ms=1.2, color="C0")
    a = ax[2, 1]; bins = np.linspace(0.62, 1.08, 24); xc_ = 0.5 * (bins[1:] + bins[:-1])
    sel = q & (P0[:, 1] < 0.40)
    for k, c, lb in [("FTLE_bwd", "C0", "FTLE backward"), ("AFRC_H_bwd", C_GAL, "AFRC$_H$ backward, censored"),
                     ("AFRC_H_bwd_uncensored", "C7", "AFRC$_H$ backward, uncensored"), ("Stokes_LAVD", "C4", "Stokes-LAVD")]:
        F = d[k] / np.nanquantile(d[k][d["struct"]], 0.95)
        prof = [np.nanmean(F[sel & (x0 >= bins[i]) & (x0 < bins[i + 1])]) for i in range(len(bins) - 1)]
        a.plot(xc_, prof, "-o", ms=2, color=c, label=lb)
    a.axvline(0.85, color="C1", lw=0.8, ls="--"); a.text(0.853, a.get_ylim()[1] * 0.92, "window-edge line", fontsize=6, color="C1")
    a.set_xlabel(r"initial position $x_0$ (quiet tracers)"); a.set_ylabel("field / structure level")
    a.set_title("(j) cut across the window-edge line\n(nothing material happens there)"); a.legend(fontsize=5.5)
    pf(ax[2, 2], P, S["node_share"], cmap="coolwarm", vmin=0, vmax=1, tr=tr, title="(k) gradient share of pair stretch,\nclean frames, $[0,1.4]$")
    pf(ax[2, 3], P, S["node_share_raw"], cmap="coolwarm", vmin=0, vmax=1, tr=tr, title="(l) gradient share incl.\nshock frames, $[0,1.4]$")
    for a_ in [ax[i, j] for i in range(3) for j in range(4) if not (i == 2 and j == 1)]:
        a_.set_xlim(np.quantile(P[:, 0], 0.002), np.quantile(P[:, 0], 0.998))
    fig.tight_layout(); fig.savefig("fig_sb_phys.png"); plt.close(fig)
    out = {}
    for k in ("FTLE_bwd", "AFRC_H_bwd", "AFRC_H_bwd_uncensored", "Stokes_LAVD", "FTLE_fwd", "AFRC_H_fwd"):
        F = d[k] / np.nanquantile(d[k][d["struct"]], 0.95)
        lo = sel & (x0 > 0.79) & (x0 < 0.85); hi = sel & (x0 > 0.85) & (x0 < 0.91)
        out[k] = float(np.nanmean(F[hi]) - np.nanmean(F[lo]))
    return out
