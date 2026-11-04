import sys, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import Delaunay

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "figure.dpi": 150, "savefig.dpi": 200,
                     "font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif"})


def tri_of(P, fac=3.0):
    import lhdx
    T = lhdx.complex_peeled(np.asarray(P, float))["tri"]
    return mtri.Triangulation(P[:, 0], P[:, 1], T)


def pf(ax, P, v, cmap="RdBu_r", vmin=None, vmax=None, sym=False, title=None, tr=None, cb=True, q=0.99):
    v = np.asarray(v, float).copy()
    fin = np.isfinite(v)
    if sym:
        a = np.nanquantile(np.abs(v[fin]), q) if vmax is None else vmax
        vmin, vmax = -a, a
    else:
        vmin = np.nanquantile(v[fin], 1 - q) if vmin is None else vmin
        vmax = np.nanquantile(v[fin], q) if vmax is None else vmax
    v[~fin] = 0.0 if sym else vmin
    tr = tri_of(P) if tr is None else tr
    im = ax.tripcolor(tr, v, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud", rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    if cb:
        c = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02); c.ax.tick_params(labelsize=6)
    return im


def norm01(v, q=0.995):
    v = np.asarray(v, float); hi = np.nanquantile(v, q); lo = np.nanquantile(v, 0.005)
    return np.clip((v - lo) / max(hi - lo, 1e-30), 0, 1)

# =============================================================================== DG


def fig_dg():
    d = np.load("res_dg.npz"); R = json.load(open("res_dg.json"))
    Pm, P0 = d["Pm"], d["P0"]; trm, tr0 = tri_of(Pm), tri_of(P0)
    fig, ax = plt.subplots(3, 4, figsize=(11.5, 5.4))
    a = np.nanquantile(np.abs(d["om_ex"]), 0.99)
    pf(ax[0, 0], Pm, d["om_ex"], sym=True, vmax=a, tr=trm, title=r"(a) exact $\omega$ ($t=5$)")
    pf(ax[0, 1], Pm, d["om_st"], sym=True, vmax=a, tr=trm, title=r"(b) LHD Stokes $\omega=B_2^\top f/A$  r=%.3f" % R["vort"]["stokes"]["r"])
    pf(ax[0, 2], Pm, d["om_ls"], sym=True, vmax=a, tr=trm, title=r"(c) LS-kNN $\omega$  r=%.3f" % R["vort"]["ls"]["r"])
    pf(ax[0, 3], Pm, d["div_cot"], sym=True, vmax=a, tr=trm,
       title=r"(d) LHD $\nabla\!\cdot u$ (cotan), same scale" + "\nrms/rms$\\omega$=%.3f" % R["div_noise_over_rms_vort"]["cotan"])
    for j, (k, ttl) in enumerate([("lavd_ex", "(e) exact LAVD"), ("lavd_st", "(f) Stokes-LAVD"), ("afrcE", "(g) AFRC$_E$ (paper input)"),
                                  ("afrcE2", "(h) AFRC$_E$, Stokes input")]):
        pf(ax[1, j], P0, norm01(d[k]), cmap="magma", vmin=0, vmax=1, tr=tr0, title=ttl)
    for j, (k, ttl) in enumerate([("ftle_ex", "(i) exact FTLE"), ("ftle_ls", "(j) LS-kNN FTLE"), ("afrcH", "(k) AFRC$_H$ (paper weights)"),
                                  ("afrcH_rb", "(l) AFRC$_H$, coexact-part weights")]):
        pf(ax[2, j], P0, norm01(d[k]), cmap="Blues", vmin=0, vmax=1, tr=tr0, title=ttl)
    fig.tight_layout(); fig.savefig("fig_dg.png"); plt.close(fig)

# =============================================================================== Sod


def fig_sod():
    d = np.load("res_sod.npz"); R = json.load(open("res_sod.json"))
    P, inner = d["P"], d["inner"]
    x = P[:, 0]; o = np.argsort(x); m = inner[o]
    xs = x[o][m]
    fig = plt.figure(figsize=(11.5, 6.2))
    g = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(g[0, 0])
    ax.plot(xs, d["rho_ex"][o][m], "k-", lw=1.2, label="exact")
    for k, c, mk in [("rho_lhd", "C3", "."), ("rho_dtfe", "C0", "x"), ("rho_sph", "C2", "+")]:
        ax.plot(xs[::3], d[k][o][m][::3], mk, ms=2.2, color=c, label={"rho_lhd": r"LHD $\rho(t_0)/J$", "rho_dtfe": "DTFE", "rho_sph": "SPH"}[k])
    ax.set_xlabel("x"); ax.set_ylabel(r"$\rho$"); ax.set_title(r"(a) density at $t=0.24$"); ax.legend(fontsize=6.5)
    ax.set_xlim(-0.45, 0.48)
    ax2 = ax.inset_axes([0.52, 0.45, 0.45, 0.33]); sel = (xs > 0.15) & (xs < 0.47)
    ax2.plot(xs[sel], d["rho_ex"][o][m][sel], "k-", lw=1)
    for k, c, mk in [("rho_lhd", "C3", "."), ("rho_dtfe", "C0", "x"), ("rho_sph", "C2", "+")]:
        ax2.plot(xs[sel], d[k][o][m][sel], mk, ms=1.5, color=c)
    ax2.tick_params(labelsize=5)
    ax = fig.add_subplot(g[0, 1])
    Pm = d["Pm"]; xm = Pm[:, 0]; om_ = np.argsort(xm); mm = inner[om_]
    ax.plot(xm[om_][mm], d["div_ex"][om_][mm], "k-", lw=1.2, label="exact (shock: $\\delta$)")
    ax.plot(xm[om_][mm][::2], d["div_cot"][om_][mm][::2], ".", ms=2, color="C3", label="LHD cotan")
    ax.plot(xm[om_][mm][::2], d["div_unit"][om_][mm][::2], ".", ms=2, color="C1", label="unit weights")
    ax.plot(xm[om_][mm][::2], d["div_ls"][om_][mm][::2], "+", ms=2, color="C2", label="LS-kNN")
    ax.set_ylim(-2, 6); ax.set_xlim(-0.45, 0.48); ax.set_xlabel("x"); ax.set_title(r"(b) $\nabla\cdot u$ at $t=0.235$"); ax.legend(fontsize=6.5)
    ax = fig.add_subplot(g[0, 2])
    tt = d["t"]; Fr = d["frames_sub"]
    lnJ = d["lnJ"][:, ::7]
    for i in range(0, Fr.shape[1], 9):
        ax.scatter(Fr[:, i, 0], tt, c=lnJ[:, i], s=0.6, cmap="viridis", vmin=-1.3, vmax=1.0, rasterized=True)
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_title(r"(c) tracer paths coloured by LHD $\ln J$")
    ax = fig.add_subplot(g[1, 0])
    err = np.abs(d["lnJ"][-1] - d["lnJ_ex"][-1])
    ax.semilogy(x[o][m], err[o][m] + 1e-16, ".", ms=1.5, color="C3", label="chained material (LHD)")
    ax.semilogy(x[o][m], np.abs(d["lnJ_fix"] - d["lnJ_ex"][-1])[o][m] + 1e-16, ".", ms=1.5, color="C7", label="fixed $t_0$ complex")
    ax.set_xlabel("x"); ax.set_ylabel(r"$|\ln J-\ln J_{exact}|$"); ax.set_title(r"(d) $\ln J$ error at $t=0.24$"); ax.legend(fontsize=6.5)
    ax.set_xlim(-0.45, 0.48)
    ax = fig.add_subplot(g[1, 1])
    ax.plot(xm[om_][mm], d["vort_raw"][om_][mm], ".", ms=1.5, color="C1", label="Stokes, raw")
    ax.plot(xm[om_][mm], d["vort_c"][om_][mm], ".", ms=1.5, color="C3", label="Stokes, censored")
    ax.plot(xm[om_][mm], d["om_ls"][om_][mm], "+", ms=1.5, color="C2", label="LS-kNN")
    ax.set_xlim(-0.45, 0.48); ax.set_ylim(-3, 3); ax.set_xlabel("x"); ax.set_title(r"(e) spurious $\omega$ (exact 0)"); ax.legend(fontsize=6.5)
    ax = fig.add_subplot(g[1, 2])
    for k, c in [("cotan", "C3"), ("bary", "C4"), ("unit", "C1"), ("gauss", "C0"), ("invlen2", "C5")]:
        E = d["Ef_" + k]; ok = np.isfinite(E[:, 1])
        ax.semilogy(tt[ok], E[ok, 1] + 1e-12, "-o", ms=2, color=c, label=k)
    ax.set_xlabel("t"); ax.set_ylabel("coexact energy fraction"); ax.set_title("(f) Hodge leakage (exact: pure gradient)"); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig("fig_sod.png"); plt.close(fig)

# =============================================================================== blast


def fig_blast():
    d = np.load("res_blast.npz"); R = json.load(open("res_blast.json"))
    P, inner = d["P"], d["inner"]; tr = tri_of(P)
    fig, ax = plt.subplots(2, 4, figsize=(11.5, 5.6))
    pf(ax[0, 0], P, d["lnJ"][-1], cmap="viridis", vmin=-1.8, vmax=1.0, tr=tr, title=r"(a) LHD $\ln J$ ($t=25$)")
    pf(ax[0, 1], P, np.abs(d["lnJ"][-1] - d["lnJ_ex"][-1]), cmap="magma", vmin=0, vmax=0.1, tr=tr, title=r"(b) $|\ln J$ error$|$")
    pf(ax[0, 2], P, d["lavd"], cmap="magma", vmin=0, vmax=np.quantile(d["lavd"], 0.99), tr=tr,
       title="(c) Stokes-LAVD, raw\n(exact 0) p99=%.2g rad" % R["spurious_lavd_p99_rad"]["raw"])
    pf(ax[0, 3], P, d["lavd_c"], cmap="magma", vmin=0, vmax=np.quantile(d["lavd"], 0.99), tr=tr,
       title="(d) Stokes-LAVD, censored\np99=%.2g rad" % R["spurious_lavd_p99_rad"]["censored"])
    r = np.hypot(P[:, 0], P[:, 1]); o = np.argsort(r); m = inner[o]
    ax[1, 0].remove(); ax[1, 1].remove()
    a = fig.add_subplot(2, 4, (5, 6))
    a.plot(r[o][m], d["rho_ex"][o][m], "k-", lw=1.2, label="exact")
    for k, c, mk, lb in [("rho_lhd", "C3", ".", r"LHD $\rho(t_0)/J$"), ("rho_dtfe", "C0", "x", "DTFE"), ("rho_sph", "C2", "+", "SPH")]:
        a.plot(r[o][m][::3], d[k][o][m][::3], mk, ms=2, color=c, label=lb)
    a.set_xlabel("r"); a.set_ylabel(r"$\rho$"); a.set_title(r"(e) density, radial ($\eta=6$)"); a.legend(fontsize=6.5)
    a = ax[1, 2]
    for k, c in [("cotan", "C3"), ("bary", "C4"), ("unit", "C1"), ("gauss", "C0"), ("invlen2", "C5")]:
        E = d["Ef_" + k]; ok = np.isfinite(E[:, 1])
        a.semilogy(d["t"][ok], E[ok, 1] + 1e-12, "-", color=c, label=k)
    a.set_xlabel("t"); a.set_title("(f) coexact energy fraction"); a.legend(fontsize=6)
    pf(ax[1, 3], d["Pm"], d["vort_c"], sym=True, vmax=0.05, title=r"(g) Stokes $\omega$, censored", cb=True)
    fig.tight_layout(); fig.savefig("fig_blast.png"); plt.close(fig)

# =============================================================================== shock-bubble


def fig_sb():
    S = pickle.load(open("res_sb.pkl", "rb")); R = json.load(open("res_sb.json"))
    grids, xc, yc = S["grids"], S["xc"], S["yc"]
    # ---- (1) DNS schlieren + tracer cloud
    fig, ax = plt.subplots(1, 4, figsize=(11.5, 2.5))
    keys = sorted(grids)
    sh = S["r0"] < 0.30
    frames = {0.3: 15, 0.6: 30, 1.0: 50, 1.4: 70}
    for j, tk in enumerate(keys):
        Wg = grids[tk]["W"]; gy, gx = np.gradient(Wg[0], yc, xc)
        sch = np.exp(-12 * np.hypot(gx, gy) / np.hypot(gx, gy).max())
        ax[j].imshow(np.vstack([sch[::-1], sch]), extent=[xc[0], xc[-1], -yc[-1], yc[-1]], cmap="gray", vmin=0, vmax=1, aspect="equal")
        s = frames.get(tk)
        if s is not None and s in S["keep"]:
            P = S["keep"][s]["P"]
            ax[j].plot(P[sh][::3, 0], P[sh][::3, 1], ",", color="C3", alpha=0.6)
            ax[j].plot(P[sh][::3, 0], -P[sh][::3, 1], ",", color="C3", alpha=0.6)
        ax[j].set_xlim(0.2, 2.0); ax[j].set_ylim(-0.6, 0.6); ax[j].set_title("DNS schlieren, t=%.1f" % tk)
        ax[j].set_xticks([]); ax[j].set_yticks([])
    fig.tight_layout(); fig.savefig("fig_sb_dns.png"); plt.close(fig)
    # ---- (2) instantaneous fields at t=1.0 and t=0.3
    for s in (50, 15):
        K = S["keep"][s]; P = K["P"]; tr = tri_of(P)
        fr_ = R["frames"][str(s)]
        fig, ax = plt.subplots(3, 4, figsize=(11.5, 6.6))
        a = np.nanquantile(np.abs(K["om"]), 0.99)
        pf(ax[0, 0], P, K["om"], sym=True, vmax=a, tr=tr, title=r"(a) DNS $\omega$ at tracers")
        pf(ax[0, 1], P, K["om_st"], sym=True, vmax=a, tr=tr, title=r"(b) LHD Stokes $\omega$ (censored)" + "\nr=%.3f" % fr_["vort_r"]["stokes_cens"])
        pf(ax[0, 2], P, K["om_raw"], sym=True, vmax=a, tr=tr, title="(c) LHD Stokes $\\omega$, raw\nr=%.3f" % fr_["vort_r"]["stokes_raw"])
        pf(ax[0, 3], P, K["om_ls"], sym=True, vmax=a, tr=tr, title="(d) LS-kNN $\\omega$\nr=%.3f" % fr_["vort_r"]["ls"])
        b = np.nanquantile(np.abs(K["th"]), 0.99)
        pf(ax[1, 0], P, K["th"], sym=True, vmax=b, tr=tr, title=r"(e) DNS $\nabla\cdot u$")
        pf(ax[1, 1], P, K["div"], sym=True, vmax=b, tr=tr, title="(f) LHD div, cotan\nr=%.3f" % fr_["div_r"]["cotan"])
        sc_ = np.nanmedian(np.abs(K["div"])) / max(np.nanmedian(np.abs(K["div_unit"])), 1e-30)
        pf(ax[1, 2], P, K["div_unit"] * sc_, sym=True, vmax=b, tr=tr, title="(g) unit weights (rescaled)\nr=%.3f" % fr_["div_r"]["unit"])
        pf(ax[1, 3], P, K["div_ls"], sym=True, vmax=b, tr=tr, title="(h) LS-kNN div\nr=%.3f" % fr_["div_r"]["ls"])
        lo, hi = np.nanquantile(K["rho"], 0.005), np.nanquantile(K["rho"], 0.995)
        pf(ax[2, 0], P, K["rho"], cmap="viridis", vmin=lo, vmax=hi, tr=tr, title=r"(i) DNS $\rho$ at tracers")
        for j, (k, nm) in enumerate([("rho_lhd", "LHD"), ("rho_dt", "DTFE"), ("rho_sph", "SPH")]):
            e = fr_["rho_mederr"][nm]
            pf(ax[2, j + 1], P, K[k], cmap="viridis", vmin=lo, vmax=hi, tr=tr,
               title="(%s) %s  med.err %.3f\n(interface %.3f)" % ("jkl"[j], nm, e["all"], e["interface"]))
        for a_ in ax.ravel():
            a_.set_xlim(np.nanquantile(P[:, 0], 0.002), np.nanquantile(P[:, 0], 0.998))
        fig.suptitle("shock-bubble, t = %.1f" % (s * 0.02), y=0.995)
        fig.tight_layout(); fig.savefig("fig_sb_t%02d.png" % s); plt.close(fig)
    # ---- (3) window fields on final positions
    P = S["Pend"]; tr = tri_of(P); w = S["win"]; Rw = R["window"]
    fig, ax = plt.subplots(2, 5, figsize=(12.5, 4.4))
    for j, k in enumerate(["DNS_LAVD", "Stokes_LAVD", "LS_LAVD", "AFRC_E", "AFRC_E_stokes"]):
        ttl = k.replace("_", " ") + ("" if k == "DNS_LAVD" else "\nsp=%.2f sal=%.2f" % (Rw[k]["sp_vs_DNSLAVD"], Rw[k]["salience"]))
        pf(ax[0, j], P, norm01(w[k]), cmap="magma", vmin=0, vmax=1, tr=tr, title=ttl)
    for j, k in enumerate(["FTLE_LS", "AFRC_H", "AFRC_H_rebuilt"]):
        ttl = k.replace("_", " ") + ("\nsal=%.2f" % Rw[k]["salience"] if k == "FTLE_LS" else "\nsp(FTLE)=%.2f sal=%.2f" % (Rw[k]["sp_vs_FTLE"], Rw[k]["salience"]))
        pf(ax[1, j], P, norm01(w[k]), cmap="Blues", vmin=0, vmax=1, tr=tr, title=ttl)
    pf(ax[1, 3], P, S["node_share"], cmap="coolwarm", vmin=0, vmax=1, tr=tr, title="pair stretch: gradient share\n(clean frames)")
    pf(ax[1, 4], P, S["node_share_raw"], cmap="coolwarm", vmin=0, vmax=1, tr=tr, title="gradient share incl.\nshock frames")
    for a_ in ax.ravel():
        a_.set_xlim(np.quantile(P[:, 0], 0.002), np.quantile(P[:, 0], 0.998))
    fig.tight_layout(); fig.savefig("fig_sb_window.png"); plt.close(fig)
    # ---- (4) Kelvin budget + energy fractions + box Helmholtz
    t = S["t"]; GE = S["GamE"]
    fig, ax = plt.subplots(1, 4, figsize=(12.5, 2.9))
    a = ax[0]
    a.plot(GE["t"], GE["Y2"], "k-", lw=1.2, label=r"Eulerian $\int_{Y_2>1/2}\omega$")
    a.plot(GE["t"], GE["bar"], "k--", lw=1, label=r"Eulerian $\int_0^t\!\int (\nabla\rho\times\nabla p)/\rho^2$")
    a.plot(t, S["Gam"]["sheet"][:, 0], "o", ms=2.2, color="C3", label="LHD region, all triangles")
    a.plot(t, S["Gam"]["sheet"][:, 1], "s", ms=1.8, color="C0", label="LHD region, clean triangles")
    a.set_xlabel("t"); a.set_ylabel(r"$\Gamma$"); a.set_title("(a) Kelvin budget, bubble + 0.05 sheet"); a.legend(fontsize=5.5)
    a = ax[1]
    a.plot(GE["t"], GE["Y1"], "k-", lw=1.2, label=r"Eulerian $Y_1>1/2$")
    a.plot(t, S["Gam"]["bubble"][:, 0], "o", ms=2.2, color="C3", label="LHD bubble tracers")
    a2 = a.twinx(); a2.plot(t, S["Gam"]["sheet"][:, 2] / S["Gam"]["sheet"][0, 2], "-", color="C2", lw=0.8)
    a2.set_ylabel("perimeter ratio (sheet)", color="C2", fontsize=6.5); a2.tick_params(labelsize=6)
    a.set_xlabel("t"); a.set_title("(b) bubble-only region"); a.legend(fontsize=5.5, loc="lower left")
    a = ax[2]
    for k, c in [("cotan", "C3"), ("unit", "C1"), ("gauss", "C0"), ("rho_cotan", "C5")]:
        E = S["Efrac"][k]; a.plot(t, E[:, 0], "-", color=c, label=k)
    a.set_xlabel("t"); a.set_ylabel("gradient energy fraction"); a.set_title("(c) Hodge split over the cloud"); a.legend(fontsize=6)
    a = ax[3]
    EH, Ps, inb = S["EH"], S["Psub"], S["inb"]
    from figs import tri_of as _t
    i = np.clip(np.round((Ps[:, 0] - EH["xc"][0]) / (EH["xc"][1] - EH["xc"][0])).astype(int), 0, len(EH["xc"]) - 1)
    j = np.clip(np.round((Ps[:, 1] - EH["yc"][0]) / (EH["yc"][1] - EH["yc"][0])).astype(int), 0, len(EH["yc"]) - 1)
    udp = EH["ud"][:, j, i].T
    a.plot(udp[inb].ravel(), S["ugb"][inb].ravel(), ",", color="C3", alpha=0.5, label="LHD $u_g$ (unweighted)")
    a.plot(udp[inb].ravel(), S["ugr"][inb].ravel(), ",", color="C0", alpha=0.3, label=r"LHD $u_g$ ($\rho$-weighted)")
    lim = np.nanquantile(np.abs(udp[inb]), 0.995); a.plot([-lim, lim], [-lim, lim], "k-", lw=0.6)
    a.set_xlim(-lim, lim); a.set_ylim(-1.6 * lim, 1.6 * lim)
    a.set_xlabel("Eulerian dilatational $u_d$ (box)"); a.set_title("(d) Hodge vs Helmholtz, t=1.0\nr=%.3f (unw.)  r=%.3f ($\\rho$-w.)" % (
        R["box"]["ug_vs_ud_r"], R["box"]["ugrho_vs_ud_r"])); a.legend(fontsize=5.5, markerscale=10)
    fig.tight_layout(); fig.savefig("fig_sb_budget.png"); plt.close(fig)


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        globals()["fig_" + nm]()
        print("ok", nm)
