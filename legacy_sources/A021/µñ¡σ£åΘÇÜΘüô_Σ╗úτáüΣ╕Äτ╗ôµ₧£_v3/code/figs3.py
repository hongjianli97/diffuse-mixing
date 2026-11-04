import json, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "savefig.dpi": 200, "font.family": "DejaVu Serif"})


def project(P, v, nx=320, bbox=None, fac=1.3):
    """Shepard-normalised Wendland C2 projection of particle values onto a grid (as in the paper's projector).
    Support radius = fac x the 8th-neighbour distance (adaptive to the local tracer spacing)."""
    P = np.asarray(P, float); v = np.asarray(v, float)
    ok = np.isfinite(v); P, v = P[ok], v[ok]
    if bbox is None:
        bbox = (P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max())
    x0, x1, y0, y1 = bbox; ny = max(int(nx * (y1 - y0) / (x1 - x0)), 10)
    gx, gy = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny); GX, GY = np.meshgrid(gx, gy)
    tree = cKDTree(P); d8 = tree.query(P, k=9)[0][:, -1]; R = fac * np.median(d8)
    G = np.c_[GX.ravel(), GY.ravel()]
    idx = tree.query_ball_point(G, R)
    lens = np.fromiter((len(i) for i in idx), int, len(idx))
    gi = np.repeat(np.arange(len(G)), lens); pi = np.fromiter((j for i in idx for j in i), int, lens.sum())
    q = np.linalg.norm(G[gi] - P[pi], axis=1) / R
    w = (1 - q) ** 4 * (1 + 4 * q)
    num = np.bincount(gi, w * v[pi], len(G)); den = np.bincount(gi, w, len(G))
    F = np.where(den > 0.15, num / np.maximum(den, 1e-300), np.nan).reshape(ny, nx)
    return F, (x0, x1, y0, y1)


def show(ax, P, v, cmap="magma", vmin=None, vmax=None, title=None, sym=False, bbox=None, cb=True):
    F, ext = project(P, v, bbox=bbox)
    fin = np.isfinite(F)
    if sym:
        a = np.nanquantile(np.abs(F[fin]), 0.99) if vmax is None else vmax; vmin, vmax = -a, a
    else:
        vmin = np.nanquantile(F[fin], 0.005) if vmin is None else vmin
        vmax = np.nanquantile(F[fin], 0.995) if vmax is None else vmax
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("white")
    im = ax.imshow(F, origin="lower", extent=ext, cmap=cm, vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if cb:
        c = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02); c.ax.tick_params(labelsize=6)
    return F, ext


def n01(v, q=0.995):
    v = np.asarray(v, float); lo = np.nanquantile(v, 0.005); hi = np.nanquantile(v, q)
    return np.clip((v - lo) / max(hi - lo, 1e-30), 0, 1)


def ridge(v, q=0.995):
    v = np.asarray(v, float); b = np.maximum(v - np.nanmedian(v), 0)
    return np.clip(b / max(np.nanquantile(b, q), 1e-30), 0, 1)


def fig_afrcE():
    F = pickle.load(open("res_afrcE_study.pkl", "rb")); R = json.load(open("res_afrcE_study.json"))
    Rr = json.load(open("res_afrcE_res.json"))
    cs = [("LO_Rd6.5", "Lamb-Oseen pair"), ("SB", "shock-bubble, $[0.6,1.4]$"), ("DG", "double gyre, $[0,10]$")]
    fig, ax = plt.subplots(3, 5, figsize=(13.5, 7.2))
    for i, (c, nm) in enumerate(cs):
        P = F[c]["P"]; r = R[c]
        show(ax[i, 0], P, n01(F[c]["lavd"]), vmin=0, vmax=1, title="%s\n(a%d) LAVD (reference physics)" % (nm, i + 1))
        show(ax[i, 1], P, n01(F[c]["weight"]), vmin=0, vmax=1,
             title="(b%d) AFRC$_E$ input: co-rotation weight\ntop-10%% Jaccard vs LAVD %.2f" % (i + 1, r["weight"]["J"]))
        show(ax[i, 2], P, n01(F[c]["AFRC_E_median"]), vmin=0, vmax=1,
             title="(c%d) AFRC$_E$ output (curvature excess)\nJaccard %.2f" % (i + 1, r["AFRC_E_median"]["J"]))
        Fw, ext = show(ax[i, 3], P, n01(F[c]["weight"]), vmin=0, vmax=1, title="(d%d) proposed: core = weight,\nrim = AFRC$_E$ contour (cyan)" % (i + 1))
        Fe, _ = project(P, n01(F[c]["AFRC_E_median"]))
        ax[i, 3].contour(np.linspace(ext[0], ext[1], Fe.shape[1]), np.linspace(ext[2], ext[3], Fe.shape[0]), np.nan_to_num(Fe), levels=[0.5], colors="c", linewidths=0.7)
        if i == 0:
            a = ax[i, 4]; rr = np.arange(0, 2.5, 0.25) + 0.125
            a.plot(rr, Rr["4000"]["lavd"], "k-", lw=1.4, label="LAVD")
            a.plot(rr, Rr["4000"]["weight"], "-", color="C0", label="co-rotation weight")
            for N, ls in (("1000", ":"), ("4000", "--"), ("16000", "-")):
                a.plot(rr, Rr[N]["AFRC_E"], ls, color="C3", label="AFRC$_E$, N=%s (core = %.0f spacings)" % (N, Rr[N]["core_radius_in_spacings"]))
            a.set_xlabel("distance from vortex centre / $a$"); a.set_ylabel("normalised"); a.legend(fontsize=5.5)
            a.set_title("(e) radial profiles, Lamb-Oseen ($k$ fixed)")
        else:
            ax[i, 4].axis("off")
    fig.tight_layout(); fig.savefig("fig_afrcE.png"); plt.close(fig)


def fig_afrcH():
    F = pickle.load(open("res_afrc_improve.pkl", "rb")); R = json.load(open("res_afrc_improve.json"))["DG"]
    d = F["DG"]; fig, ax = plt.subplots(2, 3, figsize=(11.5, 4.6))
    show(ax[0, 0], d["P0"], ridge(d["ftle_f"]), cmap="Blues", vmin=0, vmax=1, title="(a) exact forward FTLE (repelling), at $x(0)$")
    show(ax[0, 1], d["P0"], d["H_abs"], cmap="Blues", vmin=0, vmax=1, title="(b) AFRC$_H$, $|$median$|$ (paper), at $x(0)$\nSpearman vs (a) %.2f" % R["H_abs"]["sp_fwdFTLE"])
    show(ax[0, 2], d["P0"], d["H_sf"], cmap="Blues", vmin=0, vmax=1, title="(c) AFRC$_H$, signed: separating pairs only\nSpearman vs (a) %.2f" % R["H_sf"]["sp_fwdFTLE"])
    show(ax[1, 0], d["P1"], ridge(d["ftle_b"]), cmap="Reds", vmin=0, vmax=1, title="(d) backward FTLE (attracting), at $x(10)$")
    show(ax[1, 1], d["P1"], d["H_abs"], cmap="Reds", vmin=0, vmax=1, title="(e) AFRC$_H$, $|$median$|$, same values at $x(10)$\nSpearman vs (d) %.2f" % R["H_abs"]["sp_bwdFTLE"])
    show(ax[1, 2], d["P1"], d["H_sb"], cmap="Reds", vmin=0, vmax=1, title="(f) AFRC$_H$, signed, reversed time\nSpearman vs (d) %.2f" % R["H_sb"]["sp_bwdFTLE"])
    fig.tight_layout(); fig.savefig("fig_afrcH.png"); plt.close(fig)


def fig_sv():
    S = pickle.load(open("res_sv.pkl", "rb")); R = json.load(open("res_sv.json")); F = S["F"]
    fig = plt.figure(figsize=(13.5, 8.4)); g = fig.add_gridspec(3, 5)
    xc, yc = S["xc"], S["yc"]
    for j, tk in enumerate((0.2, 0.6)):
        a = fig.add_subplot(g[0, j]); rho = S["grids"][tk]["rho"]; gy, gx = np.gradient(rho, yc, xc); Gm = np.hypot(gx, gy)
        a.imshow(np.exp(-15 * Gm / Gm.max()), origin="lower", extent=[xc[0], xc[-1], yc[0], yc[-1]], cmap="gray")
        a.set_xlim(0.3, 1.5); a.set_ylim(0.1, 0.9); a.set_xticks([]); a.set_yticks([]); a.set_title("DNS schlieren, t = %.1f" % tk)
    a = fig.add_subplot(g[0, 2]); show(a, F["off"]["P_b"], S["posdiff"] / 0.005, cmap="viridis", vmin=0, vmax=1.5,
                                       title="(c) |offline - in-loop| tracer position\nat t=0.6, in tracer spacings")
    a = fig.add_subplot(g[0, 3]); P0 = S["P0"]; q = S["quiet"]; x0 = P0[:, 0]
    bins = np.arange(0.47, 1.03, 0.005); xb = 0.5 * (bins[1:] + bins[:-1])
    for k, c in (("FTLE_bwd", "C0"), ("AFRC_H_bwd", "C3"), ("Stokes_LAVD", "C4")):
        for mode, ls in (("in", "-"), ("off", ":")):
            v = F[mode][k]; v = v / np.nanquantile(np.abs(F["in"][k]), 0.99)
            pr = [np.nanmean(v[q & (x0 >= bins[i]) & (x0 < bins[i + 1])]) for i in range(len(xb))]
            a.plot(xb, pr, ls, color=c, lw=0.9, label=k.replace("_", " ") + " (" + mode + ")")
    a.axvline(S["xe"], color="C1", ls="--", lw=0.8); a.set_xlabel("$x_0$ (quiet tracers)"); a.legend(fontsize=5)
    a.set_title("(d) cut across the window-edge line\nsolid: in-loop, dotted: offline")
    a = fig.add_subplot(g[0, 4]); ks = ["FTLE_fwd", "FTLE_bwd", "AFRC_H_bwd", "Stokes_LAVD", "LS_LAVD"]
    xx = np.arange(len(ks)); w = 0.38
    a.bar(xx - w / 2, [R[k]["stripe_in"] for k in ks], w, color="C0", label="in-loop")
    a.bar(xx + w / 2, [R[k]["stripe_off"] for k in ks], w, color="C3", label="offline")
    a.set_yscale("log"); a.set_xticks(xx); a.set_xticklabels([k.replace("_", "\n") for k in ks], fontsize=6); a.legend(fontsize=6)
    a.set_title("(e) stripe amplitude at the shock's\ntravel per snapshot (%.3f)" % S["lam"])
    for i, mode in enumerate(("in", "off")):
        f = F[mode]
        for j, (k, P, cm, ttl) in enumerate([("FTLE_fwd", f["P_a"], "Blues", "FTLE forward, at $x(0.2)$"), ("FTLE_bwd", f["P_b"], "Reds", "FTLE backward, at $x(0.6)$"),
                                             ("AFRC_H_bwd", f["P_b"], "Reds", "AFRC$_H$ (time-symmetric), at $x(0.6)$"),
                                             ("Stokes_LAVD", f["P_b"], "magma", "Stokes-LAVD $[0.2,0.6]$"), ("LS_LAVD", f["P_b"], "magma", "LS-kNN LAVD")]):
            a = fig.add_subplot(g[i + 1, j])
            v = ridge(f[k]) if "LAVD" not in k else n01(f[k])
            show(a, P, v, cmap=cm, vmin=0, vmax=1, title=("in-loop: " if mode == "in" else "offline: ") + ttl, cb=False)
    fig.tight_layout(); fig.savefig("fig_sv.png"); plt.close(fig)


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        globals()["fig_" + nm](); print("ok", nm)


def fig_afrcH2():
    O = pickle.load(open("res_afrcH_paper.pkl", "rb"))
    fig, ax = plt.subplots(3, 4, figsize=(13, 8.0))
    d = O["DG"]
    show(ax[0, 0], d["P"], ridge(d["ftle"]), cmap="Blues", vmin=0, vmax=1, title="double gyre: exact forward FTLE")
    show(ax[0, 1], d["P"], n01(d["raw"]), cmap="Blues", vmin=0, vmax=1, title="AFRC$_H$, 3 scales, max-combine")
    show(ax[0, 2], d["P"], n01(d["post"]), cmap="Blues", vmin=0, vmax=1, title="AFRC$_H$ + paper post-processing\n(soft closing + graph TV)")
    ax[0, 3].axis("off")
    s = O["SB"]
    for j, (k, cm, ttl) in enumerate([("ftle_f", "Blues", "shock-bubble: FTLE forward $[1.0\\to1.4]$"), ("fwd", "Blues", "AFRC$_H$ forward (paper post)"),
                                      ("ftle_b", "Reds", "FTLE backward $[1.0\\to0.4]$"), ("bwd", "Reds", "AFRC$_H$ backward (paper post)")]):
        show(ax[1, j], s["P"], ridge(s[k]) if "ftle" in k else n01(s[k]), cmap=cm, vmin=0, vmax=1, title=ttl)
    for j, (md, k, cm, ttl) in enumerate([("SV_in", "ftle_b", "Reds", "shock-vortex, in-loop: FTLE bwd"), ("SV_in", "bwd", "Reds", "in-loop: AFRC$_H$ bwd"),
                                          ("SV_off", "ftle_b", "Reds", "offline: FTLE bwd"), ("SV_off", "bwd", "Reds", "offline: AFRC$_H$ bwd")]):
        d = O[md]; show(ax[2, j], d["P"], ridge(d[k]) if "ftle" in k else n01(d[k]), cmap=cm, vmin=0, vmax=1, title=ttl)
    fig.tight_layout(); fig.savefig("fig_afrcH2.png"); plt.close(fig)
