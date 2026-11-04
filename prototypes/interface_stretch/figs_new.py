import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.tri as mtri, matplotlib.font_manager as fm
from matplotlib.colors import TwoSlopeNorm, LogNorm
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": ["DejaVu Sans", "Noto Sans CJK JP"], "font.size": 8.5, "axes.titlesize": 9,
                     "axes.unicode_minus": False, "figure.dpi": 150})
FIG = "/tmp/work/report/"
import os; os.makedirs(FIG, exist_ok=True)
COL = dict(ref="k", vector="#1f6fd1", covector="#e8770e", combined="#7b3fb5", pair="#2a9d2a", poly="#9a9a9a")
SH = dict(ref="参照", vector="张量", covector="Nanson", combined="组合", pair="点对", poly="折线")
LAB = dict(ref="参照", vector="链式张量（新）", covector="Nanson 余向量（新）", combined="组合（新）", pair="点对中位数（旧）", poly="折线（旧）")
CEN = dict(DG=None, LO=(-3.0, 0.0), SOD=None, BLAST=(0.0, 0.0), SB=(0.75, 0.0), SV=(0.7, 0.5))


def param(name, P0):
    if name in ("DG", "SOD"):
        return P0[:, 1], "初始 y"
    c = CEN[name]; return np.degrees(np.arctan2(P0[:, 1] - c[1], P0[:, 0] - c[0])), "初始极角（度）"


def tri_plot(ax, F, val, cmap, norm=None, vmin=None, vmax=None, crop=None):
    P, tri = F["P"], F["tri"]
    v = np.where(np.isfinite(val), val, np.nan)
    T = mtri.Triangulation(P[:, 0], P[:, 1], tri)
    T.set_mask(~np.isfinite(v[tri]).all(1))
    im = ax.tripcolor(T, np.nan_to_num(v), shading="gouraud", cmap=cmap, norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if crop:
        ax.set_xlim(crop[0], crop[1]); ax.set_ylim(crop[2], crop[3])
    return im


def overlay(ax, r, name, extra=None):
    if r.get("refX") is not None:
        X = r["refX"]; ax.plot(X[:, 0], X[:, 1], "k-", lw=0.7)
    if extra is not None:
        for Xs, ls in extra:
            ax.plot(Xs[:, 0], Xs[:, 1], "k", ls=ls, lw=0.5)


def case_figure(name, g, kf, crop=None, hist_frames=None, title=""):
    O = pickle.load(open("/tmp/work/run/out_%s.pkl" % g, "rb")); B = O["base"]; res = B["res"]
    ff = max(B["fields"]); F = B["fields"][ff]; r = res[kf]
    P_ = F["P"]; cr = crop or (P_[:, 0].min(), P_[:, 0].max(), P_[:, 1].min(), P_[:, 1].max())
    asp = (cr[3] - cr[2]) / (cr[1] - cr[0]); hr = max(0.35, min(1.0, asp))
    fig, AX = plt.subplot_mosaic([["a", "b", "c"], ["d", "e", "f"]], figsize=(10.5, 3.3 * hr + 3.6), layout="constrained",
                                 height_ratios=[hr * 1.05, 1.0])
    axs = [AX["a"], AX["b"], AX["c"]]
    extra = None
    if name == "SB":
        extra = [(O["mside"][ff][0], "--"), (O["mside"][ff][2], ":")]
    # (a) context
    ax = axs[0]
    if name in ("SB", "SV"):
        Gd = O["grids"][ff] if ff in O["grids"] else O["grids"][max(O["grids"])]
        xc, yc = O["xc"], O["yc"]
        im = ax.pcolormesh(xc, yc, Gd["rho"], cmap="viridis", shading="auto", rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(cr[0], cr[1]); ax.set_ylim(cr[2], cr[3])
        ax.set_title("(a) DNS 密度 ρ，t=%.2f；黑线：参照界面" % r["t"]); plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    else:
        lab = B["phi0"]
        im = tri_plot(ax, F, lab, "twilight_shifted", vmin=np.quantile(lab, 0.01), vmax=np.quantile(lab, 0.99), crop=crop)
        ax.set_title("(a) 物质标签 φ₀（初始符号距离），t=%.2f" % r["t"]); plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    overlay(ax, r, name, extra)
    # (b) stretch field (vector)
    ax = axs[1]; v = F["vec"]; q = np.nanquantile(np.abs(v), 0.98); q = max(q, 0.05)
    im = tri_plot(ax, F, v, "RdBu_r", norm=TwoSlopeNorm(0, -q, q), crop=crop)
    overlay(ax, r, name, extra); plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    ax.set_title("(b) 拉伸场 ln λ（链式张量，φ₀ 等值线族）")
    # (c) compression field
    ax = axs[2]; cp = F["comp"]; q = max(np.nanquantile(np.abs(cp), 0.98), 0.05)
    im = tri_plot(ax, F, cp, "PuOr_r", norm=TwoSlopeNorm(0, -q, q), crop=crop)
    overlay(ax, r, name, extra); plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    ax.set_title("(c) 压缩场 −ln J（物质账本）")
    # (d) anisotropy
    ax = AX["d"]; A = np.clip(F["A"], 1, 1e3)
    im = tri_plot(ax, F, A, "magma", norm=LogNorm(1, max(10, np.nanquantile(A, 0.99))), crop=crop)
    overlay(ax, r, name, extra); plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    ax.set_title("(d) 各向异性 A = σmax(F)/λ（条件数）")
    # (e) profile
    ax = AX["e"]; per = r["per"]; s, xl = param(name, B["P0band"]); o = np.argsort(s)
    if "exact" in per:
        ax.plot(s[o], per["exact"][o], "k-", lw=1.2, label="精确（逐点）")
    elif r.get("ref_s") is not None:
        rs = r["ref_s"]; rl = r["ref_ln"]
        if name in ("SB", "SV", "LO", "BLAST"):
            rs = np.degrees(rs)
        nb = 120; edges = np.linspace(rs.min(), rs.max(), nb + 1); idx = np.clip(np.digitize(rs, edges) - 1, 0, nb - 1)
        mb = np.array([np.mean(rl[idx == k]) if np.any(idx == k) else np.nan for k in range(nb)])
        ax.plot(0.5 * (edges[1:] + edges[:-1]), mb, "k-", lw=1.2, label="参照曲线（分箱）")
    for k, mk in (("pair", "x"), ("poly", "."), ("covector", "s"), ("vector", "o")):
        ax.plot(s[o], per[k][o], mk, ms=2.6 if k != "poly" else 2.0, color=COL[k], alpha=0.8, label=LAB[k], mfc="none" if k in ("vector", "covector") else None, mew=0.7)
    lo_, hi_ = np.nanquantile(np.r_[per["vector"], per["covector"]], [0.01, 0.99]); pad = 0.3 * (hi_ - lo_) + 0.1
    ax.set_ylim(lo_ - pad, hi_ + pad); ax.set_xlabel(xl); ax.set_ylabel("ln λ"); ax.grid(alpha=0.3)
    ax.set_title("(e) 沿界面的局部拉伸，t=%.2f" % r["t"]); ax.legend(fontsize=6.3, loc="best", ncol=1)
    # (f) history
    ax = AX["f"]; fr = sorted(res); t = [res[f]["t"] for f in fr]
    refm = [res[f].get("exact_mean", res[f].get("ref_mean")) for f in fr]
    ax.plot(t, refm, "k-", lw=1.6, label=LAB["ref"])
    if "ref_side" in res[fr[0]] and len(res[fr[0]]["ref_side"]) == 3:
        lo = [min(a for a, b in res[f]["ref_side"]) for f in fr]; hi = [max(a for a, b in res[f]["ref_side"]) for f in fr]
        ax.fill_between(t, lo, hi, color="k", alpha=0.12, lw=0, label="涡片两侧 ±0.01 参照")
    for k in ("vector", "covector", "combined", "pair", "poly"):
        ax.plot(t, [res[f]["mean"][k] for f in fr], "-o" if k in ("vector",) else "--", ms=2.5, lw=1.1, color=COL[k], label=LAB[k])
    ax.set_xlabel("t"); ax.set_ylabel("⟨ln λ⟩（界面平均）"); ax.grid(alpha=0.3); ax.legend(fontsize=6.3)
    ax.set_title("(f) 界面平均拉伸的时间历程")
    fig.suptitle(title, fontsize=10.5)
    fig.savefig(FIG + "fig_%s.png" % g, bbox_inches="tight", dpi=170); plt.close(fig)


case_figure("DG", "dg", 50, title="双环流（不可压）：物质线 x₀ = 0.5")
case_figure("LO", "lo", 40, crop=(-6.0, 6.0, -6.0, 6.0), title="Lamb–Oseen 共转涡对（不可压）：绕涡圆环 r₀ = 1.5")
case_figure("SOD", "sod", 40, crop=(-0.25, 0.45, 0.0, 0.12), title="Sod 激波管（可压）：竖直物质线 x₀ = 0.1")
case_figure("BLAST", "blast", 60, title="柱面爆炸波（可压）：圆环 r₀ = 0.5")
case_figure("SB", "sb", 70, crop=(0.98, 1.52, 0.0, 0.38), title="激波–重气泡（可压，DNS）：气泡界面 r₀ = R")
case_figure("SV", "sv", 30, crop=(0.66, 1.04, 0.3, 0.7), title="激波–等熵涡（可压，DNS）：绕涡圆环 r₀ = 2R_v")

# ---------------------------------------------------------------- mechanism tests
O = pickle.load(open("/tmp/work/run/out_toy.pkl", "rb"))
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
ax = axs[0]
for key, ls in (("A_h0.05_n0", "-"), ("A_h0.025_n0", "--")):
    v = O[key]; fr = sorted(v["res"]); gt = v["gt"][fr]
    for k in ("pair", "vector", "covector"):
        ax.plot(gt, [v["res"][f]["mean"][k] for f in fr], ls, color=COL[k], lw=1.2, label=SH[k] + (" h" if ls == "-" else " h/2"))
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("γ̇t"); ax.set_ylabel("⟨ln λ⟩（精确值 0）"); ax.set_title("(a) 同心圆环：纯切向剪切"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
ax = axs[1]
for key, ls in (("A_h0.05_n0", "-"), ("A_h0.05_n0.05", "--"), ("A_h0.05_n0.1", ":")):
    v = O[key]; fr = sorted(v["res"]); gt = v["gt"][fr]
    for k in ("vector", "covector"):
        ax.plot(gt, [v["res"][f]["medabs"][k] for f in fr], ls, color=COL[k], lw=1.2, label="%s 噪声 %s" % (SH[k], key.split("_n")[1] + "h"))
ax.set_yscale("log"); ax.set_xlabel("γ̇t"); ax.set_ylabel("逐点误差中位数"); ax.set_title("(b) 同心圆环：位置噪声"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
ax = axs[2]
for key, ls in (("B_h0.04", "-"), ("B_h0.02", "--")):
    v = O[key]; fr = sorted(v["res"]); t = [v["res"][f]["t"] for f in fr]
    ax.plot(t, [v["res"][f]["exact_mean"] for f in fr], ls, color="k", lw=1.0, label="精确" if ls == "-" else None)
    for k in ("pair", "vector", "covector"):
        ax.plot(t, [v["res"][f]["mean"][k] for f in fr], ls, color=COL[k], lw=1.2, label=SH[k] + (" h" if ls == "-" else " h/2"))
ax.set_xlabel("t"); ax.set_ylabel("⟨ln λ⟩"); ax.set_title("(c) 偏心圆环卷成螺旋"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG + "fig_toy.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- convergence & noise
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
ax = axs[0]
D = pickle.load(open("/tmp/work/run/out_dg.pkl", "rb")); L = pickle.load(open("/tmp/work/run/out_lo.pkl", "rb")); Bl = pickle.load(open("/tmp/work/run/out_blast.pkl", "rb"))
series = [("DG t=10", [(2 / 80, D["nx80"]), (2 / 160, D["base"]), (2 / 320, D["nx320"])], 50),
          ("LO t=18.8", [(L["N2000"]["h"], L["N2000"]), (L["base"]["h"], L["base"]), (L["N32000"]["h"], L["N32000"])], 40)]
for (lab, ser, s), mk in zip(series, ("o", "s")):
    hs = np.array([a for a, _ in ser]); hs = hs / hs.max()
    for k in ("pair", "vector", "covector"):
        ax.loglog(hs, [max(v["res"][s]["medabs"][k], 1e-4) for _, v in ser], "-" + mk, color=COL[k], ms=3.5, lw=1.1, label="%s %s" % (lab, SH[k]))
ax.set_xlabel("示踪点间距（相对）"); ax.set_ylabel("逐点误差中位数"); ax.set_title("(a) 分辨率收敛"); ax.legend(fontsize=5.6, ncol=2); ax.grid(alpha=0.3, which="both")
ax = axs[1]
for lab, Oc, s, mk in (("DG t=10", D, 50, "o"), ("LO t=18.8", L, 40, "s")):
    for k in ("pair", "vector", "covector", "combined"):
        vals = [abs(Oc[key]["res"][s]["mean"][k] - Oc[key]["res"][s]["exact_mean"]) for key in ("base", "noise0.05", "noise0.10")]
        ax.plot([0, 0.05, 0.1], vals, "-" + mk, color=COL[k], ms=3.5, lw=1.1, label="%s %s" % (lab, SH[k]))
ax.set_xlabel("位置噪声（×h）"); ax.set_ylabel("|界面平均误差|"); ax.set_title("(b) 位置噪声（不可压）"); ax.set_yscale("log"); ax.legend(fontsize=5.6, ncol=2); ax.grid(alpha=0.3)
ax = axs[2]
S = pickle.load(open("/tmp/work/run/out_sb.pkl", "rb"))
for k in ("pair", "vector", "covector"):
    vals = [S[key]["res"][70]["mean"][k] for key in ("base", "noise0.05", "noise0.10")]
    ax.plot([0, 0.05, 0.1], vals, "-o", color=COL[k], ms=3.5, lw=1.1, label=LAB[k])
vals = [S[key]["res"][70]["mean"]["comp"] for key in ("base", "noise0.05", "noise0.10")]
ax.plot([0, 0.05, 0.1], vals, "-^", color="#b5302a", ms=3.5, lw=1.1, label="压缩 −ln J（账本）")
ax.axhline(S["base"]["res"][70]["ref_mean"], color="k", lw=1, label="拉伸参照")
ax.set_xlabel("位置噪声（×h）"); ax.set_title("(c) 激波–重气泡 t=1.4：噪声"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG + "fig_robust.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- SB attribution
fr = sorted(S["base"]["res"]); R_ = S["base"]["res"]; t = [R_[f]["t"] for f in fr]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.2))
ax = axs[0]
comp = np.array([R_[f]["mean"]["comp"] for f in fr]); vec = np.array([R_[f]["mean"]["vector"] for f in fr]); ref = np.array([R_[f]["ref_mean"] for f in fr])
lo = [min(a for a, b in R_[f]["ref_side"]) for f in fr]; hi = [max(a for a, b in R_[f]["ref_side"]) for f in fr]
ax.plot(t, comp, "-^", color="#b5302a", ms=3, label="压缩 −ln J（账本）")
ax.plot(t, vec, "-o", color=COL["vector"], ms=3, label="拉伸（链式张量）")
ax.plot(t, ref, "k-", lw=1.4, label="拉伸参照（界面中线）"); ax.fill_between(t, lo, hi, color="k", alpha=0.12, lw=0, label="涡片两侧参照范围")
ax.plot(t, comp + vec, "-", color="#444", lw=2.2, alpha=0.6, label="总增长 ln|∇c|/|∇c₀|")
old_t = [0.3, 0.6, 1.0, 1.4]; ax.plot(old_t, [-0.07, 0.46, 0.77, 1.05], "x:", color=COL["poly"], label="旧报告拉伸（折线法）")
ax.plot(old_t, [-0.08, -0.08, 0.20, 0.60], "+:", color=COL["pair"], label="旧报告拉伸（边测法）")
ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("t"); ax.set_title("(a) 气泡界面掺混归因（界面示踪点均匀平均）"); ax.legend(fontsize=6.2, ncol=2); ax.grid(alpha=0.3)
ax = axs[1]
ks = [15, 30, 50, 70]; x = np.arange(4); wdt = 0.36
c_ = [R_[k]["mean"]["comp"] for k in ks]; s_ = [R_[k]["mean"]["vector"] for k in ks]; so = [-0.07, 0.46, 0.77, 1.05]; co = [0.32, 0.48, 0.49, 0.45]
ax.bar(x - wdt / 2, co, wdt, color="#e7a29e", label="旧：压缩"); ax.bar(x - wdt / 2, so, wdt, bottom=[max(0, a) if b > 0 else 0 for a, b in zip(co, so)], color="#c8c8c8", label="旧：拉伸（折线）")
ax.bar(x + wdt / 2, c_, wdt, color="#b5302a", label="新：压缩")
ax.bar(x + wdt / 2, s_, wdt, bottom=[a if b > 0 else 0 for a, b in zip(c_, s_)], color=COL["vector"], label="新：拉伸（张量）")
ax.set_xticks(x); ax.set_xticklabels(["t=0.3", "t=0.6", "t=1.0", "t=1.4"]); ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("对 ln|∇c| 增长的贡献"); ax.set_title("(b) 新旧归因对比"); ax.legend(fontsize=6.2); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(FIG + "fig_sb_attr.png", dpi=170); plt.close(fig)
print("figures done")
