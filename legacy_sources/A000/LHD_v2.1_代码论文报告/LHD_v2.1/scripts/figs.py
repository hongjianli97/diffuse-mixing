import numpy as np, pickle, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Noto Sans CJK JP"], "font.size": 8.5, "axes.unicode_minus": False, "figure.dpi": 150})
LC = {1: "#c0392b", 2: "#2471a3", 3: "#1e8449", 4: "#9e9e9e", 5: "#d68910"}
LN = {1: "双曲", 2: "剪切", 3: "激波", 4: "已停止", 5: "光滑波"}
def labmap(ax, P, lab, cand, B=None, s=2.0):
    ax.scatter(P[:, 0], P[:, 1], s=0.3, c="#e6e6e6", lw=0, rasterized=True)
    for v in [4, 5, 2, 1, 3]:
        m = cand & (lab == v)
        if m.any(): ax.scatter(P[m, 0], P[m, 1], s=s, c=LC[v], lw=0, label="%s %d%%" % (LN[v], round(100*m.sum()/cand.sum())), rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=6.5, markerscale=3, loc="best", framealpha=0.8)
def field(ax, P, B, title, cmap="magma", q=0.995):
    ax.scatter(P[:, 0], P[:, 1], s=0.6, c=np.clip(B, 0, np.quantile(B, q)), cmap=cmap, lw=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8)

# ---------------- fig 1: analytic growth law and signature
from scipy.stats import rankdata
def Fss(l, g, t):
    if l == 0: return np.array([[1, g*t], [0, 1.]])
    return np.array([[np.exp(l*t), g/l*np.sinh(l*t)], [0, np.exp(-l*t)]])
def beta_star(l, g, T, nd=24, ns=40):
    th = np.linspace(0, np.pi, nd, endpoint=False); r0 = np.c_[np.cos(th), np.sin(th)]
    ell = np.array([np.log(np.linalg.norm(r0@Fss(l, g, tt).T, axis=1)) for tt in np.linspace(0, T, ns+1)]).T
    e = np.argmax(ell[:, -1]-ell[:, 0]); return (ell[e, -1]-ell[e, ns//2])/np.log(2), (ell[e, -1]-ell[e, ns//2])/(ell[e, ns//2]-ell[e, 0])
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
gr = np.logspace(-1, 2.3, 40)
for lT, col in [(2, "#7fb3d5"), (3, "#2471a3"), (5, "#1b2631")]:
    bb = np.array([beta_star(1.0, g, lT) for g in gr])
    ax[0].semilogx(gr, bb[:, 1], color=col, label=r"$\lambda T=%d$" % lT)
    ax[0].semilogx(gr, lT/(lT+np.log(1+gr**2/4)), "--", color=col, lw=0.8)
ax[0].set_xlabel(r"$\gamma/\lambda$"); ax[0].set_ylabel(r"$R$"); ax[0].set_title("(a) 鞍点+剪切：半窗比 $R$（虚线为渐近式）", fontsize=7.5); ax[0].legend(fontsize=6.5)
gT = np.logspace(0.5, 2.6, 40)
bb = np.array([beta_star(0, g, 1.0) for g in gT])
ax[1].semilogx(gT, bb[:, 0], color="#2471a3", label=r"纯剪切 $\beta$")
ax[1].semilogx(gT, [beta_star(np.log(g)/1.0 if g > 1 else 0.1, 0, 1.0)[0] for g in gT], color="#c0392b", label=r"纯鞍点 $\beta$（同 $\sigma_1$）")
ax[1].axhspan(0.5, 1.5, color="#d6eaf8", zorder=0); ax[1].set_xlabel(r"$\gamma T$（或 $\sigma_1$）"); ax[1].set_ylabel(r"$\beta$"); ax[1].legend(fontsize=6.5)
ax[1].set_title(r"(b) 增长指数 $\beta$：剪切 $\to1$，双曲 $=\lambda T/(2\ln 2)$", fontsize=7.5)
for nm, col, mk in [("sod", "#1e8449", "o"), ("blast", "#d68910", "s"), ("sv", "#7d3c98", "^")]:
    d = pickle.load(open(nm+".pkl", "rb")); sg = d["sig"]; ok = np.isfinite(sg[:, 1])
    sub = np.random.default_rng(0).choice(np.where(ok)[0], min(1500, ok.sum()), replace=False)
    ax[2].scatter(sg[sub, 0], sg[sub, 1], s=1.5, c=col, lw=0, label={"sod": "Sod", "blast": "爆炸波", "sv": "激波–涡"}[nm], rasterized=True)
x = np.linspace(-1.5, 2, 200); ax[2].plot(x, np.tanh(0.5*np.abs(x)), "k-", lw=0.8, label=r"$\tanh(\frac{1}{2}|\ln J|)$")
ax[2].set_xlim(-1.2, 2.0); ax[2].set_ylim(0, 0.8); ax[2].set_xlabel(r"$-\ln J_B$"); ax[2].set_ylabel(r"$|\mu_B|$"); ax[2].legend(fontsize=6, markerscale=3)
ax[2].set_title("(c) 合并穿越块的激波签名", fontsize=7.5)
plt.tight_layout(); plt.savefig("fig1_theory.pdf"); plt.close()

# ---------------- fig 2: benchmarks
b = pickle.load(open("bench.pkl", "rb"))
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
for r, nm, tt in [(0, "shear_layer", "平行剪切层 $u=\\tanh(y/\\delta)$"), (1, "saddle", "局部鞍点")]:
    d = b[nm]; field(ax[r, 0], d["P0"], d["B"], "(%s1) %s：AFRC$_\\mathrm{H}$" % ("ab"[r], tt))
    labmap(ax[r, 1], d["P0"], d["lab"], d["cand"]); ax[r, 1].set_title("(%s2) 候选点的分类" % "ab"[r], fontsize=8)
    m = d["cand"]; ax[r, 2].scatter(np.clip(d["Rex"][m], -1, 4), np.clip(d["R"][m], -1, 4), s=2, c="k", lw=0)
    ax[r, 2].plot([-1, 4], [-1, 4], "r-", lw=0.6); ax[r, 2].axhspan(0.5, 1.5, color="#d6eaf8", zorder=0)
    ax[r, 2].set_xlabel(r"精确 $\beta$（$\sigma_1$）"); ax[r, 2].set_ylabel(r"网络 $\beta$（伸长边）"); ax[r, 2].set_title("(%s3) $\\beta$：网络 vs 精确" % "ab"[r], fontsize=8)
plt.tight_layout(); plt.savefig("fig2_bench.pdf"); plt.close()

# ---------------- fig 3: DG and LO
fig, ax = plt.subplots(2, 4, figsize=(7.4, 3.9), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1.25, 0.9]))
for r, nm in [(0, "dg"), (1, "lo")]:
    d = pickle.load(open(nm+".pkl", "rb")); rs = d["rs"]
    tri_c = rs["cen"]; ax[r, 0].scatter(tri_c[:, 0], tri_c[:, 1], s=0.3, c=rs["om_hat"], cmap="RdBu_r", vmin=-np.nanquantile(np.abs(rs["om_ex"]), .99), vmax=np.nanquantile(np.abs(rs["om_ex"]), .99), lw=0, rasterized=True)
    ax[r, 0].set_aspect("equal"); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([]); ax[r, 0].set_title("(%s1) 只凭位置的 $\\hat\\omega=2\\,\\mathrm{Im}\\,p/\\Delta t$" % "ab"[r], fontsize=7.5)
    field(ax[r, 1], d["P0"], d["B"], "(%s2) AFRC$_\\mathrm{H}$（初始位置）" % "ab"[r])
    labmap(ax[r, 2], d["P0"], d["lab"], d["cand"]); ax[r, 2].set_title("(%s3) 候选点分类" % "ab"[r], fontsize=7.5)
    m = d["cand"]; ax[r, 3].scatter(np.clip(d["bex"][m], -1, 5), np.clip(d["beta"][m], -1, 5), s=1.5, c="k", lw=0)
    ax[r, 3].plot([-1, 5], [-1, 5], "r-", lw=0.6); ax[r, 3].axhspan(0.5, 1.5, color="#d6eaf8", zorder=0)
    ax[r, 3].set_xlabel(r"精确 $\beta$", fontsize=7); ax[r, 3].set_ylabel(r"网络 $\beta$", fontsize=7); ax[r, 3].set_title("(%s4)" % "ab"[r], fontsize=7.5)
plt.tight_layout(); plt.savefig("fig3_dg_lo.pdf"); plt.close()

# ---------------- fig 4: Sod / blast
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.4))
for r, nm in [(0, "sod"), (1, "blast")]:
    d = pickle.load(open(nm+".pkl", "rb")); P0 = d["P0"]
    if nm == "sod":
        sl = (P0[:, 0] > -0.45) & (P0[:, 0] < 0.45)
        ax[r, 0].scatter(P0[:, 0], P0[:, 1], s=0.4, c=d["hotcount"], cmap="Greens", lw=0, rasterized=True); ax[r, 0].set_aspect("auto"); ax[r, 0].set_xlim(-0.5, 0.5)
    else:
        ax[r, 0].scatter(P0[:, 0], P0[:, 1], s=0.4, c=d["hotcount"], cmap="Greens", lw=0, rasterized=True); ax[r, 0].set_aspect("equal")
    ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([]); ax[r, 0].set_title("(%s1) 热步数（面积单步跳跃）" % "ab"[r], fontsize=7.5)
    bins = np.logspace(-16, 0.3, 50)
    ax[r, 1].hist(np.maximum(d["pst"], 1e-16), bins=bins, color="#999", alpha=0.8, label="逐步 $\\sum|\\arg p_k|$")
    ax[r, 1].hist(np.maximum(d["pbl"], 1e-16), bins=bins, color="#1e8449", alpha=0.7, label="合并块 $|\\arg p_B|$")
    ax[r, 1].set_xscale("log"); ax[r, 1].set_xlabel("伪转角 (rad)"); ax[r, 1].legend(fontsize=6.5); ax[r, 1].set_title("(%s2) 跨激波伪转角" % "ab"[r], fontsize=7.5)
    f = d["ftle"]; labmap(ax[r, 2], P0, f["lab"], f["cand"], s=1.5); ax[r, 2].set_title("(%s3) 后向 FTLE 候选点分类" % "ab"[r], fontsize=7.5)
    if nm == "sod": ax[r, 2].set_aspect("auto"); ax[r, 2].set_xlim(-0.5, 0.5)
plt.tight_layout(); plt.savefig("fig4_shock.pdf"); plt.close()

# ---------------- fig 5: SV
d = pickle.load(open("sv.pkl", "rb")); rsv = json.load(open("res_sv.json"))
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.6))
sn = d["snap"]; hk = sn["hk"]
ax[0].scatter(sn["om_ref"][~hk], sn["om_hat"][~hk], s=0.5, c="k", lw=0, rasterized=True, label="非热三角形")
ax[0].scatter(sn["om_ref"][hk], sn["om_hat"][hk], s=0.8, c="#1e8449", lw=0, rasterized=True, label="跨激波三角形")
lim = np.quantile(np.abs(sn["om_ref"]), .999); ax[0].plot([-lim, lim], [-lim, lim], "r-", lw=0.6); ax[0].set_xlabel("DNS $\\omega$"); ax[0].set_ylabel("$\\hat\\omega$"); ax[0].legend(fontsize=6, markerscale=4)
ax[0].set_title("(a) $t=0.25$ 的步涡量", fontsize=7.5)
tm = 0.5*(d["t"][1:]+d["t"][:-1])
ax[1].plot(tm, d["Gpos"], "k-", lw=1, label="位置形式 $\\sum 2A\\,\\mathrm{Im}\\,p/\\Delta t$"); ax[1].plot(tm, d["Gdns"], "r--", lw=1, label="DNS $\\int\\omega\\,dA$")
hs = np.r_[d["hstep"], False]
for i in np.where(hs)[0]: ax[1].axvspan(tm[i]-0.01, tm[i]+0.01, color="#d5f5e3", lw=0)
ax[1].set_xlabel("$t$"); ax[1].legend(fontsize=6); ax[1].set_title("(b) 涡核物质区域的环量", fontsize=7.5)
labmap(ax[2], d["P0"], d["labf"], d["candf"], s=1.2); ax[2].set_title("(c) 后向 FTLE 候选点分类", fontsize=7.5)
H = [rsv["hodge_raw"], rsv["hodge_raw_boost"], rsv["hodge_meanfree"], rsv["hodge_aligned"], rsv["hodge_aligned_boost"]]
nm = ["原始", "原始\n+平移", "去均\n值", "Kab-\nsch", "Kabsch\n+平移"]
ax[3].bar(range(5), [h[0] for h in H], color="#d68910", label="梯度"); ax[3].bar(range(5), [h[1] for h in H], bottom=[h[0] for h in H], color="#2471a3", label="余正合")
ax[3].set_xticks(range(5)); ax[3].set_xticklabels(nm, fontsize=6); ax[3].set_ylim(0, 1); ax[3].legend(fontsize=6); ax[3].set_title("(d) 位移上链的 Hodge 能量", fontsize=7.5)
plt.tight_layout(); plt.savefig("fig5_sv.pdf"); plt.close()
print("ok")
