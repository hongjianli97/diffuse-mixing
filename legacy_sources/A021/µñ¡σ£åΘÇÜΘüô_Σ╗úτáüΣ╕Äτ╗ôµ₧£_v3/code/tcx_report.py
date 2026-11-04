import json, pickle, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_p = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; fm.fontManager.addfont(_p); fam = fm.FontProperties(fname=_p).get_name()
plt.rcParams.update({"font.family": [fam, "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 8.5})
OUT = "/home/claude/tcxrep"; os.makedirs(OUT, exist_ok=True)
CASES = ["DG", "LO", "SOD", "BLAST", "SB", "SV"]; LAB = dict(DG="双环流", LO="Lamb–Oseen", SOD="Sod", BLAST="爆炸波", SB="激波–气泡", SV="激波–涡")
R = {c: json.load(open(f"/home/claude/bench/tcx_res_{c}.json")) for c in CASES}
F = {c: pickle.load(open(f"/home/claude/bench/tcx_fields_{c}.pkl", "rb")) for c in CASES}
vort = ["DG", "LO", "SB", "SV"]; irr = ["SOD", "BLAST"]

# ---------------- figure 1: phi network vs reference (vortical), spurious rotation (irrotational)
fig, ax = plt.subplots(1, 6, figsize=(15, 2.8), constrained_layout=True)
for i, c in enumerate(vort):
    f = F[c]; m = f["inner"]; x, y = f["phi_ref"][m], f["phi"][m]
    sel = np.random.default_rng(0).choice(len(x), min(6000, len(x)), replace=False)
    ax[i].scatter(x[sel], y[sel], s=1, c="#1f5fa8", alpha=0.4, rasterized=True); lim = np.abs(x).max() * 1.05
    ax[i].plot([-lim, lim], [-lim, lim], "k:", lw=0.8); ax[i].set_xlim(-lim, lim); ax[i].set_ylim(-lim, lim)
    p = R[c]["P3_phi"]["new"]; ax[i].set_title(f"{LAB[c]}：r = {p['pearson']:.4f}", fontsize=9, loc="left")
    ax[i].set_xlabel("参照 ½∫(ω−ω̄)dt"); ax[i].set_ylabel("网络 φ") if i == 0 else None
for j, c in enumerate(irr):
    a = ax[4 + j]; v = R[c]["P3_phi"]
    keys = [("old_circ_retri", "旧：环量\n逐帧重剖"), ("argp_retri", "arg p\n逐帧重剖"), ("new_nomerge", "锚定\n不合并"), ("new", "锚定\n合并穿越")]
    a.bar(range(4), [v[k]["p99"] for k, _ in keys], color=["#7f7f7f", "#9fb7d6", "#6a9fd4", "#1f5fa8"])
    a.set_xticks(range(4)); a.set_xticklabels([l for _, l in keys], fontsize=7); a.set_ylabel("伪转角 p99（rad）")
    a.set_title(f"{LAB[c]}：参照为 0", fontsize=9, loc="left")
fig.savefig(f"{OUT}/fig_phi.pdf")

# ---------------- figure 2: objectivity, signature, noise
fig, ax = plt.subplots(1, 3, figsize=(15, 3.3), constrained_layout=True)
x = np.arange(6); w = 0.38
ax[0].bar(x - w / 2, [max(R[c]["P1_objectivity"]["phi_old_circ"], 1e-16) for c in CASES], w, color="#7f7f7f", label="旧：c/A 环量")
ax[0].bar(x + w / 2, [max(R[c]["P1_objectivity"]["phi_new"], 1e-16) for c in CASES], w, color="#1f5fa8", label="新：arg p")
ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels([LAB[c] for c in CASES], fontsize=7.5)
ax[0].set_ylabel("旋转参考系下 φ 的最大相对变化"); ax[0].legend(fontsize=7.5); ax[0].set_title("(a) 客观性（0.6 rad 转动 + 平移）", loc="left", fontsize=9)
xx = np.linspace(0, 2.0, 100); ax[1].plot(xx, np.tanh(xx / 2), "k-", lw=1, label="单轴压缩曲线")
cols = dict(SOD="#d62728", BLAST="#ff7f0e", SB="#2ca02c", SV="#9467bd")
for c in ["SOD", "BLAST", "SB", "SV"]:
    s = F[c].get("sig")
    if s is None:
        continue
    tc = s[:, 3] > 0.05
    ax[1].scatter(-s[tc, 0], s[tc, 1], s=2, c=cols[c], alpha=0.5, label=LAB[c], rasterized=True)
ax[1].set_xlim(0, 2.0); ax[1].set_ylim(0, 0.8); ax[1].set_xlabel("块压缩 −ln J_B"); ax[1].set_ylabel("|μ_B|"); ax[1].legend(fontsize=7, markerscale=4)
ax[1].set_title("(b) 激波签名：参照压缩 > 0.05 的块", loc="left", fontsize=9)
na = [c for c in vort]
ax[2].bar(np.arange(4) - w / 2, [R[c]["P4_noise"]["phi_old_circ_retri"]["rel_rms"] for c in na], w, color="#7f7f7f", label="旧：环量逐帧重剖")
ax[2].bar(np.arange(4) + w / 2, [R[c]["P4_noise"]["phi_new_anchored"]["rel_rms"] for c in na], w, color="#1f5fa8", label="新：锚定")
ax[2].set_xticks(range(4)); ax[2].set_xticklabels([LAB[c] for c in na]); ax[2].set_ylabel("φ 相对 RMS 误差（0.1h 噪声）"); ax[2].legend(fontsize=7.5)
ax[2].set_title("(c) 位置噪声下的转角", loc="left", fontsize=9)
fig.savefig(f"{OUT}/fig_obj_sig_noise.pdf")

# ---------------- figure 3: circulation jump vs cell size; persistence
fig, ax = plt.subplots(1, 2, figsize=(12, 3.2), constrained_layout=True)
for c in vort:
    P7 = R[c]["P7_circulation"]
    ax[0].plot([1, 2, 4], [P7[f"clean_chain5_m{m}"]["pearson"] for m in (1, 2, 4)], "o-", label=f"{LAB[c]} 干净")
    ax[0].plot([1, 2, 4], [P7[f"noise_single_m{m}"]["pearson"] for m in (1, 2, 4)], "s--", label=f"{LAB[c]} 噪声（单次锚定）")
ax[0].set_xlabel("每个胞腔的三角形数 m"); ax[0].set_ylabel("环量跳变与参照的 Pearson"); ax[0].set_xticks([1, 2, 4]); ax[0].legend(fontsize=6.5, ncol=2)
ax[0].set_title("(a) 胞腔合并与环量跳变", loc="left", fontsize=9)
xc = np.arange(6)
ax[1].bar(xc - w / 2, [R[c]["P10_persistence"]["cores_ref"] for c in CASES], w, color="#7f7f7f", label="参照场的核数")
ax[1].bar(xc + w / 2, [R[c]["P10_persistence"]["cores_net"] for c in CASES], w, color="#1f5fa8", label="网络 φ 的核数")
ax[1].set_xticks(xc); ax[1].set_xticklabels([LAB[c] for c in CASES], fontsize=7.5); ax[1].set_ylabel("持续性 > 阈值的核数"); ax[1].legend(fontsize=7.5)
ax[1].set_title("(b) 持续性核数（阈值 = 2·max(√(2lnN)·粗糙度, 系统误差)）", loc="left", fontsize=9)
fig.savefig(f"{OUT}/fig_circ_persist.pdf")
print("figures done")
