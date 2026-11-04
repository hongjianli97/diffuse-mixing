exec(open("figs.py").read().split("# ---------------- fig 1")[0])
import json, sys
what = sys.argv[1:]
if "rm" in what:
    d = pickle.load(open("rm.pkl", "rb")); r = json.load(open("res_rm.json"))
    fig = plt.figure(figsize=(7.4, 4.6)); gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    for j, tk in enumerate([0.8, 1.6, 2.0]):
        ax = fig.add_subplot(gs[0, j]); W = d["Ws"][[k for k in d["Ws"] if abs(float(k)-tk) < 1e-6][0]]
        ax.imshow(W[0], origin="lower", extent=[d["xc"][0], d["xc"][-1], d["yc"][0], d["yc"][-1]], cmap="viridis", vmin=1, vmax=7, aspect="equal")
        s = int(round(tk/0.04)); P = d["frames_sel"][s]; hv = d["heavy"]
        ax.scatter(P[hv, 0], P[hv, 1], s=0.08, c="w", lw=0, alpha=0.5, rasterized=True)
        ax.set_xlim(P[:, 0].min()-0.05, P[:, 0].max()+0.05); ax.set_ylim(0, 0.5); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("(a%d) DNS 密度与重气体示踪点，$t=%.1f$" % (j+1, tk), fontsize=7)
    ax = fig.add_subplot(gs[0, 3])
    e = np.abs(d["rho0"]*np.exp(-d["lnJ_final"])/d["rho_final"]-1)
    ax.scatter(d["PT"][:, 0], d["PT"][:, 1], s=0.4, c=np.log10(np.maximum(e, 1e-4)), cmap="magma", vmin=-3, vmax=0, lw=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("(a4) 账本密度误差 $\\log_{10}$，$t=2$", fontsize=7)
    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(d["t"], d["amp"], "k-", lw=1.2, label="界面示踪点（只凭位置）")
    ad = np.array(d["amp_dns"]); ax.plot(ad[:, 0], ad[:, 1], "rs", ms=4, label="DNS $Y=0.5$ 等值线")
    rc = d["rich"]; tt = np.linspace(0.5, 1.5, 10)
    ax.plot(tt, d["amp"][15]+rc["rate_model"]*(tt-0.6), "--", color="#2471a3", lw=1, label="Richtmyer（参数取自位置）")
    ax.plot(tt, d["amp"][15]+rc["rate_from_circulation"]*(tt-0.6), ":", color="#1e8449", lw=1.2, label="由沉积环量 $k\\Gamma/4$")
    ax.set_xlabel("$t$"); ax.set_ylabel("振幅 $a$"); ax.legend(fontsize=6); ax.set_title("(b) 单模 RM 振幅", fontsize=7.5)
    ax = fig.add_subplot(gs[1, 2])
    tm = 0.5*(d["t"][1:]+d["t"][:-1]); ax.plot(tm, d["Gp"], "k-", lw=1, label="位置形式"); ax.plot(tm, d["Gd"], "r--", lw=1, label="DNS")
    ax.set_xlabel("$t$"); ax.set_title("(c) 界面物质带的环量", fontsize=7.5); ax.legend(fontsize=6)
    ax = fig.add_subplot(gs[1, 3]); T9 = r["T9"]; ks = sorted(T9, key=lambda k: float(k[2:]))
    tt = [float(k[2:]) for k in ks]
    ax.plot(tt, [T9[k]["compression"] for k in ks], "o-", color="#c0392b", ms=3, label="压缩 $-\\ln J$")
    ax.plot(tt, [T9[k]["stretch_mean"] for k in ks], "s-", color="#2471a3", ms=3, label="拉伸 $\\ln\\lambda$")
    ax.plot(tt, [T9[k]["ln_mean_lambda"] for k in ks], "^-", color="#7fb3d5", ms=3, label="$\\ln\\langle\\lambda\\rangle$")
    ax.plot(tt, [T9[k]["ln_L_dns"] for k in ks], "k--", lw=0.8, label="DNS $\\ln(L/L_0)$")
    ax.set_xlabel("$t$"); ax.legend(fontsize=5.5); ax.set_title("(d) 界面掺混归因", fontsize=7.5)
    plt.tight_layout(); plt.savefig("figR_rm.pdf"); plt.close()
if "base" in what:
    b = json.load(open("res_b1_density.json")); b2 = json.load(open("res_b2_diff.json"))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    cs = ["dg", "lo", "sod", "blast", "sv"]+(["sb"] if "sb" in b else [])+(["rm"] if "rm" in b else [])
    nm = dict(dg="双环流", lo="LO 涡对", sod="Sod", blast="爆炸波", sv="激波–涡", sb="激波–气泡", rm="RM")
    for j, (meth, col) in enumerate([("ledger", "#1f5fa8"), ("sph", "#d68910"), ("dtfe", "#999999")]):
        v = [max(b[c][meth]["med"], 1e-5) for c in cs]
        ax[0].bar(np.arange(len(cs))+0.27*(j-1), v, 0.26, color=col, label={"ledger": "账本", "sph": "SPH", "dtfe": "DTFE"}[meth])
    ax[0].set_yscale("log"); ax[0].set_xticks(range(len(cs))); ax[0].set_xticklabels([nm[c] for c in cs], fontsize=7); ax[0].set_ylabel("密度中位相对误差")
    ax[0].legend(fontsize=6.5); ax[0].set_title("(a) 密度：账本 vs DTFE / SPH（同一组质量）", fontsize=7.5); ax[0].set_ylim(1e-5, 1)
    cs2 = [c for c in ["dg", "sod", "blast", "sb"] if c in b2]
    for j, (meth, col, lab) in enumerate([("network_mass", "#1f5fa8", "网络·质量形式"), ("sph", "#d68910", "SPH 传导"), ("network_volume", "#999999", "网络·体积形式")]):
        ax[1].bar(np.arange(len(cs2))+0.27*(j-1), [b2[c][meth] for c in cs2], 0.26, color=col, label=lab)
    ax[1].set_yscale("log"); ax[1].set_xticks(range(len(cs2))); ax[1].set_xticklabels([nm[c] for c in cs2], fontsize=7); ax[1].set_ylabel("相对 $L^2$ 误差")
    ax[1].legend(fontsize=6); ax[1].set_title("(b) 扩散：网络 vs SPH", fontsize=7.5)
    plt.tight_layout(); plt.savefig("figR_base.pdf"); plt.close()
print("ok")
if "dns" in what:
    dsb = json.load(open("res_dnsref_sb.json")); dsv = json.load(open("res_dnsref_sv.json"))
    bands = ["-0.06..-0.03", "-0.03..-0.015", "-0.015..-0.005", "-0.003..0.003", "0.005..0.015", "0.015..0.03", "0.03..0.06"]
    xc = [np.mean([float(v) for v in b.replace("..", " ").split()]) for b in bands]
    lv = [k for k in ["sb_dx0.01", "sb_dx0.005", "sb_dx0.0025"] if k in dsb]; cols = {"sb_dx0.01": "#aed6f1", "sb_dx0.005": "#2e86c1", "sb_dx0.0025": "#1b2631"}
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for j, tt in enumerate(["t=0.6", "t=1.4"]):
        for k in lv:
            b = dsb[k][tt]["bands"]; dxs = k.split("dx")[1]
            ax[j].plot(xc, [b[x][0] for x in bands], "o-", color=cols[k], ms=3, lw=1, label="账本 $\\Delta x=%s$" % dxs)
            ax[j].plot(xc, [b[x][1] for x in bands], "s--", color=cols[k], ms=3, lw=1, mfc="none", label="DNS $\\Delta x=%s$" % dxs)
        ax[j].axvline(0, color="k", lw=0.5); ax[j].set_xlabel("初始距界面 $\\phi_0$（负为重气体）"); ax[j].set_ylabel("压缩 $-\\ln J$ / $\\ln(\\rho/\\rho_0)$")
        ax[j].set_title("(%s) 激波–重气泡，$%s$" % ("ab"[j], tt), fontsize=7.5)
    ax[0].legend(fontsize=5.2, ncol=2)
    lv2 = [k for k in ["sv_dx0.01", "sv_dx0.005", "sv_dx0.0025"] if k in dsv]; dd = [float(k.split("dx")[1]) for k in lv2]
    ax[2].semilogx(dd, [dsv[k]["med"]*100 for k in lv2], "o-", color="#1f5fa8", label="激波–涡 中位")
    ax[2].semilogx(dd, [dsv[k]["p95"]*100 for k in lv2], "s-", color="#c0392b", label="激波–涡 95 分位")
    ax[2].semilogx(dd, [dsv[k]["core_p95"]*100 for k in lv2], "^--", color="#c0392b", label="激波–涡 涡核 95 分位")
    lvb = [k for k in ["sb_dx0.01", "sb_dx0.005", "sb_dx0.0025"] if k in dsb]; db = [float(k.split("dx")[1]) for k in lvb]
    ax[2].semilogx(db, [dsb[k]["t=1.4"]["p95"]*100 for k in lvb], "o-", color="#7d3c98", label="激波–气泡 95 分位")
    ax[2].semilogx(db, [dsb[k]["t=1.4"]["p95_outside_interface"]*100 for k in lvb], "^--", color="#7d3c98", label="激波–气泡 离界面 95 分位")
    ax[2].set_xlabel("DNS 网格 $\\Delta x$"); ax[2].set_ylabel("密度误差 (%)"); ax[2].invert_xaxis(); ax[2].legend(fontsize=5.5); ax[2].set_title("(c) 误差随 DNS 加密", fontsize=7.5)
    plt.tight_layout(); plt.savefig("figR_dns.pdf"); plt.close()
