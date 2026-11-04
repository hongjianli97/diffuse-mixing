exec(open("figs.py").read().split("# ---------------- fig 1")[0])
import json
# ---------- T-doc figure 1: shock / ledger / signature
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.25), gridspec_kw=dict(width_ratios=[1.3, 1, 1, 1]))
d = pickle.load(open("sod.pkl", "rb")); P0 = d["P0"]
ax[0].scatter(P0[:, 0], P0[:, 1], s=0.4, c=d["hotcount"], cmap="Greens", lw=0, rasterized=True); ax[0].set_xlim(-.5, .5)
ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].set_title("(a) Sod：每点热步数", fontsize=7.5)
bins = np.logspace(-16, 0.3, 50)
for k, nm in [(1, "sod"), (2, "blast")]:
    d = pickle.load(open(nm+".pkl", "rb"))
    ax[k].hist(np.maximum(d["pst"], 1e-16), bins=bins, color="#999", alpha=0.8, label="逐步求和")
    ax[k].hist(np.maximum(d["pbl"], 1e-16), bins=bins, color="#1e8449", alpha=0.7, label="合并块")
    ax[k].set_xscale("log"); ax[k].set_xlabel("伪转角 (rad)"); ax[k].legend(fontsize=6)
    ax[k].set_title("(%s) %s：跨激波伪转角" % ("bc"[k-1], {"sod": "Sod", "blast": "爆炸波"}[nm]), fontsize=7.5)
for nm, col in [("sod", "#1e8449"), ("blast", "#d68910"), ("sv", "#7d3c98")]:
    d = pickle.load(open(nm+".pkl", "rb")); sg = d["sig"]; ok = np.isfinite(sg[:, 1])
    sub = np.random.default_rng(0).choice(np.where(ok)[0], min(1500, ok.sum()), replace=False)
    ax[3].scatter(sg[sub, 0], sg[sub, 1], s=1.5, c=col, lw=0, label={"sod": "Sod", "blast": "爆炸波", "sv": "激波–涡"}[nm], rasterized=True)
x = np.linspace(-1.5, 2, 200); ax[3].plot(x, np.tanh(0.5*np.abs(x)), "k-", lw=0.8, label=r"$\tanh(\frac{1}{2}|\ln J|)$")
ax[3].set_xlim(-1.2, 2.0); ax[3].set_ylim(0, 0.8); ax[3].set_xlabel(r"$-\ln J_B$"); ax[3].set_ylabel(r"$|\mu_B|$"); ax[3].legend(fontsize=5.5, markerscale=3)
ax[3].set_title("(d) 合并块签名", fontsize=7.5)
plt.tight_layout(); plt.savefig("figT1_shock.pdf"); plt.close()
# ---------- T-doc figure 2: SV readouts, Kelvin, Hodge
d = pickle.load(open("sv.pkl", "rb")); rsv = json.load(open("res_sv.json"))
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
sn = d["snap"]; hk = sn["hk"]
ax[0].scatter(sn["om_ref"][~hk], sn["om_hat"][~hk], s=0.5, c="k", lw=0, rasterized=True, label="非跨激波三角形")
ax[0].scatter(sn["om_ref"][hk], sn["om_hat"][hk], s=0.8, c="#1e8449", lw=0, rasterized=True, label="跨激波三角形")
lim = np.quantile(np.abs(sn["om_ref"]), .999); ax[0].plot([-lim, lim], [-lim, lim], "r-", lw=0.6)
ax[0].set_xlabel("DNS $\\omega$"); ax[0].set_ylabel("$\\hat\\omega=2A\\,\\mathrm{Im}\\,p/(\\bar A\\Delta t)$"); ax[0].legend(fontsize=6, markerscale=4)
ax[0].set_title("(a) 步涡量与 DNS（$t\\approx0.25$）", fontsize=7.5)
tm = 0.5*(d["t"][1:]+d["t"][:-1])
ax[1].plot(tm, d["Gpos"], "k-", lw=1, label="位置形式"); ax[1].plot(tm, d["Gdns"], "r--", lw=1, label="DNS $\\int\\omega\\,dA$")
hs = np.r_[d["hstep"], False]
for i in np.where(hs)[0]: ax[1].axvspan(tm[i]-0.01, tm[i]+0.01, color="#d5f5e3", lw=0)
ax[1].set_xlabel("$t$"); ax[1].legend(fontsize=6); ax[1].set_title("(b) 涡核物质区域环量", fontsize=7.5)
H = [rsv["hodge_raw"], rsv["hodge_raw_boost"], rsv["hodge_meanfree"], rsv["hodge_aligned"], rsv["hodge_aligned_boost"]]
nm = ["原始", "原始\n+平移", "去均\n值", "Kab-\nsch", "Kabsch\n+平移"]
ax[2].bar(range(5), [h[0] for h in H], color="#d68910", label="梯度"); ax[2].bar(range(5), [h[1] for h in H], bottom=[h[0] for h in H], color="#2471a3", label="余正合")
ax[2].set_xticks(range(5)); ax[2].set_xticklabels(nm, fontsize=6); ax[2].set_ylim(0, 1); ax[2].legend(fontsize=6); ax[2].set_title("(c) 位移上链的 Hodge 能量", fontsize=7.5)
plt.tight_layout(); plt.savefig("figT2_sv.pdf"); plt.close()
# ---------- LCS-doc figure 4: classification on compressible cases
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.4, 1, 1]))
for k, nm, tt in [(0, "sod", "Sod"), (1, "blast", "爆炸波")]:
    d = pickle.load(open(nm+".pkl", "rb")); f = d["ftle"]; labmap(ax[k], d["P0"], f["lab"], f["cand"], s=1.5)
    ax[k].set_title("(%s) %s：后向 FTLE 候选点" % ("ab"[k], tt), fontsize=7.5)
    if nm == "sod": ax[k].set_aspect("auto"); ax[k].set_xlim(-.5, .5)
d = pickle.load(open("sv.pkl", "rb")); labmap(ax[2], d["P0"], d["labf"], d["candf"], s=1.2); ax[2].set_title("(c) 激波–涡：后向 FTLE 候选点", fontsize=7.5)
plt.tight_layout(); plt.savefig("figL4_comp.pdf"); plt.close()
print("ok")
