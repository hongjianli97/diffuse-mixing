"""fig_v4.pdf: (a) ledger v2/v3/v4 95th percentiles, (b) viscous NS planar shocks, (c,d) T4 merged-block split vs DNS torque"""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
v8 = json.load(open("results/res_v8_ledger_v4.json")); v9 = json.load(open("results/res_v9_viscous_shock.json")); v10 = json.load(open("results/res_v10_t4.json"))
fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
cs = [("planar_mid", "planar\n(mid)"), ("cyl", "cyl."), ("sod", "Sod"), ("blast", "blast"), ("dg", "DG"), ("sv", "shock-\nvortex"),
      ("sb", "SB"), ("sb", "SB\nclean"), ("rm", "RM\nclean"), ("sbv", "SBv\nclean")]
x = np.arange(len(cs)); w = 0.27
for j, (k, lab, c) in enumerate((("old_v2", "v2", "#999999"), ("v3_hybrid", "v3 hybrid", "#d9a441"), ("v4", "v4", "#2c6fbb"))):
    vals = []
    for i, (nm, l) in enumerate(cs):
        cl = "clean" in l; d = v8[nm]
        r = (d["final"]["v4_clean" if cl else "v4"] if k == "v4" else d[k + ("_clean" if cl else "")])
        vals.append(max(100 * r["p95"], 1e-4))
    ax[0].bar(x + (j - 1) * w, vals, w, label=lab, color=c)
ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels([l for _, l in cs], fontsize=7); ax[0].set_ylabel("95th pct. density error (%)")
ax[0].set_title("(a) ledger estimators"); ax[0].legend(fontsize=7)
ks = ["Ms2.24_d1_Cs0.5", "Ms2.24_d2_Cs1", "Ms2.24_d4_Cs1", "Ms2.24_d4_Cs0.25", "Ms1.5_d2_Cs0.5", "Ms1.5_d4_Cs0.25"]; x = np.arange(len(ks))
for j, (k, lab, c) in enumerate((("v2", "v2", "#999999"), ("v3_hybrid", "v3 hybrid", "#d9a441"), ("v4", "v4", "#2c6fbb"))):
    ax[1].bar(x + (j - 1) * w, [max(100 * v9[q]["final"][k]["med"], 1e-4) for q in ks], w, label=lab, color=c)
for i, q in enumerate(ks):
    ax[1].text(i - w, max(100 * v9[q]["final"]["v2"]["med"], 1e-4) * 1.3, "%d%%" % round(100 * v9[q]["final"]["v2"]["detected"]), ha="center", fontsize=6)
ax[1].set_yscale("log"); ax[1].set_xticks(x)
ax[1].set_xticklabels(["$M_s$%g\n$\\delta/h$=%.1f\n$C_s$=%g" % (v9[q]["Ms"], v9[q]["delta_measured_over_h"], v9[q]["Cs"]) for q in ks], fontsize=6.5)
ax[1].set_ylabel("median density error (%)"); ax[1].set_title("(b) viscous planar shock (NS DNS)"); ax[1].legend(fontsize=7)
b = v10["sb"]["band 0.15"]; t = np.array(b["t"]); sr = b["series"]; a = ax[2]
a.plot(t, -np.array(sr["dG"]), "k-", label="total (positions)")
a.plot(t, -np.array(sr["imp"]), "-", c="#c0392b", label="impulsive, merged blocks")
a.plot(t, -np.array(sr["smo"]), "-", c="#2c6fbb", label="smooth")
a.plot(t, -np.array(sr["dns_vel_imp"]), "--", c="#c0392b", label="DNS velocity, same triangles")
a.plot(t, -np.array(sr["dns_vel_smo"]), "--", c="#2c6fbb")
a.plot(b["t_step"], -np.array(sr["dns_step_imp"]), "-.", c="#e6a0a0", lw=0.8, label="DNS torque, solver steps")
a.plot(b["t_step"], -np.array(sr["dns_step_smo"]), "-.", c="#a0b8e0", lw=0.8)
a.plot(t, -np.array(sr["imp_old"]), ":", c="#777777", label="old split: impulsive")
a.set_xlabel("t"); a.set_title("(c) T4, shock-bubble $|\\phi_0|<0.15$")
z = b["SZ"]; th = np.degrees(z["theta"]); G1 = z["Gamma1prime"]; a = ax[3]
a.step(th, np.array(z["imp_per_length"]) / G1, where="mid", c="#c0392b", label="impulsive (merged blocks)")
a.plot(th, np.array(z["SZ1"]) / G1, "--", c="#555555", label="SZ1 local law")
bv = v10["sbv"]["band 0.15"]["SZ"]; a.step(th, np.array(bv["imp_per_length"]) / G1, where="mid", c="#e67e22", lw=0.8, label="impulsive, viscous")
a.set_xlabel("angle from upstream point (deg)"); a.set_ylabel("$\\Gamma'/\\Gamma'_1$"); a.set_title("(d) impulsive deposition per unit length"); a.legend(fontsize=6.5); a.set_ylim(0, 1.15)
ax[2].set_ylabel("$|\\Delta\\Gamma|$ (cumulative)"); ax[2].legend(fontsize=6.5)
plt.tight_layout(); plt.savefig("paper/fig_v4.pdf"); plt.savefig("report/fig_v4.pdf"); plt.savefig("/tmp/fig_v4.png", dpi=80); print("ok")
