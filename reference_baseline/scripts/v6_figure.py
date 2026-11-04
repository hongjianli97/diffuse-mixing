"""Verification figure for the paper (fig_verification.pdf) + order study of polygon vs tensor ledger."""
import sys, json, pickle, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path[:0] = ["lhd", "scripts"]
import mfm, classify as C, lhdx as X
from ledger_ub import chained_lnJ_ind, tensor_lnJ, bad_from_hot
from lhdp import disp_cochain_tri, tri_area
R2, R4, R5 = (json.load(open("results/%s.json" % f)) for f in ("res_v2_mfm", "res_v4_ledger_v3", "res_v5_diffusion"))
# ---- order study (smooth MFM, mid-time where the map is most deformed)
od = {}
for h in (0.04, 0.02, 0.01):
    c = mfm.smooth_map(h=h); fr = c["frames"]; m = c["interior"](fr[0]); mid = len(fr) // 2; bad = bad_from_hot(C.hot_steps(fr)[0])
    lp = chained_lnJ_ind(fr, bad)[0][mid]; lt = tensor_lnJ(fr, bad, keep=[mid])[mid]
    od[h] = (float(np.median(np.abs(lp - c["lnJ_ex"][mid])[m])), float(np.median(np.abs(lt - c["lnJ_ex"][mid])[m])))
hs = np.array(sorted(od)); P = np.array([od[h][0] for h in hs]); T = np.array([od[h][1] for h in hs])
order = dict(h=hs.tolist(), polygon=P.tolist(), tensor=T.tolist(), p_polygon=float(np.polyfit(np.log(hs), np.log(P), 1)[0]), p_tensor=float(np.polyfit(np.log(hs), np.log(T), 1)[0]))
json.dump(order, open("results/res_v6_order.json", "w"), indent=1); print(order)
# ---- circulation time series (band 0.15 and 0.04)
D = pickle.load(open("runs/sb_dns_dx0.005.pkl", "rb")); fr = D["frames"]; t = D["t"]; E = D["E"]; P0 = fr[0]
phi = np.hypot(P0[:, 0] - D["xb"], P0[:, 1]) - D["R"]; tri0 = X.complex_peeled(P0)["tri"]; tm = 0.5 * (t[1:] + t[:-1]); circ = {}
for wb in (0.04, 0.15):
    triR = tri0[(np.abs(phi) < wb)[tri0].all(1)]
    e = np.vstack([triR[:, [0, 1]], triR[:, [1, 2]], triR[:, [2, 0]]]); _, inv, cnt = np.unique(np.sort(e, 1), axis=0, return_inverse=True, return_counts=True); be = e[cnt[inv] == 1]
    G = []
    for s in range(len(t) - 1):
        cc, _ = disp_cochain_tri(fr[s], fr[s + 1], triR); Pm = 0.5 * (fr[s] + fr[s + 1])
        U = 0.5 * np.c_[E["u"][s] + E["u"][s + 1], E["v"][s] + E["v"][s + 1]]; w = 0.5 * (E["om"][s] + E["om"][s + 1])
        G.append((cc.sum() / (t[s + 1] - t[s]), (0.5 * (U[be[:, 0]] + U[be[:, 1]]) * (Pm[be[:, 1]] - Pm[be[:, 0]])).sum(), (w[triR].mean(1) * tri_area(Pm, triR)).sum()))
    circ[wb] = -np.array(G)
# ---- figure
fig, ax = plt.subplots(2, 3, figsize=(13, 7.6))
a = ax[0, 0]; E1 = R2["E1_planar"]
det = [v["err"]["max"] for k, v in E1.items() if v["detected"] > 0.99]; nod = [v["no_deferral_err"]["max"] for v in E1.values() if "no_deferral_err" in v]
bars = [("planar, detected\n(all $\\theta,\\eta,C_s$)", max(det)), ("no deferral", max(nod)),
        ("weak ($\\eta$=1.5,$C_s$=0.5)\nsingle-step", R4["weak"]["old_v2"]["max"]), ("weak, multi-window", max(R4["weak"]["v3_polygon"]["max"], 1e-16)),
        ("slow ($C_s$=0.25)\nsingle-step", R4["slow"]["old_v2"]["max"]), ("slow, multi-window", max(R4["slow"]["v3_polygon"]["max"], 1e-16))]
a.bar(range(len(bars)), [max(b[1], 1e-16) for b in bars], color=["C0", "C3", "C3", "C0", "C3", "C0"]); a.set_yscale("log")
a.set_xticks(range(len(bars))); a.set_xticklabels([b[0] for b in bars], fontsize=6.5, rotation=30, ha="right"); a.set_ylabel("max density error"); a.set_title("(a) planar manufactured shocks", fontsize=9)
a = ax[0, 1]
for k, mk in zip(("h0.04", "h0.02", "h0.01"), "os^"):
    tab = R2["E2_cylinder"][k]["by_curvature"]; x = [np.mean(b["h_over_Rs"]) for b in tab]
    a.loglog(x, [b["med"] for b in tab], mk + "-", label="ledger median, h=" + k[1:])
    sg = R2["E2_cylinder"][k]["signature"]; a.loglog([np.mean(b["h_over_Rs"]) for b in sg], [b["argp_med"] for b in sg], mk + ":", color="gray")
a.set_xlabel("$h/R_s$"); a.set_title("(b) converging cylindrical shock: collapse on $h/R_s$\n(dotted: block $|\\arg p_B|$)", fontsize=9); a.legend(fontsize=7)
a = ax[0, 2]
a.loglog(hs, P, "o-", label="polygon ledger (order %.2f)" % order["p_polygon"]); a.loglog(hs, T, "s-", label="tensor ledger (order %.2f)" % order["p_tensor"])
a.set_xlabel("tracer spacing h"); a.set_ylabel("median $|\\ln\\hat J-\\ln J|$"); a.set_title("(c) smooth compressible map", fontsize=9); a.legend(fontsize=7)
a = ax[1, 0]
for wb, ls in ((0.15, "-"), (0.04, "--")):
    a.plot(tm, circ[wb][:, 0], "k" + ls, label="position form, band %.2f" % wb); a.plot(tm[::3], circ[wb][::3, 1], "o", mfc="none", color="C0", ms=4, label="DNS line integral, same boundary" if wb == 0.15 else None)
    a.plot(tm, circ[wb][:, 2], ls, color="C3", lw=1, label="DNS vertex-vorticity quadrature" if wb == 0.15 else None)
a.set_xlabel("t"); a.set_ylabel("$|\\Gamma|$ upper half"); a.set_title("(d) shock-bubble circulation", fontsize=9); a.legend(fontsize=6.5)
a = ax[1, 1]; labs = [("oracle_exact_rho", "exact $\\rho$"), ("ledger_elem_arith (paper)", "ledger $\\rho$"), ("sph_rho", "SPH $\\rho$"), ("dtfe_rho", "DTFE $\\rho$"), ("area_metric", "area metric")]
w = 0.38
for j, case in enumerate(("sod", "blast")):
    a.bar(np.arange(len(labs)) + (j - 0.5) * w, [100 * R5[case][k]["relL2"] for k, _ in labs], w, label=case)
a.set_yscale("log"); a.set_xticks(range(len(labs))); a.set_xticklabels([l for _, l in labs], fontsize=7, rotation=20); a.set_ylabel("diffusion error (%)")
a.set_title("(e) diffusion: equation vs density estimate", fontsize=9); a.legend(fontsize=7)
a = ax[1, 2]; E4 = R2["E4_shear"]; r = [float(k.split("=")[1]) for k in E4]
a.loglog(r, [100 * v["rel_err_med_layer"] for v in E4.values()], "o-"); a.axhline(5, color="gray", ls=":")
a.set_xlabel("layer thickness $\\delta/h$"); a.set_ylabel("chained-tensor stretch error (%)"); a.set_title("(f) thin shear layer: information limit", fontsize=9)
plt.tight_layout(); plt.savefig("figures/fig_verification.pdf"); plt.savefig("/home/claude/tex/fig_verification.pdf"); print("fig ok")
