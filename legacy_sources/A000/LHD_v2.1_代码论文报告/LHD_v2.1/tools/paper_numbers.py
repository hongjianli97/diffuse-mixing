"""Single source of truth for the numbers quoted in the paper: every value is computed from results/*.json.
tools/check_paper_numbers.py verifies that the paper quotes exactly these strings."""
import json, numpy as np, os
R = lambda f: json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", f + ".json")))


def numbers():
    b2, v1, v4, v4b, v5, v2, t1, rm, b3, v6 = (R(f) for f in ("res_b2_diff", "res_v1_circ_line", "res_v4_ledger_v3", "res_v4b_hybrid",
                                                           "res_v5_diffusion", "res_v2_mfm", "res_t_metamorphic", "res_rm", "res_b3_ledger", "res_v6_order"))
    pc = lambda x, d=2: ("%%.%df" % d) % (100 * x)
    N = {}
    rat = {k: b2[k]["network_volume"] / b2[k]["network_mass"] for k in ("sod", "blast", "sb")}
    N["area_ratio_range"] = "%.1f--%.1f" % (min(rat.values()), max(rat.values()))
    N["area_ratio_sod"], N["area_ratio_blast"], N["area_ratio_sb"] = ("%.1f" % rat[k] for k in ("sod", "blast", "sb"))
    for b in ("0.15", "0.08", "0.04"):
        N["circ_med_" + b] = pc(v1["band" + b]["relerr_med_position_vs_line"])
    N["circ_max"] = pc(max(v1["band" + b]["relerr_max_position_vs_line"] for b in ("0.15", "0.08", "0.04")), 1)
    N["narrow_band"] = "%.3f\\to%.3f" % (-v1["band0.04"]["G_position"][0], -v1["band0.04"]["G_position"][2])
    for k in ("sod", "blast"):
        N["diff_oracle_" + k] = pc(v5[k]["oracle_exact_rho"]["relL2"]); N["diff_ledger_" + k] = pc(v5[k]["ledger_elem_arith (paper)"]["relL2"])
        N["diff_sph_" + k] = pc(v5[k]["sph_rho"]["relL2"]); N["diff_dtfe_" + k] = pc(v5[k]["dtfe_rho"]["relL2"]); N["diff_area_" + k] = pc(v5[k]["area_metric"]["relL2"])
        N["diff_overshoot_max_" + k] = max(v["overshoot"] for kk, v in v5[k].items() if isinstance(v, dict) and "overshoot" in v and "edge" not in kk)
    N["diff_edge_sod"] = pc(v5["sod"]["ledger_edge_harm"]["relL2"])
    N["diff_time_sod"] = "%.2f" % (100 * max(abs(v5["sod"][k]["relL2"] - v5["sod"]["ledger_elem_arith (paper)"]["relL2"]) for k in ("ledger_nsub2", "ledger_nsub4", "ledger_CN")))
    T = v5["sod"]["rhoD_T07"]; N["rhoD_mass"] = pc(T["mass_T_from_positions"], 3); N["rhoD_exact"] = pc(T["mass_T_exact"], 3); N["rhoD_area"] = pc(T["area_T_from_positions"], 1)
    N["T_p95"] = pc(v5["sod"]["thermo_T_error"]["p95"], 1)
    E1 = v2["E1_planar"]
    N["planar_exact_max"] = "%.0e" % max(v["err"]["max"] for v in E1.values() if v["detected"] > 0.99)
    N["planar_block_max"] = "%.0e" % max(max(v["block_argp_max"], v["block_mu_dev_max"]) for v in E1.values() if "block_argp_max" in v)
    N["planar_nodefer_max"] = pc(max(v["no_deferral_err"]["max"] for v in E1.values() if "no_deferral_err" in v), 0)
    N["weak_old_max"], N["slow_old_max"] = pc(v4["weak"]["old_v2"]["max"], 0), pc(v4["slow"]["old_v2"]["max"], 0)
    meds = [b["med"] for k in v2["E2_cylinder"] for b in v2["E2_cylinder"][k]["by_curvature"]]
    N["cyl_med_range"] = "%s--%s" % (pc(min(meds)), pc(max(meds)))
    N["order_polygon"], N["order_tensor"] = "%.2f" % v6["p_polygon"], "%.2f" % v6["p_tensor"]
    for k in ("sod", "sb", "blast"):
        N["auc_" + k] = "%.2f" % v4[k]["indicator_auc_top5"]
    N["auc_cyl"], N["auc_smooth"] = "%.2f" % v4["cyl"]["indicator_auc_top5"], "%.2f" % v4["smooth"]["indicator_auc_top5"]
    h = lambda k: v4b[k]["hybrid_tol0.1"]
    N["hyb_dg_med"], N["hyb_blast_med"], N["hyb_sb_med"], N["hyb_sod_p95"] = pc(h("dg")["med"]), pc(h("blast")["med"]), pc(h("sb")["med"]), pc(h("sod")["p95"])
    N["hyb_share"] = "%d--%d" % (round(100 * min(v["hybrid_tol0.1"]["share"] for v in v4b.values())), round(100 * max(v["hybrid_tol0.1"]["share"] for v in v4b.values())))
    N["noise_hyb_sod"], N["noise_hyb_sb"], N["noise_poly_sb"] = pc(h("sod_noise0.02h")["med"]), pc(h("sb_noise0.02h")["med"], 1), pc(v4b["sb_noise0.02h"]["polygon"]["med"], 1)
    N["meta_max"] = "%.1e" % max(t1["rigid_motion_max_dlnJ"], t1["permutation_max_dlnJ"], t1["scaling_max_dlnJ"])
    N["meta_rev_p95"] = pc(t1["time_reversal_rho0_p95"], 1)
    a = np.array(rm["amp_track"]); m = (a[:, 0] >= 0.6 - 1e-9) & (a[:, 0] <= 1.4 + 1e-9); N["rm_rate"] = "%.4f" % np.polyfit(a[m, 0], a[m, 1], 1)[0]
    d = b3; N["tau_dmed_max"] = "%.3f" % (100 * max(abs(d[t][c]["med"] - d["default"][c]["med"]) for t in ("tau=0.1", "tau=0.3") for c in ("sod", "blast", "sv")))
    N["block3_blast"] = "%s\\%%$ and its 95th percentile from $%s\\%%$ to $%s" % (pc(d["block=3"]["blast"]["med"]), pc(d["default"]["blast"]["p95"], 0), pc(d["block=3"]["blast"]["p95"], 0))
    return N


if __name__ == "__main__":
    for k, v in numbers().items(): print("%-24s %s" % (k, v))
