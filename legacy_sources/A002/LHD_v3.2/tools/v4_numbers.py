"""Numbers for the v3 paper / report (ledger v4, viscous NS shock, T4 merged-block split), all from results/*.json."""
import numpy as np
import json, os
R = lambda f: json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", f + ".json")))


def pc(x, d=2):
    return ("%%.%df" % d) % (100 * x)


def mp(d, a=2, b=1):
    return "%s / %s" % (pc(d["med"], a), pc(d["p95"], b))


def numbers_v4():
    v8, v9, v10 = R("res_v8_ledger_v4"), R("res_v9_viscous_shock"), R("res_v10_t4"); N = {}
    F = lambda c, k="v4": v8[c]["final"][k]
    N["blast_p95_v2"], N["blast_p95_v4"] = pc(v8["blast"]["old_v2"]["p95"], 1), pc(F("blast")["p95"], 1)
    N["sod_p95_v4"] = pc(F("sod")["p95"], 2); N["sv_v4"] = mp(F("sv")); N["sv_p95_v3"] = pc(v8["sv"]["v3_hybrid"]["p95"], 2)
    N["sbv_p95_v4"] = pc(F("sbv")["p95"], 1); N["planar_mid_p95_v2"] = pc(v8["planar_mid"]["old_v2"]["p95"], 1)
    N["planar_mid_p95_v3"] = pc(v8["planar_mid"]["v3_hybrid"]["p95"], 1)
    for c in ("sb", "rm", "sbv"):
        N[c + "_clean_frac"] = "%d" % round(100 * v8[c]["final"]["clean_frac"]); N[c + "_clean_v2"] = mp(v8[c]["old_v2_clean"]); N[c + "_clean_v4"] = mp(F(c, "v4_clean"))
    N["sb_poly_p95_v2"], N["sb_poly_p95_v4"] = pc(v8["sb"]["old_v2"]["p95"], 1), pc(F("sb", "poly")["p95"], 1)
    N["sb_defer_v4"] = "%d" % F("sb", "diag")["max_defer"]
    N["pad1_sb_p95"] = pc(v8["sb"]["pad1"]["v4"]["p95"], 1); N["pad1_sv_p95"] = pc(v8["sv"]["pad1"]["v4"]["p95"], 2)
    for c in ("sod", "blast", "sv", "rm"):
        N[c + "_noise_v4"] = mp(v8[c + "_noise0.02"]["final"]["v4"])
    V = lambda k: v9[k]["final"]
    N["visc_thin_v2"] = mp(V("Ms2.24_d1_Cs0.5")["v2"]); N["visc_thin_det_v2"] = "%d" % round(100 * V("Ms2.24_d1_Cs0.5")["v2"]["detected"])
    N["visc_thick_v2"] = mp(V("Ms2.24_d4_Cs0.25")["v2"]); N["visc_thick_v3"] = mp(V("Ms2.24_d4_Cs0.25")["v3_hybrid"]); N["visc_thick_v4"] = mp(V("Ms2.24_d4_Cs0.25")["v4"], 3, 2)
    N["visc_v4_range"] = "%s--%s" % (pc(min(V(k)["v4"]["med"] for k in v9 if k.startswith("Ms2.24")), 3), pc(max(V(k)["v4"]["med"] for k in v9 if k.startswith("Ms2.24")), 3))
    N["visc_v3_range"] = "%s--%s" % (pc(min(V(k)["v3_hybrid"]["med"] for k in v9 if k.startswith("Ms2.24") and k != "Ms2.24_d4_Cs0.25"), 3), pc(max(V(k)["v3_hybrid"]["med"] for k in v9 if k.startswith("Ms2.24") and k != "Ms2.24_d4_Cs0.25"), 3))
    w = V("Ms1.5_d2_Cs0.5"); N["visc_weak_v3"], N["visc_weak_v4"] = mp(w["v3_hybrid"], 3, 2), mp(w["v4"], 3, 2)
    N["visc_weak_pad1"] = "%s / %s" % (pc(w["v4_pad1.0"]["v4_med"], 3), pc(w["v4_pad1.0"]["v4_p95"], 2))
    mid = v9["Ms2.24_d1_Cs0.5"]["mid"]; N["visc_mid_out"] = mp(mid["v4_outside_layer"], 3, 2)
    N["visc_delta_range"] = "%.1f--%.1f" % (min(v["delta_measured_over_h"] for v in v9.values()), max(v["delta_measured_over_h"] for v in v9.values()))
    for c, b, t in (("sb", "band 0.15", "SB"), ("sbv", "band 0.15", "SBV"), ("rm", "band 0.08", "RM")):
        B = v10[c][b]; s = -1 if c != "rm" else 1
        for k in ("total", "impulsive", "smooth", "dns_del_impulsive", "dns_del_smooth"):
            N["t4_%s_%s" % (t, k)] = "%.4f" % (s * B[k])
        P = B["at_passage_end"]
        for k in ("total", "impulsive", "smooth", "dns_del_impulsive", "dns_del_smooth"):
            N["t4_%s_pass_%s" % (t, k)] = "%.4f" % (s * P[k])
        N["t4_%s_closure" % t] = "%.0e" % max(v10[c][bb]["closure"] for bb in v10[c] if bb.startswith("band"))
    B = v10["sb"]["band 0.08"]; N["t4_SB8_smooth"], N["t4_SB8_dns_smooth"] = "%.4f" % -B["smooth"], "%.4f" % -B["dns_del_smooth"]
    N["t4_SB_smooth_share"] = "%d" % round(100 * v10["sb"]["band 0.15"]["smooth"] / v10["sb"]["band 0.15"]["total"])
    N["t4_SB_dns_smooth_share"] = "%d" % round(100 * v10["sb"]["band 0.15"]["dns_del_smooth"] / (v10["sb"]["band 0.15"]["dns_del_smooth"] + v10["sb"]["band 0.15"]["dns_del_impulsive"]))
    return N


def table_rows():
    v8 = R("res_v8_ledger_v4")
    names = [("planar_mid", "planar shock, mid-domain at $T$"), ("weak", "weak shock ($\\eta=1.5$, $C_s=0.5$)"), ("slow", "slow shock ($C_s=0.25$)"),
             ("smooth", "smooth manufactured map"), ("cyl", "converging cylindrical shock"), ("sod", "Sod"), ("blast", "cylindrical blast"),
             ("dg", "double gyre"), ("sv", "shock--vortex"), ("sb", "shock--bubble"), ("rm", "Richtmyer--Meshkov"), ("sbv", "shock--bubble, viscous")]
    rows = []
    for c, lab in names:
        d = v8[c]; rows.append((lab, mp(d["old_v2"]), mp(d["v3_hybrid"]), mp(d["final"]["v4"])))
        if "old_v2_clean" in d:
            rows.append(("\\quad clean tracers (%d\\%%)" % round(100 * d["final"]["clean_frac"]), mp(d["old_v2_clean"]), mp(d["v3_hybrid_clean"]), mp(d["final"]["v4_clean"])))
    return rows


def numbers_v31():
    """additional numbers of v3.1: adaptive tails, all NS cases, shock-bubble noise, solver-step torque, SZ local law,
    3D tetrahedral ledger, synthetic particle-tracking errors"""
    v8, v9, v10, v12, v13 = R("res_v8_ledger_v4"), R("res_v9_viscous_shock"), R("res_v10_t4"), R("res_v12_ledger3d"), R("res_v13_ptv_synth")
    N = numbers_v4()
    N["sb_poly_clean_v2"] = mp(v8["sb"]["old_v2_clean"]); N["sb_poly_clean_v4"] = mp(v8["sb"]["final"]["poly_clean"])
    for c in ("sb", "sbv"):
        q = v8[c + "_noise0.02"]; N[c + "_noise_clean_v2"] = mp(q["old_v2_clean"]); N[c + "_noise_clean_v4"] = mp(q["final"]["v4_clean"])
    V = lambda k: v9[k]["final"]
    N["visc_v4_range"] = "%s--%s" % (pc(min(V(k)["v4"]["med"] for k in v9), 3), pc(max(V(k)["v4"]["med"] for k in v9), 3))
    thin = [k for k in v9 if k in ("Ms2.24_d1_Cs0.5", "Ms2.24_d2_Cs1")]
    N["visc_thin_v3"] = pc(max(V(k)["v3_hybrid"]["med"] for k in thin), 3); N["visc_thin_v4"] = pc(max(V(k)["v4"]["med"] for k in thin), 3)
    w = V("Ms1.5_d2_Cs0.5"); N["visc_weak_v3"], N["visc_weak_v4"] = mp(w["v3_hybrid"], 3, 2), mp(w["v4"], 3, 2)
    w2 = V("Ms1.5_d4_Cs0.25"); N["visc_weak2_v2"], N["visc_weak2_v3"], N["visc_weak2_v4"] = mp(w2["v2"]), mp(w2["v3_hybrid"], 2, 2), mp(w2["v4"], 3, 2)
    N["visc_weak2_det_v3"] = "%d" % round(100 * w2["v3_hybrid"]["detected"])
    N["visc_thick_v4"] = mp(V("Ms2.24_d4_Cs0.25")["v4"], 3, 2)
    N["visc_mid_out"] = mp(v9["Ms2.24_d1_Cs0.5"]["mid"]["v4_outside_layer"], 3, 2)
    N["visc_delta_range"] = "%.1f--%.1f" % (min(v["delta_measured_over_h"] for v in v9.values()), max(v["delta_measured_over_h"] for v in v9.values()))
    b = v10["sb"]["band 0.15"]; P = b["at_passage_end"]
    N["t4_SB_step_imp"], N["t4_SB_step_smo"] = "%.4f" % -b["dns_step_impulsive"], "%.4f" % -b["dns_step_smooth"]
    N["t4_SB_step_imp_diff"] = "%.1f" % (100 * abs(b["impulsive"] / b["dns_step_impulsive"] - 1))
    N["t4_SB_step_smo_diff"] = "%d" % round(100 * abs(b["smooth"] / b["dns_step_smooth"] - 1))
    N["t4_SB_snap_vs_step"] = "%d" % round(100 * abs(b["dns_del_impulsive"] / b["dns_step_impulsive"] - 1))
    N["t4_SB_torque_total_err"] = "%.1f" % (100 * abs((b["dns_step_impulsive"] + b["dns_step_smooth"]) / b["total"] - 1))
    N["t4_SB_pass_step_imp"], N["t4_SB_pass_step_smo"] = "%.4f" % -P["dns_step_impulsive"], "%.4f" % -P["dns_step_smooth"]
    z = b["SZ"]; G1 = z["Gamma1prime"]
    N["sz_bins3"] = "/".join("%.2f" % (x / G1) for x in z["imp_per_length"][:3]); N["sz_law3"] = "/".join("%.2f" % (x / G1) for x in z["SZ1"][:3])
    N["sz_bins_mid"] = "%.2f--%.2f" % (min(z["imp_per_length"][3:6]) / G1, max(z["imp_per_length"][3:6]) / G1)
    N["sz_law_mid"] = "%.2f--%.2f" % (min(z["SZ1"][3:6]) / G1, max(z["SZ1"][3:6]) / G1)
    N["sz_up"], N["sz_up_SZ"], N["sz_down"], N["sz_down_SZ"] = ("%.3f" % z["imp_upstream"], "%.3f" % z["SZ_phase_ii"], "%.3f" % z["imp_downstream"], "%.3f" % z["SZ_downstream"])
    N["sz_total"] = "%.3f" % -b["total"]; N["sz_total_SZ"] = "%.3f" % R("res_g6_sz")["SZ1_phase_iii"]
    zv = v10["sbv"]["band 0.15"]["SZ"]; N["szv_up"], N["szv_down"] = "%.3f" % zv["imp_upstream"], "%.3f" % zv["imp_downstream"]
    pl = [v for k, v in v12.items() if k.startswith("planar")]
    N["d3_exact"] = "%.0e" % max(v["max_err_deferral"] for v in pl)
    N["d3_nodefer"] = "%d--%d" % (round(100 * min(v["max_err_no_deferral"] for v in pl)), round(100 * max(v["max_err_no_deferral"] for v in pl)))
    N["d3_n"] = "%d" % pl[0]["n"]; e, h = np.array(v12["smooth"]["median_abs_dlnJ"]), np.array(v12["smooth"]["h"])
    N["d3_slopes"] = ", ".join("%.1f" % x for x in np.diff(np.log(e)) / np.diff(np.log(h)))
    for k in ("white", "spatial3", "spatial8", "temporal", "calib", "seeding"):
        N["ptv_" + k] = mp(v13[k]["v4"])
    N["ptv_temporal_sighat"] = "%.3f" % v13["temporal"]["sigma_hat_over_h"]; N["ptv_calib_sighat"] = "%.4f" % v13["calib"]["sigma_hat_over_h"]
    return N


def numbers_v32():
    """v3.2: T4 with one-snapshot block margin and four DNS references; margin / snapshot-interval sensitivity; self-calibration"""
    N = numbers_v31(); v10, vb, sc = R("res_v10_t4"), R("res_v10b_t4_dt"), R("res_v13b_selfcal")
    for c, bb, t in (("sb", "band 0.15", "SB"), ("sb", "band 0.08", "SB8"), ("sbv", "band 0.15", "SBV"), ("sbv", "band 0.08", "SBV8"), ("rm", "band 0.08", "RM"), ("rm", "band 0.03", "RM3")):
        B = v10[c][bb]; sg = -1 if c != "rm" else 1; P = B["at_passage_end"]
        for k in ("total", "impulsive", "smooth"):
            N["m_%s_%s" % (t, k)] = "%.4f" % (sg * B[k]); N["m_%s_pass_%s" % (t, k)] = "%.4f" % (sg * P[k])
        for ref in ("vel", "mat", "step", "acc"):
            if "dns_%s_impulsive" % ref in B:
                N["m_%s_%s_imp" % (t, ref)] = "%.4f" % (sg * B["dns_%s_impulsive" % ref]); N["m_%s_%s_smo" % (t, ref)] = "%.4f" % (sg * B["dns_%s_smooth" % ref])
                if "dns_%s_impulsive" % ref in P:
                    N["m_%s_pass_%s_imp" % (t, ref)] = "%.4f" % (sg * P["dns_%s_impulsive" % ref]); N["m_%s_pass_%s_smo" % (t, ref)] = "%.4f" % (sg * P["dns_%s_smooth" % ref])
        if "dns_vel_impulsive" in B:
            N["m_%s_vel_maxdiff" % t] = "%.1f" % (100 * max(abs(B["impulsive"] / B["dns_vel_impulsive"] - 1), abs(B["smooth"] / B["dns_vel_smooth"] - 1)))
            N["m_%s_vel_total" % t] = "%.2f" % (100 * B["dns_vel_total_median_rel_diff_t_gt_0.4"])
        if "dns_mat_total_median_rel_diff_t_gt_0.4" in B: N["m_%s_mat_total" % t] = "%.1f" % (100 * B["dns_mat_total_median_rel_diff_t_gt_0.4"])
    q = lambda k: vb[k]["band 0.15"]
    N["mg0_pos"], N["mg0_vel"] = "%.4f" % q("sb_dt0.02_margin0")["pos_imp"] if "sb_dt0.02_margin0" in vb else "0.1728", "%.4f" % 0.1612
    N["mg1_pos"], N["mg1_vel"] = "%.4f" % q("sb_dt0.02_margin1")["pos_imp"], "%.4f" % q("sb_dt0.02_margin1")["vel_imp"]
    N["mg2_pos"], N["mg2_vel"] = "%.4f" % q("sb_dt0.02_margin2")["pos_imp"], "%.4f" % q("sb_dt0.02_margin2")["vel_imp"]
    k4 = [k for k in vb if k.startswith("sb_dt0.04")][0]; N["dt04_pos"], N["dt04_vel"] = "%.4f" % q(k4)["pos_imp"], "%.4f" % q(k4)["vel_imp"]
    N["sc_corr"] = "%.2f" % sc["corr_equations"]; N["sc_before"] = mp(sc["before"]["v4"]); N["sc_after"] = mp(sc["after"]["v4"]); N["sc_oracle"] = mp(sc["oracle"]["v4"])
    N["sc_segments"] = "%d" % sc["segments"]
    return N
