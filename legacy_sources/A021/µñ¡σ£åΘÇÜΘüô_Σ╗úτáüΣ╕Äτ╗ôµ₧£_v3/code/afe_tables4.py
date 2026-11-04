import json, glob, numpy as np
O = "report_rev/"; NM = dict(DG="双环流", LO="Lamb–Oseen", SOD="Sod", BLAST="爆炸波", SB="激波–气泡", SV="激波–涡")
# sheet
rows = json.load(open("sheet_0_2.json")) + json.load(open("sheet_2_8.json"))
L = [r"\begin{tabular}{cccccccc}\toprule 层厚 $\delta_s$ & $h$ & 帧数 & 节点 $\max|\varphi|$ & 层内 $|\varphi|$ 中位数 & 区域积分 $\int\varphi\,dA$ & $\Gamma(t_0)$ & $\Gamma(t_{\rm end})$\\\midrule"]
for r in sorted(rows, key=lambda r: (-r["ds"], r["frames"], -r["h"])):
    L.append(f"{r['ds']:g} & {r['h']:g} & {r['frames']} & {r['phi_max']:.2f} & {r['phi_band_median']:.2f} & {r['Q_phi_box']:.3f} & {r['Gamma_t0']:.4f} & {r['Gamma_tend']:.4f}\\\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "t4_sheet.tex", "w").write("\n".join(L))
# circulation strength
L = [r"\begin{tabular}{llccccc}\toprule 算例 & 涡 & 面积 & $S_\Gamma$（环量） & $S_\Gamma$（旋转参考系） & $S_\varphi$（内减环带） & $\varphi_{\rm core}$\\\midrule"]
for f in ["circ_strength_LO_DG.json", "circ_strength_SV.json", "circ_strength_SB.json"]:
    try: D = json.load(open(f))
    except Exception: continue
    for c, rs in D.items():
        for r in rs:
            L.append(f"{NM[c]} & ({r['pos'][0]:.2f}, {r['pos'][1]:.2f}) & {r['area']:.4f} & {r['S_circ']:.3f} & {r['S_circ_rotframe']:.3f} & {r['S_phi_inside_minus_ring']:.3f} & {r['phi_core']:.2f}\\\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "t4_circ.tex", "w").write("\n".join(L))
# merge tree
L = [r"\begin{tabular}{lcccccc}\toprule 算例 & 涡区占比 & Spearman $P$ 干净 / 噪声 & Spearman $\varphi$ 干净 / 噪声 & Pearson $P$ / $\varphi$（噪声） & 背景 $|\varphi|$/核 & 背景 $|P|$/核\\\midrule"]
for c in ["LO", "DG", "SV", "SB"]:
    r = json.load(open(f"sync_{c}.json")); v = r["in_vortex_region"]; b = r["background"]
    L.append(f"{NM[c]} & {v['frac']:.2f} & {v['spearman_P_net_ref']:.3f} / {v['spearman_P_noise_ref']:.3f} & {v['spearman_phi_net_ref']:.3f} / {v['spearman_phi_noise_ref']:.3f} & {v['pearson_P_noise_ref']:.3f} / {v['pearson_phi_noise_ref']:.3f} & {b['median_absphi_over_core']:.3f} & {b['median_absP_over_core']:.3f}\\\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "t4_merge.tex", "w").write("\n".join(L))
L = [r"\begin{tabular}{lccccc}\toprule 算例 & 保留示踪点 & $\varphi$ 平移（中位数 / 四分位距） & 核心 $\varphi$ 全集 / 子集 & 核心 $P$ 全集 / 子集\\\midrule"]
for c in ["LO", "DG"]:
    e = json.load(open(f"sync_{c}.json"))["ensemble"]
    L.append(f"{NM[c]} & {e['kept']} & {e['phi_shift_median']:+.3f} / {e['phi_shift_iqr']:.0e} & {e['core_phi_full']:.3f} / {e['core_phi_sub']:.3f} & {e['core_P_full']:.3f} / {e['core_P_sub']:.3f}\\\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "t4_ens.tex", "w").write("\n".join(L))
# neff
L = [r"\begin{tabular}{lccc}\toprule 算例 & $S-1$ & 显著核（持续性 $>0.5$）：单步占比 / $n_{\rm eff}$ & 其余核：单步占比 / $n_{\rm eff}$\\\midrule"]
for f in ["neff_SOD_SV.json", "neff_LO_DG.json", "neff_SB.json"]:
    D = json.load(open(f))
    for c, d in D.items():
        r = d["res"]["clean"]; top = [x for x in r if x["pers"] > 0.5]; low = [x for x in r if x["pers"] <= 0.5]
        g = lambda xs, k: f"{min(x[k] for x in xs):.2f}–{max(x[k] for x in xs):.2f}" if xs else "—"
        L.append(f"{NM[c]} & {d['S']-1} & {g(top,'share')} / {g(top,'neff')} & {g(low,'share')} / {g(low,'neff')}\\\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "t4_neff.tex", "w").write("\n".join(L))
print("ok")
