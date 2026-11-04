import json, numpy as np
C = ["DG", "LO", "SOD", "BLAST", "SB", "SV"]; NM = dict(DG="双环流", LO="Lamb–Oseen", SOD="Sod", BLAST="爆炸波", SB="激波–气泡", SV="激波–涡")
R = {c: json.load(open(f"tcx_res_{c}.json")) for c in C}; L = []
f3 = lambda x: f"{x:+.3f}" if x is not None else "—"
# T1 strength
L.append(r"\begin{tabular}{lccccc}\toprule 算例 & 偏差：不扣 / 全局 / \textbf{能量} & 相对 RMS：不扣 / 全局 / \textbf{能量} & Spearman：全局 / \textbf{能量} & 干净：能量=不扣 \\\midrule")
for c in ["DG", "LO", "SB", "SV"]:
    p = R[c]["P4_noise"]; p5 = R[c]["P5_lavd"]
    same = abs(p5["energy"]["rel_rms"] - p5["new"]["rel_rms"]) < 1e-3
    L.append(f"{NM[c]} & {p['lavd_bias_nofloor']:+.2f} / {p['lavd_bias_floor']:+.2f} / \\textbf{{{p['lavd_bias_energy']:+.2f}}} & "
             f"{p['lavd_nofloor']['rel_rms']:.3f} / {p['lavd_floor']['rel_rms']:.3f} / \\textbf{{{p['lavd_energy']['rel_rms']:.3f}}} & "
             f"{p['lavd_floor']['spearman']:.2f} / {p['lavd_energy']['spearman']:.2f} & {'是' if same else '否'} \\\\")
for c in ["SOD", "BLAST"]:
    p = R[c]["P4_noise"]
    L.append(f"{NM[c]}（伪量 p99） & {p['lavd_bias_nofloor']:+.2f} / {p['lavd_bias_floor']:+.2f} / {p['lavd_bias_energy']:+.2f} & "
             f"{p['lavd_nofloor']['p99']:.2f} / {p['lavd_floor']['p99']:.2f} / {p['lavd_energy']['p99']:.2f} & — & — \\\\")
L.append(r"\bottomrule\end{tabular}"); open("report_rev/t_strength.tex", "w").write("\n".join(L)); L = []
# T2 cores / vortices
L.append(r"\begin{tabular}{lcccccccc}\toprule 算例 & 参照核 & 旧法 干净 & 旧法 噪声（匹配） & \textbf{软阈值 噪声}（匹配） & 持续门剔除 干净/噪声 & 参照涡 & \textbf{涡 干净}（匹配） & \textbf{涡 噪声}（匹配）\\\midrule")
for c in C:
    p = R[c]["P10_persistence"]
    L.append(f"{NM[c]} & {p['cores_ref']} & {p['cores_net']} & {p['cores_net_noise']}（{p['matched_noise_in_ref']}） & "
             f"\\textbf{{{p['cores_shrink_noise']}}}（{p['matched_shrink_noise']}） & {p['sustained_rejected_clean']}/{p['sustained_rejected_noise']} & {p['vortices_ref']} & "
             f"\\textbf{{{p['vortices_new']}}}（{p['vortices_new_matched']}） & \\textbf{{{p['vortices_new_noise']}}}（{p['vortices_new_noise_matched']}） \\\\")
L.append(r"\bottomrule\end{tabular}"); open("report_rev/t_cores.tex", "w").write("\n".join(L)); L = []
# T3 boundaries
L.append(r"\begin{tabular}{llcccccc}\toprule 算例 & 核 & 判据 & 找到（网络/参照/噪声） & IoU 参照 & IoU 噪声 & 面积 & 网络 $\ln L$ / 凸性亏损 \\\midrule")
sel = [("DG", 0), ("DG", 1), ("LO", 0), ("LO", 1), ("SV", 0), ("SV", 2), ("SB", 1), ("SB", 0)]
yn = lambda b: "是" if b else "否"
for c, k in sel:
    b = R[c]["P11_boundary"][k]
    for q, lab in [("cdef<=0.02", "凸性 $\\le 0.02$"), ("lnLd<=0.1", "周长 $\\delta=0.1$"), ("lnLd<=0.2", "周长 $\\delta=0.2$")]:
        d = b[q]; g = lambda key, fmt: (fmt % d[key]) if key in d else "—"
        L.append(f"{NM[c] if q.startswith('cdef') else ''} & {('(%.2f, %.2f)' % tuple(b['pos'])) if q.startswith('cdef') else ''} & {lab} & "
                 f"{yn(d['net_found'])}/{yn(d['ref_found'])}/{yn(d['noise_found'])} & {g('iou_ref','%.2f')} & {g('iou_noise','%.2f')} & "
                 f"{g('net_area','%.4f')} & {g('net_lnL','%.2f')} / {g('net_cdef','%.3f')} \\\\")
    L.append(r"\addlinespace[2pt]")
L.append(r"\bottomrule\end{tabular}"); open("report_rev/t_bnd.tex", "w").write("\n".join(L)); L = []
# T4 objectivity
L.append(r"\begin{tabular}{lcccccc}\toprule 量 & " + " & ".join(NM[c] for c in C) + r"\\\midrule")
for key, lab in [("phi_new", "$\\varphi$"), ("lavd_new", "LAVD（分段）"), ("lavd_energy", "LAVD（能量去偏）"), ("phi_old_circ", "旧环量法 $\\varphi$")]:
    L.append(lab + " & " + " & ".join("%.1e" % R[c]["P1_objectivity"][key] for c in C) + r"\\")
L.append(r"\bottomrule\end{tabular}"); open("report_rev/t_obj.tex", "w").write("\n".join(L)); L = []
# T5 verify
V = json.load(open("res_verify_perim.json"))
L.append(r"\begin{tabular}{lccc}\toprule 边界 & 网络 $\ln L$ & 精确（2000 点） & 精确（8000 点）\\\midrule")
for k, v in V.items():
    nm, core, q = k.split(" "); q = q.replace("cdef<=0.02", "凸性").replace("lnLd<=", "周长 δ=")
    L.append(f"{NM[nm]} {core.replace('core','核')} {q} & {v['network']:.3f} & {v['exact_2k']:.3f} & {v['exact_8k']:.3f}\\\\")
L.append(r"\bottomrule\end{tabular}"); open("report_rev/t_verify.tex", "w").write("\n".join(L))
print("ok")
