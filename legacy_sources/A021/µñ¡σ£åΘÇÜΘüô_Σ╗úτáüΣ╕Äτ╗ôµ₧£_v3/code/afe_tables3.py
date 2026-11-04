import json, numpy as np
O = "report_rev/"; NM = dict(DG="双环流", LO="Lamb–Oseen", SOD="Sod", BLAST="爆炸波", SB="激波–气泡", SV="激波–涡")
SW = {c: json.load(open(f"sweep_{c}.json")) for c in ["LO", "DG", "SV", "SOD", "SB"]}; SS = json.load(open("sweep_summary.json"))
# kernel width
L = [r"\begin{tabular}{l" + "c"*7 + r"}\toprule 算例 & " + " & ".join(f"$n={n}$" for n in (2, 3, 4, 6, 8, 12, 16)) + r"\\\midrule"]
for c in ["LO", "DG", "SV", "SB"]:
    L.append(NM[c] + " & " + " & ".join(f"{SW[c]['kernel_width'][str(n)]:.2f}" for n in (2, 3, 4, 6, 8, 12, 16)) + r"\\")
L.append(r"$0.55\sqrt n$ & " + " & ".join(f"{0.55*np.sqrt(n):.2f}" for n in (2, 3, 4, 6, 8, 12, 16)) + r"\\\bottomrule\end{tabular}")
open(O + "t3_kernel.tex", "w").write("\n".join(L))
# rules
names = list(SS["rules"]["0.1"].keys()); names = [n for n in names if n != "n_vortices"]
lab = {"fixed n=2": "固定 2 次", "fixed n=8 (noise rule)": "固定 8 次（噪声规则）", "cap 4(r/3h)^2 (rev2)": "第 2 版上限 $4(r/3h)^2$",
       "cap w<=r/3": "$w\\le r/3$", "cap w<=r/4": "$w\\le r/4$", "cap w<=r/6": "$w\\le r/6$（第 3 版）", "oracle (best n per vortex)": "逐涡最优（不可实现）"}
L = [r"\begin{tabular}{lcc}\toprule 平滑规则 & $\delta=0.1$ & $\delta=0.2$\\\midrule"]
for n in names:
    L.append(f"{lab[n]} & {SS['rules']['0.1'][n]:.3f} & {SS['rules']['0.2'][n]:.3f}\\\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "t3_rules.tex", "w").write("\n".join(L))
# per-vortex best n
L = [r"\begin{tabular}{llcc" + "c"*7 + r"}\toprule 算例 & 涡 & $r_{\rm eq}/h$ & 最优 $n$ & " + " & ".join(f"{n}" for n in (2, 3, 4, 6, 8, 12, 16)) + r"\\\midrule"]
for c in ["LO", "DG", "SV", "SB"]:
    for v in SW[c]["vortices"]:
        x = v["by_delta"].get("0.1")
        if not x: continue
        mn = {int(n): float(np.mean(y)) for n, y in x["iou_noise"].items()}; best = max(mn, key=mn.get)
        L.append(f"{NM[c]} & ({v['pos'][0]:.2f}, {v['pos'][1]:.2f}) & {x['req_over_h']:.1f} & {best} & " + " & ".join(f"{mn[n]:.2f}" for n in (2, 3, 4, 6, 8, 12, 16)) + r"\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "t3_bestn.tex", "w").write("\n".join(L))
# delta
L = [r"\begin{tabular}{ll" + "c"*5 + r"}\toprule 算例 & 涡 & " + " & ".join(f"$\\delta={d}$" for d in (0.05, 0.1, 0.15, 0.2, 0.3)) + r"\\\midrule"]
for k, v in SS["delta"].items():
    c, pos = k.split(" ", 1); pos = json.loads(pos)
    cells = []
    for d in ("0.05", "0.1", "0.15", "0.2", "0.3"):
        cells.append("{:.1f} / {:.2f} / {:.2f}".format(*v[d]) if d in v else "—")
    L.append(f"{NM[c]} & ({pos[0]:.2f}, {pos[1]:.2f}) & " + " & ".join(cells) + r"\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "t3_delta.tex", "w").write("\n".join(L))
# J_max
L = [r"\begin{tabular}{lll}\toprule 算例 & 干净：全部显著核的单步占比 & 噪声（3 个种子）：最大值\\\midrule"]
for c in ["LO", "DG", "SV", "SB", "SOD"]:
    a = ", ".join(f"{s:.2f}" for s in sorted(SW[c]["shares_clean"])[:12]); b = max([max(s) for s in SW[c]["shares_noise"] if s], default=float("nan"))
    L.append(f"{NM[c]} & {a} & {'—' if np.isnan(b) else f'{b:.2f}'}\\\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "t3_jmax.tex", "w").write("\n".join(L))
print("ok")
