"""Tables for the rev2 sections."""
import json, numpy as np
O = "report_rev/"; NM = dict(DG="双环流", LO="Lamb–Oseen", SOD="Sod", BLAST="爆炸波", SB="激波–气泡", SV="激波–涡")
# topo
L = [r"\begin{tabular}{lcccc}\toprule 算例与参数 & 激活边比例 干净 / 噪声 & 涡 干净 & 涡 噪声 & 干净涡的大小 \\\midrule"]
LO = json.load(open("topo_res_LO.json"))
rows = [("LO", "delta=0.05 conformal=True", LO), ("LO", "delta=0.1 conformal=True", LO), ("LO", "delta=0.1 conformal=False", LO), ("LO", "delta=0.2 conformal=True", LO),
        ("DG", "delta=0.1 conformal=True", json.load(open("topo_res_DG.json"))), ("SV", "delta=0.1 conformal=True", json.load(open("topo_res_SV.json")))]
for c, k, R in rows:
    d = R[k]; vs = d["clean"]["vortices"]
    size = "、".join(f"{v['n_tri']} 个三角形" for v in vs[:2]) if vs else "—"
    if c == "SV" and vs: size += "（背景小块，非涡）"
    dl = k.split()[0].split("=")[1]; cf = "扣 ln J/2" if "True" in k else "不扣"
    L.append(f"{NM[c]}，$\\delta={dl}$，{cf} & {100*d['active_clean']:.3g}\\% / {100*d['active_noise']:.3g}\\% & {d['clean']['n_vortex']} & {d['noise']['n_vortex']} & {size}\\\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "t_topo.tex", "w").write("\n".join(L))
# perimeter estimators
P1 = json.load(open("res_iss_perim.json")); P2 = json.load(open("res_iss_perim2.json"))
L = [r"\begin{tabular}{lcccc}\toprule 算例 & P1 插值像 & 当前复形上的等值线 & 二次重构 & P1 $\ge$ 精确的比例\\\midrule"]
for key in ["LO N=8000", "LO N=32000", "DG N=12800"]:
    a = P1[key]; e1 = np.median([abs(x["P1"] - x["exact"]) for x in a]); e2 = np.median([abs(x["Kt"] - x["exact"]) for x in a if x["Kt"] is not None])
    b = np.array(P2[key]); e3 = np.median(np.abs(b[:, 2] - b[:, 4])); up = np.mean(b[:, 1] >= b[:, 4] - 1e-3)
    L.append(f"{key.replace('LO', 'Lamb–Oseen').replace('DG', '双环流')} & {e1:.3f} & {e2:.3f} & {e3:.3f} & {up:.2f}\\\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "t_perimest.tex", "w").write("\n".join(L))
# sustained shares
L = [r"\begin{tabular}{lll}\toprule 算例 & 干净核的单步占比 & 噪声核的单步占比（种子 5）\\\midrule"]
for c in ["SOD", "DG", "LO", "SV", "SB"]:
    r = json.load(open(f"iss_cores_{c}.json"))
    a = ", ".join(f"{x['share']:.2f}" for x in r["cores_clean"][:6]) or "—"
    b = ", ".join(f"{x['share']:.2f}" for x in r["noise"][0]["cores"][:6]) or "—"
    L.append(f"{NM[c]} & {a} & {b}\\\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "t_share.tex", "w").write("\n".join(L))
# SNR and smoothing
L = [r"\begin{tabular}{llcccccc}\toprule 算例 & 参照涡 & 持续性 & 信噪比（种子 5/6/7） & \multicolumn{4}{c}{噪声 IoU（平滑 2 / 4 / 8 / 16 次；三个种子的范围）}\\\midrule"]
for c in ["DG", "LO", "SV", "SB"]:
    r = json.load(open(f"iss_cores_{c}.json")); nv = len(r["ref_vortices"])
    for i in range(nv):
        snr = "/".join(f"{rec['ref_vortex_snr'][i]['pers_over_noise']:.1f}" for rec in r["noise"])
        rng = []
        for npass in ("2", "4", "8", "16"):
            v = [rec["iou_vs_ref_by_smoothing"][npass][i] for rec in r["noise"]]
            rng.append(f"{min(v):.2f}–{max(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}")
        L.append(f"{NM[c] if i == 0 else ''} & ({r['ref_vortices'][i][0]:.2f}, {r['ref_vortices'][i][1]:.2f}) & {r['noise'][0]['ref_vortex_snr'][i]['ref_pers']:.2f} & {snr} & " + " & ".join(rng) + r"\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "t_snr.tex", "w").write("\n".join(L))
# SB windows
W = json.load(open("res_iss_sb.json"))
L = [r"\begin{tabular}{lcccc}\toprule 窗口 & 主涡持续性 & 最内层等值线 $\ln\mathcal L$ & 相干边界 & 等效半径 / $h$\\\midrule"]
for k, rows in W.items():
    x = rows[0]
    a_, b_ = [float(v) for v in k.strip("[]").split(",")]
    L.append(f"{{}}$[{a_:.1f},\\,{b_:.1f}]$ & {x['pers']:.1f} & {x['innermost_lnL']:.2f} & {'有' if x['coherent_area'] else '无'} & {x['coherent_req_over_h'] if x['coherent_req_over_h'] else '—'}\\\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "t_sbwin.tex", "w").write("\n".join(L))
print("ok")
