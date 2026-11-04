import json, numpy as np
CASES = ["DG", "LO", "SOD", "BLAST", "SB", "SV"]; LAB = dict(DG="双环流", LO="Lamb–Oseen", SOD="Sod", BLAST="爆炸波", SB="激波–气泡", SV="激波–涡")
R = {c: json.load(open(f"/home/claude/bench/tcx_res_{c}.json")) for c in CASES}


def e(x, d=3):
    if x is None:
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(x)
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        m, ex = f"{x:.1e}".split("e"); return f"${m}\\times10^{{{int(ex)}}}$"
    return f"{x:.{d}f}"


def row(name, vals):
    return name + " & " + " & ".join(vals) + r"\\"


def phi_cell(c, key, sub="P3_phi"):
    v = R[c][sub][key]
    return e(v["pearson"], 4) + " / " + e(v["rel_rms"]) if "pearson" in v else "p99 " + e(v["p99"])


L = []
hdr = "量 & " + " & ".join(LAB[c] for c in CASES) + r"\\\midrule"
# ---- table A: identities, objectivity, chain
L.append(r"\begin{table}[h]\centering\small\caption{P1 与锚定链：恒等式、客观性与热步}\label{tab:p1}")
L.append(r"\begin{tabularx}{\linewidth}{@{}P{4.2cm}" + "L" * 6 + r"@{}}\toprule"); L.append(hdr)
L.append(row("恒等式误差 $\\det F-(|p|^2-|q|^2)$", [e(R[c]["P1_identities"]["det_err"]) for c in CASES]))
L.append(row("恒等式误差 $c_t/A_t-2\\,\\mathrm{Im}\\,p$", [e(R[c]["P1_identities"]["circ_err"]) for c in CASES]))
L.append(row("客观性：新 $\\varphi$ 相对变化", [e(R[c]["P1_objectivity"]["phi_new"]) for c in CASES]))
L.append(row("客观性：新 LAVD 相对变化", [e(R[c]["P1_objectivity"]["lavd_new"]) for c in CASES]))
L.append(row("客观性：旧环量法 $\\varphi$ 相对变化", [e(R[c]["P1_objectivity"]["phi_old_circ"]) for c in CASES]))
L.append(row("热步比例", [e(R[c]["chain"]["hot_frac"]) for c in CASES]))
L.append(row("每节点平均段数", [e(R[c]["chain"]["nseg_mean"], 2) for c in CASES]))
L.append(r"\bottomrule\end{tabularx}\end{table}")
# ---- table B: rotation and LAVD
L.append(r"\begin{table}[h]\centering\small\caption{P3、P5：离散相对转角与 LAVD（有涡算例为 Pearson / 相对 RMS；无旋算例为伪量 p99，单位 rad）}\label{tab:p3}")
L.append(r"\begin{tabularx}{\linewidth}{@{}P{3.4cm}" + "L" * 6 + r"@{}}\toprule"); L.append(hdr)
for k_, nm in (("new", "新：锚定 + 合并穿越"), ("new_nomerge", "锚定，不合并"), ("argp_retri", "arg p，逐帧重剖"), ("old_circ_retri", "旧：环量，逐帧重剖")):
    L.append(row("$\\varphi$ " + nm, [phi_cell(c, k_) for c in CASES]))
L.append(row("LAVD 新", [phi_cell(c, "new", "P5_lavd") for c in CASES]))
L.append(r"\bottomrule\end{tabularx}\end{table}")
# ---- table C: signature and ledger
L.append(r"\begin{table}[h]\centering\small\caption{P2、P6：激波签名与块链账本}\label{tab:p2}")
L.append(r"\begin{tabularx}{\linewidth}{@{}P{4.0cm}" + "L" * 6 + r"@{}}\toprule"); L.append(hdr)
g = lambda c, k_: R[c]["P2_signature"].get(k_)
L.append(row("参照压缩 $>0.05$ 的块数", [e(g(c, "n_true_compression")) if g(c, "n_true_compression") is not None else "0" for c in CASES]))
L.append(row("偏离单轴曲线（中位数）", [e(g(c, "dev_median")) for c in CASES]))
L.append(row("块极转角 $|\\arg P_B|$（中位数）", [e(g(c, "argP_median")) for c in CASES]))
L.append(row("$\\ln\\eta$ 相对误差（中位数）", [e(g(c, "ln_eta_err_rel")) for c in CASES]))
L.append(row("误判为平面激波的块比例", [e(g(c, "false_shock_rate")) for c in CASES]))
L.append(row("账本 $|\\Delta\\ln J|$ 中位 / p95：新", [e(R[c]["P6_ledger"]["new_med"]) + " / " + e(R[c]["P6_ledger"]["new_p95"]) for c in CASES]))
L.append(row("新，锚定 5 帧", [e(R[c]["P6_ledger"]["new_R5_med"]) + " / " + e(R[c]["P6_ledger"]["new_R5_p95"]) for c in CASES]))
L.append(row("旧，C 通道推迟重锚", [e(R[c]["P6_ledger"]["old_Cchannel_med"]) + " / " + e(R[c]["P6_ledger"]["old_Cchannel_p95"]) for c in CASES]))
L.append(r"\bottomrule\end{tabularx}\end{table}")
# ---- table D: noise
L.append(r"\begin{table}[h]\centering\small\caption{P4：0.1$h$ 位置噪声}\label{tab:p4}")
L.append(r"\begin{tabularx}{\linewidth}{@{}P{3.4cm}" + "L" * 6 + r"@{}}\toprule"); L.append(hdr)
L.append(row("估计噪声 $\\hat\\sigma/h$", [e(R[c]["P4_noise"]["sigma_hat_over_h"]) for c in CASES]))
L.append(row("$\\varphi$ 新（锚定）", [phi_cell(c, "phi_new_anchored", "P4_noise") for c in CASES]))
L.append(row("$\\varphi$ arg p 逐帧重剖", [phi_cell(c, "phi_argp_retri", "P4_noise") for c in CASES]))
L.append(row("$\\varphi$ 旧环量逐帧重剖", [phi_cell(c, "phi_old_circ_retri", "P4_noise") for c in CASES]))
L.append(row("LAVD 扣噪底", [phi_cell(c, "lavd_floor", "P4_noise") for c in CASES]))
L.append(row("LAVD 不扣噪底", [phi_cell(c, "lavd_nofloor", "P4_noise") for c in CASES]))
L.append(row("LAVD 偏差：扣 / 不扣", [e(R[c]["P4_noise"]["lavd_bias_floor"]) + " / " + e(R[c]["P4_noise"]["lavd_bias_nofloor"]) for c in CASES]))
L.append(row("账本 $|\\Delta\\ln J|$ 中位 / p95", [e(R[c]["P4_noise"]["lnJ_med"]) + " / " + e(R[c]["P4_noise"]["lnJ_p95"]) for c in CASES]))
L.append(r"\bottomrule\end{tabularx}\end{table}")
# ---- table E: circulation jump
L.append(r"\begin{table}[h]\centering\small\caption{P7：物质胞腔上的环量跳变（Pearson；无旋算例为伪量 p99）}\label{tab:p7}")
L.append(r"\begin{tabularx}{\linewidth}{@{}P{3.4cm}" + "L" * 6 + r"@{}}\toprule"); L.append(hdr)
for cond in ("clean", "noise"):
    for blk in ("single", "chain5"):
        for m in (1, 2, 4):
            k_ = f"{cond}_{blk}_m{m}"
            cells = []
            for c in CASES:
                v = R[c]["P7_circulation"][k_]
                cells.append(e(v["pearson"]) if "pearson" in v else "p99 " + e(v["p99"]))
            L.append(row(("干净" if cond == "clean" else "噪声") + "，" + ("单次锚定" if blk == "single" else "每 5 帧重锚") + f"，$m={m}$", cells))
L.append(r"\bottomrule\end{tabularx}\end{table}")
# ---- table F: cells, Hodge, persistence, boundaries, stretch
L.append(r"\begin{table}[h]\centering\small\caption{P8–P12：胞腔、Hodge、持续性、涡边界与切向拉伸}\label{tab:p8}")
L.append(r"\begin{tabularx}{\linewidth}{@{}P{4.2cm}" + "L" * 6 + r"@{}}\toprule"); L.append(hdr)
L.append(row("非仿射度 $>0.01$ 的胞腔比例", [e(R[c]["P8_cells"]["hollow_frac"]["0.01"]) for c in CASES]))
L.append(row("非仿射度：核外 / 核内（中位比）", [e(R[c]["P8_cells"].get("enrichment")) for c in CASES]))
L.append(row("非仿射度与 $|\\nabla\\varphi_{\\mathrm{ref}}|$ 的 Spearman", [e(R[c]["P8_cells"].get("spearman_eps_gradphi")) for c in CASES]))
L.append(row("调和维数（3 个空心胞腔）：对角 / Whitney / 随机", ["/".join(str(R[c]["P9_hodge"][k_]["harmonic_dim"]) for k_ in ("diag_cotan", "whitney", "random_spd")) for c in CASES]))
L.append(row("Hodge 正交性残差（三种度量最大）", [e(max(R[c]["P9_hodge"][k_]["orth"] for k_ in ("diag_cotan", "whitney", "random_spd"))) for c in CASES]))
L.append(row("Galerkin 梯度份额：单步 / 5 帧块", [e(R[c]["P9_hodge"]["galerkin_step1"]["Eg"]) + " / " + e(R[c]["P9_hodge"]["galerkin_block5"]["Eg"]) for c in CASES]))
L.append(row("持续性核数：网络 / 参照", [f"{R[c]['P10_persistence']['cores_net']} / {R[c]['P10_persistence']['cores_ref']}" for c in CASES]))
L.append(row("噪声下核数（与参照匹配）", [f"{R[c]['P10_persistence']['cores_net_noise']}（{R[c]['P10_persistence']['matched_noise_in_ref']}）" for c in CASES]))
def bnd(c):
    b = [x for x in R[c]["P11_boundary"] if "iou" in x]
    return f"{len(b)} 个，IoU {np.mean([x['iou'] for x in b]):.3f}" if b else ("无参照涡" if not any(x["ref_found"] for x in R[c]["P11_boundary"]) else "未找到")
L.append(row("凸边界：与参照匹配数，平均 IoU", [bnd(c) for c in CASES]))
L.append(row("界面拉伸均值：网络 / 参照", [e(R[c]["P12_stretch"]["net_mean"]) + " / " + e(R[c]["P12_stretch"]["ref_mean"]) for c in CASES]))
L.append(row("逐点误差中位数", [e(R[c]["P12_stretch"].get("point_err_median")) for c in CASES]))
L.append(r"\bottomrule\end{tabularx}\end{table}")
open("/home/claude/tcxrep/tables.tex", "w").write("\n".join(L))
print("tables done")
