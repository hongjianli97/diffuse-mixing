"""v2.1 -> v3 full report (Chinese): new part 7 (ledger v4, NS viscous shock, T4 merged-block split), updated status
row, T4 section status, open-issue list.  All numbers from tools/v4_numbers.py."""
import re, sys, json
sys.path.insert(0, "tools"); import v4_numbers as V
N = V.numbers_v32(); N['m_vel_maxdiff_all'] = '%.0f' % max(float(N[k]) for k in N if k.endswith('_vel_maxdiff')); v9 = V.R("res_v9_viscous_shock"); v10 = V.R("res_v10_t4"); used = set()
b = v10["sb"]["band 0.15"]
N["t4_SB_imp_diff"] = "%.1f" % (100 * abs(b["impulsive"] / b["dns_del_impulsive"] - 1)); N["t4_SB_smo_diff"] = "%d" % round(100 * abs(b["smooth"] / b["dns_del_smooth"] - 1))
N["t4_RM_smooth_share"] = "%d" % round(100 * v10["rm"]["band 0.08"]["smooth"] / v10["rm"]["band 0.08"]["total"])
def F(s):
    def sub(m):
        k = m.group(1)
        if k not in N: return m.group(0)
        used.add(k); return N[k].replace("--", r"\mbox{--}")   # en dash in text and math mode
    return re.sub(r"\{([A-Za-z0-9_.]+)\}", sub, s)
p = "report/LHD_full_report.tex"; s = open(p).read()
def rep(old, new):
    global s
    assert old in s, old[:50]; s = s.replace(old, new, 1)
ZH = {"planar shock, mid-domain at $T$": "平面制造激波（评估时停在区域中部）", "weak shock ($\\eta=1.5$, $C_s=0.5$)": "弱激波（$\\eta=1.5,C_s=0.5$）",
      "slow shock ($C_s=0.25$)": "慢激波（$C_s=0.25$）", "smooth manufactured map": "光滑制造映射", "converging cylindrical shock": "会聚柱面制造激波",
      "Sod": "Sod", "cylindrical blast": "柱面爆炸波", "double gyre": "双环流", "shock--vortex": "激波–等熵涡", "shock--bubble": "激波–气泡（无粘）",
      "Richtmyer--Meshkov": "单模 RM", "shock--bubble, viscous": "激波–气泡（粘性）"}
rows = []
for r in V.table_rows():
    lab = r[0]; lab = ZH.get(lab, lab).replace("\\quad clean tracers", "\\quad DNS 未混合的示踪点")
    rows.append("%s & %s\\%% & %s\\%% & %s\\%%\\\\" % ((lab,) + tuple(x.replace(" / ", "\\% / ") for x in r[1:])))
vk = ["Ms2.24_d1_Cs0.5", "Ms2.24_d2_Cs1", "Ms2.24_d4_Cs1", "Ms2.24_d4_Cs0.25", "Ms1.5_d2_Cs0.5", "Ms1.5_d4_Cs0.25"]
vrows = []
for k in vk:
    o = v9[k]; f = o["final"]
    vrows.append("$M_s$=%g，$\\delta/h$=%.1f，$C_s$=%g & %s（%d\\%%） & %s & %s\\\\" % (o["Ms"], o["delta_measured_over_h"], o["Cs"],
                 V.mp(f["v2"], 2, 1), round(100 * f["v2"]["detected"]), V.mp(f["v3_hybrid"], 3, 2), V.mp(f["v4"], 3, 2)))
trows = []
for c, bb, lab in (("sb", "band 0.15", "激波–气泡（无粘），$|\\phi_0|<0.15$"), ("sb", "band 0.08", "激波–气泡（无粘），$|\\phi_0|<0.08$"),
                   ("sbv", "band 0.15", "激波–气泡（粘性），$|\\phi_0|<0.15$"), ("rm", "band 0.08", "单模 RM，$|\\phi_0|<0.08$")):
    B = v10[c][bb]; sg = -1 if c != "rm" else 1; P = B["at_passage_end"]
    trows.append("%s & %.4f & %.4f / %.4f & %.4f / %.4f & %.4f / %.4f & %.4f / %.4f\\\\" % (lab, sg * B["total"], sg * B["impulsive"], sg * B["smooth"],
                 sg * B["dns_del_impulsive"], sg * B["dns_del_smooth"], sg * P["impulsive"], sg * P["smooth"], sg * P["dns_del_impulsive"], sg * P["dns_del_smooth"]))
    if "dns_step_impulsive" in B:
        trows.append("\\quad 同上，逐求解器步积分的力矩 & & & %.4f / %.4f & & %.4f / %.4f\\\\" % (sg * B["dns_step_impulsive"], sg * B["dns_step_smooth"], sg * P["dns_step_impulsive"], sg * P["dns_step_smooth"]))
    if "dns_vel_impulsive" in B:
        trows.append("\\quad 同上，同一三角形上的 DNS 速度环量 & & & %.4f / %.4f & & %.4f / %.4f\\\\" % (sg * B["dns_vel_impulsive"], sg * B["dns_vel_smooth"], sg * P["dns_vel_impulsive"], sg * P["dns_vel_smooth"]))
PART7 = F(r"""\clearpage\part{第七部分　v3–v3.2：账本 v4、可分辨粘性激波、T4 合并块拆分、三维与合成 PTV}

本部分处理 v2.1 列出的三项未解决问题：张量账本的离群值与激波处的 95 分位；多窗口检测在可分辨粘性激波上的检验；T4 冲击/光滑拆分改在合并块上定义并与 DNS 斜压力矩直接比较。四个 DNS（激波–气泡、激波–等熵涡、单模 RM、粘性激波–气泡，$\Delta x=0.005$）全部重跑。所有数字由 \texttt{tools/patch\_report\_v3.py} 从 \texttt{results/res\_v8\_ledger\_v4.json}、\texttt{res\_v9\_viscous\_shock.json}、\texttt{res\_v10\_t4.json} 生成。

\section{账本 v4}
v3 的混合估计只在末时刻比较一次张量值与多边形值，一个坏的张量增量会使此后整段记录都退回多边形值。账本是各锚定段增量之和，两种估计都可以在同一个锚定段上计算，v4 据此做了三处改动（\texttt{lhd/ledger\_v4.py}）。
\begin{itemize}
\item \textbf{几何窗口检测。}窗口取 $w\in\{2,3,4,6,8,12,16,24,32\}$。只有当窗口内已有热步解释的面积下降不到一半时才使用这个窗口，并且只标记真正承担压缩的那几步（单步对数比低于窗口均值的 0.1），再向两侧延伸，直到星的单步压缩不再可测（单步对数比高于 $-\max(3\hat\sigma_R,0.02R_{\rm peak})$，$\hat\sigma_R$ 为该示踪点单步比值的稳健背景，$R_{\rm peak}$ 为窗口内最大压缩）。这样可分辨剖面的尾部能被覆盖，单步就能检出的激波也不会拉长推迟；推迟上限由热段长度的 99 分位给出，最多 60 块。
\item \textbf{分段加保护的张量增量。}某一段使用张量增量的条件是：拟合良态（$F$ 的相对标准误差 $<0.05$）；近邻不跨物质界面（参考密度在 $\ln$ 上的跳跃 $>0.1$ 即视为跨界面）；与多边形增量之差在无热步段不超过 0.1，在热步段不超过 $0.02+2|\delta|+3\hat\sigma/h$。其中 $\delta$ 是该段多边形的非仿射量，$\hat\sigma$ 由三阶时间差分估计。被拒的段只回退这一段，离群值不再往后传。
\item \textbf{单侧弧修复。}跨物质界面的星，用同种物质的最大连续弧扇形计算；评估时刻正被激波扫过的星，用与中心一起作仿射运动的最长环弧计算。后者要求弧的仿射残差低于整环的 1/4 且低于 $0.01+3\hat\sigma/h$，同时整环残差大于 0.03。对锐利激波或接触间断，两种修复都精确；对可分辨激波或光滑的非仿射运动，不会有弧满足条件，结果不变。质量仍由多边形账本给出。
\end{itemize}
\paragraph{回归检查中发现并修正的三个问题。}第一版 v4 在激波–等熵涡上不如 v3。原因有三：(1) 弧修复在光滑的非仿射涡核里被误用了 5465 次，加上整环残差下限后为 0–1 次；(2) 包含单步激波的长窗口使整窗被标为热步，推迟上限从 6 块拉长到 13 块，加上“已解释”判据和裁剪后解决；(3) 激波段固定容差 0.02 拒用了涡核里的张量，改为 $0.02+2|\delta|$ 后解决。加噪声时激波段的拒用次数增加约 30 倍，所以容差里又加了 $3\hat\sigma/h$。
\begin{table}[!htbp]\centering\small\caption{末时刻密度误差（中位 / 95 分位）：账本 v2、v3 混合估计与 v4。DNS 未混合的示踪点：DNS 被动标量 $|\Delta Y_1|\le0.01$。}
\begin{tabularx}{\linewidth}{@{}L r r r@{}}\toprule
算例 & v2 & v3 混合 & v4\\\midrule
""") + "\n".join(rows) + F(r"""
\bottomrule\end{tabularx}\end{table}
\paragraph{读数。}
\begin{itemize}
\item 凡是账本自身造成的激波误差都明显下降：平面制造激波停在区域中部时，95 分位从 ${planar_mid_p95_v2}\%$（v2）和 ${planar_mid_p95_v3}\%$（v3）降到精确；爆炸波从 ${blast_p95_v2}\%$ 降到 ${blast_p95_v4}\%$；Sod 降到 ${sod_p95_v4}\%$；激波–等熵涡为 ${sv_v4}\%$（v3 的 95 分位为 ${sv_p95_v3}\%$）；粘性激波–气泡的 95 分位降到 ${sbv_p95_v4}\%$。
\item 无粘激波–气泡与 RM 中仍有 30\%–35\% 的 95 分位，这部分来自参照：它们落在 DNS 已把两种气体数值混合的示踪点上，那里的 DNS 密度是混合值。只看未混合的示踪点（${sb_clean_frac}\%$、${rm_clean_frac}\%$、${sbv_clean_frac}\%$），v4 为 ${sb_clean_v4}\%$、${rm_clean_v4}\%$、${sbv_clean_v4}\%$，v2 为 ${sb_clean_v2}\%$、${rm_clean_v2}\%$、${sbv_clean_v2}\%$。
\item 0.02$h$ 噪声下，v4 为 ${sod_noise_v4}\%$（Sod）、${blast_noise_v4}\%$（爆炸波）、${sv_noise_v4}\%$（激波–涡）、${rm_noise_v4}\%$（RM）。中位误差与 v3 相差不超过 0.11 个百分点；爆炸波的 95 分位约为 v3（34\%）的三分之一，Sod 略高于 v3（3.3\%）。
\item 0.02$h$ 噪声下的未混合示踪点：激波–气泡 v4 为 ${sb_noise_clean_v4}\%$（v2 为 ${sb_noise_clean_v2}\%$），粘性激波–气泡 v4 为 ${sbv_noise_clean_v4}\%$（v2 为 ${sbv_noise_clean_v2}\%$）。
\item 新检测没有使多边形账本变差：推迟上限 ${sb_defer_v4}$ 块时，激波–气泡未混合示踪点上为 ${sb_poly_clean_v4}\%$，v2 检测下为 ${sb_poly_clean_v2}\%$。全部示踪点的 95 分位为 ${sb_poly_p95_v4}\%$，v2 检测下为 ${sb_poly_p95_v2}\%$，差别落在参照已被混合的示踪点上。上一版用固定余量时，推迟上限为 12 块，95 分位为 31.1\%。
\end{itemize}

\section{可分辨粘性激波：一维 Navier–Stokes DNS}
平面粘性激波是一维问题，所以用一维 Navier–Stokes DNS 计算（\texttt{lhd/ns1d.py}）。格式与物性和激波–气泡的粘性 DNS 相同：MUSCL–HLLC，常粘性，$\mathrm{Pr}=0.72$，补上了显式粘性步长限制，网格取 $\delta/10$。示踪点在求解器内用 Heun 法推进，经旋转（$\theta=0.5$）并加均匀切向滑移嵌入二维。实测激波厚度 $\delta/h={visc_delta_range}$（$M_s=2.24$ 时 $\delta\approx2.3\mu$）。末时刻全部示踪点已越过激波，参照为 DNS 在示踪点处的密度（\texttt{scripts/v9\_viscous\_shock.py}）。
\begin{table}[!htbp]\centering\small\caption{可分辨粘性激波（NS DNS）的末时刻密度误差（\%，中位 / 95 分位），括号内为单步检测的检出率。}
\begin{tabularx}{\linewidth}{@{}L r r r@{}}\toprule
算例 & v2（检出率） & v3 & v4\\\midrule
""") + "\n".join(vrows) + F(r"""
\bottomrule\end{tabularx}\end{table}
\paragraph{读数。}
\begin{itemize}
\item 单步检测在薄而慢的激波上只检出 ${visc_thin_det_v2}\%$，在厚激波上检出率为零。v3 的 6 帧窗口能检出 $M_s=2.24$ 的激波，但厚而慢的激波上推迟不够长，误差为 ${visc_thick_v3}\%$。弱、厚且慢的激波（$M_s=1.5$，$\delta/h=4$，$C_s=0.25$）v3 完全漏检：检出率 ${visc_weak2_det_v3}\%$，误差 ${visc_weak2_v3}\%$；v2 为 ${visc_weak2_v2}\%$。
\item 几何窗口加自适应尾部之后，6 个算例的中位误差为 ${visc_v4_range}\%$：厚而慢的激波为 ${visc_thick_v4}\%$，两个弱激波为 ${visc_weak_v4}\%$ 和 ${visc_weak2_v4}\%$。在薄激波上 v3 与 v4 相差不超过 0.001 个百分点（${visc_thin_v3}\%$ 与 ${visc_thin_v4}\%$）。激波正在区域中部时，激波层以外的误差为 ${visc_mid_out}\%$。
\item 上一版用固定余量（标记跨度的一半），在弱激波（$M_s=1.5$，$\delta/h=2$）上为 0.133\%；把余量加到全部跨度能修好这个算例，但会使激波–气泡的 95 分位升到 31.2\%。自适应尾部在两边都没有代价，这个折中已经不存在。
\end{itemize}

\section{T4：合并块上的冲击/光滑拆分与 DNS 斜压力矩}
\paragraph{定义。}三角形在快照 $n$ 为热，当且仅当某个顶点在第 $n-1$ 或第 $n$ 步为热。在它的每个热块 $[n_0,n_1]$ 上，由定理 2 望远相消，冲击沉积等于 $\Gamma_t^{n_1+1/2}-\Gamma_t^{n_0-1/2}$，是块两端帧的有限步量，不含逐步求和的伪转动。不压缩三角形的热块（面积对数比不低于 $-0.15$）不计入冲击部分，因为卷起的涡片中相邻三角形会通过数值或物理粘性通量交换环量。每个保留的块两侧各加一个快照的余量（理由见下）。冲击与光滑两部分之和精确等于区域环量的变化，闭合误差为 ${t4_SB_closure}$（\texttt{lhd/t4split.py}）。旧拆分把“区域内任何地方有激波”的快照整步记为冲击部分；在这些物质带上总有一部分正被激波扫过，所以旧拆分把全部变化都记为冲击部分，光滑部分为零。
\paragraph{参照。}DNS 斜压力矩 $\rho^{-2}\nabla\rho\times\nabla p$ 由求解器场在示踪点上的值经 P1 梯度计算，并用同一分类拆分，有两种形式：
\begin{itemize}
\item 在快照上采样，在带内当前 Delaunay 三角形上计算；
\item 在每个求解器步上积分（梯形公式），在最近快照的 Delaunay 三角形上计算，按示踪点和快照区间累加（\texttt{scripts/run\_sb\_torque.py}）。重跑的 DNS 与存档的示踪点位置完全一致。
\end{itemize}
后者在时间上分辨了冲击沉积，不依赖快照采样。
\begin{table}[!htbp]\centering\small\caption{物质带环量变化 $|\Delta\Gamma|$ 的拆分：位置形式（冲击 / 光滑）与 DNS 斜压力矩（冲击 / 光滑），全程与激波扫过结束时（激波–气泡 $t=0.6$，RM $t=0.8$）。}
\begin{tabularx}{\linewidth}{@{}L r r r r r@{}}\toprule
区域 & 总变化 & 位置形式 & DNS 力矩 & 位置形式（扫过结束） & DNS 力矩（扫过结束）\\\midrule
""") + "\n".join(trows) + F(r"""
\bottomrule\end{tabularx}\end{table}
\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{fig_v4.pdf}
\caption{(a) 账本 v2、v3、v4 的末时刻 95 分位误差；(b) 可分辨粘性激波（NS DNS）的中位误差，灰柱上方数字为单步检出率；(c) 激波–气泡宽带的环量变化在合并块上的拆分（实线）、同一三角形上的 DNS 速度环量（虚线）与逐求解器步积分的 DNS 斜压力矩（点划线），三者用同一分类，点线为旧拆分（与总变化重合）；(d) 单位初始界面长度上的冲击沉积随偏离上游点角度的分布，与 SZ1 局部标度律对比。}\end{figure}
\paragraph{读数。}
\begin{itemize}
\item \textbf{主参照}：同一物质三角形上的 DNS 速度环量。求解器速度存储在示踪点上，用与位置形式相同的帽形核 $\Delta\Gamma^n_{\rm DNS}=\tfrac12(\Gamma_{n+1}-\Gamma_{n-1})$，按同一组块拆分。它的总量与位置形式相差 ${m_SB_vel_total}\%$（$t>0.4$ 的中位；窄带为 ${m_SB8_vel_total}\%$，因为它包含了数值耗散）。
\item 无粘激波–气泡宽带：位置形式为 $|\Delta\Gamma|={m_SB_total}={m_SB_impulsive}+{m_SB_smooth}$（冲击 + 光滑），DNS 速度为 ${m_SB_vel_imp}+{m_SB_vel_smo}$；激波扫过结束时（$t=0.6$）为 ${m_SB_pass_impulsive}+{m_SB_pass_smooth}$ 对 ${m_SB_pass_vel_imp}+{m_SB_pass_vel_smo}$。粘性激波–气泡为 ${m_SBV_impulsive}+{m_SBV_smooth}$ 对 ${m_SBV_vel_imp}+{m_SBV_vel_smo}$；RM 为 ${m_RM_impulsive}{m_RM_smooth}$ 对 ${m_RM_vel_imp}{m_RM_vel_smo}$。六条带（每个流动两条）上，两部分的最大相差为 ${m_vel_maxdiff_all}\%$。
\item \textbf{块边界余量是关键}：每个保留的块两侧各加一个快照。不加余量时，宽带冲击部分在位置形式下为 ${mg0_pos}$，在 DNS 速度下为 ${mg0_vel}$，原因是捕捉激波有几个网格宽，其前沿比星压缩越过检测阈值早约一个快照到达三角形。加一个快照后为 ${mg1_pos}$ 与 ${mg1_vel}$，加两个快照后为 ${mg2_pos}$ 与 ${mg2_vel}$。快照间隔加倍只使位置形式的冲击部分改变 0.5\%（$\Delta t=0.04$、不加余量时为 ${dt04_pos}$）。
\item \textbf{以加速度构造的三种参照}：同一三角形上 DNS 加速度 $-\nabla p/\rho\,(+\nabla\!\cdot\!\tau/\rho)$ 的线积分（逐求解器步积分、同样的核）为 ${m_SB_mat_imp}+{m_SB_mat_smo}$，其总量误差为 ${m_SB_mat_total}\%$；逐求解器步积分的斜压力矩为 ${m_SB_step_imp}+{m_SB_step_smo}$；另有快照采样的力矩。三者给出的数值把位置形式包在中间，但各自带有由捕捉激波造成的、几个百分点的误差。
\item 激波气泡约有三分之一的环量不是在三角形自己的激波块内沉积的，而是由波后压力场沉积；RM 的光滑部分是来自来回反射波的小的负修正。在物质三角形上直接对力矩做 P1 求积，三角形被拉伸后会失效（后期变号）。
\item \textbf{与 SZ 局部标度律对比}：把宽带的冲击沉积按初始时刻偏离上游点的角度 $\theta$ 分组，除以 $R\,\Delta\theta$，得到单位初始界面长度上的沉积（图 d）。总量所用的 SZ1 模型假定上游四分之一圆为 $\Gamma'_1\sin\theta$，下游四分之一圆为 $\Gamma'_1$。实测结果（以 $\Gamma'_1$ 为单位）：
  \begin{itemize}
  \item 前 45° 与 $\sin\theta$ 一致（${sz_bins3}$ 对 ${sz_law3}$）；
  \item 45°–90° 低于模型（${sz_bins_mid}$ 对 ${sz_law_mid}$）；
  \item 在下游驻点处降到零，而不是保持 $\Gamma'_1$。
  \end{itemize}
  积分值：上游四分之一圆为 ${sz_up}$（SZ 第二阶段为 ${sz_up_SZ}$），下游四分之一圆为 ${sz_down}$（SZ 为 ${sz_down_SZ}$）；粘性气泡分别为 ${szv_up}$ 与 ${szv_down}$。总量 ${sz_total}$ 接近 SZ1 的 ${sz_total_SZ}$，原因是光滑沉积补足了下游的缺口，也就是说 SZ1 的下游项实际上同时包含了冲击沉积和后续沉积。光滑部分按角度分组没有意义，因为它主要是相邻三角形之间沿涡片输运的环量。

\end{itemize}

\section{三维：四面体账本（制造映射）}
账本可以原样推广到四面体 Delaunay 复形（\texttt{lhd/ledger3d.py}）：
\begin{itemize}
\item 星面积改为星体积，推迟规则相同；
\item 命题 2（跨平面激波望远相消）仍然成立，因为仿射映射下多面体的体积比就是 $\det F$。
\end{itemize}
检验结果（\texttt{scripts/v12\_ledger3d.py}）：
\begin{itemize}
\item 三维平面制造激波：法向斜置、有切向滑移，$\eta=1.5,3,6$，${d3_n}$ 个内部示踪点。带推迟时误差不超过 ${d3_exact}$，不推迟时为 ${d3_nodefer}\%$。
\item 光滑可压缩有旋映射：相邻两级分辨率之间的收敛阶依次为 ${d3_slopes}$，趋于一阶，与二维相同。
\end{itemize}
三维流动（DNS）仍未计算。

\section{合成粒子追踪误差}
没有真实的粒子追踪数据，所以在激波–涡 DNS 的示踪点上模拟它的误差结构（$\sigma=0.02h$，\texttt{scripts/v13\_ptv\_synth.py}）。v4 的结果（中位 / 95 分位）：
\begin{itemize}
\item \textbf{白噪声}：${ptv_white}\%$。
\item \textbf{空间相关噪声}：相关长度 $3h$、$8h$ 时为 ${ptv_spatial3}\%$、${ptv_spatial8}\%$，危害小于白噪声，因为星面积只取决于相对位置。
\item \textbf{时间相关噪声}（AR(1)，系数 0.9）：${ptv_temporal}\%$。三阶时间差分估计的噪声只有 $\hat\sigma/h={ptv_temporal_sighat}$，约为真实值的三分之一。
\item \textbf{播种不均匀}（密度相差 3.3 倍）：${ptv_seeding}\%$，不比均匀播种差。
\item \textbf{静态标定畸变}（rms $0.1h$，相关长度 $10h$）：${ptv_calib}\%$。噪声估计完全看不到它（$\hat\sigma/h={ptv_calib_sighat}$）：它以“畸变场散度沿轨迹的变化”直接进入 $\ln J$，任何只用位置的估计都无法消除。因此对标定误差必须直接限制畸变场的散度。
\end{itemize}

\section{仍未解决}
\begin{enumerate}
\item \textbf{静态标定畸变}：账本无法消除。只用位置的自标定在本流动中也不可行：平移段方程与真值的相关系数只有 ${sc_corr}$，反演又放大了误差，校正后误差变为 ${sc_after}\%$，校正前为 ${sc_before}\%$（用真值校正可达 ${sc_oracle}\%$）。因此必须在粒子图像处理阶段去除畸变。
\item \textbf{时间相关噪声}：会被三阶差分低估。这只影响容差中的噪声项，误差本身不比白噪声大。
\item \textbf{尚未做}：三维流动与真实 PTV 数据。
\end{enumerate}

""")
i = s.index(r"\section{仍未解决}"); j = s.index(r"\clearpage\part{附录部分"); s = s[:i] + r"""\paragraph{说明。}v2.1 在这里列出的前三项未解决问题由第七部分处理。

""" + PART7 + s[j:]
rep(r"T4 Kelvin & 位置形式严格成立；区域收支 & 同边界线积分中位相差 0.03–0.25\%；与 SZ 模型 8\% & 论文 A\\",
    F(r"T4 Kelvin & 位置形式严格成立；区域收支；合并块冲击/光滑拆分（v3） & 同边界线积分中位相差 0.03–0.25\%；与 SZ 模型 8\%；两部分与同一三角形上的 DNS 速度环量相差至多 {m_vel_maxdiff_all}\% & 论文 A\\"))
rep(r"(3) 拆分应改在合并块上定义，目前按“激波位于区域内的步”划分，会把这些步内的光滑沉积也算进冲击部分。",
    r"(3) 拆分应改在合并块上定义，目前按“激波位于区域内的步”划分，会把这些步内的光滑沉积也算进冲击部分。\textbf{v3 更新}：缺口 (1)(3) 已在第七部分解决。")
k = "Nanson 恒等式把界面掺混精确分解为切向拉伸与压缩（T9）。"
rep(k, k + F(r"\textbf{v3/v3.1（第七部分）}：分段加保护的账本 v4 去掉了激波处由账本自身造成的误差（爆炸波 95 分位 ${blast_p95_v2}\%\to{blast_p95_v4}\%$）；几何窗口加自适应尾部检测，使望远相消在 6 个可分辨粘性激波（NS DNS）上成立（中位 ${visc_v4_range}\%$）；T4 的冲击/光滑拆分改在合并块上定义，两部分之和精确，两部分与同一物质三角形上的 DNS 速度环量相差不超过 ${m_vel_maxdiff_all}\%$；四面体账本在三维制造平面激波上精确。"))
open(p, "w").write(s); json.dump(sorted(used), open("tools/v4_report_keys.json", "w")); print("report patched; keys", len(used))
