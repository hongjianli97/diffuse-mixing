"""Revise the Chinese full report (v2): new Part VI built from results/*.json, plus updated summary/status/conclusions."""
import sys, json
sys.path.insert(0, "tools"); from paper_numbers import numbers
N = numbers(); R = lambda f: json.load(open("results/%s.json" % f))
v1, v4, v4b, v5, v7, v2 = R("res_v1_circ_line"), R("res_v4_ledger_v3"), R("res_v4b_hybrid"), R("res_v5_diffusion"), R("res_v7_dns3"), R("res_v2_mfm")
f = sys.argv[1]; s = open(f).read(); fails = []
def rep(old, new):
    global s
    if old not in s: fails.append(old[:60]); return
    s = s.replace(old, new, 1)
P = lambda x, d=2: ("%%.%df" % d) % (100 * x)
pm = lambda d: "%s\\%% / %s\\%%" % (P(d["med"]), P(d["p95"], 1))

# ---------- tables computed from JSON
circ_rows = []
for b in ("0.15", "0.08", "0.04"):
    d = v1["band" + b]; circ_rows.append("无粘激波–气泡，$|\\phi_0|<%s$ & %s & %s & %s\\\\" % (b, P(d["relerr_med_position_vs_line"]), P(d["relerr_max_position_vs_line"], 1), P(d["relerr_med_areaquad_vs_line"], 1)))
for nm, lab in (("sbv", "粘性激波–气泡（$\\Delta x=0.005$）"), ("sv", "激波–等熵涡"), ("rm", "单模 RM")):
    for reg, d in v7[nm]["circulation"].items():
        circ_rows.append("%s，%s & %s & %s & %s\\\\" % (lab, reg.replace("band ", "$|\\phi_0|<$").replace("core r0<2Rv", "涡核 $r_0<2R_v$").replace("core r0<Rv", "涡核 $r_0<R_v$"),
                                                       P(d["med_rel_pos_vs_line"], 3), P(d["max_rel_pos_vs_line"], 1), P(d["med_rel_quad_vs_line"], 1)))
diff_rows = []
for lab, k in (("精确 $\\rho$（oracle）", "oracle_exact_rho"), ("账本 $\\rho$（论文默认）", "ledger_elem_arith (paper)"), ("SPH $\\rho$", "sph_rho"), ("DTFE $\\rho$", "dtfe_rho"),
               ("面积度量", "area_metric"), ("账本 $\\rho$，每帧 4 个子步", "ledger_nsub4"), ("账本 $\\rho$，Crank–Nicolson", "ledger_CN"), ("账本 $\\rho$，边电导 M 矩阵", "ledger_edge_harm")):
    diff_rows.append("%s & %s\\%% & %s\\%%\\\\" % (lab, P(v5["sod"][k]["relL2"], 3), P(v5["blast"][k]["relL2"])))
T = v5["sod"]["rhoD_T07"]
led_rows = []
for nm, lab in (("smooth", "光滑制造映射"), ("cyl", "会聚柱面制造激波"), ("weak", "弱激波（$\\eta=1.5,C_s=0.5$）"), ("slow", "慢激波（$C_s=0.25$）"), ("sod", "Sod"), ("blast", "柱面爆炸波"),
                ("dg", "双环流"), ("sb", "激波–气泡（无粘）"), ("sv", "激波–等熵涡"), ("rm", "单模 RM"), ("sbv", "激波–气泡（粘性）")):
    if nm in v7: old, hyb, auc = v7[nm]["old_v2"], v7[nm]["v3_hybrid"], v7[nm]["indicator_auc_top5"]
    elif nm in ("weak", "slow"): old, hyb, auc = v4[nm]["old_v2"], v4[nm]["v3_hybrid"], v4[nm]["indicator_auc_top5"]
    else: old, hyb, auc = v4[nm]["old_v2"], v4b[nm]["hybrid_tol0.1"], v4[nm]["indicator_auc_top5"]
    led_rows.append("%s & %s & %s & %.2f\\\\" % (lab, pm(old), pm(hyb), auc))
noise_rows = []
for nm, lab in (("smooth", "光滑制造映射"), ("sod", "Sod"), ("blast", "柱面爆炸波"), ("sb", "激波–气泡（无粘）"), ("sv", "激波–等熵涡"), ("rm", "单模 RM"), ("sbv", "激波–气泡（粘性）")):
    k = nm + "_noise0.02h"
    if k in v7: a, b = v7[k]["v3_polygon"], v7[k]["v3_hybrid"]
    else: a, b = v4b[k]["polygon"], v4b[k]["hybrid_tol0.1"]
    noise_rows.append("%s & %s & %s\\\\" % (lab, pm(a), pm(b)))
E1 = v2["E1_planar"]; E4 = v2["E4_shear"]
shear = "、".join("%s\\%%" % P(v["rel_err_med_layer"], 1) for v in E4.values())

part = r"""
\clearpage\part{第六部分　v2：修正、补证与验证}
\noindent\fcolorbox{accent!40}{tint}{\begin{minipage}{0.965\linewidth}\vspace{0.4em}
\textbf{\sffamily 本部分的结论}\quad
(1) 九条代数恒等式全部通过精确符号验证；定理 1 推广到任意多边形，单轴签名证明为充要条件，定理 2 改称运动学恒等式并补出带误差界的离散 Kelvin 推论。
(2) 在制造流映射上：平面激波处账本精确到 $10^{-13}$ 量级；弯曲激波处误差只取决于 $h/R_s$；光滑流中多边形账本是一阶，新增的点态张量账本是二阶。
(3) 同区域环量比较：在同一物质边界上，位置形式环量与 DNS 速度线积分的中位相差为 $0.03\%$–$0.25\%$，四个 DNS 算例都如此；旧的“与 DNS 相差 1.5\%”比较的是不同区域，已废弃。
(4) 扩散的“质量形式准 4–10 倍”实为 """ + N["area_ratio_range"] + r""" 倍，差距来自方程本身，不来自账本；在 $\rho D\propto T^{0.7}$ 下差距依然存在。
(5) 账本 v3（多窗口激波检测、自动推迟上限、张量/混合估计）：弱激波和慢激波由失效变为精确，八个流动算例的中位误差下降约 30\%–90\%，激波处的 95 分位基本不变。
\vspace{0.3em}\end{minipage}}

\section{本轮修正的不一致}
\begin{table}[!htbp]\centering\small\caption{论文与报告中已修正的表述。数字由 \texttt{tools/paper\_numbers.py} 从结果文件计算，\texttt{tools/check\_paper\_numbers.py} 核对：v1 有 34 处问题，v2 全部通过。}
\begin{tabularx}{\linewidth}{@{}l L L@{}}\toprule
位置 & 旧表述 & 现行表述\\\midrule
总摘要 & 密度中位误差 $<1\%$ & 至多 $1.3\%$（双环流 1.3\%，Sod 为机器精度）\\
扩散 & 质量形式比体积形式准 4–10 倍 & """ + N["area_ratio_range"] + r""" 倍（Sod """ + N["area_ratio_sod"] + "、爆炸波 " + N["area_ratio_blast"] + "、激波–气泡 " + N["area_ratio_sb"] + r"""）；误差定义为 RMS 误差除以参照的标准差\\
环量 & 与 DNS 相差 1.5\%（物质带对整个上半区域） & 与同一物质边界上的 DNS 速度线积分比较，见表~\ref{v2:tab:circ}\\
Kelvin & 物质带环量守恒 & 只有宽带近似守恒；窄带在位置形式和线积分中一致下降（""" + N["narrow_band"].replace("\\to", "$\\to$") + r"""），原因是无粘 DNS 在涡片附近的数值耗散\\
参数敏感性 & 锚定周期 3–8 帧对中位误差影响 $<0.03$ 个百分点 & 锚定周期取 3 帧时，爆炸波中位误差 0.58\%$\to$0.77\%、95 分位 32\%$\to$46\%；$\tau$ 的影响 $\le$""" + N["tau_dmed_max"] + r""" 个百分点\\
RM & 增长率 0.0421 & """ + N["rm_rate"] + r"""（由振幅序列在 $t\in[0.6,1.4]$ 上拟合）；Richtmyer 参数与一维激波折射精确解（$\Delta u=0.5112$，$\rho=2.178/6.413$，$A^+=0.493$）相差 $\le0.7\%$\\
T9 估计量 & 两者同时失效“当且仅当” & 改为充分条件，并给出信息极限的数值（第~\ref{v2:sec:mfm} 节）\\\bottomrule
\end{tabularx}\end{table}

\section{证明的补强}
\begin{theorem}[有限步环量与面积变化，多边形形式]\status{已证+符号验证}
记 $u^{\rm m}_e=\frac12(\Delta x_a+\Delta x_b)$，$q^{\rm m}_e=\frac12(q^n_e+q^{n+1}_e)$，$c_e=\overline{u^{\rm m}_e}\,q^{\rm m}_e$（上横线只表示共轭）。对任意由示踪点构成的闭合多边形 $\Pi$，有 $\sum_{\partial\Pi}\im c_e=A^{n+1}_\Pi-A^n_\Pi$；且对 $\Pi$ 的任意三角剖分，$\sum_{\partial\Pi}\re c_e=\sum_t2A^n_t\im p_t$。
\end{theorem}
证明的要点：顶点为 $x^n+s\Delta x$ 的多边形，其鞋带面积是 $s$ 的二次式，所以 $A(1)-A(0)=A'(\tfrac12)$ 精确成立。虚部就是 ALE 方法中的二维离散几何守恒律（Thomas–Lombard 1979；Farhat 等 2001），本工作的新意在于把它与环量配成一个复数，并用作质量账本。

\begin{corollary}[离散 Kelvin 定理]\status{已证}
定理 4.2（位置形式）本身是时间上的分部求和，不含动力学，所以改称“离散运动学环量恒等式”。若示踪点沿光滑流映射运动，且 $a=-\nabla\Psi$（正压、无粘），则对周长为 $L$、最长边为 $h$ 的物质多边形，
$\big|\sum_{t\in R}(\Gamma^{n+1/2}_t-\Gamma^{n-1/2}_t)\big|\le\frac{\Delta t^2L}{12}\big(h^2\|\nabla^2a\|_\infty+\Delta t^2\|D_t^2a\|_\infty\big)+O(\Delta t^6L)$。
\end{corollary}
推论的证明：主项是 $\oint a\cdot dx=0$ 的梯形公式，每条边的误差不超过 $\ell^3\|\nabla^2a\|/12$。

\begin{proposition}[单轴签名是充要条件]\status{已证+符号验证}
对迹为正的 $F$，$\arg p=0$ 当且仅当 $F$ 对称；对对称正定的 $F$，$|\mu|=\tanh(\frac12|\ln J|)$ 当且仅当某个特征值等于 1。两者之差的平方的分子分解为 $-4\lambda_1\lambda_2(\lambda_1-1)(\lambda_1+1)(\lambda_2-1)(\lambda_2+1)$。
\end{proposition}
另有两点。第一，合并块就等于块两端帧之间的直接三点映射。第二，偏离签名量度的是激波曲率和波后流动的不均匀性（弯曲激波上的涡量跃变，Hayes 1957）；在会聚柱面激波上，$|\arg p_B|\approx0.16\,h/R_s$。

\paragraph{账本估计的是什么。}账本给出 $\hat\rho_i=\rho_i(t_c)A^\ast_i(t_c)/A^\ast_i(t)$，目标是 $\rho(X_i,t_c)/J(X_i,t)$。多边形面积是顶点的二次型，所以对任意仿射预测 $\hat x$ 和残差 $e$，精确成立
$A^\ast(\hat x+e)=\det F\,A^\ast(t_a)+\sum_k\frac12e_k\times(\hat x_{k+1}-\hat x_{k-1})+A^\ast(e)$。
由此可得四点：
\begin{itemize}
\item 星作仿射运动时，账本精确。
\item 取 $\delta=\ln[A^\ast(t)/A^\ast(t_a)]-\ln\det F_{\rm LS}$，它恰好是一个增量中仿射运动解释不了的部分，可作为逐点指示量。
\item 噪声律是同一展开取 $e$ 为噪声的特例。
\item 光滑流中，多边形面积比估计的是星面积质心处的 $J$，而不是示踪点处的 $J$，所以账本是一阶。
\end{itemize}
望远相消命题的假设写精确为：平面激波速度恒定，两侧为均匀态。在这些假设下，穿越时刻是初始位置的仿射函数，所以无论激波用了几帧扫过星，面积比都精确。

\paragraph{Dirac 一节的表述。}定理 1、2 与账本只用到与度量无关的 $d_1$，度量只进入 $L_0$ 与 T7；$M_1$ 取 Whitney 质量阵（对角余切 Hodge 星给出同一个 $d_0^\top M_1d_0$，但 $L_1$ 不同）。方差无条件衰减的前提补为 $\rho\kappa\ge0$ 且三角形正定向。

\paragraph{M 矩阵版本。}若改用边电导（相邻三角形电导的调和平均），并把剥离边界边上为负的余切权截断为零，$K$ 就是 M 矩阵，对任何电导场都满足离散极值原理且半正定。注意不截断时它不是半正定的，实测会发散。实际效果是：单元形式的越界量 $\le3\times10^{-8}$，而截断版在狭长的 Sod 区域上误差升到 """ + N["diff_edge_sod"] + r"""\%（单元形式为 """ + N["diff_ledger_sod"] + r"""\%），所以默认仍用单元形式。

\paragraph{T9 的补充。}$\kappa>0$ 时等值线不再是物质线，Nanson 分解只是梯度收支的运动学部分。薄层的标量耗散 $\propto\rho\kappa\lambda^2/J$，因此按积分混合率计算时拉伸项要计两次；RM 中为 $2\times0.054$ 对 $0.80$，压缩仍约占 88\%。

\section{制造流映射上的验证}\label{v2:sec:mfm}
为把方法误差与 DNS 误差分开，我们直接指定解析流映射，使密度、$J$ 与拉伸都有精确值（\texttt{lhd/mfm.py}）。结果见图~\ref{v2:fig:verif}。
\begin{itemize}
\item \textbf{平面激波}（$\theta\in\{0,0.5\}$，$\eta\in\{1.5,3,6\}$，含切向滑移，$C_s=0.5$–$4$）：只要激波被检出，账本误差 $\le""" + N["planar_exact_max"].replace("e-", "\\times10^{-") + r"""}$，合并块单轴到 $""" + N["planar_block_max"].replace("e-", "\\times10^{-") + r"""}$；不推迟重锚时误差最大 """ + N["planar_nodefer_max"] + r"""\%。单步检测会漏掉弱激波和慢激波：$\eta=1.5$、$C_s=0.5$ 时最大误差 """ + N["weak_old_max"] + r"""\%；$\eta=3$、$C_s=0.25$ 时 """ + N["slow_old_max"] + r"""\%。
\item \textbf{会聚柱面激波}（后方为汇流）：三种 $h$ 的误差曲线按 $h/R_s$ 重合，中位误差 """ + N["cyl_med_range"].replace("--", "–") + r"""\%（$h/R_s=0.01$–$0.12$）。
\item \textbf{光滑可压缩旋转映射}：多边形账本观测阶 """ + N["order_polygon"] + r"""，张量账本观测阶 """ + N["order_tensor"] + r"""。
\item \textbf{薄剪切层}：$\delta/h=0.5,1,2,4,8$ 时，链式张量拉伸的相对误差为 """ + shear + r"""，信息极限约为 $\delta\gtrsim2$–$4h$。
\item \textbf{变形关系检验}（Sod）：刚体运动（含随时间变化的转动）、编号置换与缩放下，$\ln J$ 的最大变化为 """ + N["meta_max"].replace("e-", "$\\times10^{-") + r"""}$；时间反演恢复初始密度的中位误差为机器精度，95 分位 """ + N["meta_rev_p95"] + r"""\%。
\end{itemize}
\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{fig_verification.pdf}
\caption{验证（图中文字为英文，与论文 v2 相同）。(a) 平面制造激波的最大密度误差；(b) 会聚柱面激波：账本中位误差与块转角随 $h/R_s$ 的变化；(c) 光滑映射：多边形账本与张量账本；(d) 激波–气泡环量：位置形式、同一边界上的 DNS 线积分、旧的顶点涡量求积；(e) 扩散：电导中用精确、账本、SPH、DTFE 密度，以及面积度量；(f) 薄剪切层的信息极限。}\label{v2:fig:verif}\end{figure}

\section{同区域环量比较}
旧比较把物质带的位置形式环量与 DNS 在另一区域上的涡量积分相比。正确的参照是 DNS 速度沿同一条物质多边形边界的线积分。为此重跑了四个 DNS（激波–等熵涡补存了示踪点速度）。
\begin{table}[!htbp]\centering\small\caption{位置形式环量对同边界 DNS 线积分的相对差（取 $|\Gamma|$ 超过其最大值 20\% 的时刻）；最后一列是旧参照（顶点涡量面积求积）对线积分的相对差。}\label{v2:tab:circ}
\begin{tabularx}{\linewidth}{@{}L r r r@{}}\toprule
算例与区域 & 中位（\%） & 最大（\%） & 旧参照中位（\%）\\\midrule
""" + "\n".join(circ_rows) + r"""
\bottomrule\end{tabularx}\end{table}
在四个 DNS 中，位置形式与线积分的中位相差都在 0.25\% 以内，而旧的面积求积在卷起后偏离可达数十个百分点。这说明位置形式给出的是精确的边界环量，此前的差异来自参照本身。需要说明，示踪点是用同一速度场积分的，所以这项检验确认的是“位置形式环量 = 同一回路上的速度环量”；环量的物理正确性仍依赖 DNS。

\section{扩散：方程与密度估计的分解}
\begin{table}[!htbp]\centering\small\caption{末时刻扩散误差（RMS 误差除以参照标准差）。}
\begin{tabularx}{0.8\linewidth}{@{}L r r@{}}\toprule
变体 & Sod & 爆炸波\\\midrule
""" + "\n".join(diff_rows) + r"""
\bottomrule\end{tabularx}\end{table}
读数如下。
\begin{itemize}
\item 质量度量与面积度量相差 10 倍以上，而电导里换用不同的密度估计只差 1.1–1.5 倍（DTFE 在爆炸波中为 4.9 倍）。所以“4–10 倍”检验的是方程的选择，账本贡献的是让这一选择只凭位置就能实现。
\item 子步与 Crank–Nicolson 对结果影响很小，快照间隔不是误差的主要来源。
\item 改用气体动理学电导 $\rho D\propto T^{0.7}$，$T$ 由位置重建（沿示踪点熵守恒，检出激波时按实测密度比给出 Rankine–Hugoniot 熵跃；$T$ 的中位误差为机器精度，95 分位 """ + N["T_p95"] + r"""\%）。结果：质量度量 """ + N["rhoD_mass"] + r"""\%（用精确 $T$ 为 """ + N["rhoD_exact"] + r"""\%），面积度量 """ + N["rhoD_area"] + r"""\%，差距依然存在。
\end{itemize}

\section{账本 v3：多窗口检测、自动推迟、张量/混合估计}
\paragraph{改动。}
\begin{itemize}
\item \textbf{多窗口激波检测}：一个星要经过多帧才被激波扫过时（$C_s\lesssim1$、弱激波或粘性激波），在长度 $w\le6$ 帧的窗口内，若星面积下降超过 $\max(\tau,3w\,\mathrm{med}_t|R_i|)$，并且该窗口上的最小二乘仿射映射有一个主伸长等于 1（容差 $0.1$，即签名的转动不变形式），就把窗口内各步标为热步。第一版没有签名确认，卷起区被误标，激波–气泡的 95 分位从 29.7\% 升到 39.4\%；加确认后为 30.3\%。
\item \textbf{自动推迟上限}：取热步持续长度 99 分位的 2 倍，限制在 20–50 帧之间。
\item \textbf{张量账本}：对链式最小二乘变形梯度（12 近邻二次拟合，推迟规则相同）取 $\ln\det$。光滑流中是二阶，跨平面激波精确，但偶有离群值。
\item \textbf{混合估计}：张量值与多边形值在 $\ln J$ 上相差不超过 $0.1$ 时用张量值（""" + N["hyb_share"] + r"""\% 的示踪点），否则用多边形值；质量仍由多边形账本给出。
\end{itemize}
\begin{table}[!htbp]\centering\small\caption{末时刻密度误差（中位 / 95 分位）：旧账本与混合估计，以及指示量识别最差 5\% 误差的 AUC。激波–等熵涡取 $t=0.6$，粘性激波–气泡取 $\Delta x=0.005$。}
\begin{tabularx}{\linewidth}{@{}L r r r@{}}\toprule
算例 & 旧账本（v2） & 混合估计（v3） & 指示量 AUC\\\midrule
""" + "\n".join(led_rows) + r"""
\bottomrule\end{tabularx}\end{table}
\begin{table}[!htbp]\centering\small\caption{0.02$h$ 位置噪声下的末时刻密度误差（中位 / 95 分位）。}
\begin{tabularx}{0.85\linewidth}{@{}L r r@{}}\toprule
算例 & 多边形账本 & 混合估计\\\midrule
""" + "\n".join(noise_rows) + r"""
\bottomrule\end{tabularx}\end{table}
\paragraph{读数。}
\begin{itemize}
\item 重跑的三个 DNS 都复现了旧数值：激波–涡 0.34\%/3.4\%，RM 0.73\%/35\%，粘性激波–气泡 0.53\%/8.9\%。
\item 混合估计在所有流动算例中都降低了中位误差。激波–等熵涡的 95 分位也明显下降（涡核内的非仿射误差正是张量账本能纠正的）；激波、接触间断与界面附近的 95 分位基本不变。
\item 噪声下混合估计的中位误差普遍降低约 30\%。
\item 指示量在激波流中能很好地识别最大误差（AUC 0.86–0.97），在光滑流中没有信息，它不是误差上界。
\end{itemize}

\section{仍未解决}
\begin{enumerate}
\item 张量账本有离群值，目前靠混合规则兜底；激波处的 95 分位没有改善。
\item 多窗口检测只在制造激波上验证过，尚未在可分辨的粘性激波（SBLI 类）上检验。
\item T4 的冲击/光滑拆分还没有改在合并块上定义，也没有与 DNS 斜压力矩直接比较。
\item 粘性激波–气泡这次只重跑了 $\Delta x=0.005$；三维与真实 PTV 数据仍未做。
\end{enumerate}
"""
rep(r"\clearpage\part{附录部分", part + "\n" + r"\clearpage\part{附录部分")
# ---------- front matter & summary
rep(r"位置形式的离散 Kelvin 定理严格成立；以示踪点质量为度量的 Dirac 算子把物质账本、环量收支、变密度扩散（$L_0=\delta_1d_0$，T8）与加权动态 Laplacian（T7）统一为同一算子的不同块；",
    r"位置形式的离散环量恒等式严格成立，由它可推出离散 Kelvin 定理；这些恒等式与度量无关，给出物质账本；以账本质量为度量的 Dirac 算子的 Laplacian 块，就是变密度扩散（$L_0=\delta_1d_0$，T8）与加权动态 Laplacian（T7）；")
rep(r"密度中位误差 $<1\%$，比 DTFE、SPH 准 1–2 个数量级；扩散的质量形式比体积形式准 4–10 倍、比 SPH 准 2–5 倍；位置形式环量与 DNS 相差 0.2\%–1.5\%，与 Samtaney–Zabusky 环量沉积模型相差 8\%；",
    r"密度中位误差至多 1.3\%（v3 混合估计为 0.11\%–0.47\%），比 DTFE、SPH 准 1–2 个数量级；扩散的质量形式比体积形式准 " + N["area_ratio_range"] + r" 倍，差距来自方程，在 $\rho D\propto T^{0.7}$ 下依然存在；在同一物质边界上，位置形式环量与 DNS 速度线积分的中位相差 0.03\%–0.25\%，与 Samtaney–Zabusky 模型相差 8\%；全部代数恒等式通过精确符号验证，账本在制造流映射上得到验证；")
rep(r"\textbf{边界}：无粘 DNS 卷起后的界面带不收敛（粘性 DNS 中收敛）；",
    r"\textbf{边界}：多边形账本在光滑流中为一阶（张量账本为二阶）；无粘 DNS 卷起后的界面带不收敛（粘性 DNS 中收敛）；")
rep(r"第四部分是断轨下的逐轨账本；附录部分是后续工作（脊分类）。",
    r"第四部分是断轨下的逐轨账本；第五部分是场图；第六部分（v2）是修正、补证与验证；附录部分是后续工作（脊分类）。")
rep(r"T1 激波检测 & 单步面积台阶；合并块签名 $|\mu_B|=\tanh(\frac12|\ln J_B|)$ & 检出 0.96–1.00；伪转角经合并块降一个数量级",
    r"T1 激波检测 & 单步面积台阶＋多窗口检测（签名确认）；合并块签名（充要） & 检出 0.96–1.00；弱激波和慢激波制造算例由失效变为精确")
rep(r"密度中位 $<1\%$；比 DTFE/SPH 准 1–2 量级；噪声律；自适应锚定；逐轨断轨账本",
    r"密度中位 $\le1.3\%$（混合估计 0.11–0.47\%）；平面激波处精确；弯曲激波误差随 $h/R_s$；比 DTFE/SPH 准 1–2 量级；噪声律；逐轨断轨账本")
rep(r"与 DNS 0.16\%–2.5\%；与 SZ 模型 8\%", r"同边界线积分中位相差 0.03–0.25\%；与 SZ 模型 8\%")
rep(r"质量形式 0.2\%–6\%；体积形式大 4–10 倍", r"质量形式 0.2\%–6\%；体积形式大 " + N["area_ratio_range"] + r" 倍（来自方程）")
rep(r"两者同时失效当且仅当界面既是强剪切面、物质层又薄于 $h$，这是只凭位置的信息极限。",
    r"当界面既是强剪切面、物质层又薄于 $h$ 时，两者都会失效（充分条件；数值见第六部分），这是只凭位置的信息极限。")
rep(r"\item \textbf{度量即测度。}在可压缩流中，以质量而不是面积为度量是可检验的物理选择：扩散误差相差 4–10 倍，组分按计数偏差 2.5 倍。",
    r"\item \textbf{度量即测度。}在可压缩流中，以质量而不是面积为度量是可检验的物理选择：扩散误差相差 " + N["area_ratio_range"] + r" 倍（来自方程，在 $\rho D\propto T^{0.7}$ 下依然存在），组分按计数偏差 2.5 倍。")
rep(r"\item \textbf{边界。}卷起后界面带的结论", r"\item \textbf{验证。}全部代数恒等式通过精确符号验证；账本在制造流映射上平面激波处精确、弯曲激波处误差随 $h/R_s$、光滑流中为一阶（张量账本为二阶）；环量在同一物质边界上与 DNS 线积分一致到 0.25\% 以内。\item \textbf{边界。}卷起后界面带的结论")
rep(r"（可运行演示：有限步恒等式、Sod 账本与激波签名、质量形式扩散、爆炸波掺混归因、动态 Laplacian、断轨账本）。",
    r"（可运行演示：有限步恒等式、Sod 账本与激波签名、质量形式扩散、爆炸波掺混归因、动态 Laplacian、断轨账本）。v2 新增：\texttt{lhd/mfm.py}（制造流映射）、\texttt{lhd/ledger\_ub.py}（指示量、多窗口检测、\texttt{ledger\_v3}、张量/混合账本、热力学重建）、\texttt{tests/}（精确符号检验与变形关系检验）、\texttt{tools/}（数字单一来源与核对）、\texttt{scripts/v1\_*}–\texttt{v7\_*}（验证脚本）。")
rep(r"{\color{quiet}2026 年 10 月}", r"{\color{quiet}2026 年 10 月（v2：修正与验证版）}")
open(f, "w").write(s); print("FAILED:", fails if fails else "none")
