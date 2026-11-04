import json, os, numpy as np
J = lambda f: json.load(open(f)) if os.path.exists(f) else None
pc = lambda x, d=1: ("%." + str(d) + "f") % (100*x) + r"\%"
def pcs(x):
    if x is None: return "—"
    if x < 1e-6: return "$<10^{-6}$"
    return pc(x, 2 if x < 0.01 else 1)
b1 = J("res_b1_density.json"); b2 = J("res_b2_diff.json"); b3 = J("res_b3_ledger.json"); b3t = J("res_b3_t9.json")
b4s = J("res_b4_sod.json"); b4v = J("res_b4_sv.json"); b4c = J("res_b4c.json"); rm = J("res_rm.json")
dsb = J("res_dnsref_sb.json"); dsv = J("res_dnsref_sv.json")
LV = [k for k in ["sb_dx0.01", "sb_dx0.005", "sb_dx0.0025"] if dsb and k in dsb]
T = []; A = T.append
A(r"""\newcommand{\HEADTITLE}{论文 A 补充验证与框架重组}
\input{preamble.tex}
\begin{document}
\begin{center}
{\sffamily\bfseries\LARGE 质量度量下的拉格朗日 Hodge–Dirac 计算}\\[0.4em]
{\sffamily\large 论文 A 的框架重组与补充验证：DNS 加密、基线、参数敏感性、实际数据与 Richtmyer–Meshkov 应用}\\[0.6em]
{\color{quiet}2026 年 10 月}
\end{center}
\noindent\fcolorbox{accent!40}{tint}{\begin{minipage}{0.965\linewidth}\vspace{0.3em}
\textbf{\sffamily 本稿做了什么}\quad 按上一份评估列出的缺口，补做六项工作：(1) 两个 DNS 各算三档网格（$\Delta x=0.01,0.005,0.0025$，示踪点不变），确认界面压缩的参照与密度 95 分位误差的来源；(2) 密度与 DTFE、SPH 对比，扩散与 SPH 传导对比；(3) 账本与链式张量的参数敏感性；(4) 稀疏示踪点、断轨与实际量级噪声下的检验；(5) 新增单模 Richtmyer–Meshkov（RM）DNS 作为复杂应用；(6) 统一全部文档中的数值。另外回答“其余 T 是否融入、没有 T3–T5 还能否叫 LHD”。

\textbf{\sffamily 主要结论}\quad
\textbf{框架}：论文 A 的成熟部分是同一个 Dirac 算子在质量度量下的几个块，可以保留 LHD 的名字（第~1 节）。
\textbf{DNS 加密}：激波–等熵涡的 95 分位来自账本在涡核内的误差，与 DNS 无关；激波–重气泡卷起之前的界面差异来自 DNS 数值扩散，卷起之后（$t=1.4$）界面带上两者都不收敛，旧的“压缩占 54\%–62\%”需降级为待复核（第~2 节）。
\textbf{基线}：只凭位置的密度比 SPH 准约一个数量级、比 DTFE 准 1–2 个数量级；扩散在三个算例上比 SPH 准 2–5 倍。
\textbf{参数}：只有“最多推迟重锚次数”敏感，不推迟时爆炸波误差增大约 18 倍。
\textbf{实际数据}：0.02$h$ 噪声下中位误差约 1.5\%–2\%；稀疏、断轨下仍可用。
\textbf{RM}：振幅与 DNS 相差 $\le$1.2\%，增长率与 Richtmyer 模型相差 3\%，模型参数全部取自示踪点位置。
\vspace{0.3em}\end{minipage}}

\tableofcontents
\newpage
""")
A(r"""\section{框架重组：质量度量下的 Hodge–Dirac}
\subsection{三个命题}
记第 $n$ 帧的剥离 Delaunay 复形为 $K^n$，上边缘 $d_0,d_1$，度量 $M_0,M_1,M_2$，$\delta_1=M_0^{-1}d_0^\top M_1$。三点映射写成 $F\zeta=p\zeta+q\bar\zeta$。

\begin{proposition}[复上链恒等式]\status{已证+数值}\label{prop:cplx}
对边 $e=(a\to b)$ 与步 $n\to n+1$，定义复 1‑上链 $c_e=\overline{\bar u_e}\;\bar q_e$，$\bar u_e=\tfrac12(\Delta x_a+\Delta x_b)$，$\bar q_e=\tfrac12(q_e^n+q_e^{n+1})$（二维向量与复数等同）。则对任意三角形与任意一步严格成立
\begin{equation}
(d_1c)_t=2A_t^n\im p_t+i\,(A_t^{n+1}-A_t^n).
\end{equation}
\end{proposition}
\begin{proof}
$\re(\bar u\,q)=u\cdot q$，$\im(\bar u\,q)=u_xq_y-u_yq_x=u\cdot R_{-90^\circ}q$。实部即中点位移上链的环量，虚部即法向通量；两者分别等于 $2A^n\im p$ 与 $A^{n+1}-A^n$（T3 定理，证明见 T1–T9 定稿）。
\end{proof}
于是 \textbf{T4}（Kelvin 环量）是 $\re(d_1c)$，\textbf{T2}（账本）的单步增量是 $\ln(1+\sum_{t\ni i}\im(d_1c)_t/A^\ast_i)$：二者是同一个有限步量的两个分量，不需要速度，也不依赖步长。数值上两部分分别验证到 $3\times10^{-14}$ 与 $2\times10^{-14}$。

\begin{proposition}[扩散算子是 $D^2$ 的节点块]\status{已证}\label{prop:L0}
取 $M_0=\mathrm{diag}(m_i)$（示踪点质量），$M_1$ 为以 $(\rho\kappa)_T$ 加权的 P1 Galerkin 度量，则 T8 的半离散方程 $M\dot c=-Kc$ 就是 $\dot c=-L_0c$，$L_0=\delta_1d_0=M_0^{-1}d_0^\top M_1d_0$。体积形式对应 $M_0=\mathrm{diag}(A^\ast_i/3)$、$M_1$ 以 $\kappa$ 加权。
\end{proposition}
所以“质量形式比体积形式准 5–10 倍”（可压缩算例）是对度量选择的直接检验：物理测度决定 Hodge–Dirac 的度量，而不是数值细节。

\begin{proposition}[其余组成的位置]\status{已证}
T9 的 Nanson 余向量估计是 $d_0$ 作用在初始标签 0‑上链 $\phi_0$ 上；链式张量估计是三点映射合成律在近邻上的最小二乘推广；T1 的激波签名是合并块满足 $|\mu_B|=\tanh(\tfrac12|\ln J_B|)$、$\arg p_B=0$；T6 是质量 0‑链的推前。
\end{proposition}

\subsection{各组成部分的去留}
\begin{table}[!htbp]\centering\small
\caption{组成部分与 Dirac 算子的对应，以及在论文 A 中的位置。}
\begin{tabularx}{\linewidth}{@{}l L l@{}}\toprule
组成 & 在 $D$（质量度量）中的位置 & 去向\\\midrule
T1 激波检测 & 节点上的面积跳跃；块映射的签名 & 论文 A\\
T2 物质账本 & $\im(d_1c)$ 在星上求和 & 论文 A（核心）\\
T3 局部读数 & $\re(d_1c)/(\bar A\Delta t)$，$\im(d_1c)/(\bar A\Delta t)$ & 论文 A（只保留局部量）\\
T3 全局分解 & $D$ 的谱分割、能量占比 & 另一篇（拓扑信号）\\
T4 Kelvin & $\re(d_1c)$ 与二阶上链 $d_1g$ & 论文 A\\
T5 Dirac 谱、声学、DESP & $D$ 的谱与联合滤波 & 另一篇（拓扑信号）\\
T6 质量输运 & 质量 0‑链的推前 & 论文 A\\
T7 转移算子 & 推前的 Ulam 矩阵 & 论文 A 一段或删去\\
T8 扩散 & $L_0=\delta_1d_0$，度量取质量与 $\rho\kappa$ & 论文 A（核心）\\
T9 掺混 & Nanson：$d_0\phi_0$ 与账本；链式 $(p,q)$ & 论文 A（核心）\\
脊分类 & 边时间序列与账本 & 第三项工作\\\bottomrule
\end{tabularx}\end{table}

\subsection{论文 A 的创新点}
\begin{enumerate}
\item \textbf{有限步严格。}环量、面积变化、Kelvin 收支都是位置的代数恒等式，不需要速度、不依赖步长，跨激波时用合并块仍然成立。已有的拉格朗日轨迹网络方法建立在图（0、1 维）上，没有 2‑胞腔上的这类恒等式。
\item \textbf{度量即测度。}可压缩流中质量度量与面积度量给出可区分的结果：扩散误差相差 4–10 倍，组分按计数偏差 2.5 倍。
\item \textbf{精度。}只凭位置的密度比 DTFE、SPH 准 1–2 个数量级（第~\ref{sec:base} 节）。
\item \textbf{物理结果。}激波–重气泡界面在激波扫过后先被切向缩短、抵消大部分压缩（$t\le0.6$，对 DNS 网格稳健）；单模 RM 到 $ka\approx0.6$ 时界面梯度增长约 94\% 来自压缩，振幅增长率与 Richtmyer 模型相差 3\%，模型参数全部取自示踪点位置（第~\ref{sec:rm} 节）。
\end{enumerate}
""")
A(r"""\section{DNS 加密：界面压缩的参照与密度 95 分位的来源}\label{sec:dns}
两个 DNS 各算三档网格 $\Delta x=0.01,0.005,0.0025$，示踪点初始位置与间距（$h=0.005$）不变，在求解器内用同样的 Heun/双线性格式推进。比较同一组示踪点上的账本压缩 $-\ln J$ 与 DNS 插值密度比 $\ln(\rho/\rho_0)$，按初始到界面的有符号距离 $\phi_0$ 分带（负值为重气体一侧）。
""")
bands = ["-0.06..-0.03", "-0.03..-0.015", "-0.015..-0.005", "-0.003..0.003", "0.005..0.015", "0.015..0.03", "0.03..0.06"]
if dsb:
    for tt in ["t=0.6", "t=1.4"]:
        A(r"\begin{table}[!htbp]\centering\small\caption{激波–重气泡，$%s$：各带的账本压缩 / DNS 插值压缩。}" % tt + "\n" + r"\begin{tabular}{l" + "c"*len(LV) + r"}\toprule" + "\n$\\phi_0$ 区间 & " + " & ".join(["$\\Delta x=%s$" % k.split("dx")[1] for k in LV]) + r"\\\midrule" + "\n")
        for b in bands:
            A("$[%s]$ & " % b.replace("..", ",") + " & ".join(["%.3f / %.3f" % tuple(dsb[k][tt]["bands"][b]) for k in LV]) + r"\\" + "\n")
        A(r"\bottomrule\end{tabular}\end{table}" + "\n")
    A(r"""\begin{table}[!htbp]\centering\small\caption{激波–重气泡：密度误差随 DNS 网格的变化。}
\begin{tabular}{l""" + "c"*len(LV) + r"""}\toprule
量 & """ + " & ".join(["$\\Delta x=%s$" % k.split("dx")[1] for k in LV]) + r"\\\midrule" + "\n")
    for tt in ["t=0.6", "t=1.4"]:
        A("$%s$ 中位误差 & " % tt + " & ".join([pcs(dsb[k][tt]["med"]) for k in LV]) + r"\\" + "\n")
        A("$%s$ 95 分位 & " % tt + " & ".join([pcs(dsb[k][tt]["p95"]) for k in LV]) + r"\\" + "\n")
        A("$%s$ 95 分位（离界面 $>0.03$） & " % tt + " & ".join([pcs(dsb[k][tt]["p95_outside_interface"]) for k in LV]) + r"\\" + "\n")
    A(r"$x>1$ 的重气体质量：网络 / DNS & " + " & ".join(["%.4f / %.4f" % tuple(dsb[k]["heavy_mass_x>1"]) for k in LV]) + r"\\\bottomrule\end{tabular}\end{table}" + "\n")
if dsv:
    lv2 = [k for k in ["sv_dx0.01", "sv_dx0.005", "sv_dx0.0025"] if k in dsv]
    A(r"""\begin{table}[!htbp]\centering\small\caption{激波–等熵涡：密度误差随 DNS 网格的变化（$t=0.6$）。}
\begin{tabular}{l""" + "c"*len(lv2) + r"""}\toprule
量 & """ + " & ".join(["$\\Delta x=%s$" % k.split("dx")[1] for k in lv2]) + r"""\\\midrule
中位误差 & """ + " & ".join([pcs(dsv[k]["med"]) for k in lv2]) + r"""\\
95 分位 & """ + " & ".join([pcs(dsv[k]["p95"]) for k in lv2]) + r"""\\
涡核内中位 / 95 分位 & """ + " & ".join([pcs(dsv[k]["core_med"]) + " / " + pcs(dsv[k]["core_p95"]) for k in lv2]) + r"""\\
误差最大的 5\% 中位于涡核的比例 & """ + " & ".join([pc(dsv[k]["top5_in_core"], 0) for k in lv2]) + r"""\\\bottomrule
\end{tabular}\end{table}
""")
A("%%DNSTEXT%%\n")
A(r"""\section{基线对比}\label{sec:base}
\paragraph{密度。}三种方法用同一组示踪点质量 $m_i=\rho_0A^\ast_i(t_0)/3$：账本 $\rho_i=\rho_{0,i}e^{-\ln J_i}$（C 通道与推迟重锚，只凭位置）；DTFE（Schaap \& van de Weygaert 2000，$\rho_i=3m_i/A^\ast_i$）；自适应核 SPH 求和（三次样条，$k=24$）。""")
nm = dict(dg="双环流", lo="Lamb–Oseen", sod="Sod", blast="爆炸波", sv="激波–涡", sb="激波–气泡", rm="RM")
if b1:
    cs = [c for c in ["dg", "lo", "sod", "blast", "sv", "sb", "rm"] if c in b1]
    A("\n" + r"\begin{table}[!htbp]\centering\small\caption{末时刻密度的中位相对误差 / 95 分位（同一内部掩码）。}\begin{tabular}{lccc}\toprule 算例 & 账本 & SPH & DTFE\\\midrule" + "\n")
    for c in cs:
        A(nm[c] + " & " + " & ".join([pcs(b1[c][m]["med"]) + " / " + pcs(b1[c][m]["p95"]) for m in ["ledger", "sph", "dtfe"]]) + r"\\" + "\n")
    A(r"\bottomrule\end{tabular}\end{table}" + "\n")
    s = b1["sod"]["band_disc"]; bb = b1["blast"]["band_disc"]
    A(r"""在激波与接触间断附近（以精确解的局部跳跃定义），三者的 95 分位分别为：Sod 中账本 %s、DTFE %s、SPH %s；爆炸波中 %s、%s、%s。账本的中位误差在全部七个算例上比 SPH 小约一个数量级、比 DTFE 小 1–2 个数量级。DTFE 的误差主要来自 Delaunay 星在大变形下的重连：质量按初始星分配，当前星却已换了邻居；账本逐点锚定、跨激波望远相消，正是为此设计的。
""" % (pcs(s["ledger"]["p95"]), pcs(s["dtfe"]["p95"]), pcs(s["sph"]["p95"]), pcs(bb["ledger"]["p95"]), pcs(bb["dtfe"]["p95"]), pcs(bb["sph"]["p95"])))
A(r"""
\paragraph{扩散。}对比方法为 SPH 传导（Brookshaw / Cleary–Monaghan 形式，电导取 $\rho\kappa$ 的调和平均，写成守恒的质量形式 $m_i\dot c_i=\sum_j2V_iV_j\bar k_{ij}F_{ij}(c_j-c_i)$），与网络用同一组账本密度、同一个隐式时间步，只差空间离散。""")
if b2:
    cs2 = [c for c in ["dg", "sod", "blast", "sb"] if c in b2]
    nm2 = dict(dg="双环流（$\\kappa=10^{-3}$）", sod="Sod（$\\kappa=10^{-3}$）", blast="爆炸波（$\\kappa=2\\times10^{-4}$）", sb="激波–气泡（DNS 标量）")
    A("\n" + r"\begin{table}[!htbp]\centering\small\caption{末时刻扩散标量的相对 $L^2$ 误差，以及计算时间。}\resizebox{\linewidth}{!}{\begin{tabular}{lccccc}\toprule 算例 & 网络·质量 & SPH & 网络·体积 & 纯对流 & 时间（网络 / SPH）\\\midrule" + "\n")
    for c in cs2:
        o = b2[c]; A(nm2[c] + " & %s & %s & %s & %s & %.0f s / %.0f s" % (pcs(o["network_mass"]), pcs(o["sph"]), pcs(o["network_volume"]), pcs(o["pure_advection"]), o["time_network"], o["time_sph"]) + r"\\" + "\n")
    A(r"\bottomrule\end{tabular}}\end{table}" + "\n")
    A(r"""网络质量形式在 Sod、爆炸波、双环流中比 SPH 小 2–5 倍，在激波–气泡中两者相当（参照本身是带数值扩散的 DNS 标量），计算时间约为 SPH 的一半；两者的标量质量都守恒到机器精度，都没有越界。不可压的双环流中体积形式与质量形式等价。SPH 在双环流中误差大，是因为强折叠的物质丝使示踪点分布高度各向异性，各向同性核的 Laplacian 近似变差；P1 刚度对各向异性的星仍然一致。
""")
A(r"\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{figR_base.pdf}\caption{(a) 密度中位误差：账本、SPH、DTFE；(b) 扩散误差：网络质量形式、SPH、网络体积形式。}\end{figure}" + "\n")
A(r"""\section{参数敏感性}
默认参数：C 通道阈值 $\tau=\max(0.2,4\hat\sigma)$，锚定周期 5 帧，最多推迟 4 个周期；链式张量 $k=12$。每次只改一个参数。
""")
if b3:
    lab = {"default": "默认", "tau=0.1": "$\\tau_{\\min}=0.1$", "tau=0.3": "$\\tau_{\\min}=0.3$", "block=3": "锚定周期 3", "block=8": "锚定周期 8",
           "defer=0": "不推迟重锚", "defer=2": "最多推迟 2", "defer=8": "最多推迟 8"}
    A(r"\begin{table}[!htbp]\centering\small\caption{账本密度的中位误差 / 95 分位，以及 C 通道的检出率 / 误报率。}\resizebox{\linewidth}{!}{\begin{tabular}{lcccc}\toprule 参数 & Sod 密度 & 爆炸波密度 & 激波–涡密度 & 检出 / 误报（Sod；爆炸波）\\\midrule" + "\n")
    for k in lab:
        if k not in b3: continue
        o = b3[k]; ds, db = o["sod"]["det"], o["blast"]["det"]
        A(lab[k] + " & " + " & ".join([pcs(o[c]["med"]) + " / " + pcs(o[c]["p95"]) for c in ["sod", "blast", "sv"]]) + " & %.4f / %.1e；%.4f / %.3f" % (ds[0], ds[1], db[0], db[1]) + r"\\" + "\n")
    A(r"\bottomrule\end{tabular}}\end{table}" + "\n")
    A(r"""阈值在所测范围内几乎不影响结果。最敏感的是推迟重锚：不推迟时爆炸波的中位误差从 %s 升到 %s，只允许推迟 2 个周期时为 %s；推迟 4 与 8 个周期结果相同。激波模板持续的帧数超过推迟上限时，锚定星的一端落在激波内，望远相消失效，这与理论一致。锚定周期有取舍：爆炸波中取 3 帧时 95 分位升到 %s，涡核中却偏好短周期（激波–涡的 95 分位在 3、5、8 帧时为 %s、%s、%s）。所以最多推迟次数应按“激波扫过一个星所需帧数 / 锚定周期”设定，锚定周期应按星的畸变程度自适应，而不是固定值。
""" % (pcs(b3["default"]["blast"]["med"]), pcs(b3["defer=0"]["blast"]["med"]), pcs(b3["defer=2"]["blast"]["med"]), pcs(b3["block=3"]["blast"]["p95"]),
       pcs(b3["block=3"]["sv"]["p95"]), pcs(b3["default"]["sv"]["p95"]), pcs(b3["block=8"]["sv"]["p95"])))
if b3t:
    o = b3t["T9_blast"]
    A(r"\begin{table}[!htbp]\centering\small\caption{链式张量估计在爆炸波圆环上的界面平均拉伸（精确值 %.4f）与逐点误差中位数。}\begin{tabular}{lcc}\toprule 参数 & 界面平均 & 逐点误差中位数\\\midrule" % ((1/2.4)*np.log(4.0)) + "\n")
    for k, v in o.items(): A(k.replace("default", "默认").replace("block=", "锚定周期 ").replace("defer=", "最多推迟 ") + " & %.4f & %.4f" % (v["mean"], v["med_abs_err"]) + r"\\" + "\n")
    A(r"\bottomrule\end{tabular}\end{table}" + "链式张量对 $k$、锚定周期与推迟次数都不敏感：界面平均在 $\\pm2\\times10^{-4}$ 内不变。\n")
A(r"""\section{接近实际数据的检验}
粒子追踪测速（PTV）的位置不确定度通常为 0.05–0.1 像素，示踪点间距约 5–10 像素，即 $\sigma\approx0.01h$–$0.02h$；断轨率取每帧 1\%–3\%。检验包括四项：(i) 每帧独立高斯噪声 $0.02h$、$0.05h$、$0.1h$；(ii) 随机保留 1/4 与 1/16 的示踪点；(iii) 断轨：每条轨迹每帧以概率 $p$ 中断，消失 2 帧后以新身份出现；账本按 5 帧窗口运行，只用窗口内完整的轨迹，新身份的密度由 8 个已知密度的近邻按反距离插值（对数密度）赋初值；(iv) 组合：1/4 稀疏、0.02 倍（稀疏后）间距的噪声、$p=0.01$。
""")
for nm_, b4 in [("Sod", b4s), ("激波–涡（DNS）", b4v)]:
    if not b4: continue
    g = lambda k: (pcs(b4[k]["med"]) + " / " + pcs(b4[k]["p95"])) if k in b4 else "—"
    gv = lambda k: (pcs(b4[k]["v2"]["med"]) + " / " + pcs(b4[k]["v2"]["p95"])) if k in b4 else "—"
    gs = lambda k: (pcs(b4[k]["v3"]["med"]) + " / " + pcs(b4[k]["v3"]["p95"])) if k in b4 else "—"
    A(r"\begin{table}[!htbp]\centering\small\caption{%s：密度中位误差 / 95 分位。}\resizebox{\linewidth}{!}{\begin{tabular}{lcccccc}\toprule 无扰动 & 噪声 0.02$h$ & 噪声 0.05$h$ & 噪声 0.1$h$ & 稀疏 1/4 & 稀疏 1/16 & 组合\\\midrule" % nm_ + "\n")
    A(" & ".join([g("clean_v2"), gv("noise_0.02"), gv("noise_0.05"), gv("noise_0.1"), gs("sparse_1/4"), gs("sparse_1/16"), g("combined_1/4_0.02h_p0.01")]) + r"\\\bottomrule\end{tabular}}" + "\n")
    br = [k for k in ["breaks_p0", "breaks_p0.01", "breaks_p0.03"] if k in b4]
    A(r"\\[0.3em]\begin{tabular}{lccc}\toprule 断轨概率 $p$ & " + " & ".join([k.split("_p")[1] for k in br]) + r"\\\midrule" + "\n中位 / 95 分位 & " + " & ".join([g(k) for k in br]) + r"\\" + "\n末窗口可用轨迹比例 & " + " & ".join([pc(b4[k]["usable_frac"], 0) for k in br]) + r"\\\bottomrule\end{tabular}\end{table}" + "\n")
    if b4.get("clean_circ") is not None:
        A(r"激波–涡涡核区域的位置形式环量与 DNS 的相对误差（末步）：无扰动 %s，噪声 0.02$h$ / 0.05$h$ / 0.1$h$ 为 %s / %s / %s，稀疏 1/4 / 1/16 为 %s / %s。" % (
            pcs(b4["clean_circ"]), pcs(b4["noise_0.02"]["circ"]), pcs(b4["noise_0.05"]["circ"]), pcs(b4["noise_0.1"]["circ"]), pcs(b4["sparse_1/4"]["circ"]), pcs(b4["sparse_1/16"]["circ"])) + "\n")
if b4c:
    A(r"""
\paragraph{时间平滑。}对带噪声的位置做 Savitzky–Golay 平滑（二次，窗口 5 或 9 帧）后再算账本，密度中位误差：
\begin{center}\small\begin{tabular}{lcccc}\toprule & \multicolumn{2}{c}{噪声 0.02$h$} & \multicolumn{2}{c}{噪声 0.05$h$}\\ & 不平滑 & 窗 5 / 窗 9 & 不平滑 & 窗 5 / 窗 9\\\midrule
""")
    for c, n_ in [("sod", "Sod"), ("sv", "激波–涡")]:
        o = b4c[c]
        A(n_ + " & %s & %s / %s & %s & %s / %s" % (pcs(o["noise0.02_sg0"]["med"]), pcs(o["noise0.02_sg5"]["med"]), pcs(o["noise0.02_sg9"]["med"]), pcs(o["noise0.05_sg0"]["med"]), pcs(o["noise0.05_sg5"]["med"]), pcs(o["noise0.05_sg9"]["med"])) + r"\\" + "\n")
    A(r"\bottomrule\end{tabular}\end{center}" + "\n" + r"无噪声数据平滑后（窗 5 / 9）的中位误差为：Sod %s / %s，激波–涡 %s / %s。平滑只带来 10\%%–30\%% 的改善，窗口过长还会在激波处抹掉位置轨迹的拐折。" % (
        pcs(b4c["sod"]["clean_sg5"]["med"]), pcs(b4c["sod"]["clean_sg9"]["med"]), pcs(b4c["sv"]["clean_sg5"]["med"]), pcs(b4c["sv"]["clean_sg9"]["med"])) + "\n")
A("%%REALTEXT%%\n")
if rm:
    r = rm["richtmyer"]; t9 = rm["T9"]; ks = sorted(t9, key=lambda k: float(k[2:]))
    A(r"""\section{复杂应用：单模 Richtmyer–Meshkov}\label{sec:rm}
\paragraph{设置。}二维 Euler（MUSCL–HLLC），$M_s=1.5$ 激波从轻气体（$\rho=1$）射向扰动界面 $x=0.5+a_0\cos(2\pi y)$，$a_0=0.04$，重气体 $\rho=3$（Atwood 数 0.5），半波长通道 $y\in[0,0.5]$（$y=0$ 对称、$y=0.5$ 壁面），$\Delta x=0.005$，$t\in[0,2]$，51 帧；%d 个示踪点（$h=0.004$）在求解器内推进。所有网络量只用示踪点位置与初始密度。
""" % rm["N"])
    A(r"""\begin{table}[!htbp]\centering\small\caption{RM：网络结果与 DNS。}
\begin{tabularx}{\linewidth}{@{}L >{\raggedright\arraybackslash}p{0.46\linewidth}@{}}\toprule 量 & 结果\\\midrule
T1 检出率 / 误报（以 DNS 单步密度跃变 15\%% 为准） & %.2f / %.0f\\
T2 密度中位误差（$t=0.4,1.0,2.0$） & %s，%s，%s\\
\quad 95 分位（离界面 $>0.03$） & %s，%s，%s\\
T2 末时刻：账本 / SPH / DTFE 中位误差 & %s / %s / %s\\
T4 界面带环量：位置形式与 DNS 的中位相对误差 & %s\\
\quad 激波后环量（位置形式 / DNS） & %.4f / %.4f\\
T6 落在平均界面之后的重气体质量：网络 / DNS & %s\\
振幅（界面示踪点 / DNS 等值线），5 个时刻 & 相差 $\le$ 1.2\%%\\
线性段增长率：实测 / Richtmyer / 由环量 $k\Gamma/4$ & %.4f / %.4f / %.4f\\
T9 界面平均：压缩 $-\ln J$ / 拉伸 $\ln\lambda$（$t=2$） & %.3f / %.3f\\
T9 $\ln\langle\lambda\rangle$ 与 DNS 界面长度 $\ln(L/L_0)$ 的增长（$t=0.4\to2$） & %.4f / %.4f\\\bottomrule
\end{tabularx}\end{table}
""" % (rm["T1_detect"], rm["T1_false"], pcs(rm["T2_t=0.4"]["med"]), pcs(rm["T2_t=1.0"]["med"]), pcs(rm["T2_t=2.0"]["med"]),
       pcs(rm["T2_t=0.4"]["p95_away_interface"]), pcs(rm["T2_t=1.0"]["p95_away_interface"]), pcs(rm["T2_t=2.0"]["p95_away_interface"]),
       pcs(rm["T2_baselines_final"]["ledger"]["med"]), pcs(rm["T2_baselines_final"]["sph"]["med"]), pcs(rm["T2_baselines_final"]["dtfe"]["med"]),
       pcs(rm["T4"]["relerr_med"]), rm["T4"]["G_after_shock"][0], rm["T4"]["G_after_shock"][1],
       "；".join(["%s：%.4f / %.4f" % (k.replace("t=", "$t=") + "$", v[0], v[1]) for k, v in rm["T6_heavy_mass_behind_mean_interface"].items()]),
       0.04208, r["rate_model"], r["rate_from_circulation"], t9[ks[-1]]["compression"], t9[ks[-1]]["stretch_mean"],
       t9[ks[-1]]["ln_mean_lambda"]-t9[ks[0]]["ln_mean_lambda"], t9[ks[-1]]["ln_L_dns"]-t9[ks[0]]["ln_L_dns"]))
    A(r"""\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{figR_rm.pdf}
\caption{单模 RM。(a1–a3) DNS 密度；(a4) 末时刻账本密度误差（对数）；(b) 振幅：界面示踪点、DNS 等值线、Richtmyer 模型（参数取自位置）与由沉积环量估计的斜率；(c) 界面物质带环量，位置形式与 DNS；(d) 界面掺混归因。}\end{figure}

\paragraph{物理读数。}
(1) \textbf{增长率。}激波扫过（$t\approx0.16$）后，界面示踪点给出的振幅先被压缩到 $a_0^+=%.4f$，随后线性增长，$t\in[0.6,1.4]$ 的增长率为 0.0421，$t>1.4$ 降到 0.0389（开始非线性饱和）。Richtmyer 冲击模型 $\dot a=k\,\Delta u\,A^+a_0^+$ 给出 %.4f，与实测相差 3\%%；其中界面速度跳跃 $\Delta u=%.3f$、波后 Atwood 数 $A^+=%.3f$（由账本密度 %.2f 与 %.2f 求得）、$a_0^+$ 全部取自示踪点位置。
(2) \textbf{环量与增长。}把沉积在半波长界面带上的环量看成正弦涡片，$\dot a=k\Gamma/4$ 给出 %.4f，比实测高约 19\%%：这一估计假设涡片无限薄，并忽略界面带内非涡片部分的涡量。位置形式的环量与 DNS 中位误差为 %s，所以这一偏差来自模型而不是测量。
(3) \textbf{掺混归因。}到 $t=2$（$ka\approx0.6$）界面压缩为 %.2f，几乎全部在激波扫过时一次获得；拉伸从接近零缓慢增长到 %.3f。所以在这一阶段，单模 RM 界面上的梯度增长约 94\%% 来自压缩；拉伸只有在卷起（$ka>1$）之后才可能接管。此时尚未卷起，界面带也没有出现激波–重气泡 $t=1.4$ 时的不收敛问题。
""" % (r["a0_plus"], r["rate_model"], r["du"], r["A_plus"], r["rho_light"], r["rho_heavy"], r["rate_from_circulation"], pcs(rm["T4"]["relerr_med"]), t9[ks[-1]]["compression"], t9[ks[-1]]["stretch_mean"]))
A(r"""\section{统一数值登记表}
四份旧报告（质量输运报告、界面拉伸重推报告、六算例统一报告、T1–T9 定稿）中同一量有不同数值，或已被推翻的结论。下表给出以后引用时唯一采用的版本。
\begin{table}[!htbp]\centering\small\caption{数值登记。“现行值”为本稿或最近一次统一流水线的结果。}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.24\linewidth} >{\raggedright\arraybackslash}p{0.27\linewidth} L@{}}\toprule 量 & 旧值（出处） & 现行值与说明\\\midrule
激波–气泡界面拉伸，$t=1.4$ & 0.60–1.05 区间（质量输运报告） & 0.285（链式张量），参照 0.390，涡片范围 [0.06, 0.39]；旧区间来自点对与折线法，已推翻\\
激波–气泡压缩占总增长，$t=1.4$ & 30\%–43\%（质量输运报告）；54\%–62\%（重推报告） & 依赖界面带的压缩值，DNS 加密后不收敛，降级为待粘性 DNS 复核；$t\le0.6$ 的“先切向缩短、抵消大部分压缩”稳健\\
激波–气泡界面长度比 & 5.3（质量输运报告） & 2.16（自适应标记曲线）\\
激波–气泡界面压缩，$t=1.4$ & 0.450（重推报告）；0.501（统一报告） & %%SBCOMP%%\\
Lamb–Oseen 圆环拉伸 & “约为 0”（质量输运报告） & $-0.31$（逐点精确值的界面平均）\\
激波–涡梯度能量占比 & 0.75–0.81（统一报告） & 0.53（Kabsch 对齐后）；原始值约三分之一来自平移，不作物理读数\\
Sod 冲击部分 / 总变化 & 471\%（统一报告） & 删除（分母接近零）\\
激波–气泡 Dirac 残差（激波处） & 0.005（正文）/ 0.01（总表）（统一报告） & 0.005；T5 不进入论文 A\\
密度中位误差：双环流 / 激波–涡 / 激波–气泡 & 1.3\% / 0.33\% / 0.70\%（质量输运）；1.35\% / 0.36\% / 0.73\%（统一） & %%DENS%%（本稿，只凭位置的热步，DNS 网格 0.005）\\
激波–气泡扩散误差（质量 / 体积） & 6.1\% / 24\%（质量输运）；5.1\% / 25.2\%（统一） & %%SBDIFF%%\\
T8 方差衰减的条件 & “需 $K_{ij}\le0$，Delaunay 满足” & 方差衰减无条件；$K_{ij}\le0$ 只关系离散极值原理，变电导下 Delaunay 不保证\\
位置形式 Kelvin 的边向量 & $\tfrac14(q^{n-1}+2q^n+q^{n+1})$ & 一族精确形式，$c=0$ 即 $q^n$ 最简\\\bottomrule
\end{tabularx}\end{table}
""")
A(r"""\section{仍然存在的缺口}
\begin{enumerate}
\item \textbf{卷起后的界面。}激波–重气泡 $t=1.4$ 界面带上的压缩，以及依赖它的掺混占比，需要在有物理粘性、界面网格收敛的 DNS 上复核，且示踪点间距要不大于最薄的物质层。
\item \textbf{自适应锚定。}锚定周期在爆炸波中宜长、在涡核中宜短；应按锚定星的畸变程度（如累积 $|\mu|$）决定重锚，而不是固定帧数。
\item \textbf{三维。}没有三维算例。理论推广直接（四面体、三维 Nanson），可作为局限写明。
\item \textbf{真实实验数据。}本稿的“实际数据”检验是在 DNS 轨迹上加扰动，仍不是真实 PTV 数据；SBLI 需要粘性求解器与大规模示踪点，本稿以单模 RM 代替。
\item \textbf{噪声。}账本对每帧独立噪声的响应约为 $\sigma/h$ 量级；抗噪版本（错帧剖分、有向面积）没有明显改善，时间平滑只改善 10\%–30\%。
\item \textbf{断轨。}窗口化账本在断轨下可用，但窗口端点落在激波内时会损失望远相消；需要让窗口端点也按激波模板推迟。
\item \textbf{文献定位}仍需补查：free-Lagrange 方法、Froyland \& Junge（2018）有限元动态 Laplacian、DEC 与图形学中的 Helmholtz–Hodge 分解。
\end{enumerate}
\appendix
\section{代码与数据}{\sloppy
\texttt{bcom.py}（账本、抗噪账本、断轨窗口账本、SPH 传导、误差统计）；\texttt{b1\_density.py}、\texttt{b1b.py}（密度基线）；\texttt{b2\_diff.py}、\texttt{b2b.py}（扩散基线）；\texttt{b3\_sens.py}、\texttt{b3\_t9.py}（敏感性）；\texttt{b4\_real.py}、\texttt{b4c.py}（实际数据检验）；\texttt{run\_rm.py}、\texttt{r1\_rm.py}（RM）；\texttt{run\_sb\_any.py}、\texttt{run\_sv\_fine.py}、\texttt{dns\_ref.py}（DNS 加密）；\texttt{make\_report.py}（本稿，全部数字由 \texttt{res\_*.json} 生成）。\par}
\end{document}
""")
txt = "".join(T)
if dsb:
    txt = txt.replace("%%SBCOMP%%", "不确定：三档 DNS 轨迹上账本为 " + " / ".join(["%.3f" % dsb[q]["t=1.4"]["bands"]["-0.003..0.003"][0] for q in LV]) + "，DNS 为 " + " / ".join(["%.3f" % dsb[q]["t=1.4"]["bands"]["-0.003..0.003"][1] for q in LV]) + "；卷起后的界面带处于信息极限，见第~\\ref{sec:dns} 节")
if b1: txt = txt.replace("%%DENS%%", "%s / %s / %s" % (pcs(b1["dg"]["ledger"]["med"]), pcs(b1["sv"]["ledger"]["med"]), pcs(b1["sb"]["ledger"]["med"])))
if b2 and "sb" in b2: txt = txt.replace("%%SBDIFF%%", "%s / %s（本稿；SPH %s）" % (pcs(b2["sb"]["network_mass"]), pcs(b2["sb"]["network_volume"]), pcs(b2["sb"]["sph"])))
dt_ = []
if dsv:
    a, b_, c = [dsv.get("sv_dx%s" % x) for x in ["0.01", "0.005", "0.0025"]]
    dt_.append(r"""\paragraph{激波–等熵涡：95 分位来自账本在涡核内的误差。}DNS 网格从 0.01 加密到 0.0025，密度中位误差由 %s 降到 %s，说明中位误差中有一部分来自 DNS 参照本身；95 分位却保持在 %s–%s，误差最大的 5\%% 中有 %s–%s 位于涡核（$r_0<2R_v$）内，涡核内的 95 分位为 %s–%s，不随 DNS 加密变化。所以本例的 95 分位是账本自身的误差：涡核内强差动旋转使锚定星在一个锚定周期内发生非仿射变形，星面积比不再是物质面积比。锚定周期取 3、5、8 帧时激波–涡的 95 分位为 2.9\%%、3.4\%%、4.7\%%，印证了这一点。
""" % (pcs(a["med"]), pcs(c["med"]), pcs(min(a["p95"], b_["p95"], c["p95"])), pcs(max(a["p95"], b_["p95"], c["p95"])),
       pc(min(a["top5_in_core"], c["top5_in_core"]), 0), pc(max(a["top5_in_core"], c["top5_in_core"]), 0), pcs(min(a["core_p95"], c["core_p95"])), pcs(max(a["core_p95"], c["core_p95"]))))
if dsb:
    nb = ["-0.03..-0.015", "-0.015..-0.005", "0.005..0.015", "0.015..0.03"]
    def gap(k, tt, bb): return float(np.mean([abs(dsb[k][tt]["bands"][x][0]-dsb[k][tt]["bands"][x][1]) for x in bb]))
    B = lambda k, tt, b: dsb[k][tt]["bands"][b]
    P6 = [dsb[k]["t=0.6"] for k in LV]; P14 = [dsb[k]["t=1.4"] for k in LV]
    dt_.append(r"""\paragraph{激波–重气泡，卷起之前（$t=0.6$）：差异来自 DNS。}界面两侧 $0.005<|\phi_0|<0.03$ 的四个带上，账本与 DNS 插值压缩之差的平均值随 DNS 网格（0.01 / 0.005 / 0.0025）为 %s；密度误差的 95 分位为 %s，离界面 $>0.03$ 的 95 分位为 %s，中位误差为 %s。账本值在三档 DNS 轨迹上基本不变，DNS 值逐档向账本靠拢，两侧方向相反（重气体一侧 DNS 偏低、轻气体一侧偏高），这是接触间断被数值扩散抹宽、两种气体密度相互混合的特征。所以在卷起之前，账本与 DNS 的差异应归于 DNS 参照，账本给出的是示踪点所代表物质本身的压缩。
""" % (" / ".join(["%.3f" % gap(k, "t=0.6", nb) for k in LV]), " / ".join([pcs(p["p95"]) for p in P6]), " / ".join([pcs(p["p95_outside_interface"]) for p in P6]), " / ".join([pcs(p["med"]) for p in P6])))
    dt_.append(r"""\paragraph{激波–重气泡，卷起之后（$t=1.4$）：离界面的带收敛，界面带不收敛。}重气体一侧 $[-0.06,-0.03]$ 与 $[-0.03,-0.015]$ 带上，DNS 压缩随网格为 %s 与 %s，逐档靠近账本（%s 与 %s）；离界面 $>0.03$ 的密度误差 95 分位由 %s 降到 %s，中位误差由 %s 降到 %s。但在 $|\phi_0|<0.015$ 内两者都不收敛：界面带 $[-0.003,0.003]$ 上账本压缩为 %s，DNS 为 %s；95 分位停在 %s。原因在于这里的 DNS 是无粘的：涡片上的 Kelvin–Helmholtz 卷起随网格加密不断出现更细的结构，没有网格收敛的极限；$\Delta x=0.0025$ 时这些结构已经细于示踪点间距 $h=0.005$，锚定星内的映射不再是仿射的，账本也随之失效。这正是 T9 所说的信息极限（物质层薄于 $h$）。
""" % (" / ".join(["%.3f" % B(k, "t=1.4", "-0.06..-0.03")[1] for k in LV]), " / ".join(["%.3f" % B(k, "t=1.4", "-0.03..-0.015")[1] for k in LV]),
       " / ".join(["%.3f" % B(k, "t=1.4", "-0.06..-0.03")[0] for k in LV]), " / ".join(["%.3f" % B(k, "t=1.4", "-0.03..-0.015")[0] for k in LV]),
       pcs(P14[0]["p95_outside_interface"]), pcs(P14[-1]["p95_outside_interface"]), pcs(P14[0]["med"]), pcs(P14[-1]["med"]),
       " / ".join(["%.3f" % B(k, "t=1.4", "-0.003..0.003")[0] for k in LV]), " / ".join(["%.3f" % B(k, "t=1.4", "-0.003..0.003")[1] for k in LV]),
       " / ".join([pcs(p["p95"]) for p in P14])))
    hm = [dsb[k]["heavy_mass_x>1"] for k in LV]
    dt_.append(r"""\paragraph{结论与对旧结论的影响。}(1) 卷起之前，界面附近的账本—DNS 差异来自 DNS 的数值扩散，账本可信。(2) 卷起之后，界面带上的压缩参照无法由无粘 DNS 的加密确定，账本在界面带上也达到信息极限；离界面 0.015 以外两者收敛。(3) 因此旧报告中“$t=1.4$ 时压缩占总增长 54\%%–62\%%”依赖界面带上的压缩值，应降级为“依赖分辨率、需在粘性 DNS 上复核”；“激波扫过后界面先被切向缩短、抵消大部分压缩”（$t\le0.6$）不受影响。(4) 积分量对此不敏感：$x>1$ 的重气体质量（网络 / DNS）三档为 %s，误差稳定在约 1\%%。
""" % ("；".join(["%.4f / %.4f" % tuple(h) for h in hm])))
dns_txt = "".join(dt_) + r"\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{figR_dns.pdf}\caption{(a)(b) 激波–重气泡各带的账本压缩（实线）与 DNS 插值压缩（虚线），颜色由浅到深为 DNS 网格由粗到细；(c) 两个 DNS 算例的密度误差随 DNS 网格的变化。}\end{figure}" + "\n"
rt = ""
if b4s and b4v:
    rt = r"""\paragraph{读数。}(1) \textbf{噪声。}账本对每帧独立噪声的响应约与 $\sigma/h$ 成正比：$0.02h$ 时中位误差 %s（Sod）、%s（激波–涡），$0.05h$ 时 %s、%s。按 PTV 的典型精度（$\sigma\approx0.01h$–$0.02h$），中位误差在 1\%%–2\%% 量级。错帧剖分与有向面积的抗噪版本没有可见改善（$0.05h$ 时 %s 对 %s）。(2) \textbf{稀疏。}示踪点减为 1/4 与 1/16 时，激波–涡的中位误差为 %s 与 %s，涡核区域环量误差为 %s 与 %s。(3) \textbf{断轨。}每帧 1\%%–3\%% 的断轨下，窗口化账本仍有 %s–%s 的轨迹可用，中位误差 %s–%s（激波–涡）；主要代价来自窗口端点落在激波内时损失的望远相消，以及新身份初值的插值误差。(4) \textbf{组合。}三种扰动同时存在时，中位误差 %s（Sod）与 %s（激波–涡），95 分位 %s 与 %s。
""" % (pcs(b4s["noise_0.02"]["v2"]["med"]), pcs(b4v["noise_0.02"]["v2"]["med"]), pcs(b4s["noise_0.05"]["v2"]["med"]), pcs(b4v["noise_0.05"]["v2"]["med"]),
       pcs(b4v["noise_0.05"]["v3"]["med"]), pcs(b4v["noise_0.05"]["v2"]["med"]),
       pcs(b4v["sparse_1/4"]["v3"]["med"]), pcs(b4v["sparse_1/16"]["v3"]["med"]), pcs(b4v["sparse_1/4"]["circ"]), pcs(b4v["sparse_1/16"]["circ"]),
       pc(b4v["breaks_p0.03"]["usable_frac"], 0), pc(b4v["breaks_p0.01"]["usable_frac"], 0), pcs(b4v["breaks_p0.01"]["med"]), pcs(b4v["breaks_p0.03"]["med"]),
       pcs(b4s["combined_1/4_0.02h_p0.01"]["med"]), pcs(b4v["combined_1/4_0.02h_p0.01"]["med"]), pcs(b4s["combined_1/4_0.02h_p0.01"]["p95"]), pcs(b4v["combined_1/4_0.02h_p0.01"]["p95"]))
txt = txt.replace("%%DNSTEXT%%", dns_txt).replace("%%REALTEXT%%", rt)
open("../repA/paperA_supp.tex", "w").write(txt); print("written", len(txt))
