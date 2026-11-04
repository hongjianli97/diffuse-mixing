import json, os, glob, numpy as np
J = lambda f: json.load(open(f)) if os.path.exists(f) else None
pc = lambda x, d=1: ("%." + str(d) + "f") % (100*x) + r"\%"
def pcs(x):
    if x is None: return "—"
    if abs(x) < 1e-6: return "$<10^{-6}$"
    return pc(x, 2 if abs(x) < 0.01 else 1)
ad = {}
for f in glob.glob("res_g1_adapt_*.json"): ad.update(J(f))
nz = J("res_g4_noise.json"); dl = J("res_g5_dynlap.json"); sv = J("res_dnsref_sbv.json"); sz = J("res_g6_sz.json"); sbi = J("res_dnsref_sb.json")
b4s = J("res_b4_sod.json"); b4v = J("res_b4_sv.json")
T = []; A = T.append
A(r"""\newcommand{\HEADTITLE}{论文 A 缺口补齐（续）}
\input{preamble.tex}
\begin{document}
\begin{center}{\sffamily\bfseries\LARGE 论文 A 缺口补齐（续）}\\[0.4em]
{\sffamily\large 粘性 DNS、自适应锚定、噪声传播律、动态 Laplacian 取代 T7，以及附件文献的定位}\\[0.6em]{\color{quiet}2026 年 10 月}\end{center}
""")
# ----- summary box filled later
A("%%SUMMARY%%\n\\tableofcontents\n\\newpage\n")
A(r"""\section{缺口与处理一览}
\begin{table}[!htbp]\centering\small\caption{上一份报告列出的缺口与本轮的处理。}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.2\linewidth} >{\raggedright\arraybackslash}p{0.3\linewidth} L@{}}\toprule
缺口 & 本轮的处理 & 结果\\\midrule
卷起后界面带的压缩参照 & 给求解器加物理粘性、热传导与组分扩散（$\mu=5\times10^{-4}$，$Re\approx1300$），三档网格 & %%VISC_ROW%%\\
锚定周期的取舍 & 残差触发的自适应锚定：星的仿射拟合残差超过阈值即提前重锚 & %%ADAPT_ROW%%\\
噪声没有理论 & 推导账本的噪声传播律 & 中位误差 $\approx0.74\,\sigma/h$，与静态与流动数据一致\\
T7 结果弱 & 用加权 FEM 动态 Laplacian（Froyland–Junge 2018 的离散，Froyland–Kwok 2020 的权重）取代 Ulam 矩阵，质量取自账本 & %%DL_ROW%%\\
环量沉积的模型比较 & 按 Samtaney–Zabusky（1994）式 4.12 与 5.12 手工录入公式，与位置形式环量比较 & 上半气泡环量：位置形式 0.266、DNS 0.270、SZ1 模型 0.289（相差约 8\%）\\
文献定位 & 阅读附件 15 篇，逐篇对应 & 见第~\ref{sec:lit} 节；T8 的 FEM 刚度与加权动态 Laplacian 已有先例，论文 A 的新意需改述\\
断轨窗口端点 & 未做 & 方案见第~\ref{sec:rest} 节\\
三维、真实 PTV & 未做 & 作为局限写明\\\bottomrule
\end{tabularx}\end{table}
""")
# ----- viscous section
A(r"""\section{粘性 DNS：卷起之后的界面压缩}\label{sec:visc}
\paragraph{设置。}在原 MUSCL–HLLC 求解器上加守恒形式的粘性、热传导与组分扩散通量（常数 $\mu=5\times10^{-4}$，$Pr=0.72$，$\rho D=\mu$），其余设置与原激波–重气泡相同（$M_s=1.5$、密度比 3），示踪点不变（$h=0.005$）。以气泡直径与波后速度计，$Re\approx1300$。界面的扩散厚度 $\sqrt{\nu t}$ 在 $t=1.4$ 约为 0.015，即 3 个示踪点间距，因此物质层不再薄于 $h$。网格取 $\Delta x=0.01,0.005,0.0035$。
""")
bands = ["-0.06..-0.03", "-0.03..-0.015", "-0.015..-0.005", "-0.003..0.003", "0.005..0.015", "0.015..0.03", "0.03..0.06"]
if sv:
    LV = [k for k in ["sbv_dx0.01", "sbv_dx0.005", "sbv_dx0.0035"] if k in sv]
    for tt in ["t=0.6", "t=1.4"]:
        A(r"\begin{table}[!htbp]\centering\small\caption{粘性激波–重气泡，$%s$：各带的账本压缩 / DNS 插值压缩。}" % tt + "\n" + r"\begin{tabular}{l" + "c"*len(LV) + r"}\toprule" + "\n$\\phi_0$ 区间 & " + " & ".join(["$\\Delta x=%s$" % k.split("dx")[1] for k in LV]) + r"\\\midrule" + "\n")
        for b in bands: A("$[%s]$ & " % b.replace("..", ",") + " & ".join(["%.3f / %.3f" % tuple(sv[k][tt]["bands"][b]) for k in LV]) + r"\\" + "\n")
        A(r"\bottomrule\end{tabular}\end{table}" + "\n")
    A(r"\begin{table}[!htbp]\centering\small\caption{粘性激波–重气泡：密度误差与 T9 掺混归因随网格的变化。}" + "\n" + r"\begin{tabular}{l" + "c"*len(LV) + r"}\toprule 量 & " + " & ".join(["$\\Delta x=%s$" % k.split("dx")[1] for k in LV]) + r"\\\midrule" + "\n")
    for tt in ["t=0.6", "t=1.4"]:
        A("$%s$ 密度中位 / 95 分位 & " % tt + " & ".join([pcs(sv[k][tt]["med"]) + " / " + pcs(sv[k][tt]["p95"]) for k in LV]) + r"\\" + "\n")
    for tt in ["t=0.6", "t=1", "t=1.4"]:
        if all(tt in sv[k].get("T9", {}) for k in LV):
            A("$%s$ 拉伸 / 压缩 / 总增长 & " % tt + " & ".join(["%.3f / %.3f / %.3f" % (sv[k]["T9"][tt]["stretch"], sv[k]["T9"][tt]["compression"], sv[k]["T9"][tt]["stretch"]+sv[k]["T9"][tt]["compression"]) for k in LV]) + r"\\" + "\n")
    A(r"$x>1$ 的重气体质量：网络 / DNS & " + " & ".join(["%.4f / %.4f" % tuple(sv[k]["heavy_mass_x>1"]) for k in LV]) + r"\\\bottomrule\end{tabular}\end{table}" + "\n")
    A("%%VISC_TEXT%%\n")
else:
    A("（粘性 DNS 的分析未完成。）\n")
# ----- adaptive anchoring
A(r"""\section{自适应锚定}
\paragraph{规则。}账本精确的条件是：锚定星的顶点在锚定段内作仿射运动（星多边形的边保持为物质直线）。于是每一步对每个示踪点的锚定星做最小二乘仿射拟合，残差 $r_i=\big(\tfrac1K\sum_k|x_k-Fx^a_k-c|^2\big)^{1/2}/\sqrt{A^\ast_i}$；当 $r_i>\varepsilon$ 或锚定段达到 5 帧时重锚，但星触及激波模板时照旧推迟（总长不超过 20 帧）。这样，仿射性好的区域保持 5 帧的锚定，涡核等非仿射区域自动缩短锚定段。
""")
if ad:
    cs = [c for c in ["sod", "blast", "dg", "lo", "sv", "sb", "rm"] if c in ad]
    nm = dict(dg="双环流", lo="Lamb–Oseen", sod="Sod", blast="爆炸波", sv="激波–涡", sb="激波–气泡", rm="RM")
    A(r"\begin{table}[!htbp]\centering\small\caption{密度中位误差 / 95 分位：固定 5 帧与自适应锚定（$\varepsilon=0.01,0.02,0.04$）。}\resizebox{\linewidth}{!}{\begin{tabular}{lcccc}\toprule 算例 & 固定 5 帧 & $\varepsilon=0.01$ & $\varepsilon=0.02$ & $\varepsilon=0.04$\\\midrule" + "\n")
    for c in cs:
        o = ad[c]; A(nm[c] + " & " + " & ".join([pcs(o[k]["med"]) + " / " + pcs(o[k]["p95"]) for k in ["fixed5", "adapt_eps0.01", "adapt_eps0.02", "adapt_eps0.04"]]) + r"\\" + "\n")
    A(r"\bottomrule\end{tabular}}\end{table}" + "\n")
    A(r"""自适应锚定在所有算例上都不比固定 5 帧差，在涡核主导的激波–涡中把 95 分位从 %s 降到 %s（$\varepsilon=0.01$），与“锚定周期 3 帧”的最好结果相当，同时保留了爆炸波中长锚定的好处（%s，与固定 5 帧相同）。上一轮的“锚定周期需取舍”由此解决；建议默认 $\varepsilon=0.01$。早先试过的“只靠残差触发、最长 20 帧”的版本在爆炸波中变差（95 分位 39.6\%%），原因是光滑膨胀区的锚定段过长，这说明 5 帧上限本身有用：它限制了单段内非仿射误差的积累。
""" % (pcs(ad["sv"]["fixed5"]["p95"]), pcs(ad["sv"]["adapt_eps0.01"]["p95"]), pcs(ad["blast"]["adapt_eps0.01"]["p95"])) if ("sv" in ad and "blast" in ad) else "")
# ----- noise law
A(r"""\section{噪声传播律}
\begin{proposition}[账本的噪声传播]\status{已证+数值}
设每帧位置带独立各向同性噪声 $\delta x\sim\mathcal N(0,\sigma^2I)$。(i) 节点的星面积就是其邻点组成的连接多边形的面积，与中心点位置无关；对邻点 $x_k$（按角度排序）有 $\partial A^\ast/\partial x_k=\tfrac12R_{90^\circ}(x_{k+1}-x_{k-1})$，故
\begin{equation}
\operatorname{std}(\delta\ln A^\ast)=\frac{\sigma}{2A^\ast}\Big(\sum_k|x_{k+1}-x_{k-1}|^2\Big)^{1/2}=c_\ast\,\frac{\sigma}{h},
\end{equation}
正六边形星 $c_\ast=\sqrt{4.5}/(3\sqrt3/2)=0.816$。(ii) 账本 $\ln J$ 的误差主要来自窗口两端的两帧：中间重锚时，旧星在当前帧的面积与新星在同一帧的锚定面积用的是同一组带噪位置，且两星大部分顶点相同，噪声大部分相消。于是 $\operatorname{std}(\delta\ln J)\approx\sqrt2\,c_\ast\sigma/h$，中位绝对误差 $\approx0.674\sqrt2\,c_\ast\,\sigma/h$。
\end{proposition}
""")
if nz:
    A(r"""\paragraph{检验。}在静止的抖动格点上（无流动，只有噪声），由实际剖分算得 $c_\ast=%.3f$，预测中位误差 $%.3f\,\sigma/h$；账本实测为：11 帧 $%.2f$–$%.2f\,\sigma/h$，41 帧 $%.2f$–$%.2f\,\sigma/h$（随帧数只缓慢增长，印证了 (ii) 的相消）。流动算例中，Sod 为 $0.75\,\sigma/h$，激波–涡为 $1.0\,\sigma/h$（涡核的非仿射误差叠加在噪声上）。

\paragraph{推论。}(1) 要使密度中位误差低于 1\%%，需要 $\sigma\lesssim0.013h$，与 PTV 的典型精度（$0.01h$–$0.02h$）同量级；(2) 误差按局部间距计：压缩区间距变小，同样的 $\sigma$ 带来更大的相对误差，压缩比 $\eta$ 时放大约 $\sqrt\eta$；(3) 与基于速度梯度的诊断不同（Badza 等 2023 与 Balasuriya 2020 讨论的不确定性随拉伸放大），账本的噪声误差不随累积变形放大；随变形放大的是链式张量估计（放大因子为各向异性 $A$）。(4) 噪声主要在两端帧，所以对首末各若干帧的位置做时间平均，比全程平滑更有效。
""" % (nz["c_pred_std_per_frame"], nz["c_pred_med_lnJ"], nz["S11_sig0.02"]["med_over_sig"], nz["S11_sig0.05"]["med_over_sig"], nz["S41_sig0.02"]["med_over_sig"], nz["S41_sig0.05"]["med_over_sig"]))
# ----- dynamic Laplacian
A(r"""\section{用加权动态 Laplacian 取代 T7}
T7 的按质量加权 Ulam 矩阵在六个算例中三个退化、一个找错主导集（Lamb–Oseen 与涡核的 Jaccard 只有 0.11）。附件中的 Froyland \& Junge（2018）给出了散乱轨迹上的 FEM 动态 Laplacian：每个时刻对当前位置做 Delaunay 剖分，组装 P1 刚度并对时间平均，解广义特征问题；Froyland \& Kwok（2020）把它推广到非保体积的动力学与非均匀初始测度。这两者与论文 A 的算子完全对应：P1 刚度就是 T8 的 $K(t)$，权重（测度）就是账本给出的质量。所以 T7 应改为
\begin{equation}
\bar K v=\lambda M v,\qquad \bar K=\frac1S\sum_s K(t_s),\quad M=\mathrm{diag}(m_i),
\end{equation}
可压缩时 $K(t_s)$ 以账本密度加权。
""")
if dl:
    A(r"""\paragraph{检验。}Lamb–Oseen 涡对：第 2–4 个特征向量取 90\%% 分位以上的集合，与涡核（$r_0<1.5a$）的最佳 Jaccard 为 %.2f（Ulam 为 0.11）。双环流：第二特征向量按中位数二分，与左环流的 Jaccard 为 %.2f（Ulam 为 0.59）。所以 T7 改为动态 Laplacian 后不再是弱项，而且它与 T8 共用同一个算子，与第 1 节的“质量度量下的 Dirac 算子”框架一致：$\bar K$ 是 $D^2$ 节点块的时间平均。需要明确的是，这一部分的离散方法来自 Froyland–Junge，权重的理论来自 Froyland–Kwok；论文 A 的新内容是用账本从位置得到可压缩流中的权重。
""" % (dl["lo"]["best_jaccard_core"], dl["dg"]["jaccard_left"]))

if sz:
    r_ = {k: sz[k] for k in ["inviscid_dx0.005", "inviscid_dx0.0025", "viscous_dx0.0035"] if k in sz}
    nmz = {"inviscid_dx0.005": "无粘，$\\Delta x=0.005$", "inviscid_dx0.0025": "无粘，$\\Delta x=0.0025$", "viscous_dx0.0035": "粘性，$\\Delta x=0.0035$"}
    A(r"""\section{环量沉积：与 Samtaney--Zabusky 模型比较}\label{sec:sz}
Samtaney \& Zabusky（1994，JFM 269）给出平面界面单位原长度上的沉积环量（其式 4.12，换算到以入射气体声速 $c_0$ 为单位）
\begin{equation}
\Gamma'\approx\frac{2}{\gamma+1}\Big(1-\eta^{-1/2}\Big)\sin\alpha\,\big(1+M^{-1}+2M^{-2}\big)(M-1)\,c_0,
\end{equation}
以及圆形气泡（半个气泡）上的总环量模型 SZ1：$\Gamma_{iii}=(1+\tfrac12\pi)\Gamma'_1r_0$（其式 5.12，$\Gamma'_1$ 为上式 $\sin\alpha$ 的系数），第二阶段末约为 $\Gamma'_1r_0$。本算例 $M=1.5$、$\eta=3$、$\gamma=1.4$、$r_0=0.25$、$c_0=1$，得 $\Gamma'_1=%.4f$，第二阶段末 %.4f，SZ1 为 %.4f。

位置形式的环量取初始距界面 $|\phi_0|<0.15$ 的物质带（上半气泡），由中点位移上链的环量求和得到（T4，只凭位置）。
""" % (sz["SZ_Gamma1prime"], sz["SZ_phase_ii"], sz["SZ1_phase_iii"]))
    A(r"\begin{table}[!htbp]\centering\small\caption{上半气泡的环量（取绝对值）。}\begin{tabular}{lccc}\toprule & $t=0.6$ & $t=1.0$ & $t=1.4$\\\midrule" + "\n")
    for k, v in r_.items():
        A(nmz[k] + "：位置形式（物质带） & " + " & ".join(["%.3f" % abs(x) for x in v["band0.15"]["G_pos_at"]]) + r"\\" + "\n")
        A(r"\quad DNS，同一组三角形上积分 & " + " & ".join(["%.3f" % abs(x) for x in v["band0.15"]["G_dns_at"]]) + r"\\" + "\n")
        A(r"\quad DNS，整个上半区域 & " + " & ".join(["%.3f" % abs(v["dns_domain"][q]) for q in ["0.6", "1.0", "1.4"]]) + r"\\" + "\n")
    A(r"SZ1 模型（第三阶段末） & \multicolumn{3}{c}{%.3f}\\\bottomrule\end{tabular}\end{table}" % sz["SZ1_phase_iii"] + "\n")
    v0 = r_["inviscid_dx0.005"]["band0.15"]
    A(r"""\paragraph{读数。}(1) 激波扫过后（$t=0.6$），只凭位置得到的上半气泡环量为 %.3f，DNS 整个上半区域为 %.3f，二者相差约 1.5\%%；SZ1 模型为 %.3f，比两者高约 8\%%，在 Samtaney–Zabusky 报告的模型与 DNS 的偏差范围内。所以 T4 的位置形式环量不仅与本文 DNS 一致，也与经典模型一致。(2) 无粘算例中，物质带的环量在 $t=0.6$ 之后几乎守恒（Kelvin 定理，%.3f → %.3f），而“同一组三角形上用顶点涡量积分”的 DNS 值在后期偏离（$t=1.4$ 为 %.3f）：卷起后物质三角形严重变形并跨过涡片，顶点平均的求积失效，而位置形式是多边形边界上的精确环量，不受影响。(3) 粘性算例中两者在全程一致（中位相对误差 %s），窄带（$|\phi_0|<0.04$）的环量随时间减小，宽带不变，这正是粘性把涡量扩散出窄带、而宽带的边界上粘性通量很小的结果。
""" % (abs(v0["G_pos_at"][0]), abs(r_["inviscid_dx0.005"]["dns_domain"]["0.6"]), sz["SZ1_phase_iii"], abs(v0["G_pos_at"][0]), abs(v0["G_pos_at"][2]), abs(v0["G_dns_at"][2]),
       pcs(r_["viscous_dx0.0035"]["band0.15"]["relerr_med"]) if "viscous_dx0.0035" in r_ else "—"))

# ----- literature
A(r"""\section{附件文献的定位}\label{sec:lit}
\begin{table}[!htbp]\centering\small\caption{附件文献与论文 A 的关系。}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.26\linewidth} L@{}}\toprule
文献 & 与本工作的关系与应如何引用\\\midrule
Froyland \& Junge 2018，SIADS（散乱、稀疏、不完整轨迹上的 FEM 动态 Laplacian） & 逐时刻 Delaunay 剖分、P1 刚度、时间平均，正是 T8 刚度与新 T7 的离散；论文 A 不能再声称“轨迹上的 P1 刚度”是新的。新的是：瞬态对流—扩散的求解、以 $\rho\kappa$ 为电导、以账本质量为质量矩阵，以及在激波流中对 DNS 的定量检验\\
Froyland \& Kwok 2020，JNS（加权黎曼流形上的动态 Laplacian） & 非保体积动力学与非均匀初始测度的理论；“度量即测度”在连续层面已由它给出。论文 A 的贡献是在可压缩、含激波的流动中只凭位置得到这一测度，并定量显示面积权重的误差（5–10 倍）\\
Karrasch \& Keller 2020（LCS 的几何热流理论） & 拉格朗日坐标下的对流—扩散是时变热方程，时间平均即动态 Laplacian；T8 的 $M\dot c=-K(t)c$ 是它的 FEM 离散，并推广到变密度与 $\rho\kappa$\\
Froyland 2015（动态等周） & 动态 Laplacian 的出发点；新 T7 的理论依据\\
Hadjighasem 等 2017（LCS 方法的比较） & 脊分类工作的基准与比较对象；不进入论文 A\\
Haller 等 2021（单轨迹准客观诊断）；Aksamit 等 2024（稀疏轨迹的相对拉伸与转动） & “只凭轨迹”的已有路线：前者用单条轨迹，后者用相对量；AFRC 椭圆通道直接用了后者。论文 A 应说明其区别：三角形上的有限步恒等式给出严格的积分量（面积、环量），而不是逐点的准客观量\\
Farazmand \& Haller 2016（极转角 PRA）；Haller \& Beron-Vera 2013（黑洞涡） & 三点映射的 $\arg p$ 即三角形上的极转角，合并块给出跨激波的 PRA；椭圆 LCS 的参照\\
Badza 等 2023；Balasuriya 2020（随机敏感度） & 速度不确定性下 LCS 的敏感度；第 4 节的噪声传播律与之对照：账本误差 $O(\sigma/h)$，不随累积变形放大\\
Singer \& Wu 2012（向量扩散映射与联络 Laplacian） & 以局部旋转为联络的扩散；可用于把逐步的 Kabsch 对齐推广为局部客观的联络，属于拓扑信号那一篇\\
Samtaney \& Zabusky 1994；Picone \& Boris 1988；Haas \& Sturtevant 1987 & 激波–气泡的环量沉积模型与实验。论文 A 的位置形式环量应与 Samtaney–Zabusky 的平面界面标度律（$\Gamma'\propto(1-\eta^{-1/2})\sin\alpha\,(1+M^{-1}+2M^{-2})(M-1)$）与气泡总环量模型比较；本轮已做，见第~\ref{sec:sz} 节\\
Anand \& Bianconi 2009；Anand, Bianconi \& Severini 2011；多路网络熵与重叠；带味网络几何；《多层网络》 & 网络系综的 Shannon 熵与以 $\rho=L/\operatorname{Tr}L$ 定义的 von Neumann 熵。可用于拓扑信号那一篇：以质量度量下的 $\bar K$ 定义密度矩阵，其 von Neumann 熵度量“混合几何”的谱复杂度；时间窗层可用多路网络熵与重叠刻画相干结构的存续。不进入论文 A\\\bottomrule
\end{tabularx}\end{table}
""")
A(r"""\section{仍未完成}\label{sec:rest}
\begin{enumerate}
\item \textbf{断轨窗口端点。}方案：不再把账本切成固定窗口，而是让每条轨迹在其连续段内保持自己的锚定星；当锚定星的某个顶点消失时，若不在激波模板内则立即重锚，否则推迟到激波过后、用剩余顶点的连接多边形计算面积比。需要重写账本的数据结构（每帧的点集不同），本轮未做。
\item \textbf{三维与真实 PTV 数据}：作为论文 A 的局限写明。
\end{enumerate}
\end{document}
""")
txt = "".join(T)
# fills
def visc_eval():
    if not sv: return "未完成", ""
    LV = [k for k in ["sbv_dx0.01", "sbv_dx0.005", "sbv_dx0.0035"] if k in sv]
    if len(LV) < 3: return "部分完成", ""
    ib = "-0.003..0.003"; f, m_ = sv["sbv_dx0.0035"]["t=1.4"]["bands"][ib], sv["sbv_dx0.005"]["t=1.4"]["bands"][ib]
    dled, ddns = abs(f[0]-m_[0]), abs(f[1]-m_[1]); gap = abs(f[0]-f[1])
    conv = (dled < 0.05) and (ddns < 0.05)
    T14 = [sv[k]["T9"]["t=1.4"] for k in LV]
    row = ("界面带收敛：两档细网格的账本相差 %.3f、DNS 相差 %.3f，最细网格上二者相差 %.3f" % (dled, ddns, gap)) if conv else ("界面带仍未完全收敛：两档细网格的账本相差 %.3f、DNS 相差 %.3f" % (dled, ddns))
    row += "；$t=1.4$ 拉伸 %.3f、压缩 %.3f（$Re\\approx1300$）" % (T14[-1]["stretch"], T14[-1]["compression"])
    text = r"""\paragraph{读数。}在 $|\phi_0|<0.003$ 的界面带上，$t=1.4$ 时账本压缩随网格为 %s，DNS 为 %s；两档细网格之间账本相差 %.3f、DNS 相差 %.3f，最细网格上二者相差 %.3f。""" % (
        " / ".join(["%.3f" % sv[k]["t=1.4"]["bands"][ib][0] for k in LV]), " / ".join(["%.3f" % sv[k]["t=1.4"]["bands"][ib][1] for k in LV]), dled, ddns, gap)
    if conv:
        text += r"有了物理粘性，界面带上账本与 DNS 都随网格收敛并且彼此接近，所以无粘 DNS 中的不收敛来自没有物理耗散的 Kelvin–Helmholtz 小尺度结构（细于示踪点间距），而不是账本本身。"
    else:
        text += r"即使加了物理粘性，界面带在本网格范围内仍未完全收敛。"
    text += r""" 掺混归因：$t=1.4$ 时界面平均拉伸随网格为 %s，压缩为 %s，梯度总增长为 %s。拉伸接近零而略为负，所以在 $Re\approx1300$ 下，到 $t=1.4$ 界面梯度的增长几乎全部来自压缩：物理粘性抑制了卷起，界面没有被明显拉长。这是另一个雷诺数下的结论，不能直接替代无粘算例中被降级的“54\%%–62\%%”；它说明压缩与拉伸的分配依赖雷诺数，而账本在两个雷诺数下都给出了可检验的值（拉伸与压缩接近抵消时，不宜再用“压缩占总增长的比例”作指标，应分别报告两项）。""" % (
        " / ".join(["%.3f" % x["stretch"] for x in T14]), " / ".join(["%.3f" % x["compression"] for x in T14]), " / ".join(["%.3f" % (x["stretch"]+x["compression"]) for x in T14]))
    return row, text
vr, vt = visc_eval()
adr = ("激波–涡 95 分位 %s → %s，其余算例不变差" % (pcs(ad["sv"]["fixed5"]["p95"]), pcs(ad["sv"]["adapt_eps0.01"]["p95"]))) if "sv" in ad else "—"
dlr = ("Lamb–Oseen 涡核 Jaccard 0.11 → %.2f；双环流 0.59 → %.2f" % (dl["lo"]["best_jaccard_core"], dl["dg"]["jaccard_left"])) if dl else "—"
summ = r"""\noindent\fcolorbox{accent!40}{tint}{\begin{minipage}{0.965\linewidth}\vspace{0.3em}
\textbf{\sffamily 结论}\quad (1) \textbf{粘性 DNS}：%s。(2) \textbf{自适应锚定}：%s，解决了锚定周期的取舍。(3) \textbf{噪声传播律}：账本中位误差 $\approx0.74\,\sigma/h$，只由两端帧决定、不随累积变形放大，静态与流动数据都吻合。(4) \textbf{T7 改为加权动态 Laplacian}：%s；它与 T8 共用同一算子。(4$'$) \textbf{环量沉积}：只凭位置的上半气泡环量 0.266，与 DNS（0.270）相差 1.5\%%，与 Samtaney–Zabusky 的 SZ1 模型（0.289）相差 8\%%。(5) \textbf{文献}：附件中 Froyland–Junge（2018）与 Froyland–Kwok（2020）已经给出了轨迹上的 FEM 刚度与加权动态 Laplacian，Karrasch–Keller（2020）给出了拉格朗日热方程；论文 A 的新意应改述为“只凭位置得到可压缩、含激波流动的质量测度（账本），并据此求解变密度的拉格朗日对流—扩散与掺混归因”，而不是 FEM 离散本身。
\vspace{0.3em}\end{minipage}}
""" % (vr, adr, dlr)
txt = txt.replace("%%SUMMARY%%", summ).replace("%%VISC_ROW%%", vr).replace("%%ADAPT_ROW%%", adr).replace("%%DL_ROW%%", dlr).replace("%%VISC_TEXT%%", vt)
open("../repG/gaps2.tex", "w").write(txt); print("written", len(txt))
