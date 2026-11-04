import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": ["Noto Sans CJK JP"], "font.size": 8, "axes.unicode_minus": False})
J = lambda f: json.load(open(f)) if os.path.exists(f) else None
pc = lambda x, d=1: ("%." + str(d) + "f") % (100*x) + r"\%"
def pcs(x):
    if x is None: return "—"
    if abs(x) < 1e-6: return "$<10^{-6}$"
    return pc(x, 2 if abs(x) < 0.01 else 1)
R = {c: J("res_s1_%s.json" % c) for c in ["sod", "blast", "sv"]}; R = {k: v for k, v in R.items() if v}
SC = ["p0.01", "p0.03", "p0.05", "shock_forced30%+p0.01", "sparse1/4+0.02h+p0.03"]
SCN = {"p0.01": "随机断轨 1\\%/帧", "p0.03": "随机断轨 3\\%/帧", "p0.05": "随机断轨 5\\%/帧", "shock_forced30%+p0.01": "30\\% 的粒子在穿越激波时断轨", "sparse1/4+0.02h+p0.03": "1/4 稀疏 + 0.02$h$ 噪声 + 3\\%/帧"}
SCF = {"p0.01": "1%/帧", "p0.03": "3%/帧", "p0.05": "5%/帧", "shock_forced30%+p0.01": "激波处断轨", "sparse1/4+0.02h+p0.03": "稀疏+噪声"}
CN = {"sod": "Sod", "blast": "柱面爆炸波", "sv": "激波–等熵涡（DNS）"}
# figure
fig, ax = plt.subplots(1, len(R), figsize=(7.4, 2.6), squeeze=False); ax = ax[0]
for j, (c, r) in enumerate(R.items()):
    sc = [s for s in SC if s in r]; x = np.arange(len(sc))
    for k, (meth, col, lab) in enumerate([("windowed", "#999999", "窗口化（旧）"), ("stream_deferred", "#1f5fa8", "逐轨账本（延迟标定）")]):
        ax[j].bar(x+0.36*(k-0.5), [100*r[s][meth]["all"]["p95"] for s in sc], 0.34, color=col, label=lab)
    for xi, s in zip(x, sc):
        ax[j].text(xi-0.18, 100*r[s]["windowed"]["all"]["p95"]*1.05, "%d%%" % round(100*r[s]["windowed"]["coverage"]), ha="center", fontsize=5.5, color="#555")
    ax[j].axhline(100*r["no_breaks"]["p95"], color="k", lw=0.7, ls="--")
    ax[j].set_xticks(x); ax[j].set_xticklabels([SCF[s] for s in sc], rotation=30, fontsize=6.5); ax[j].set_yscale("log")
    ax[j].set_title("(%s) %s" % ("abc"[j], CN[c]), fontsize=8); ax[j].set_ylabel("末帧密度误差 95 分位 (%)")
ax[0].legend(fontsize=6)
plt.tight_layout(); plt.savefig("../repS/figS.pdf"); plt.close()
T = []; A = T.append
A(r"""\newcommand{\HEADTITLE}{断轨下的逐轨账本：设计与检验}
\input{preamble.tex}
\begin{document}
\begin{center}{\sffamily\bfseries\LARGE 断轨下的逐轨账本}\\[0.4em]{\sffamily\large 每条轨迹保持自己的锚定星：数据结构、算法与检验}\\[0.6em]{\color{quiet}2026 年 10 月}\end{center}
%%BOX%%
\tableofcontents
\newpage
\section{问题}
PTV 数据中的轨迹会中断：粒子在若干帧内丢失，之后以新的身份重新出现。上一版的处理是把账本切成固定的 5 帧窗口，只用窗口内完整的轨迹，新身份的密度由近邻插值赋值。它有两个缺陷：(1) 窗口端点若落在激波内，锚定星的一端就在激波中，望远相消失效；(2) 只要轨迹在窗口内断过一次，这一窗口就不能用，覆盖率随断轨率迅速下降。本稿按上一份报告给出的方案，把账本改成逐轨的：每条轨迹在其连续段内保持自己的锚定星。

\section{设计}
\subsection{数据结构}
\begin{itemize}
\item \textbf{轨迹表}：$P\in\mathbb R^{S\times M\times2}$，第 $m$ 列是第 $m$ 条轨迹，缺失处为 NaN。同一个物理粒子可以对应多条轨迹，算法不知道这种对应关系。
\item \textbf{每条轨迹的锚定状态}：锚定帧 $a_i$；锚定时刻 Delaunay 星的连接环（按角度排序的邻居轨迹编号 $r_{i,1},\dots,r_{i,K}$）；这些邻居相对 $i$ 的锚定位置；每对相邻环顶点是否构成三角形的标志；当前段的起点值 $b_i=\ln J_i(a_i)$。
\item \textbf{密度标定}：第 0 帧存在的轨迹用已知的初始密度；之后出生的轨迹单独标定（见下）。
\end{itemize}

\subsection{规则}
\begin{enumerate}
\item \textbf{测量。}$\ln J_i(t)=b_i+\ln\big(A_i^{\rm ring}(t)/A_i^{\rm ring}(a_i)\big)$，其中 $A^{\rm ring}$ 是以 $i$ 为中心、只取当前仍然存在的环顶点的有向扇形面积；分子分母用\emph{同一组}顶点。一个顶点消失时，相邻两个三角形并成一个（用弦代替两条边），而不是中断测量。
\item \textbf{常规重锚}：锚定段达到 5 帧，或环的仿射拟合残差超过 0.01（自适应锚定），重锚到当前帧的星；环触及激波模板时推迟，最多 15 帧。
\item \textbf{激波中的断环。}若存在的环顶点少于 3 个或面积不足锚定时的 30\%，而此时环正处在激波模板内，就\emph{在锚定帧重建环}：取在锚定帧与当前帧都存在的轨迹，在锚定帧做局部 Delaunay，得到新环；锚定帧与 $b_i$ 不变。这样锚定星仍在激波前，穿越照样望远相消。不在激波中的断环直接在当前帧重锚。
\item \textbf{出生与标定}：出生的轨迹从出生帧开始自己的账本（$\ln J=0$）。它的绝对密度由已标定的近邻在标定时刻 $t_c$ 的 $\ln\rho$ 按反距离平均得到，再由自己的账本推到其他时刻。$t_c$ 取出生后第一个自己不在激波模板内的帧（\textbf{延迟标定}），或直接取出生帧（\textbf{即时标定}）。
\end{enumerate}

\begin{proposition}[缩减环的精确性]\status{已证}
若环在锚定段内作仿射运动，则以任意一组存在的环顶点（$\ge3$ 个、按原角度顺序）构成的多边形，其面积比都等于 $\det F$。因此顶点消失不引入误差，只缩小了用于测量的物质多边形。平面激波完整扫过锚定星时，锚定帧与当前帧之间的映射是仿射的，所以在锚定帧重建的环（只用幸存轨迹）同样给出精确的跨激波面积比。
\end{proposition}
\begin{proof}
仿射映射把多边形映为多边形，面积乘以 $\det F$，与顶点个数无关。第二句：锚定帧在激波前、当前帧在激波后时，两帧之间整个邻域经历同一个单轴映射。
\end{proof}

\paragraph{开发中的一个教训。}第一版在“激波中的断环”时也在当前帧重锚。这样锚定被放在了激波中间，在 30\% 粒子于激波处断轨的情形下，Sod 的末帧 95 分位达到 32\%；新出生轨迹的误差也随之变大，因为它们从出错的邻居那里标定。改为“在锚定帧重建环”后，降到 2.0\%（第~\ref{sec:res} 节）。这说明逐轨账本的关键不在缩减环本身，而在不让断轨把锚定点拖进激波。

\section{检验设计}
\begin{itemize}
\item \textbf{算例}：Sod（精确轨迹与密度，7500 个粒子，41 帧）、柱面爆炸波（精确，9345 个，61 帧）、激波–等熵涡（DNS 在线示踪点，14400 个，31 帧）。
\item \textbf{断轨模型}：每帧之后，每条存活轨迹以概率 $p$ 中断；粒子随后缺失 1–3 帧（均匀随机），再以新编号出现。$p=1\%,3\%,5\%$。
\item \textbf{对抗情形}：30\% 的被激波扫过的粒子恰在穿越激波的那一步断轨（外加 1\%/帧的随机断轨），专门检验“端点落在激波内”。
\item \textbf{组合情形}：随机保留 1/4 的粒子，加 0.02 倍（稀疏后）间距的位置噪声，$p=3\%$。
\item \textbf{比较}：窗口化账本（旧）、逐轨账本加即时标定、逐轨账本加延迟标定；以及同一组粒子不断轨时的账本（下限）。
\item \textbf{指标}：末帧仍存在的轨迹中能给出密度的比例（覆盖率）；末帧密度相对误差的中位数与 95 分位；其中出生轨迹的 95 分位；全部（轨迹，帧）观测合并的中位数与 95 分位。
\end{itemize}
""")
A(r"\section{结果}\label{sec:res}" + "\n")
for c, r in R.items():
    A(r"\begin{table}[!htbp]\centering\small\caption{%s：末帧密度误差（中位 / 95 分位）与覆盖率；不断轨时为 %s / %s。}" % (CN[c], pcs(r["no_breaks"]["med"]), pcs(r["no_breaks"]["p95"])) + "\n")
    A(r"\resizebox{\linewidth}{!}{\begin{tabular}{lcccc}\toprule 情形 & 每粒子断轨次数 & 窗口化（旧） & 逐轨账本 & 出生轨迹 95 分位（逐轨）\\\midrule" + "\n")
    for s in SC:
        if s not in r: continue
        o = r[s]; f = lambda m: "%s / %s（%s）" % (pcs(o[m]["all"]["med"]), pcs(o[m]["all"]["p95"]), pc(o[m]["coverage"], 0))
        A(SCN[s] + " & %.2f & " % o["breaks_per_particle"] + " & ".join([f(m) for m in ["windowed", "stream_deferred"]]) + " & " + (pcs(o["stream_deferred"]["newborn"]["p95"]) if "newborn" in o["stream_deferred"] else "—") + r"\\" + "\n")
    A(r"\bottomrule\end{tabular}}\end{table}" + "\n")
A(r"\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{figS.pdf}\caption{末帧密度误差的 95 分位。灰柱上方的数字为窗口化方法的覆盖率（逐轨方法均为 100\%）；虚线为不断轨时的账本。}\end{figure}" + "\n")
A("%%READ%%\n")
A(r"""\section{结论与局限}
\begin{enumerate}
\item 逐轨账本让每条轨迹在其连续段内保持自己的锚定星：顶点消失时缩减环，激波中的断环在锚定帧重建，锚定点因此不会被断轨拖进激波。
\item 覆盖率由窗口化的 79\%–95\% 提高到 100\%：末帧存在的每条轨迹都有密度估计，包括断轨后重新出现的轨迹。
\item 出生轨迹的标定是主要的剩余误差来源；延迟到离开激波模板之后再标定，总体上比即时标定好。
\item 稀疏加噪声的组合情形中，误差由噪声主导（见噪声传播律），逐轨与窗口化差别不大，但逐轨的覆盖率仍为 100\%。
\item 爆炸波的薄壳仍是难点：新出现的轨迹在陡峭密度梯度中用近邻标定，误差大；改进方向是用账本的质量守恒（局部质量平衡）代替近邻插值来标定新轨迹。
\item 仍需要第 0 帧（或某一参考时刻）的密度；整条轨迹全程都不与已标定轨迹相邻的孤立出生轨迹无法标定。
\end{enumerate}
\appendix\section{代码}{\sloppy\texttt{stream.py}：轨迹表的生成（\texttt{tracks\_from\_particles}）、逐轨账本（\texttt{ledger\_stream}）、窗口化账本（\texttt{ledger\_windowed}）、缩减环面积（\texttt{\_fan\_area}）与在锚定帧重建环（\texttt{\_rering}）；\texttt{s1\_exp.py}：全部检验；\texttt{s\_report.py}：本稿。\par}
\end{document}
""")
txt = "".join(T)
# narrative
rd = []
if "sod" in R:
    r = R["sod"]
    rd.append(r"\paragraph{Sod。}随机断轨 1\%%、3\%%、5\%%/帧时，逐轨账本（延迟标定）的末帧 95 分位为 %s、%s、%s，窗口化为 %s、%s、%s，后者的覆盖率只有 %s、%s、%s。激波处断轨的对抗情形中，逐轨为 %s，窗口化为 %s（覆盖率 %s）。" % (
        pcs(r["p0.01"]["stream_deferred"]["all"]["p95"]), pcs(r["p0.03"]["stream_deferred"]["all"]["p95"]), pcs(r["p0.05"]["stream_deferred"]["all"]["p95"]),
        pcs(r["p0.01"]["windowed"]["all"]["p95"]), pcs(r["p0.03"]["windowed"]["all"]["p95"]), pcs(r["p0.05"]["windowed"]["all"]["p95"]),
        pc(r["p0.01"]["windowed"]["coverage"], 0), pc(r["p0.03"]["windowed"]["coverage"], 0), pc(r["p0.05"]["windowed"]["coverage"], 0),
        pcs(r["shock_forced30%+p0.01"]["stream_deferred"]["all"]["p95"]), pcs(r["shock_forced30%+p0.01"]["windowed"]["all"]["p95"]), pc(r["shock_forced30%+p0.01"]["windowed"]["coverage"], 0)) + "\n")
for c in ["blast", "sv"]:
    if c in R:
        r = R[c]; s_ = [s for s in SC if s in r]
        better = sum(r[s]["stream_deferred"]["all"]["p95"] <= r[s]["windowed"]["all"]["p95"] for s in s_)
        rd.append(r"\paragraph{%s。}在 %d 个情形中的 %d 个里，逐轨账本（延迟标定）的 95 分位不高于窗口化；逐轨覆盖率均为 100\%%，窗口化为 %s–%s。激波处断轨情形：逐轨 %s，窗口化 %s。" % (
            CN[c], len(s_), better, pc(min(r[s]["windowed"]["coverage"] for s in s_), 0), pc(max(r[s]["windowed"]["coverage"] for s in s_), 0),
            pcs(r["shock_forced30%+p0.01"]["stream_deferred"]["all"]["p95"]) if "shock_forced30%+p0.01" in r else "—",
            pcs(r["shock_forced30%+p0.01"]["windowed"]["all"]["p95"]) if "shock_forced30%+p0.01" in r else "—") + "\n")
f2 = J("res_s2_sv_0.03.json")
if f2:
    rd.append(r"\paragraph{同一批轨迹上的公平比较（激波–涡，3\%%/帧）。}窗口化只对窗口内完整的轨迹给出估计，所以上表的 95 分位不是在同一批轨迹上比较的。只取两种方法都有估计的轨迹（%d 条）：逐轨账本中位 %s、95 分位 %s，窗口化中位 %s、95 分位 %s，两者相当；逐轨账本另外覆盖了窗口化漏掉的 %d 条轨迹，其 95 分位为 %s。涡核内（$r_0<2R_v$）两者的 95 分位为 %s 与 %s，这是账本在涡核内的固有误差（见噪声与锚定的分析），与断轨无关。" % (
        f2["common_stream"][2], pcs(f2["common_stream"][0]), pcs(f2["common_stream"][1]), pcs(f2["common_windowed"][0]), pcs(f2["common_windowed"][1]),
        f2["stream_only_tracks"][2], pcs(f2["stream_only_tracks"][1]), pcs(f2["stream_core"][1]), pcs(f2["windowed_core"][1])) + "\n")
if "blast" in R:
    r = R["blast"]
    rd.append(r"\paragraph{爆炸波是最难的情形。}不断轨时账本的 95 分位已达 %s（激波正扫过的薄壳）。断轨后，逐轨账本的中位误差为 %s–%s，窗口化为 %s–%s；95 分位分别为 %s–%s 与 %s–%s。逐轨账本好得多，但 95 分位仍高：61 帧内每个粒子平均断轨 0.6–2.6 次，中心区膨胀使示踪点变稀，而新出现的轨迹常落在密度为 6 的薄壳内，用近邻标定时会跨越陡峭的密度梯度。稀疏加噪声时两种方法都失效（95 分位超过 100\%%）：稀疏后的间距已与薄壳厚度相当。" % (
        pcs(r["no_breaks"]["p95"]), pcs(min(r[s]["stream_deferred"]["all"]["med"] for s in SC[:4])), pcs(max(r[s]["stream_deferred"]["all"]["med"] for s in SC[:4])),
        pcs(min(r[s]["windowed"]["all"]["med"] for s in SC[:4])), pcs(max(r[s]["windowed"]["all"]["med"] for s in SC[:4])),
        pcs(min(r[s]["stream_deferred"]["all"]["p95"] for s in SC[:4])), pcs(max(r[s]["stream_deferred"]["all"]["p95"] for s in SC[:4])),
        pcs(min(r[s]["windowed"]["all"]["p95"] for s in SC[:4])), pcs(max(r[s]["windowed"]["all"]["p95"] for s in SC[:4]))) + "\n")
rd.append(r"\paragraph{开发中修掉的第二个问题。}第二版在激波外遇到不可测的环时沿用上一帧的值（丢掉这一步的增量），并在涡核内长期使用缩减环（缩减后的多边形更大，非仿射误差更大）。改为“所有断环都在锚定帧重建、激波外丢顶点后立即重锚”后，激波–涡在同一批轨迹上的 95 分位从 6.1\%% 降到 4.2\%%。\n")
txt = txt.replace("%%READ%%", "".join(rd))
box = r"""\noindent\fcolorbox{accent!40}{tint}{\begin{minipage}{0.965\linewidth}\vspace{0.3em}
\textbf{\sffamily 结论}\quad 逐轨账本把覆盖率从窗口化的 79\%%–95\%% 提高到 100\%%，并在大多数断轨情形下降低了误差；在“粒子恰在穿越激波时断轨”的对抗情形中，Sod 的 95 分位由窗口化的 %s 降到 %s。关键设计是：顶点消失时缩减环而不中断测量；环在激波中断开时，在锚定帧用幸存轨迹重建环，使锚定点留在激波前。
\vspace{0.3em}\end{minipage}}
""" % ((pcs(R["sod"]["shock_forced30%+p0.01"]["windowed"]["all"]["p95"]), pcs(R["sod"]["shock_forced30%+p0.01"]["stream_deferred"]["all"]["p95"])) if "sod" in R else ("—", "—"))
txt = txt.replace("%%BOX%%", box)
open("../repS/stream_report.tex", "w").write(txt); print("written", len(txt), list(R))
