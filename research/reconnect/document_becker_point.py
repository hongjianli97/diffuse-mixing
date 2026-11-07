"""Honest unresolved Becker report, PDF source and read-only next budget gate."""
from pathlib import Path
import shutil
from audit_route_decision import ROOT,read,save,digest


def main():
    out=ROOT/'runs/reconnect-becker-point';r=read(out/'results.json');tc=r['time_control'];ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'numerical-audit.json');rm=read(out/'reference-indicators.json')
    assert audit['graphs_and_full_actions_verified']==28 and audit['prior_formal_manifests_verified']==101
    assert not r['registered_moving_Becker_oracle_discriminator_passed'] and tc['state_time_decrease_resolved'] and not tc['H_time_decrease_resolved']
    for name,source in [('completion.log','completion'),('audit.log','audit')]:shutil.copyfile(f'/tmp/ltn-becker-point-{source}.log',out/name)
    sources={'runs/reconnect-contact-spatial/sha256.json':'parent cumulative scientific seal6576entries',
      'runs/reconnect-contact-spatial/next-gate-spec.json':'immutable original movingBecker registration,oldexecutedfalse retained',
      'research/reconnect/BECKER_BOUNDARY_DURATION_CONTRACT.md':'boundary/time/numeric criteria fixedbefore science',
      'research/reconnect/periodic_voronoi_fv.py':'unchanged Euclidean clip/moments originalperiodicbuilder',
      'research/reconnect/voronoi_time_action.py':'unchanged exponential/fullrawuniformization',
      'research/reconnect/voronoi_time16_core.py':'unchanged oldT.05helper,unitbitwiseequivalence control',
      'research/rebuild/becker.py':'unchanged prescribedBecker flow',
      'research/rebuild/REPORT.md':'originalrebuild centre-definition distinctions,not recoveredv080',
      'runs/rebuild-stage3-space/inputs-cells32.npz':'original32column true mass/c0/reference andsaved uncertaintyvectors',
      'runs/rebuild-stage3-space/reference-n8192-k8192.npz':'saved independentreference,not solvedagain',
      'docs/METHOD.md':'original materialunknown,H,time/spatial/physicalcriteria',
      'docs/CODEX_TASK.md':'originalT4 andthree-densitysourceclosure'}
    save(out/'evidence-index.json',{'sources':[{'path':p,'sha256':digest(ROOT/p),'role':role} for p,role in sources.items()],
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'original_parser_failure_preserved':True})
    future=['runs/reconnect-becker-point/results.json','runs/reconnect-becker-point/reference-indicators.json','runs/reconnect-becker-point/source-column-inputs.npz',
      'runs/reconnect-becker-point/source-reference-n8192-k8192.npz','runs/reconnect-becker-point/density-volume-diagnostics.json',
      'research/reconnect/BECKER_BOUNDARY_DURATION_CONTRACT.md','research/reconnect/becker_point_protocol.json','research/reconnect/becker_point_core.py','research/reconnect/t1_uniform_core.py']
    save(out/'next-gate-spec.json',{'already_executed':False,'type':'read_only_saved_moving_Becker_H_reference_budget_review',
      'reason':'state time-decrease resolved butHdecrease belowunchangedoriginalHpositiveindicator; originalstoprule retains unresolvedandforbids automaticexpansion',
      'sources':[{'path':p,'sha256':digest(ROOT/p)} for p in future],
      'scope':'savedfiles only; separate reference-space/time/projection andmachine/tail/fullsequence/bridge contributions; quantify originalHindicator dominantterms andwhethernecessary independentreferencechecking could resolve it',
      'required_outputs':['source-linked component table','unchanged originalHformula anddecision preserved','separate pointdensity/materialvolume diagnostics without causal attribution','concrete next necessary work only if justified before new inputs'],
      'forbidden':'no candidateN/K/row/centre/physicalparameter change,no newPDEorrefine solve thisreview,no tighterHformula,thresholdretune,signedcancellation,clipping/normalization/F_Cfit orRichardsonstate',
      'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_PDE_solves':0,'new_software_checks':0,'old_candidate_reruns':0,'recorded_checks_total':500,
      'main_wall_seconds_cap':120,'main_peak_RSS_MiB_cap':1024,'original_unresolved_decision_must_remain':True,
      'stage3_physical_acceptance':'incomplete','moving_shock_T4_closed':False,'all_three_density_sources_closed':False,'fluid_orc_enabled':False,'kappa_computed':False})
    obs=r['observations'];table='\n'.join(f"| {v['K']} | {v['native_error_relative']:.12g} | {v['column_observation_error_relative']:.12g} | {v['H']:.15g} | {v['raw_mass_change']:.12g} | {v['primary_control_difference_relative']:.12g} |" for v in obs)
    gap=tc['H4_8']-tc['H8_16'];recovery=read(out/'completion-lock.json');npre=len(read(out/'pre-completion-files-sha256.json'))
    report=f'''# 移动Becker点FV oracle：外边界/显式时长接口与未决判别

2026-11-04（Asia/Shanghai）。执行[唯一登记项](../../runs/reconnect-contact-spatial/next-gate-spec.json)，先固定[边界/时长契约](BECKER_BOUNDARY_DURATION_CONTRACT.md)、[实现协议](becker_point_protocol.json)及四项检查，再加载新科学输入。

**决定：{r['comparison_decision']}。本例未达到预定全部通过条件。** 状态时间误差下降可分辨，H下降尚未超过原固定正项指标。独立几何/传播审计通过不等于物理判别通过。按停止规则保留未决，不增加N/K、改代表中心、削减H指标或调整阈值制造通过。Stage3 incomplete，fluid ORC关闭，κ未计算。

## 固定物理问题与新接口

δ=.08、原Dfloat.02、T=1/10、原32物质列，单位横截面复制8行（256节点）。K4/8/16共28新图、3实际及3完整独立路径；没有新PDE求解或旧候选重跑。点沿原固定质量标签中点μc的轨迹X(μc,t)，不强迫成真实物质格当前几何中心。只在y单位周期，左右最外物质端点零组分通量；激波不是壁，不删除跨激波正面，x不周期。

新增可选bounded_periodic_voronoi分享原精确欧氏polygon clip/moments原语，旧periodic构造器不改。近邻KD树仅作初围界；随后对全部点最近y镜像作精确world半径检查。每个最终单元满足4R_circ²<Ly²/4，故距离至少Ly/2的非最近镜像不能切割；不满足则拒绝。外边界面单列零Neumann通量，内面互反正g装配一次。

普通内面g=(|f|/d)/[1/(2ρiD)+1/(2ρjD)]。构造器只收点、ID、给定点ρ、D、外界两端点及y周期，不收真实内部物质面、m/c0/参考或F/C。全部28图先封存，再加载真实质量、c0、参考或旧状态；质量标签仅用于给定点轨迹。真实物质单元平均仍为m_i=∫ρdV、c̄_i=m_i⁻¹∫ρc dV，不能把点Voronoi体积或m/点ρ替代真实体积。

旧ordered_path硬编码T=.05；新增可选ordered_path_duration仅把精确dt换为T/K，复用原指数、完整统一化、机器/1e−16尾项、H与代数检查。在T=.05非对易夹具下所有状态、半步和元数据逐位匹配旧路径；T=.1确实使用dt=.1/K，D0严格恒等。

## 原始质量平均与保存参考

原32列质量/c0逐字节沿用；横向每行m/8、相同c0，保存聚合桥而不修补。终时只复用原8192空间/8192时间独立参考投影至32列的均值；不把旧真物质面FV当成当前点后端已经通过，也不使用未取回v080数字作比较。误差分母固定原32列初始质量波动范数{r['original_column_initial_fluctuation_norm']:.17g}。

保存参考空间差范数{rm['components_absolute']['saved_reference_space_difference']:.12g}，时间差{rm['components_absolute']['saved_reference_time_difference']:.12g}，合计{rm['reference_space_time_indicator_absolute']:.12g}；它们是经验细化指标，不是认证误差界。独立审计用精确Fraction重叠积分核对原保存投影，未替换原参考。

| K | 原生相对PDE误差 | 回原32列观测误差 | H | 原始质量漂移 | 主/完整独立序列差/σ0 |
|---|---:|---:|---:|---:|---:|
{table}

原参考H={obs[-1]['reference_H']:.15g}。K16实际误差{obs[-1]['native_error_relative']:.12g}是一个网格的测量，不创造绝对PDE精度阈值或空间阶声明。

## 预定判别为何未通过

状态4/8差/σ0={tc['state4_8_relative']:.12g}，8/16差/σ0={tc['state8_16_relative']:.12g}，下降超过固定{tc['fixed_state_decrease_indicator_relative']:.12g}，故状态时间控制分辨。

H4/8差={tc['H4_8']:.12g}，H8/16差={tc['H8_16']:.12g}，两者下降仅{gap:.12g}，低于原H指标{tc['fixed_H_decrease_indicator']:.12g}，故H未分辨。原正项预算含两倍三路径机器/尾项/完整独立差、两倍已存参考空间/时间指标及原始初值/横向观测桥；H使用原h_indicator，未因有符号误差抵消或更紧后验公式而改判。

参考项在纯路径差中本可相消，但原登记明确保守保留；不能在看到结果后删除它以制造通过。参考空间差主导现预算不等于全部物理误差由参考造成，点密度/真实物质平均密度与点Voronoi/真实物质体积也不等同，另存逐时刻诊断且不作因果分摊。

![原保存状态、误差、H未决与密度/体积诊断](../../runs/reconnect-becker-point/becker-point.png)

## 首次诊断故障与追加补全

主科学计算在{ex['elapsed_main_to_diagnostic_failure_seconds']:.3f}秒/{ex['peak_main_RSS_MiB']:.2f}MiB内保存全部图、路径和数值结果；随后体积诊断把有理数字符串直接float转换，漏了Fraction解析而退出1。冻结主源码、原数值JSON与失败日志保留，不能称主流程首次成功。

追加becker_point_completion.py仅在诊断输出中用float(Fraction(value))，从已存数据补全诊断/状态/执行记录，不重建图或重算传播，不修改指标/阈值。{npre}个补全前文件哈希保持；补全{ex['elapsed_completion_seconds']:.3f}秒。正式重放入口replay_becker_point.py重放科学路径并验证诊断及原失败保留。

四项新软件检查首次通过（旧496未重跑，记录500）。完整独立审计首次{ar['elapsed_audit_seconds']:.3f}秒/{ar['peak_auditor_RSS_MiB']:.2f}MiB通过：28图全部单元外界/半平面/镜像/互反面，精确率、最小尾项、原权重与完整独立序列逐位核对，库存/极值/方差/耗散和原H指标验证。782原文件、15附件、101份既有正式科学清单保持，4份历史诊断既有差异保留。

## 下一项与下载

下一项只登记[保存H/参考预算的只读分量复核](../../runs/reconnect-becker-point/next-gate-spec.json)，确认参考空间/时间/投影与数值/桥项来源，保留原H公式和未决结论，判断是否有必要在新输入前另行制定独立参考核验。该复核不求PDE、不新建图、不运行候选、不加N/K、不改H/阈值；原停止规则继续生效。

[PDF报告](../../exports/LTN_becker_point_report_20261104.pdf) · [代码与结果ZIP](../../exports/LTN_becker_point_code_results_20261104.zip)。ZIP含代码/文档、782原始文件、本例全图/路径/参考/审计与失败记录；不含约39GB全工作区或15附件，旧v080仍未取回。原README/METHOD/CODEX_TASK/PHASE和旧代码/数据/报告不改，只更新状态与备份交接；PDF/GitHub备份按用户已授权范围，不发布环境或版本。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/replay_becker_point.py --directory runs/reconnect-becker-point
```

新解压/克隆的逐位重放依赖平台与库构建；不一致时保留诊断，不能放宽阈值。下载CRC/SHA及重放以交付JSON为准，完整历史保全审计仍需原工作区。
'''
    with (ROOT/'research/reconnect/BECKER_POINT_COMPARISON.md').open('x') as f:f.write(report)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write(f'''\n\n102. [移动Becker点FV oracle：外边界/显式时长接口与未决判别](../research/reconnect/BECKER_POINT_COMPARISON.md)（2026-11-04）。\n\n执行原δ.08/D.02/T.1/32列×8行、K4/8/16单例；新增可选有界x/周期y精确点几何与T显式包装，旧构造器/指数/统一化/H代码不改，T.05夹具逐位匹配旧路径。28图全部先封存再读原m/c0/参考；3实际/完整独立路径，参考不新解。原生K16误差{obs[-1]['native_error_relative']:.12g}。状态时间控制已分辨，H下降{gap:.12g}未超过原指标{tc['fixed_H_decrease_indicator']:.12g}，决定{r['comparison_decision']}，不宣告通过，不扩N/K或改H/阈值。\n\n4项新软件检查及完整独立审计首次通过，记录500，旧496不重跑。主流程在体积诊断有理数解析时失败，全部科学结果已存；原失败/源码/JSON保留，追加补全只修解析，{npre}个原文件保持、不重算传播。原782文件、15附件、101旧正式清单保持。下一项只登记保存H/参考预算分量只读复核，0新图/候选/PDE/软件检查。Stage3 incomplete，fluid ORC关闭，κ未算；按授权提供PDF/代码ZIP与GitHub备份，不发布环境/版本。\n''')
    make_tex(out,r,rm,ex,ar,npre)


def make_tex(out,r,rm,ex,ar,npre):
    tc=r['time_control'];rows='\n'.join(f"{v['K']} & {v['native_error_relative']:.8g} & {v['H']:.12f} & {v['raw_mass_change']:.6g} \\\\" for v in r['observations'])
    tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setsansfont{Noto Sans CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=22mm]{geometry}\usepackage{amsmath,amssymb,booktabs,graphicx,hyperref,fancyhdr}
\hypersetup{pdftitle={移动Becker点FV oracle：未决判别与接口验证},pdfauthor={diffuse-mixing research},colorlinks=true,urlcolor=blue}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\setlength{\headheight}{16pt}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-04}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries LTN扩散与掺混}\par\vspace{3mm}{\Large 移动Becker点FV oracle：未决判别}\par\vspace{3mm}2026年11月4日（Asia/Shanghai）\end{center}
\section*{1. 本例未达到预定全部通过条件}
\textbf{状态时间误差下降已分辨，H时间差下降尚未超过原固定指标。几何和传播审计通过，不能替代物理判别通过。Stage3仍incomplete，fluid ORC关闭，κ未算。}

执行上轮唯一登记项：$\delta=.08,D=.02,T=1/10$，原32物质列沿单位横截面复制8行，共256节点；K4/8/16、28新图、3实际及3完整独立路径，不作参数扫描。复用已存8192空间/时间独立参考，不新解PDE或重跑旧候选。

\begin{center}\begin{tabular}{rrrr}\toprule
K & 终时相对PDE误差 & 实际H & 原始质量漂移\\\midrule
@ROWS@
\bottomrule\end{tabular}\end{center}
原参考$H=0.018932968708588$。分母为原32列初始质量波动范数$\sigma_0=0.3373623336859549$，二维原生与回原32列的物质观测分别记录。单N误差没有独立空间阶或新绝对PDE验收阈值，不据4.13e-4宣告Stage3通过。

\begin{center}\begin{tabular}{p{53mm}rr}\toprule
时间判别量 & 状态范数$/\sigma_0$ & H\\\midrule
4/8端点差 & 1.59124812e-4 & 6.46754987e-7\\
8/16端点差 & 3.96940149e-5 & 7.04753874e-8\\
差的下降 & 1.19430797e-4 & 5.76279600e-7\\
固定下降指标 & 4.29078238e-7 & 4.85814409e-6\\
决定 & 已分辨 & 未分辨\\\bottomrule
\end{tabular}\end{center}
\textbf{按停止规则保留未决：不增加K32/N/横向行数、不换代表点中心、不调整H指标或阈值。} 下一项仅为保存预算的只读复核，不能在看到数值后移除参考项以制造通过。

此前给定静止接触oracle的N16/32/64空间控制保持原通过记录；该证据不能自动覆盖移动外边界与激波。本轮四项新增软件检查首次通过，累计记录500项（旧496未重跑）；这不等于500个物理算例，也不是旧v080/246的恢复。
\newpage
\section*{2. 固定物质状态、边界与物理时长}
\[
m_i=\int_{K_i(t)}\rho\,dV,\qquad \bar c_i=\frac1{m_i}\int_{K_i(t)}\rho c\,dV.
\]
点身份为固定质量标签区间中点$\mu_c$，$x=X(\mu_c,t)$，$y=(j+.5)/8$；不在每个时刻把点改成真实物质格几何中点。原32列m/c0逐字节沿用，横向每行m/8、相同c0；原聚合舍入桥保留、不修补。

外域为$[X(\mu_0,t),X(\mu_M,t)]\times[0,1)$。\textbf{只有y周期，最外x物质端点零组分通量；激波不是壁，x不是周期。} 新增可选点Voronoi适配器分享原精确欧氏polygon clip/moments原语，旧periodic构造器不改。此clip只构造几何，不裁剪系数或状态。

KD树近点只提供初围界，随后检查所有点的可能最近y镜像约束；每个最终单元满足$4R_{circ}^2<L_y^2/4$，排除距离至少$L_y/2$的非最近镜像。独立审计从外边界直线和点间双中垂线重建全部顶点/面积/内面与外面，所有单元检查半平面。外面Neumann通量0，不装配虚构外邻边；跨激波内面不删除。
\[
g_{ij}=\frac{|f_{ij}|/d_{ij}}{1/(2\rho_iD)+1/(2\rho_jD)}.
\]
构造器只收world点、ID、给定点密度、D、外界两端点和y周期，不收m/c0/参考、真实内部物质面或F/C。全部28图先封存，再读真实质量/初场/参考或旧状态。真实物质体积与点Voronoi体积、m/点ρ是不同定义；诊断差异保留，不分配因果误差。

\subsection*{显式时长包装}
旧ordered\_path硬编码T=.05。新增可选ordered\_path\_duration仅将精确$\Delta t$改为$T/K$并验证T正；指数、完整统一化、机器/最小$10^{-16}$尾项、H及代数尺度均保持。

在T=.05非对易夹具下，新旧全部端点/半步/元数据逐位一致。T=.1确实使用$\Delta t=.1/K$，D=0严格恒等。四类软件检查还覆盖有界x/周期y互反面/零通量/非法输入、原始列积分、完整序列损坏拒绝。原指数/统一化及旧时序源文件字节不改。

本例只使用oracle点密度，不等于kernel/ledger已验收。没有归一化面积、ρ、m、权重、行或浓度，没有F/C拟合；旧质量平均阻力实验通量不取代普通两半阻力。
\newpage
\section*{3. 保存状态、误差与未决H}
\begin{center}\includegraphics[width=\textwidth]{becker-point.png}\end{center}
终时K16相对原PDE误差约$4.13042910\times10^{-4}$。状态4/8到8/16差下降可分辨；H下降约$5.76\times10^{-7}$，小于固定$4.86\times10^{-6}$，故保留未决。不能把“程序/代数审计通过”改写成“物理判别通过”。

原指标逐项正加：两倍三路径机器/尾项/完整独立序列差，两倍已存参考空间/时间差，以及初始/横向观测桥；H仍用原h\_indicator。保存参考空间差质量范数$7.22461791\times10^{-8}$，时间差$5.35431503\times10^{-11}$，空间差主导该参考指标。

参考在纯路径差中本可相消，但原登记明确保守保留；不能事后删项或换更紧H公式来改判。经验参考细化差不是认证总误差界；其主导预算也不证明所有物理误差来自参考。点密度/真实物质平均ρ、点Voronoi/真实物质体积的不同近似另存逐时刻诊断，不按误差排序推因果。

独立核验原保存8192参考的重叠积分与原32列参考，保留原值；没有新PDE求解、浓度外推或回归拟合。图中坐标是固定质量标签，未知量仍为真实物质单元平均。
\newpage
\section*{4. 审计、诊断故障与恢复}
主科学计算在@MAIN@秒／@RSS@ MiB保存全部图、三路径和数值结果。随后体积诊断把有理数字符串直接转float，漏了Fraction解析而退出1。\textbf{原失败日志、冻结主源码、原JSON和@NPRE@个补全前文件保持，不能称主流程首次成功。}

追加becker\_point\_completion仅在诊断中采用float(Fraction(value))，从保存数据补全密度/体积诊断、状态和执行记录，@COMP@秒完成。没有重建图、重算传播、修改判别指标或阈值。正式重放入口另验证补全诊断及原失败文件。

完整独立审计首次@AUDIT@秒／@ARSS@ MiB通过。28图的全部外界/顶点/半平面/镜像/互反面/普通阻力逐项核验；精确率、最小Poisson尾项、原权重与完整独立序列逐位一致。库存、极值、方差、共享面耗散及原H公式按原尺度检查，不调平质量漂移。

782原文件、15附件、101份先前正式科学清单保持；四份历史诊断既有差异保留。原README/METHOD/CODEX\_TASK/PHASE和旧代码/数据/报告不改，只更新研究状态与备份交接文件。PDF、代码ZIP与GitHub备份延续用户授权，不是环境/软件版本发布。

\subsection*{下一项仅作保存预算只读复核}
保持原未决结论，核对参考空间/时间/投影、机器/尾项/序列和桥项的来源与占比，判断是否需要在新输入前另行制定独立参考核验。\textbf{该复核0新图、0候选运行、0新PDE求解、0新软件检查。} 不自动加N/K、改变点中心或H阈值，不以更紧后验公式追认本例通过。

移动激波T4完整控制与oracle/kernel/ledger三密度来源仍未闭合，Stage3 incomplete，fluid ORC保持关闭，κ未计算。旧v080数字不可当作当前测量或相对改善依据。

\subsection*{下载与重放}
ZIP含研究代码/文档、782原文件、本例全部图/状态/参考/审计及失败记录；不含约39GB完整工作区或15附件。原工作区另保留完整历史。可在新解压目录重放：
\begin{quote}\small\begin{verbatim}
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 python \
research/reconnect/replay_becker_point.py \
--directory runs/reconnect-becker-point
\end{verbatim}\end{quote}
重放从封存图重建作用，核对全部状态/半步/CSR/原权重/物质积分/原判别与诊断，不重建几何或求PDE。逐位一致受平台/数值库构建影响，不一致应保留诊断，不放宽物理阈值。CRC/SHA、解压与GitHub恢复验证以交付收据为准。
\end{document}
'''
    for key,value in {'ROWS':rows,'MAIN':f'{ex["elapsed_main_to_diagnostic_failure_seconds"]:.3f}','RSS':f'{ex["peak_main_RSS_MiB"]:.2f}',
      'NPRE':str(npre),'COMP':f'{ex["elapsed_completion_seconds"]:.3f}','AUDIT':f'{ar["elapsed_audit_seconds"]:.3f}','ARSS':f'{ar["peak_auditor_RSS_MiB"]:.2f}'}.items():tex=tex.replace('@'+key+'@',value)
    with (out/'LTN_becker_point_report_20261104.tex').open('x') as f:f.write(tex)


if __name__=='__main__':main()
