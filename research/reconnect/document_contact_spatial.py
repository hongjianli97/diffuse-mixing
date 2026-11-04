"""Additive three-grid report/PDF source and one moving-Becker interface gate."""
from pathlib import Path
import shutil
from audit_route_decision import ROOT,read,save,digest


def main():
    out=ROOT/'runs/reconnect-contact-spatial';r=read(out/'results.json');audit=read(out/'numerical-audit.json');ex=read(out/'execution.json');ar=read(out/'audit-execution.json')
    assert audit['both_new_exact_world_geometries_rawsequences_and17Pade_states_verified'] and audit['prior_formal_manifests_verified']==100
    shutil.copyfile('/tmp/ltn-contact-spatial-main.log',out/'study.log');shutil.copyfile('/tmp/ltn-contact-spatial-audit.log',out/'audit.log')
    sources={'runs/reconnect-contact-oracle/sha256.json':'parent cumulative seal6459entries','runs/reconnect-contact-oracle/next-gate-spec.json':'immutable registered three-grid control,old executedfalse retained',
      'runs/reconnect-contact-oracle/results.json':'oldN32decision andstates reused withoutaction','research/reconnect/CONTACT_ORACLE_COMPARISON.md':'previous N32 report andfailure recovery retained',
      'research/reconnect/CONTACT_PERIODIC_SPECTRAL_BASIS.md':'original weighted transmission referencebasis andtailproof',
      'research/reconnect/periodic_voronoi_fv.py':'unchanged pointgeometry/ordinary halfresistance','research/reconnect/voronoi_time_action.py':'unchanged sparseexponential/rawuniformization',
      'research/reconnect/voronoi_time16_core.py':'unchanged T.05 chronology andH/I','research/reconnect/contact_oracle_reporting.py':'original NumPy totaltype restoration reused unchanged',
      'docs/METHOD.md':'original fixedmaterial,threegrid,physicalcriteria','docs/CODEX_TASK.md':'original T4 anddensitysource closure',
      'runs/rebuild-stage3-space/inputs-cells32.npz':'nextgate: original rebuiltBecker32 materialmass/c0/projectedreference,notv080',
      'runs/rebuild-stage3-space/reference-n8192-k8192.npz':'nextgate: existing independent fineBeckerreference,not solvedagain',
      'runs/rebuild-stage3-space/config.json':'nextgate: original physicaldelta/D/T andboundaryscope',
      'runs/rebuild-stage3-space/results.json':'nextgate: existingreference uncertainty andlimitedphysicalFVcomparison',
      'research/rebuild/becker.py':'nextgate: original prescribedflow andx(mu,t)',
      'research/rebuild/REPORT.md':'nextgate: oldgeometry-centre/mass-centre distinction remains',
      'research/reconnect/T3_COVERAGE_REVIEW.md':'originalpoint-backendT4gapnotclosedbyoldtruefaceFV'}
    save(out/'evidence-index.json',{'sources':[{'path':p,'sha256':digest(ROOT/p),'role':role} for p,role in sources.items()],
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'N32_original_decision_unchanged':True})
    passed=r['registered_three_space_level_contact_oracle_control_passed']
    next_sources=['runs/rebuild-stage3-space/inputs-cells32.npz','runs/rebuild-stage3-space/reference-n8192-k8192.npz','runs/rebuild-stage3-space/config.json',
      'runs/rebuild-stage3-space/results.json','research/rebuild/becker.py','research/rebuild/REPORT.md','research/reconnect/periodic_voronoi_fv.py',
      'research/reconnect/voronoi_time_action.py','research/reconnect/voronoi_time16_core.py']
    save(out/'next-gate-spec.json',{'already_executed':False,
      'type':'one_moving_Becker_point_boundary_duration_adapter_contract_and_oracle_discriminator' if passed else 'read_only_saved_contact_spatial_budget_review',
      'reason':'static contact samebackend threegrid control passed; next originalgap is moving shock with correctly declared nonperiodic materialboundary andphysical duration',
      'scope':'one rebuiltBecker delta.08,Doriginalfloat.02,T.1,32fixedmaterialcolumns,8uniform transversecopies/unitheight; oracle pointdensity only,no density/flow/modelscan',
      'staged_dependency':'first write exact bounded-x/periodic-y outerboundary anddurationT contract; verify adapter/unit equivalence at originalT.05; freeze foursoftwarechecks andprotocolbefore anynewscientificpoint/graph/reference/state load',
      'sources':[{'path':p,'sha256':digest(ROOT/p)} for p in next_sources],
      'original_material_columns':32,'transverse_rows':8,'nodes':256,'delta':.08,'D':.02,'T_exact':'1/10','K_values':[4,8,16],
      'point_labels':'fixed mass-labelmidpoints oforiginal32muintervals;worldx=originalBecker.x_mu(mu_mid,t),y=(j+.5)/8; never forcepoints tocurrentcellgeometriccentres',
      'boundary':'givenmovingoutermaterialendpoints x_mu(mu0,t),x_mu(muM,t),zero scalarflux there; periodicunit-y only,no periodicx,shock isnotwall',
      'geometry':'new optional exactbounded-x/periodic-y pointVoronoi adapter sharing originalEuclidean clipping/moments/ordinaryhalfresistance; old periodicbuilder unchanged; never giveinternalmaterialfaces/scalar/reference/F_C to builder',
      'time':'new optional duration-explicit chronological wrapper sharing unchanged sparseexponential/fulluniformization andoriginalH; atT.05 fixture mustmatcholdordered_path bitwise; do not silentlyuseoldhardcodedT.05 atT.1',
      'coefficient_input_order':'puregivenmasslabels/points/givenrho/D/outerboundary only; all28graphs sealedbefore actualfixedmass/c0/reference oroldstateload; internalmasslabelintervals generatepointlabels only,not materialfaces passedtobuilder',
      'material_state':'reuse original32column mass/c0; extrudeeachrow withmass/8,samec0; preserveunmodifiedcolumnvalues andnormalizationbridges; truephysicalmaterialmeans notpointVoronoimeans',
      'reference':'reuse existing saved8192space/8192time projected32column reference atT.1 plus savedspace/timeuncertainty; no newPDEsolve; do not replace witholdtruefaceFV solution',
      'acceptance':'exact independent boundary/bisector/face/ordinaryresistance audits; originalmass/extrema/variance/sign/residualguards; fullindependentsequences andoriginalH; actual4/8to8/16endpointstate/Htimedifference decrease mustexceed fixedpositive numeric/reference indicators; report actualPDEerror anddensitypoint-vsfinitecell diagnostics withoutfittingphysicalthreshold',
      'physical_scope':'onepointgeometryoracle moving-shockdiscriminator only; failure/unresolved retained,not fullT4or densityadmission; no claimnewspaceconvergence',
      'new_candidate_graphs':28 if passed else 0,'new_candidate_transient_runs':3 if passed else 0,'new_complete_independent_uniformization_sequences':3 if passed else 0,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_spectral_fits':0,'new_software_checks_planned':4 if passed else 0,'recorded_checks_total_if_all_pass':500,
      'check_categories':['boundedx/periodicy Euclideanface reciprocity,noflux andillegalinput rejection','explicitT chronology equivalenceandtimeorder atoldT.05','unchanged32column materialmeans/ref reuseandtransverseintegrals','rawfullsequence/originalH/time/PDEindicator adversecase rejection'],
      'main_wall_seconds_cap':900,'main_peak_RSS_MiB_cap':1024,'tail_per_interval_exact':'1/10000000000000000','original_algebra_tolerance':1e-10,'norm_estimator_seed':20261101,
      'stop':'preservefailure/unresolved; no automaticmoreN/K/rows/delta/centreconvention/density/modelscan,positivefacedeletion,clipping,normalization,F/Cfit,thresholdorHchanges',
      'new_Becker_points_graphs_states_or_reference_evaluations_generated_this_turn':False,'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    obs=r['per_grid_observations'];rows='\n'.join(f"| {N} | {v['time']:g} | {v['native_error_native_denominator']:.12g} | {v['common_error']:.12g} | {v['native_H']:.15g} | {v['native_reference_H']:.15g} |" for N,values in obs.items() for v in values if v['time']>0)
    comparisons='\n'.join(f"| {v['coarse_N']}→{v['fine_N']} | {v['time']:g} | {name} | {v['gains'][name]:.12g} | {v['fixed_indicators'][name]:.12g} | {v['decisions'][name]} |" for v in r['adjacent_spatial_comparisons'] if v['time']>0 for name in ('native','common','native_H','common_H'))
    errors=[obs[str(N)][-1]['common_error'] for N in (16,32,64)];ratios=[errors[0]/errors[1],errors[1]/errors[2]]
    text=f'''# 同一普通FV周期接触oracle：三档物质空间对照

2026-11-04（Asia/Shanghai）。执行[上轮登记项](../../runs/reconnect-contact-oracle/next-gate-spec.json)，[固定协议](contact_spatial_protocol.json)在新科学输入前锁定。

**决定：{r['comparison_decision']}。** 同一静止周期接触N16/32/64，原生/共同浓度及原生/共同H的两次细化改善在两个时刻均超过预定正项指标。这里只完成给定接触oracle基准的三档控制；移动激波与三密度来源尚未闭合，Stage3 incomplete，fluid ORC关闭，κ未算。

## 同一实现与真实物质状态

ρ1/4分相、单位周期方形、原Dfloat.02、T1/20、K16固定。只补N16/N64两个静态点图、两条实际及两条完整独立路径；N32整条路径只加载，不重算。两个新几何及系数全部先封存，再解析质量、c0、参考或旧状态。每档真实m=ρ/N²，c0从同一连续初场逐格质量平均新推导，复用原128模态90位连续投影系数，不插值旧均值或伪称恢复v080。

原periodic_voronoi_fv、普通两半阻力、稀疏指数/完整统一化、K16时序及H公式全部字节不改。低/高相与界面g分别D、4D、(8/5)D，封存后逐面核对；独立重建顶点、双中垂线、互反面、面积和固定全点探针。图构造器不收真实物质面、F/C或参考。没有裁剪、正面删除、状态/质量/密度/行/权重归一化。

各档仍用Decimal100/120位、独立mpmath80/100位有限均值参考，两个正时刻分别核算原128模态尾界，t0直接积分连续公式。已有Cartesian全模态半离散控制、原始负零特征值及全部舍入模态保持，独立Padé核对两新网格各17端点。静态L没有中点冻结误差，不生成虚假的4/8/16时间阶或重复图；K16实际作用误差由完整独立序列和Cartesian控制隔离。

## 固定共同物质观测与误差

共同观测以原派生N16质量为容量，直接加子单元ρc积分后除该容量，不修补初值或参考。原初波动分母固定 **{r['common_initial_fluctuation_norm']:.17g}**。所有子质量容量因子、初态桥与参考投影桥保留；原生误差另列自身分母，不混同不同网格H或不同区域的误差。

| N | t | 原生误差/自身σ0 | 共同误差/N16σ0 | 原生H | 原生参考H |
|---|---:|---:|---:|---:|---:|
{rows}

终时共同误差为{errors[0]:.12g}、{errors[1]:.12g}、{errors[2]:.12g}，相邻比{ratios[0]:.8g}、{ratios[1]:.8g}。这是本例的三档观察，不拟合或认证一般收敛阶，不外推浓度。

## 预定正项预算

两相邻档分别在t=.025/.05要求四种改善都可分辨。浓度预算逐项正加原作用机器/尾项、观测完整序列差、Cartesian独立谱/转换/舍入/时标控制、原评价指标、参考序列化/截断与初态桥；共同观测另乘容量范数因子并加入参考/初态投影桥。H对每档数值项调用原h_indicator，再加直接绝对H参考/初态投影桥。旧N32单例预算继续通过原NumPy类型恢复入口精确复现；新空间比较将具名分量转为原值Pythonfloat后按事先固定sum求和，不修改旧结果或H公式。

| 相邻档 | t | 量 | 改善 | 固定正项指标 | 决定 |
|---|---:|---|---:|---:|---|
{comparisons}

![三空间档与固定指标](../../runs/reconnect-contact-spatial/contact-spatial.png)

## 检查、审计、保全

三项新增软件检查首次通过：多N界面g/有限单元均值与损坏拒绝；非等子质量直接积分/保留非零桥；四种正项分类、完整序列扰坏及原生高频误差被共同聚合掩盖的拒绝。旧493项未重跑，累计记录496。首次主计算{ex['elapsed_main_seconds']:.3f}秒/{ex['peak_RSS_MiB']:.2f}MiB，满足900秒/1024MiB；完整独立审计首次{ar['elapsed_audit_seconds']:.3f}秒/{ar['peak_auditor_RSS_MiB']:.2f}MiB通过。

精确率、最小1e−16 Poisson尾项、原权重及完整独立序列逐位核对，原始库存/极值/方差/耗散保持。共同物质审计不调用生产聚合，直接核验{audit['common_material_audit']['all_direct_component_integral_snapshots_verified']}份快照、{audit['common_material_audit']['fixed_parent_exact_integral_probes_verified']}个精确Fraction母单元积分探针；最大fsum差{audit['common_material_audit']['maximum_fsum_vs_saved_mean_difference']:.12g}。

782原文件、15附件、100份既有正式科学清单、N32原判别和其失败恢复记录均保持，4份历史诊断既有差异保留。原README/METHOD/CODEX_TASK/PHASE、旧代码/数据/报告不改，只更新RESEARCH_STATUS及备份交接文件。PDF与代码结果ZIP延续用户明确要求；GitHub备份延续已授权范围，不是环境发布或软件版本发布。

## 下一项：移动Becker的物理接口

静止周期接触基准已完成本次三档控制，不继续N128或角度/密度矩阵扫描。下一项只[登记同一移动Becker oracle判别](../../runs/reconnect-contact-spatial/next-gate-spec.json)，先写两个具体接口契约：给定移动外物质端点的x零通量/y周期点Voronoi，以及T显式的时序包装。原周期构造器与指数/统一化实现不改；旧时序helper硬编码T=.05，不能直接冒充T=.1，新增可选包装必须先在T=.05夹具逐位匹配旧路径。

计划δ=.08、D=.02、T=.1、原32物质列，沿单位横向复制8行，点沿固定μ中点轨迹，不强迫成为真实物质格几何中心。只给构图器移动最外端点，不给真实内部物质面；不把激波当壁或把x做周期。复用已保存原32列质量/c0及8192空间/时间参考，不新解PDE，也不把原真物质面FV当作当前点后端已经通过。本轮只绑定这些来源SHA，没有生成新Becker点、图、状态或参考评价。若这一步失败/未分辨，保留并停止，不调代表点或阈值制造通过。

之后仍需同后端完整T4和oracle/kernel/ledger三密度闭合；当前只完成给定静止接触oracle，不据此启用fluid ORC。

## 下载与重放

[PDF报告](../../exports/LTN_contact_spatial_report_20261104.pdf) · [代码与结果ZIP](../../exports/LTN_contact_spatial_code_results_20261104.zip)。ZIP包含研究代码/文档、782原始文件、本轮三档可重放结果及Becker下一步所需的精选保存来源；不含约39GB完整工作区、全部旧运行/导出副本或15原附件。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/contact_spatial_study.py --replay runs/reconnect-contact-spatial
```

只重放两条新路径，N32仅加载；不重建几何、解PDE或运行旧候选。逐位重放依赖平台/库构建；完整历史保全审计需原工作区。下载CRC/SHA和新解压重放以交付JSON为准。
'''
    with (ROOT/'research/reconnect/CONTACT_SPATIAL_COMPARISON.md').open('x') as f:f.write(text)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write(f'''\n\n101. [同一周期接触oracle三档物质空间对照](../research/reconnect/CONTACT_SPATIAL_COMPARISON.md)（2026-11-04）。\n\n执行原N16/32/64登记项，只补N16/N64两张静态点图及K16实际/完整独立序列；两图先封存再读m/c0/参考/旧状态，N32仅加载原结果保持。决定{r['comparison_decision']}，原生/共同浓度与原生/共同H两相邻档两个时刻改善均超过原正项指标。终时共同误差{errors[0]:.12g}、{errors[1]:.12g}、{errors[2]:.12g}，非一般阶证明。全部原几何/普通阻力/时间/H源码字节不改，128模态90位系数复用且新有限均值按100/120位评价，独立80/100位及新网格各17端点Padé审计通过。\n\n三项新检查、主计算和完整审计首次通过；旧493未重跑，记录496。782原文件、15附件、100份旧正式清单和原N32失败恢复保持。下一项只登记原δ.08/D.02/T.1/32列Becker oracle接口与最小判别：先写移动外边界x零通量/y周期、T显式时序包装并在T.05夹具匹配旧时序，再按封存协议构图/传播，复用原质量/c0/参考；本轮没有新Becker点/图/状态/参考评价。Stage3仍incomplete，fluid ORC关闭，三密度尚未闭合。按已授权生成PDF/代码结果ZIP和GitHub备份，不发布环境/版本。\n''')
    make_tex(out,r,audit,ex,ar)


def make_tex(out,r,audit,ex,ar):
    obs=r['per_grid_observations'];rows='\n'.join(f"{N} & {v['time']:g} & {v['native_error_native_denominator']:.7g} & {v['common_error']:.7g} & {abs(v['common_H']-v['common_reference_H']):.7g} \\\\" for N,values in obs.items() for v in values if v['time']>0)
    budgetrows='\n'.join(f"{v['coarse_N']}→{v['fine_N']} & {v['time']:g} & {v['gains']['common']:.6g} & {v['fixed_indicators']['common']:.6g} \\\\" for v in r['adjacent_spatial_comparisons'] if v['time']>0)
    tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setsansfont{Noto Sans CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=22mm]{geometry}\usepackage{amsmath,amssymb,booktabs,graphicx,hyperref,fancyhdr}
\hypersetup{pdftitle={LTN周期接触oracle三档物质空间对照},pdfauthor={diffuse-mixing research},colorlinks=true,urlcolor=blue}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\setlength{\headheight}{16pt}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-04}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries LTN扩散与掺混}\par\vspace{3mm}{\Large 周期接触oracle三档物质空间对照}\par\vspace{3mm}2026年11月4日（Asia/Shanghai）\end{center}
\section*{1. 本轮结论与固定问题}
\textbf{N16/32/64同一普通点FV静止周期接触的空间对照通过。Stage 3仍未完整通过，fluid ORC关闭，κ未计算。}

本轮严格执行上一轮唯一登记项：同一单位周期方形、等半两相$\rho_L=1,\rho_R=4$、原二进制$D=0.02$、$T=1/20$、$K=16$不变。只补N16/N64两张静态点图与两条实际/完整独立路径，N32全部保存状态只加载、不重算。静态$L$无中点冻结时间误差，不制造4/8/16时间阶，不继续N128或角度/模型扫描。

\textbf{两个时刻、两相邻档的原生/共同浓度与原生/共同H误差改善均超过预定正项指标。} 三项新增软件检查、主计算和完整独立审计首次通过。旧493项检查未重跑，累计记录496项，不冒称恢复旧v0.8.0/246。

\subsection*{真实物质状态与独立实现}
\[
m_i=\int_{K_i}\rho\,dV,\qquad
\bar c_i=\frac1{m_i}\int_{K_i}\rho c\,dV,\qquad m_i=\frac{\rho_i}{N^2}.
\]
原连续初场按每档有限单元积分新推导，不插值旧平均或从轨迹点值重采样。两个新系数图全部先封存，再加载质量、c0、参考或旧状态。构图仅收点、ID、给定oracle点密度、D及周期边界；不收真实物质面、参考或F/C。
\[
g_{ij}=\frac{|f_{ij}|}{\ell_i/(\rho_iD)+\ell_j/(\rho_jD)},\quad
g_L=D,\quad g_R=4D,\quad g_{LR}=\frac85D.
\]
上式仅在封存后逐面核对。原点Voronoi、普通阻力、稀疏指数/统一化、时序及H实现全部字节不改；没有裁剪、删除正面、归一化或阈值调整。

各档复用原128模态90位连续系数，Decimal100/120评价真实均值，独立mpmath80/100核验。初态直接积分原公式，两个正时刻分别核算尾项，保留所有初态/序列化桥。独立Cartesian控制保留全部离散模态与原始负零特征值；两新网格各17端点用独立Padé指数核对。

\subsection*{同一N16物质区域上的公平观测}
\[
(Pc)_I=\frac{\sum_{i\subset I}m_i c_i}{m_I^{(16)}},\qquad
\sigma_{0,16}=@SIGMA@,\qquad H=1-\frac{\mathrm{Var}_m(c)}{\mathrm{Var}_m(c_0)}.
\]
容量逐块核对，不修补质量或均值。跨网格误差比较固定上述区域与分母，原生误差另列自身分母；各自H分别与各自参考比较，避免混淆不同观测。所有容量因子、初态及参考聚合桥保留。
\newpage
\section*{2. 三档结果与固定正项比较}
\begin{center}\small\begin{tabular}{rrrrr}\toprule
N & $t$ & 原生误差/自身$\sigma_0$ & 共同误差/N16$\sigma_0$ & 共同H误差\\\midrule
@ROWS@
\bottomrule\end{tabular}\end{center}
终时共同误差：$1.76306246\times10^{-3}$、$4.17790234\times10^{-4}$、$1.02495677\times10^{-4}$。相邻误差比约4.220和4.076，仅是本例观察，不拟合或认证一般收敛阶，不外推浓度。
\begin{center}\includegraphics[width=.98\textwidth]{contact-spatial.png}\end{center}
\begin{center}\small\begin{tabular}{rrrr}\toprule
相邻档 & $t$ & 共同误差改善 & 固定正项指标\\\midrule
@BUDGETROWS@
\bottomrule\end{tabular}\end{center}
所有原生/共同浓度及原生/共同H四量均要求改善超过各自指标，不能只挑上表中共同误差放行。预算保留双方原机器/尾项、完整序列差、Cartesian谱/变换/舍入、评价、参考截断与序列化及初态桥；共同观测乘容量因子并加所有投影桥。H对数值项用原二次h\_indicator并加直接绝对H桥，无有符号抵消。
\newpage
\section*{3. 验证、保全与下一步}
三项必要软件检查首次通过：多N精确界面g及独立Gauss有限均值；不等子质量直接积分与非零桥保留；正项分类、完整序列损坏及细格高频误差被共同聚合掩盖时的原生控制拒绝。\textbf{原生可分辨变差即拒绝，即使共同观测改善。}

主计算首次@MAIN@秒／@RSS@ MiB，满足900秒／1024 MiB封顶。完整独立审计首次@AUDIT@秒／@ARSS@ MiB通过。两新图均独立重建双中垂线/互反面/半阻力/面积/固定全点探针；精确率、最小$10^{-16}$ Poisson尾项、原权重与全部独立序列逐位核对，原始库存/极值/方差/耗散按原尺度核验。

共同物质审计不调用生产聚合函数：独立分ID并直接核验165份快照和825个Fraction母单元积分探针，最大fsum与保存均值差$2.22\times10^{-16}$。N32原判别及其首次失败/类型恢复记录保持，不覆盖旧结果。

782原文件、15附件、100份先前正式科学清单保持；四份历史诊断既有差异保留。原README/METHOD/CODEX\_TASK/PHASE与旧代码/数据/报告不改，仅更新研究状态入口及备份交接文件。PDF/代码ZIP和GitHub备份延续用户已授权范围，不是环境或版本发布。

\subsection*{唯一下一项：移动Becker物理接口与单例}
静止周期接触三档oracle控制已完成，剩余T4缺口是最终同一后端的移动激波与三密度来源闭合。下一项先写两个具体接口契约，再执行一个原Becker oracle判别：
\begin{itemize}
\item 给定移动最外物质端点的x零通量边界，y单位周期；不把激波当固壁、不把x做周期，不能向点构造器提供真实内部物质面。
\item T显式的可选时序包装，共享原指数/统一化与H。现helper硬编码T=.05，不能直接用于T=.1；新包装须先在T=.05夹具逐位匹配旧路径。
\item 固定$\delta=.08,D=.02,T=.1$、原32物质列，横向复制8行；点沿固定质量标签中点轨迹，不强迫成真实单元几何中心。复用原质量/c0与已保存8192空间/时间参考，不新解PDE。
\end{itemize}
本轮仅绑定来源SHA，未生成新Becker点、图、状态或参考评价。接口/算例失败或未分辨则保留停止，不调代表位置、网格或阈值制造通过。oracle单例不代表kernel/ledger密度已验收，fluid ORC继续关闭。

\subsection*{恢复与重放}
ZIP含研究代码/文档、782原始文件、本轮三档状态/系数/参考/审计，以及下一步Becker精选已存来源；不含完整约39GB工作区或15附件。完整历史保全审计仍需原工作区。新解压目录可重放两条新路径，N32只加载：
\begin{quote}\small\begin{verbatim}
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 python \
research/reconnect/contact_spatial_study.py \
--replay runs/reconnect-contact-spatial
\end{verbatim}\end{quote}
逐位重放与平台/数值库构建有关；不一致时保留差异，不放宽物理阈值。文件CRC/SHA、新解压和GitHub克隆核验以交付收据为准。Stage3 incomplete，κ未算。
\end{document}
'''
    values={'SIGMA':f'{r["common_initial_fluctuation_norm"]:.16g}','ROWS':rows,'BUDGETROWS':budgetrows,'MAIN':f'{ex["elapsed_main_seconds"]:.3f}',
      'RSS':f'{ex["peak_RSS_MiB"]:.2f}','AUDIT':f'{ar["elapsed_audit_seconds"]:.3f}','ARSS':f'{ar["peak_auditor_RSS_MiB"]:.2f}'}
    for key,value in values.items():tex=tex.replace('@'+key+'@',value)
    with (out/'LTN_contact_spatial_report_20261104.tex').open('x') as f:f.write(tex)


if __name__=='__main__':main()
