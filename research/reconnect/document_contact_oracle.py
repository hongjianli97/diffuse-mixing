"""Source-linked progress report, explicit next space gate and Chinese PDF source."""
from pathlib import Path
from fractions import Fraction as F
import shutil
from audit_route_decision import ROOT,read,save,digest


def main():
    out=ROOT/'runs/reconnect-contact-oracle';r=read(out/'results.json');audit=read(out/'numerical-audit.json');ex=read(out/'execution.json');ar=read(out/'audit-execution.json')
    assert audit['registered_same_backend_contact_oracle_control_passed'] and audit['prior_formal_manifests_verified']==99
    for name,source in [('study.log','main'),('audit-completed.log','audit-completed'),('reporting-type-recovery.log','reporting')]:shutil.copyfile(f'/tmp/ltn-contact-oracle-{source}.log',out/name)
    sources={
      'runs/reconnect-affine-shear-spatial/sha256.json':'parent cumulative seal6374entries',
      'runs/reconnect-affine-shear-spatial/next-gate-spec.json':'immutable single N32 contact registration; old executedfalse retained',
      'runs/reconnect-affine-shear-spatial/results.json':'passed T3 affine shear three-space-level evidence',
      'research/reconnect/AFFINE_SHEAR_SPATIAL_COMPARISON.md':'previous T3 report,unchanged',
      'exports/ltn-affine-shear-spatial.zip':'recovered parent22MB delivery,unchanged',
      'research/reconnect/CONTACT_PERIODIC_SPECTRAL_BASIS.md':'original complete weighted transmission basis andtailproof',
      'runs/reconnect-contact-finite-time-reference/reference-K128-p90.json':'stored128mode90digit continuous initial projections',
      'research/reconnect/periodic_voronoi_fv.py':'unchanged pointdefined geometry andordinary half resistance',
      'research/reconnect/voronoi_time_action.py':'unchanged sparse exponential/fulluniformization',
      'research/reconnect/voronoi_time16_core.py':'unchanged K16 chronology andactual H/I',
      'research/reconnect/cartesian_fv_modes.py':'independent extra-geometry full-mode control only',
      'docs/METHOD.md':'original T1-T4,material PDE,threelevels,H andphysical accuracy criteria',
      'docs/CODEX_TASK.md':'original three densitysource andshock/contact scope'}
    save(out/'evidence-index.json',{'sources':[{'path':p,'sha256':digest(ROOT/p),'role':role} for p,role in sources.items()],
      'new_material_data_not_recoveredv080':True,'old_candidate_reruns':0,'new_reference_PDE_solves':0})
    paths=['native-results.json','trajectory-states.npz']
    next_sources=['runs/reconnect-contact-oracle/original-pure-inputs.npz','runs/reconnect-contact-oracle/source-reference-K128-p90.json',
      'runs/reconnect-contact-oracle/results.json','runs/reconnect-contact-oracle/material-reference.npz','runs/reconnect-contact-oracle/reference-parameters.json',
      'runs/reconnect-contact-oracle/Cartesian-control.npz','runs/reconnect-contact-oracle/Cartesian-parameters.json']+[f'runs/reconnect-contact-oracle/k16/{p}' for p in paths]
    save(out/'next-gate-spec.json',{'already_executed':False,'type':'one_three_space_level_same_backend_static_contact_oracle_reuse_N32',
      'reason':'N32 samebackend geometry/action bridge passed; original METHOD requires three space levels before a spatial convergence claim',
      'scope':'same static periodic contact rho1/4,originalD.02,T1/20,original continuous c0; no density/flow/model/angle scan',
      'N_values':[16,32,64],'new_N_values':[16,64],'reused_N_values':[32],'ordered_intervals':16,
      'new_candidate_graphs':2,'one_static_graph_per_new_grid_reused':True,'new_candidate_transient_runs':2,
      'new_complete_independent_uniformization_sequences':2,'reused_saved_N32_paths':1,'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_spectral_projection_fits':0,
      'sources':[{'path':p,'sha256':digest(ROOT/p)} for p in next_sources],
      'geometry':'unchanged original periodicpointVoronoi andordinaryhalfresistance; bothnewgraphs sealedbefore m/c0/reference/oldstates; preserve exactcounts,areas andinterfaceg',
      'material':'derive each new grid fromoriginalcontinuousformula andrho/N^2,not interpolateoldcellmeans; preservefloatserialization andinitialbridges; truefiniteunitaverages',
      'reference':'reuse stored128mode90digitcoefficients,Decimal100/120 plus independentmp80/100; useoriginalanalytic tails separately at.025/.05; t0directformula,not truncatedinitial',
      'reference_evaluation_indicator':'unchanged64eps*sqrt(M)*max(1,maxabs(c0))',
      'common_observation_N':16,'common_observation':'direct child component masses dividedby original derivedN16mass; exactcapacitycontrol andrawinitial/referenceprojectionbridges; no reweighting',
      'acceptance':'all three grid pointgeometry/fullsequence/Cartesian/originalH/PDE controls; bothadjacent16/32,32/64 at.025/.05 native/common concentration andnative/commonH gains exceed predeclared positivebudgets',
      'comparison_budget':'forbothgrids sum unchangedmachine/tail/fullsequence/Cartesian/evaluation/reference-rounding/tail/initialbridge terms; common multiplycapacitynormfactors andaddallinitial/referenceprojectionbridges; H usesunchangedquadratic h_indicator onnumericterms anddirectabsoluteHprojection/initialbridges; no signed cancellation',
      'static_time_policy':'no freezingerror withconstantL; K16wholeaction controlled independently,not fake4/8/16 temporalorder orduplicategraphs',
      'reporting_type_policy':'restore original np.float64 totaltype before same evaluation; preservefirstauditandtestfixturefailure; do not alterfrozen scientificcode oroldresultbits',
      'planned_new_software_checks':3,'recorded_checks_total_if_all_pass':496,
      'check_categories':['multiN exactinterfacegeometry andfinitecellreference','fixedparent component integrals andbridges','native/common/H positivebudget classification andadverse rejection'],
      'main_wall_seconds_cap':900,'main_peak_RSS_MiB_cap':1024,'tail_per_interval_exact':'1/10000000000000000','original_algebra_tolerance':1e-10,'norm_estimator_seed':20261101,
      'stop':'preservefailed/unresolved; no automaticN128/K/angle/density/modelscan,clipping,normalization,F/Cfit orphysics/H thresholdchanges',
      'new_spatial_points_mass_initial_graphs_references_or_states_generated_this_turn':False,
      'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    meta=read(out/'reference-parameters.json');sigma=r['initial_mass_fluctuation_norm'];last=r['observations'][-1]
    table='\n'.join(f"| {v['time']:g} | {v['reference_error_relative']:.12g} | {v['actual_vs_Cartesian_relative']:.12g} | {v['fixed_action_indicator_relative']:.12g} | {v['actual_H']:.15g} | {v['reference_H']:.15g} | {v['raw_mass_change']:.12g} |" for v in r['observations'])
    summary=f'''# LTN扩散与掺混：同一普通FV周期接触oracle判别

2026-11-04（Asia/Shanghai）。执行[上轮登记项](../../runs/reconnect-affine-shear-spatial/next-gate-spec.json)。用户本轮明确要求分别提供PDF报告和代码压缩包；只生成下载文件，不发布环境或版本。

**决定：{r['comparison_decision']}。** 单N32同后端周期接触实现衔接通过；不是三档空间收敛，也不等于T4/三密度完整验收。Stage3 incomplete，fluid ORC关闭，κ未计算。

## 承接与本轮范围

新环境中上轮T3三档结果及约22MB交付包实际可读，SHA核对保持。该轮原N16/32/64共同物质终时误差依次1.35713303171e−3、3.42588344563e−4、8.63454576632e−5，压缩/剪切登记覆盖已补齐，不继续角度或更细格扫描。旧v0.8.0/246未恢复，现代码与原附件分开标注，不将旧对话数字冒作本轮测量。

本轮固定单位周期方形、静止等半两相ρ=1/4、D保留原float.02、T=1/20、N32（1024物质单元）。两接触位于x=.5和周期接缝。一个静态图只构造一次，K16实际及完整独立传播复用该图；无新PDE求解、谱投影拟合或旧候选重跑。研究时间路径仍是指数作用，不冒称原默认隐式Euler的生产CLI迁移已完成。

## 质量平均状态与阻力

状态始终为m_i=∫K_iρdV及c̄_i=m_i⁻¹∫K_iρc dV。本例对齐相内m_i=ρ_i/32²，总质量2.5；c0由原连续初场逐格积分新推导，不插值旧N128平均、不伪称恢复v080数据。所有几何/导通率先封存再读实际m/c0/参考。构造器仅收到点、ID、给定oracle点ρ、D和周期边界，不收到真实物质面、局部F/C或预制g。

普通两半阻力g=面长/[ℓ_i/(ρ_iD)+ℓ_j/(ρ_jD)]。本方格精确相内g=D、4D，界面g=(8/5)D。2048共享正面中低相992、高相992、界面64，逐面几何及独立双中垂线/互反面/面积/固定全点探针核验。未删除跨界面正面，不裁剪、重归一化或修改质量/密度/状态。

## 独立物质参考与半离散控制

复用已封存128模态90位α/β，不重拟合。原连续初场为1/2+f0(x)cos(2πy)，左右f0=(1/5)cos[π(a_px+φ_p)]，(aL,φL)=(16/5,1/5)、(aR,φR)=(4/5,7/5)。完整质量加权透射基为1、cos(2πnx)、sin(2πnx)/ρ，正特征率4π²Dn²。c与ρD∂xc在两接触处连续。

实际参考评价真实有限单元积分，保留sinc因子和法向常数模态的横向衰减。t0直接积分连续公式，不用有限谱截断替换初场。Decimal100/120和独立mpmath80/100核对，固定评价指标沿用64eps√M max(1,max|c0|)，初值float序列化桥{meta['initial_serialization_bridge_absolute']:.12g}原样保留。128模态解析尾项分别按两个正时刻计算，不能把T=.05的尾界套在.025；数值精度控制不是区间认证总误差界。

另一独立控制使用既有Cartesian法向32阶对称环及全部1024离散模态，输入原始c0，保留舍入模态和原始负零特征值，不过滤或置零。独立审计从精确半阻力另建法向环，并用Padé指数核对全部17个全状态端点。实际主路径仍用原点几何和原稀疏指数/完整统一化后端，不从Cartesian先验生成主边权。

## 结果

误差分母固定原初质量波动范数 **{sigma:.17g}**。H=1−Var_m(c)/Var_m(c0)沿用原定义。

| t | FV/PDE相对误差 | FV/Cartesian相对差 | 固定作用指标 | 实际H | PDE参考H | 原始质量漂移 |
|---|---:|---:|---:|---:|---:|---:|
{table}

终时FV/PDE误差{last['reference_error_relative']:.12g}与独立半离散空间误差一致，FV/Cartesian差{last['actual_vs_Cartesian_relative']:.12g}明显更小。此处通过的是同后端实现/独立控制，不因4.34e−4这个单值制造新的物理精度阈值，也不宣称空间收敛已完成。PDE误差分解用独立半离散空间项加原数值/参考/初态正项，不从实际误差中扣除空间误差。

![保存状态、误差与H](../../runs/reconnect-contact-oracle/contact-oracle.png)

## 测试、审计与失败保留

三项新增检查复核通过，旧490项未重跑，累计记录493项。首次几何夹具N4的L1半径恰为1/4，不满足既有审计严格R<1/4；只改该夹具为N8及相应探针/面数，保留首次源码、日志和JSON，未改审计条件或N32科学算例。其他检查涵盖逐格独立Gauss积分、分时尾项、完整独立序列及Cartesian状态扰坏后的拒绝。

主计算首次{ex['elapsed_main_seconds']:.3f}秒/{ex['peak_RSS_MiB']:.2f}MiB。首次完整审计在报告字值比较处失败：原Cartesian总指标为NumPy float64，JSON读回成为Pythonfloat，Python3.12的补偿sum与原NumPy顺序sum产生末位差。保留首次审计源码/日志；追加contact_oracle_reporting.py只恢复原标量类型，数值位不变，同一冻结evaluate精确复现原JSON。48个恢复前文件保持，不改原结果、科学公式、指标或阈值，不重算科学路径。正式重放使用新入口，冻结旧study的--replay不作为交付入口。

第二次完整审计通过，耗时{ar['elapsed_audit_seconds']:.3f}秒/{ar['peak_auditor_RSS_MiB']:.2f}MiB。精确率、最小1e−16 Poisson尾项、原权重和全部独立序列逐位核对；17端点Padé、独立80/100位参考与两个时刻尾项、原始库存/极值/方差/耗散均核验。最大原始质量漂移{audit['maximum_raw_mass_drift']:.12g}。

782原文件、15附件、99份先前正式科学清单及上轮交付保持，4份历史诊断既有差异保留。原README/METHOD/CODEX_TASK/PHASE、附件与旧代码/结果/报告不改，仅更新RESEARCH_STATUS。PDF为本轮用户明确授权；无版本、提交推送、环境更改或发布。

## 下一步与完整Stage3差距

下一项只登记同一静止接触N16/32/64空间控制：复用本轮N32路径，只补N16/N64两个静态图、各K16实际/独立路径。保持同一连续初场/物理D/边界/参考系数，固定N16共同物质积分及原生浓度/H并查，两时刻两相邻档改善必须超过预定正项预算；不自动N128、K或模型扫描。本轮未生成这些新空间档材料、图、参考或状态。

之后仍需最终同一后端的移动激波T4及oracle/kernel/ledger三密度来源闭合。当前只用oracle，不能把kernel或ledger改密度后产生的差异当作同一PDE；质量/初场/物理D与边界需固定。没有原始DNS不是阻塞条件，fluid ORC在物理验收前继续关闭，不承诺未经计算的完成日期。

## 下载与重放

[PDF报告](../../exports/LTN_contact_oracle_report_20261104.pdf) · [代码与本轮结果](../../exports/LTN_contact_oracle_code_results_20261104.zip)。压缩包包含全部项目研究代码/文档、782个原始文件、本轮可重放数据及失败记录；不包含全部18GB历史运行目录或21GB历史交付副本，也不冒称恢复缺失v080。旧附件保全在原工作区核验，不重复打包15个附件。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/contact_oracle_reporting.py --directory runs/reconnect-contact-oracle --replay
```

代码包CRC/SHA及全新解压重放以exports/LTN_contact_oracle_delivery_20261104.json为准。完整历史保全审计仍需原工作区。
'''
    with (ROOT/'research/reconnect/CONTACT_ORACLE_COMPARISON.md').open('x') as f:f.write(summary)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write(f'''\n\n100. [同一普通FV周期接触oracle判别](../research/reconnect/CONTACT_ORACLE_COMPARISON.md)（2026-11-04）。\n\n执行上轮N32/ρ1:4/D=.02/T=.05登记项，一张静态点图先封存再读m/c0/参考，K16实际和完整独立序列；原几何/阻力/时间后端不改。决定{r['comparison_decision']}，终时相对PDE误差{last['reference_error_relative']:.12g}、相对Cartesian差{last['actual_vs_Cartesian_relative']:.12g}，原H和正项指标通过。复用128模态90位系数评价新真实有限均值，不插值或伪称恢复v080；独立80/100位、两个时间尾项、17端点Padé、精确率/尾项/权重和原始状态审计通过。\n\n新增3检查第二次通过，首次N4审计夹具半径边界错误保留后仅换N8，不改阈值；记录493，旧490未重跑。主计算首次通过，首次完整审计因JSON读回Pythonfloat/原NumPyfloat求和末位差失败，追加类型恢复入口后精确复现原JSON；48个恢复前文件/首次代码与日志不改，第二次审计通过。782原文件、15附件、99份既有正式清单保持。\n\n下一项仅登记同接触N16/32/64必要空间控制，只补N16/N64两静态图并复用N32；尚未执行。Stage3仍缺同后端完整T4和三密度来源闭合，fluid ORC关闭，κ未算。按用户本轮明确要求生成独立PDF与代码结果ZIP，未发布环境/版本或提交推送。\n''')
    make_tex(out,r,audit,ex,ar,meta)


def make_tex(out,r,audit,ex,ar,meta):
    last=r['observations'][-1];rows='\n'.join(f"{v['time']:g} & {v['reference_error_relative']:.6g} & {v['actual_vs_Cartesian_relative']:.6g} & {v['fixed_action_indicator_relative']:.6g} \\\\" for v in r['observations'])
    hrows='\n'.join(f"{v['time']:g} & {v['actual_H']:.12f} & {v['reference_H']:.12f} & {v['raw_mass_change']:.5g} \\\\" for v in r['observations'])
    sigma=r['initial_mass_fluctuation_norm']
    tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\setmainfont{Noto Serif CJK SC}
\setsansfont{Noto Sans CJK SC}
\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=22mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,xcolor,hyperref,fancyhdr}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue,pdftitle={LTN扩散与掺混：周期接触oracle判别},pdfauthor={diffuse-mixing research}}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-04}\fancyfoot[C]{\thepage}
\setlength{\headheight}{16pt}
\begin{document}
\begin{center}{\LARGE\bfseries LTN扩散与掺混}\par\vspace{3mm}{\Large 同一普通FV周期接触oracle判别}\par\vspace{3mm}2026年11月4日（Asia/Shanghai）\end{center}
\section*{1. 本轮结论与项目位置}
\textbf{N32同一后端静止周期接触oracle判别通过。Stage 3仍未完整通过，fluid ORC保持关闭，未计算曲率。}

本轮承接已封存的T3仿射剪切三档空间对照，执行其唯一下一项。新环境能够读取旧成果；上一轮约22 MB交付包与6374条累计科学哈希逐项核对保持。没有重新应用旧修改、重跑旧候选或发布环境/版本。PDF与代码压缩包是本轮用户明确要求的下载交付。

\begin{center}\begin{tabular}{p{35mm}p{111mm}}\toprule
项目 & 当前证据与限制\\\midrule
T1与T2 & 已封存均匀扩散、平移、D=0与精确旋转控制；本轮不重跑。\\
T3仿射压缩/剪切 & 给定算例的方向、本构与三空间档登记覆盖已补齐。不是一般点云收敛定理。\\
本轮T4接触 & 一张N32静态点几何图，实际K16传播与完整独立控制及Cartesian全模态半离散解相容。\\
剩余T4工作 & 本接触必要三档空间控制、最终同后端移动激波、三密度来源闭合。\\
Stage 3 / ORC & incomplete；fluid ORC关闭。当前研究指数作用与原默认隐式Euler交付范围区分。\\\bottomrule
\end{tabular}\end{center}

固定单位周期方形、等半两相$\rho_L=1,\rho_R=4$、原二进制$D=0.02$、$T=1/20$。$N=32$代表$32\times32=1024$个固定物质单元。接触位于$x=1/2$及周期接缝；不是开放边界移动激波。

\textbf{终时相对PDE误差：@ERR@；相对独立Cartesian传播差：@CROSS@。} 两者量级分离，支持该有限算例中的几何/阻力/传播实现衔接。单N32误差不能替代空间收敛或完整T4验收；没有据结果重新设置物理精度阈值。

本轮新增三项软件检查经复核通过，累计记录493项，旧490项未重跑。首次夹具与审计失败及其修正均保留。旧v0.8.0/246仍未恢复，不冒称当前代码是取回的旧版本。
\newpage
\section*{2. 物质平均、几何与透射参考}
\subsection*{2.1 固定质量未知量与普通阻力}
\[
m_i=\int_{K_i}\rho\,dV,\qquad \bar c_i=\frac1{m_i}\int_{K_i}\rho c\,dV,
\qquad m_i=\frac{\rho_i}{N^2},\quad\sum_i m_i=\frac52.
\]
初值由原连续公式积分新推导，不插值旧N128均值，也不从轨迹点值重采样。图构造器仅接收点、ID、给定oracle点密度、$D$和周期边界；不接收真实物质面、$F/C$或参考标量。\textbf{唯一系数图先封存，再解析实际质量、初值与参考。}
\[
g_{ij}=\frac{|f_{ij}|}{\ell_i/(\rho_iD)+\ell_j/(\rho_jD)}.
\]
本方格相内$g=D,4D$，界面$g=(8/5)D$；这些是封存后核对值，不用于代替点几何构造。2048个共享正面中低相992、高相992、界面64。独立审计重建双中垂线、互反面、两半阻力、面积和固定全点探针。没有删跨界正面、裁剪、重归一化或改质量/密度/初场。

\subsection*{2.2 连续谱与真实有限单元平均}
原初场$c_0=1/2+f_0(x)\cos(2\pi y)$，
\[
f_0(x)=\tfrac15\cos[\pi(a_px+\phi_p)],\quad
(a_L,\phi_L)=(16/5,1/5),\quad(a_R,\phi_R)=(4/5,7/5).
\]
质量加权透射算子$A=-\rho^{-1}(\rho D f')'$的完整基为$1$、$\cos(2\pi nx)$、$\sin(2\pi nx)/\rho(x)$；两接触处$[c]=[\rho D\partial_xc]=0$。令$\tau=4\pi^2Dt$，复用原存128模态90位系数$\alpha_n,\beta_n$：
\[
c=\tfrac12+e^{-\tau}\cos(2\pi y)
\left[\alpha_0+\sum_{n=1}^{128}e^{-\tau n^2}
\left(\alpha_n\cos(2\pi nx)+\beta_n\frac{\sin(2\pi nx)}{\rho}\right)\right].
\]
参考评价逐物质格积分，保留$\operatorname{sinc}$因子与法向常数模态的横向衰减。$t=0$直接积分连续初场，不以有限谱截断替换。初值序列化质量范数桥为@BRIDGE@，原样保留。

Decimal100/120位评价与独立mpmath80/100位核对。原评价指标保持为\[64\epsilon\sqrt M\max(1,\max|c_0|).\]128模态尾项的解析指数下界按$t=0.025$与$0.05$分别计算；不把终时尾界用于更早时刻。算术/舍入控制不冒充区间认证总误差界。
\newpage
\section*{3. 数值结果与误差分离}
误差按原初质量波动归一化，H保持原定义：\[\frac{\|\bar c^{FV}-\bar c^{PDE}\|_m}{\sigma_0},\quad \sigma_0=@SIGMA@,\quad H=1-\frac{\mathrm{Var}_m(c)}{\mathrm{Var}_m(c_0)}.\]
\begin{center}\small\begin{tabular}{rrrr}\toprule
$t$ & FV/PDE误差 & FV/Cartesian差 & 固定作用指标\\\midrule
@ROWS@
\bottomrule\end{tabular}\end{center}
\begin{center}\small\begin{tabular}{rrrr}\toprule
$t$ & 实际$H$ & PDE参考$H$ & 原始质量漂移\\\midrule
@HROWS@
\bottomrule\end{tabular}\end{center}
\begin{center}\includegraphics[width=.94\textwidth]{contact-oracle.png}\end{center}
主路径使用同一稀疏指数作用与完整正统一化，原权重不归一化。静态$L$没有中点冻结误差，因此不制造4/8/16时间下降要求。独立Cartesian控制保留全部1024离散模态及原始c0；法向32阶原始最小特征值不置零。另从精确两半阻力重建法向环，用Padé指数核对全部17个端点。

PDE误差分解保留独立半离散空间项、原作用数值项、参考评价/尾项及初态桥，逐项正加；$H$使用原二次扰动指标。没有有符号抵消、拟合参数或浓度外推。
\newpage
\section*{4. 审计、保全与重放}
\textbf{主计算首次完成：@MAIN@秒／@RSS@ MiB。} 在900秒／1024 MiB资源封顶内。原始库存、极值、质量方差、正性与共享面耗散按原代数尺度检查；最大原始质量漂移@DRIFT@，未调平。

\textbf{软件检查：} 首次3项中几何夹具N4的$L^1$半径恰为$1/4$，不满足旧审计严格$R<1/4$条件；只把该夹具改为N8及对应探针/面数。原失败源码、JSON与日志保留，未改审计门槛或N32科学算例。第二次3项均通过。另含独立Gauss有限均值积分、分时尾项、扰坏完整序列与Cartesian状态后的拒绝检查。

\textbf{报告类型恢复：} 首次完整审计在JSON字值核对处失败：原Cartesian总指标为NumPy float64，JSON读回Python float后，Python3.12补偿sum与原顺序NumPy sum产生末位差。原科学代码、结果和48个恢复前文件保持。追加入口仅恢复原标量类型，IEEE值位不变，同一冻结evaluate精确复现原结果。没有增大指标或改阈值，不重算科学路径。第二次完整审计通过：@AUDIT@秒／@ARSS@ MiB。

独立核验覆盖：几何/界面g，80/100位参考，两时间解析尾项，17端点Padé，精确率、最小$10^{-16}$ Poisson尾项、原始权重及全部独立序列逐位一致。782原文件、15附件、99份既有正式科学清单及上轮交付保持；四份历史诊断既有差异保留。

\subsection*{交付内容}
PDF报告单独下载；代码ZIP包含全部研究代码/文档、782个原始文件、本轮状态/系数/参考/审计与失败记录。\textbf{不包含全部18 GB历史运行数据或21 GB历史交付副本}，15个原附件仅在工作区保全核验，不重复打包。完整历史保全审计仍需原工作区；本轮完整数值重放可在新解压目录进行。

\begin{quote}\small
\begin{verbatim}
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 python \
research/reconnect/contact_oracle_reporting.py \
--directory runs/reconnect-contact-oracle --replay
\end{verbatim}
\end{quote}
上面四行可作为一条shell命令整体复制。正式重放入口是reporting文件，冻结旧study的--replay保留作诊断，不作为交付入口。

\section*{5. 唯一下一项与Stage3差距}
只登记同接触N16/32/64必要空间控制：复用N32，仅补N16/N64两个静态图，各K16实际／完整独立路径；固定N16共同物质观测，同时核对原生浓度和H。两个时刻两相邻档的四类改善必须超过固定正项预算，所有初态/参考聚合桥保留。\textbf{本轮尚未执行这些新空间档。}

后续仍需最终同一后端移动激波T4及oracle/kernel/ledger三密度来源闭合。当前oracle单例不接纳重构或ledger；不追加T3角度或全矩阵扫描。无新版本、提交推送、环境配置更改或发布。fluid ORC保持关闭，不承诺未经计算的完成日期。
\end{document}
'''
    values={'ERR':f'{last["reference_error_relative"]:.8g}','CROSS':f'{last["actual_vs_Cartesian_relative"]:.8g}',
      'BRIDGE':f'{meta["initial_serialization_bridge_absolute"]:.8g}','SIGMA':f'{sigma:.16g}','ROWS':rows,'HROWS':hrows,
      'MAIN':f'{ex["elapsed_main_seconds"]:.3f}','RSS':f'{ex["peak_RSS_MiB"]:.2f}','DRIFT':f'{audit["maximum_raw_mass_drift"]:.8g}',
      'AUDIT':f'{ar["elapsed_audit_seconds"]:.3f}','ARSS':f'{ar["peak_auditor_RSS_MiB"]:.2f}'}
    for key,v in values.items():tex=tex.replace('@'+key+'@',v)
    with (out/'LTN_contact_oracle_report_20261104.tex').open('x') as f:f.write(tex)


if __name__=='__main__':main()
