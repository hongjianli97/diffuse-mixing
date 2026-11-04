"""Additive three-level report and a single same-backend contact gate; no scientific runs."""
import shutil
from audit_route_decision import ROOT,read,save,digest


def main():
    out=ROOT/'runs/reconnect-affine-shear-spatial';r=read(out/'results.json');audit=read(out/'numerical-audit.json')
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');geo=read(out/'geometry-audit-execution.json')
    assert audit['all56_independent_world_geometries_exact_rates_and_full_raw_sequences_verified'] and audit['prior_formal_manifests_verified']==98
    for name,source in [('study.log','main'),('geometry-audit.log','geometry'),('audit.log','audit')]:shutil.copyfile(f'/tmp/ltn-affine-shear-spatial-{source}.log',out/name)
    sources={
      'runs/reconnect-affine-shear/sha256.json':'parent cumulative seal,5830 entries; original reporting failure retained',
      'runs/reconnect-affine-shear/next-gate-spec.json':'immutable N16/32/64 registration; old already_executed=false retained',
      'runs/reconnect-affine-shear/results-completed.json':'authoritative N32 valid saved result, never rerun or overwritten',
      'research/reconnect/AFFINE_SHEAR_DOMAIN_CONTRACT.md':'original given sliding-periodic domain/two-mode contract',
      'research/reconnect/AFFINE_SHEAR_COMPARISON.md':'parent N32 pass with reporting-only recovery disclosed',
      'research/reconnect/T3_AFFINE_SPATIAL_COMPARISON.md':'passed compression pair, preserved coarse decision',
      'research/reconnect/T3_COVERAGE_REVIEW.md':'original T1-T4 evidence and actual remaining gaps',
      'exports/ltn-affine-shear.zip':'parent delivery remains unchanged, omitted from new archive',
      'research/reconnect/sliding_periodic_voronoi.py':'unchanged opt-in given sliding-world geometry',
      'research/reconnect/periodic_voronoi_fv.py':'unchanged rectangle point-defined geometry/ordinary half resistance',
      'research/reconnect/voronoi_time_action.py':'unchanged primary exponential/full raw uniformization',
      'research/reconnect/voronoi_time16_core.py':'unchanged chronological actions and actual full-state I/H',
      'research/reconnect/contact_spectral_reference.py':'next gate: existing finite-cell contact reference evaluator only',
      'research/reconnect/CONTACT_PERIODIC_SPECTRAL_BASIS.md':'next gate: existing periodic contact transmission basis',
      'runs/reconnect-contact-finite-time-reference/reference-K128-p90.json':'next gate: stored spectral coefficients, hash only in this turn',
      'research/reconnect/cartesian_fv_modes.py':'next gate: existing extra-geometry contact FV cross-control, not main point builder',
      'docs/METHOD.md':'original T1-T4,physical PDE,three-scale andphysical threshold criteria',
      'docs/CODEX_TASK.md':'original oracle/kernel/ledger scope; no DNS waiting condition'}
    save(out/'evidence-index.json',{'sources':[{'path':name,'sha256':digest(ROOT/name),'role':role} for name,role in sources.items()],
      'source_copies':read(out/'source-copies.json'),'original_N32_pass_preserved':True,'old_candidate_reruns':0,'new_reference_PDE_solves':0})
    passed=r['registered_three_space_level_affine_shear_control_passed']
    next_sources=['runs/reconnect-contact-finite-time-reference/reference-K128-p90.json','research/reconnect/CONTACT_PERIODIC_SPECTRAL_BASIS.md',
      'research/reconnect/contact_spectral_reference.py','research/reconnect/cartesian_fv_modes.py',
      'research/reconnect/FV_CONTACT_FIXED_TIME_COMPARISON.md','research/reconnect/FV_CONTACT_COMMON_OBSERVATION.md',
      'runs/reconnect-affine-shear-spatial/original-pure-inputs.npz']
    save(out/'next-gate-spec.json',{'already_executed':False,
      'type':'one_same_point_geometry_action_backend_static_periodic_contact_oracle_control' if passed else 'read_only_saved_affine_shear_spatial_budget_review',
      'reason':'registered affine compression/shear coverage complete at declared cases; next original gap is same final point-geometry/action T4,not more T3 angles/grids' if passed else 'preserve failed/unresolved space comparison before expansion',
      'scope':'one original periodic contact,rho1/4,D originalfloat.02,T1/20,N32,oracle point densities only; no shock/density/space scan',
      'N':32,'T_exact':'1/20','rho_left_exact':'1','rho_right_exact':'4','interface_x_exact':'1/2',
      'boundary':'original static unit square,periodic both axes; second contact at periodic seam; no artificial shock or moving open boundary',
      'material_state':'newly evaluate original continuous initial formula and stored transmission reference on fixed N32 material squares; exactmassrho/N^2; do not label as recoveredv080 orinterpolateN128means',
      'pure_source':{'path':'runs/reconnect-affine-shear-spatial/original-pure-inputs.npz','sha256':digest(out/'original-pure-inputs.npz')},
      'sources':[{'path':p,'sha256':digest(ROOT/p)} for p in next_sources],
      'geometry':'same unchanged periodic_voronoi_fv point-only builder,ordinary half resistance; positions,IDs,givenoracle rho/D,boundary only; seal before m/c0/reference',
      'new_candidate_graphs':1 if passed else 0,'static_coefficients_cached_once_and_reused':True,
      'ordered_intervals':16,'new_candidate_transient_runs':1 if passed else 0,'new_complete_independent_uniformization_sequences':1 if passed else 0,
      'reference':'reuse stored128-mode90digit coefficients; evaluate N32 finite material means at t0,.025,.05,independently check arithmetic precision and rounding/initial bridge; evaluate original analytic truncation bound separately at each positive time,never reuseT.05tail at.025; t0initial directcontinuousformula,not truncatedspectralinitial; no new spectral fit or PDE solve',
      'cross_control':'original extra-geometry Cartesian full-mode FV serves independent semidiscrete spatial/action oracle only; no coefficients from it supplied to point builder',
      'acceptance':'exact aligned worldface/rho halfresistance laws and independent geometry; actual full-state path vs independently evaluated Cartesian semidiscrete solution within locked positive solver/evaluation/initial-bridge terms,raw inventory/extrema/variance/H; independently evaluated PDE error and original H indicator reported without thresholds fitted to result',
      'time_policy':'staticL gives no midpoint-freezing error; full independent sequence and Cartesian semidiscrete reference isolate actionerror; do not require spurious4/8/16 time-freezing decrease or regenerate identicalstaticgraphs',
      'density_policy':'oracle alone establishes backend plumbing; kernel andledger still separate later controls withsame m/c0/D/boundary; no F correction or source-dependent normalization',
      'software_checks_planned':3 if passed else 0,'check_categories':['point geometry/contact half-resistance and rejection','true cell-mean/ref evaluation and initial bridge','unchanged action/fullsequence/PDE-H comparison and adverse control rejection'],
      'recorded_checks_total_if_all_pass':493,'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_spectral_projection_fits':0,
      'tail_per_interval_exact':'1/10000000000000000','original_algebra_tolerance':1e-10,'norm_estimator_seed':20261101,
      'main_wall_seconds_cap':900,'main_peak_RSS_MiB_cap':1024,
      'stop':'preserve failure/unresolved; no automaticN64/128,K/angle/density/flow scans,clipping,pointrepair,normalization,F/Cfit orphysical/H thresholdchanges',
      'new_T4_points_material_reference_graphs_or_states_generated_this_turn':False,'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,
      'general_T3_admitted':False,'general_nonorthogonal_domains_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    table='\n'.join(f"| {N} | {v['time']:g} | {v['native_error_native_denominator']:.12g} | {v['common_error']:.12g} | {v['native_H']:.15g} | {v['native_reference_H']:.15g} | {v['raw_native_mass_change']:.12g} |" for N,rows in r['per_grid_observations'].items() for v in rows if v['time']>0)
    spatial='\n'.join(f"| {v['coarse_N']}→{v['fine_N']} | {v['time']:g} | {key} | {v['gains'][key]:.12g} | {v['fixed_indicators'][key]:.12g} | {v['decisions'][key]} |" for v in r['adjacent_spatial_comparisons'] if v['time']>0 for key in ('native','common','native_H','common_H'))
    time='\n'.join(f"| {N} | {v['time']:g} | {v['state4_8_relative']:.12g} | {v['state8_16_relative']:.12g} | {v['H4_8']:.12g} | {v['H8_16']:.12g} |" for N,q in r['per_grid_controls'].items() for v in q['time_controls'] if v['time']>0)
    fine=r['per_grid_controls']['64'];direction='\n'.join(f"| {v['time']:g} | {v['actual_direction_contrast']:.12g} | {v['reference_direction_contrast']:.12g} | {v['fixed_positive_contrast_indicator']:.12g} |" for v in fine['direction_observables'] if v['time']>0)
    report=f'''# 仿射剪切：同一普通FV三档物质空间对照

2026-11-02（Asia/Shanghai）。执行[上一轮登记项](../../runs/reconnect-affine-shear/next-gate-spec.json)；[固定实现口径](affine_shear_spatial_protocol.json)在新科学输入前锁定。

**决定：{r['comparison_decision']}。** 原N16/32/64、两非零物理时刻、相邻两档的原生/共同浓度及原生/共同H改善均按原正项指标判断；所有网格冻结符号/参考/时间控制及细N64原始倾斜模态方向响应逐项核验。原N32通过记录原值保持。Stage3 incomplete，fluid ORC关闭。

## 物质状态与同一实现链

只补N16/N64，s=5t、T=1/20、ρ1、D原float.02、K4/8/16不变，共56新图、6实际及6完整独立序列；N32三路径只加载。全部56张局部/世界欧氏图先封存，再解析任何真实m/c0、参考或旧状态。给定滑移晶格B列(1,0),(s,1)，逆基坐标只作搜索索引，world距离/双中垂线/面长/中心距与普通两半阻力不变。新几何、旧矩形、正交适配、原时间推进源码均保持原字节；不重新实现或换成解析g构图。

使用原T1真实质量、原有限物质单元c0/ID，m=1/N²，不以点密度体积重定义材料，不从变形点重采样c0。原乘积初场到理想双模态的浮点桥保留，不投影实际状态。D0、平移/旋转、压缩、正弦剪切和旧核候选不重跑。无新PDE数值求解。

每张图封存后检查三组精确g与2Dh²I二阶矩；独立world审计重建全部顶点/互反面/半阻力/面积，固定五探针全点九镜像核验。最终数值审计再用不同桶细度与重建最终半径对全部单元检查所有可能切割半平面，拒绝隐含固定近邻截断。原正面不删除，面积、质量、密度、权重、行和状态不归一化或裁剪。

## 原生与共同物质误差

原生误差使用各网格原初波动分母；跨网格改善和共同观测固定原N16初波动范数 **{r['common_initial_fluctuation_norm']:.17g}**。共同观测为子单元ρc积分直接求和除以原N16质量，不调容量或初值。容量因子、每档初态桥、参考聚合桥均按正项保留。独立审计不用生产聚合函数，重新分ID并直接fsum各子积分，另对固定母单元用精确Fraction核对。

| N | t | 原生误差/自身σ0 | 共同误差/原N16σ0 | 原生H | 原生参考H | 原始质量漂移 |
|---|---:|---:|---:|---:|---:|---:|
{table}

空间对照固定K16，只比较16→32、32→64，两非零时刻都要求四量改善超过各自指标。浓度指标包含双方实际8/16差、解析中点16到连续网格时间项、双方8/16数值/完整序列/模态残差/参考评价/初态桥；共同观测另乘原容量因子，加入所有初态/参考投影桥。H沿用预先声明的T1比较结构：绝对H时间差与桥正加，数值项用原h_indicator，无有符号抵消或Richardson状态。

| 相邻档 | t | 量 | 误差改善 | 固定正项指标 | 判定 |
|---|---:|---|---:|---:|---|
{spatial}

## 时间、双模态参考与原始方向响应

同一物质PDE双时钟J±=D(2t∓5t²+25t³/3)，原质量平均振幅.1sinc(π/N)²。新N16/N64独立80/100位算术核验有限单元均值、两物理时钟、连续网格及冻结离散符号；原固定64eps√M max(1,max|c0|)评价余量保持。高精度参考的有理相位数学零/单位值精确评价，实际NumPy模态、原始权重/状态舍入值不修补。

| N | t | 状态4/8差/自身σ0 | 状态8/16差/自身σ0 | H4/8差 | H8/16差 |
|---|---:|---:|---:|---:|---:|
{time}

方向观测仍为原始I±=Σm(c−μ0)sin[2π(a_x±a_y)]，比较各自初始变化；不拟合/缩放观测或替换实际状态。细档判别使用原全状态Cauchy–Schwarz正项与实际8/16差，分别检查两观测与参考，再检查方向差超过两正项之和。

| N64时刻 | 实际ΔI+−ΔI− | 物理参考 | 固定正项指标 |
|---|---:|---:|---:|
{direction}

![三空间档误差与细档原始方向对照](../../runs/reconnect-affine-shear-spatial/affine-shear-spatial.png)

## 首次软件检查错误、主计算与审计

首次3项软件检查中1项失败：错误沿用压缩算例“细档方向未分辨”的布尔预期，而剪切小算例按原指标已分辨。保留原测试源码、JSON、日志，只把两条预期False改为True；故意扰坏完整独立序列的拒绝断言保持，科学代码/公式/阈值未因此修改。第二次3项均通过，旧487项未重跑，累计记录490项，不称测试首次成功。小型3/6/12夹具不是新增科学算例。

主计算首次完成 **{ex['elapsed_main_seconds']:.3f}秒/{ex['peak_RSS_MiB']:.2f}MiB**，在1500秒/1024MiB内。沿用父轮追加补全器的JSON标量转换器，新结果一次正常保存；父轮不完整JSON及原序列化失败仍保持原字节，不追认父流程首次成功。28张N16、28张N64纯几何审计 **{geo['elapsed_including_wait_seconds']:.3f}秒**（含等待）；最终独立数值审计首次 **{ar['elapsed_audit_seconds']:.3f}秒/{ar['peak_auditor_RSS_MiB']:.2f}MiB**。精确率、最小1e−16 Poisson尾项、原权重和全部独立序列逐位核对，另外检查共享面作用/对称作用、常量、极值、质量方差和耗散。

782原文件、15附件、98份既有正式科学清单及父轮交付保持；四份历史诊断既有差异保留。原README/METHOD/CODEX_TASK/PHASE、附件与旧代码/数据/报告均不改，仅更新RESEARCH_STATUS。无PDF、版本、提交推送、配置更改或发布。旧v0.8.0/246没有恢复，也不作为继续解析基准的等待条件。

## T3覆盖与下一项

{'本轮已补齐' if passed else '本轮尚未补齐'}原METHOD表T3中的给定仿射压缩/剪切、方向响应与恒D/恒Γ登记覆盖；不继续追加角度、模型或N128扫描。该结论是已声明算例/边界族的实验覆盖，不是一般非正交点云收敛定理，general_T3_admitted仍false；不能把通用声明的false解读为还需无限补T3算例。

Stage3仍缺原要求的**最终同一后端T4及oracle/kernel/ledger三密度来源闭合**。下一项[只登记一个同一普通点几何/时间后端的静止周期接触oracle判别](../../runs/reconnect-affine-shear-spatial/next-gate-spec.json)：N32、ρ1/4、原D=.02、T=.05，原单位周期边界不需要激波开放边界适配；一个静态系数图只构造一次，K16实际/完整独立路径，另以已有Cartesian全模态半离散FV作独立对照。真实N32质量/均值从原连续公式及已存128模态90位参考系数评价，明确是新推导材料，不插值旧N128均值或伪称恢复原数据。

静态L没有中点冻结时间误差，不制造4/8/16下降要求或重复构造同一图；原独立完整序列/半离散对照控制作用误差。先做oracle同后端连接，不把它当三密度已通过或移动激波已闭合。参考尾项须按两个正时刻分别评价，不能将T=.05的尾项用于更早时刻；初值直接用连续公式积分。该项只登记，本轮未生成T4点、材料、图、参考或状态。生产默认隐式Euler与当前研究中点指数作用的交付范围继续区分；fluid ORC关闭，κ未计算。

[保存结果](../../runs/reconnect-affine-shear-spatial/results.json) · [可重放数据包](../../exports/ltn-affine-shear-spatial.zip) · [交付核验](../../exports/ltn-affine-shear-spatial-delivery.json)。解压CRC/SHA及六条新路径逐位重放以交付JSON为准；N32三路径仅加载，不重放。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/affine_shear_spatial_study.py --replay runs/reconnect-affine-shear-spatial
```
'''
    reportpath=ROOT/'research/reconnect/AFFINE_SHEAR_SPATIAL_COMPARISON.md'
    with reportpath.open('x') as f:f.write(report)
    status=ROOT/'docs/RESEARCH_STATUS.md'
    with status.open('a') as f:f.write(f'''

99. [仿射剪切：同一普通FV三档物质空间对照](../research/reconnect/AFFINE_SHEAR_SPATIAL_COMPARISON.md)（本轮）。

执行上轮固定N16/32/64、K4/8/16登记项，只补N16/N64，56图全部先封存再读原m/c0/参考或旧状态；6实际/6完整独立路径，N32三路径仅加载、原通过记录保持。几何/普通阻力/时间后端原字节不改，原生/共同浓度与H相邻改善、各档时间/符号/参考和细N64原始方向按预定正项指标核验。决定{r['comparison_decision']}，原生/共同/H空间收益分辨={r['all_native_common_and_H_spatial_gains_resolved']}，细档方向分辨={r['fine_grid_direction_response_resolved']}。原物质均值和初态/参考投影桥保留，不拟合、裁剪、归一化或改物理/H阈值。

首次软件检查1项预期布尔错误保留源码/日志，只更正测试预期，第二次3项通过；旧487不重跑，累计490。主计算首次{ex['elapsed_main_seconds']:.3f}s/{ex['peak_RSS_MiB']:.2f}MiB，独立几何/全部单元半平面/80–100位参考/原始权重和完整序列/固定母单元质量积分及正项预算审计通过，782原文件、15附件、98份旧正式清单保持；父轮JSON失败和补全前文件不改。

通过本轮即补齐原表T3给定压缩/剪切登记覆盖，不追加角度/模型/更细格扫描，不宣称通用非正交点云定理。下一项只登记同一最终点几何/时间后端N32周期接触oracle判别，1静态图复用、K16实际/独立路径，复用原128模态90位系数评价新N32有限均值并独立Cartesian半离散控制；本轮未生成T4输入/图/参考/状态。Stage3剩同后端T4和三密度来源闭合，incomplete，fluid ORC关闭，κ未算。无PDF、版本、提交推送、配置更改或发布。有效入口runs/reconnect-affine-shear-spatial/results.json；下载exports/ltn-affine-shear-spatial-downloads.md。
''')


if __name__=='__main__':main()
