"""Additive result/provenance report and one necessary future space control."""
from pathlib import Path
import shutil
from audit_route_decision import ROOT,read,save,digest


def main():
    out=ROOT/'runs/reconnect-affine-shear';r=read(out/'results-completed.json');audit=read(out/'numerical-audit.json')
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');geo=read(out/'geometry-audit-execution.json')
    assert audit['all28_independent_world_geometries_exact_rates_and_full_raw_sequences_verified'] and audit['prior_formal_manifests_verified']==97
    for old,name in [('/tmp/ltn-affine-shear-geometry.log','geometry-audit.log'),('/tmp/ltn-affine-shear-completion.log','completion.log'),('/tmp/ltn-affine-shear-audit.log','audit.log')]:shutil.copyfile(old,out/name)
    sources={
      'runs/reconnect-t3-coverage/sha256.json':'previous cumulative scientific seal,5540 entries',
      'runs/reconnect-t3-coverage/next-gate-spec.json':'immutable affine-shear registration; original already_executed=false retained',
      'research/reconnect/AFFINE_SHEAR_DOMAIN_CONTRACT.md':'original written boundary/two-mode contract, unchanged',
      'research/reconnect/T3_COVERAGE_REVIEW.md':'source-linked reason for affine-shear gap; passed older controls retained',
      'exports/ltn-t3-coverage.zip':'previous review delivery unchanged, omitted from new ZIP',
      'research/reconnect/periodic_voronoi_fv.py':'unchanged rectangle construction and world Euclidean clipping/moment primitives',
      'research/reconnect/oriented_periodic_boundary.py':'unchanged exact orthogonal boundary mode',
      'research/reconnect/voronoi_time_action.py':'unchanged primary exponential and raw uniformization',
      'research/reconnect/voronoi_time16_core.py':'unchanged chronological solver and actual I/H',
      'docs/METHOD.md':'original material,physical PDE,three-scale and T1-T4 criteria',
      'docs/CODEX_TASK.md':'original stage acceptance and density-source scope'}
    save(out/'evidence-index.json',{'sources':[{'path':name,'sha256':digest(ROOT/name),'role':role} for name,role in sources.items()],
      'source_copies':read(out/'source-copies.json'),'completion_snapshot':'pre-completion-files-sha256.json',
      'old_N32_compression_or_sine_shear_decisions_changed':False,'new_reference_PDE_solves':0,'old_candidate_reruns':0})
    passed=r['registered_affine_shear_numerical_control_passed'];parent=read(ROOT/'runs/reconnect-t3-coverage/sha256.json')
    material_sources={}
    for N in (16,64):
        p=ROOT/f'runs/reconnect-t1-uniform/n{N}/material-reference.npz';rel=str(p.relative_to(ROOT));assert digest(p)==parent[rel]
        material_sources[str(N)]={'path':rel,'sha256':digest(p),'old_T1_static_reference_history_only':True,'mass_c0_not_parsed_this_round':True}
    records=[]
    for K in (4,8,16):
        for name in ('native-results.json','trajectory-states.npz','modal-residues.json'):
            p=out/f'k{K}'/name;records.append({'path':str(p.relative_to(ROOT)),'sha256':digest(p)})
    save(out/'next-gate-spec.json',{'already_executed':False,
      'type':'one_three_space_level_same_affine_shear_control_reuse_N32' if passed else 'read_only_saved_affine_shear_control_budget_review',
      'reason':'direction and time controls resolved; fixed N32 space symbol error dominates time term; original METHOD requires three levels for a space study' if passed else 'preserve failure/unresolved before expansion',
      'scope':'same single affine-shear flow andrho1/D=.02/T=.05,s=5t; add only N16/N64,keep K4/8/16; no angle/model/flow scan',
      'N_values':[16,32,64],'new_N_values':[16,64],'reused_N_values':[32],'ordered_interval_counts':[4,8,16],
      'new_candidate_graphs':56 if passed else 0,'new_candidate_transient_runs':6 if passed else 0,
      'new_complete_independent_uniformization_sequences':6 if passed else 0,'reused_N32_saved_paths':3,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_material_definitions':0,
      'pure_source':{'path':str((out/'original-pure-inputs.npz').relative_to(ROOT)),'sha256':digest(out/'original-pure-inputs.npz')},
      'material_sources':material_sources,'reused_N32_records':records,
      'N32_valid_result':{'path':str((out/'results-completed.json').relative_to(ROOT)),'sha256':digest(out/'results-completed.json')},
      'source_protocol':{'path':str((out/'implementation-protocol.json').relative_to(ROOT)),'sha256':digest(out/'implementation-protocol.json')},
      'reporting_policy':'retain original NumPy-bool serialization failure; use completed valid results and plain scalar JSON formatter in new study,without changing science/criteria',
      'coefficient_inputs':'same actualworldpoints,IDs,givenrho/D/boundary only; all56 new local/world graphs sealed before any material/c0/reference or reused N32 state parse',
      'geometry':'unchanged sliding-domain world Euclidean constructor; old rectangle/isometry/action code unchanged; exactfaces/rates/moments,all-cell halfplanes andfixed9image probes independently audited',
      'fixed_geometry_probe_formula':'uniqueIDs [0,N//2,N-1,N^2//2,N^2-1]',
      'references':'same true material two-mode clocks with each original cell mean; independent80/100digit evaluation,initial bridge,residue guards and raw original c0 retained',
      'common_observation_N':16,'common_observation':'direct child component mass integrals divided by original N16 mass; retain all source capacity and initial/reference projection bridges',
      'spatial_acceptance':'fixedK16,adjacent16/32 and32/64,bothnonzerotimes: native/common concentration andnative/common H error gains exceed predetermined positive time/input/numerical/reference/projection budgets',
      'spatial_budgets':'both actual8/16 differences andanalytic midpoint16-to-continuous-grid time norms,all8/16 original numerical/reference/residue/initial-bridge terms;common observations include capacity factors and all projection bridges; H uses predeclared T1 structure direct absolute time H differences plus unchanged H norm indicator for numerical/reference/residue terms',
      'time_and_direction':'all grids K4/8/16 frozen/reference/state/H controls and measuredtime decrease; fineN64 two raw tilted-mode observables and responsecontrast must pass unchanged whole-state positive CauchySchwarz plus observed8/16 indicators',
      'planned_new_software_checks':3 if passed else 0,'check_categories':['multi-N sliding direction,moments andmodeguards','original fixed-parent mass observations andnonzero bridges','native/common spatial,time andraw-direction budget classification/adversecase rejection'],
      'recorded_checks_total_if_all_pass':490,'tail_per_interval_exact':'1/10000000000000000','original_algebra_tolerance':1e-10,'norm_estimator_seed':20261101,
      'main_wall_seconds_cap':1500,'main_peak_RSS_MiB_cap':1024,
      'resource_basis':'28N32 graphs/science toreporting fault149.100s; rough(.25+4)nodescale about634s forN16/N64;hard1500s/1024MiB envelope,not measured speed oraccuracy guarantee',
      'stop':'preserve failure/unresolved; no furtherN/K/angle/flow/model scan,pointrepair,clipping,positivefacedeletion,mass/rho/state/row/weightnormalization,F_Cfit orH/tolerancechange',
      'new_spatial_inputs_graphs_references_or_states_generated_this_round':False,'general_T3_admitted':False,'general_nonorthogonal_domains_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    obs=r['observations'];direction=r['direction_observables'];tim=r['time_controls'];last=obs[-1]
    table='\n'.join(f"| {v['time']:g} | {v['reference_error_relative']:.12g} | {v['H']:.15g} | {v['reference_H']:.15g} | {v['raw_mass_change']:.12g} |" for v in obs)
    dt='\n'.join(f"| {v['time']:g} | {v['actual_direction_contrast']:.12g} | {v['reference_direction_contrast']:.12g} | {v['fixed_positive_contrast_indicator']:.12g} | {'分辨' if v['direction_contrast_resolved'] else '初始零差'} |" for v in direction)
    tt='\n'.join(f"| {v['time']:g} | {v['state4_8_relative']:.12g} | {v['state8_16_relative']:.12g} | {v['H4_8']:.12g} | {v['H8_16']:.12g} |" for v in tim if v['time']>0)
    report=f'''# T3仿射剪切：给定滑移周期域的普通欧氏FV判别

2026-11-02（Asia/Shanghai）。执行[上轮登记项](../../runs/reconnect-t3-coverage/next-gate-spec.json)，[原书面契约](AFFINE_SHEAR_DOMAIN_CONTRACT.md)与[本轮固定口径](affine_shear_protocol.json)在科学输入前锁定。

**本次N32判别结论：{r['comparison_decision']}。** 三组世界欧氏导通率/二阶矩、两倾斜模态参考、完整独立序列及4/8/16时间控制均核验；两个非零时刻的原始方向响应差与PDE相容，并超过固定正项指标。只通过这个给定滑移模式/分辨率的判别例，不声明新空间收敛曲线或一般T3通过。Stage3 incomplete，fluid ORC关闭。

## 实现与物理状态

新增可选[sliding_periodic_voronoi.py](sliding_periodic_voronoi.py)。给定B列向量(1,0)、(s,1)，s=5t∈[0,1/4]、T=.05；world点X=B a，域B[0,1)²和晶格BZ²同步变化。全局B是边界输入，不是局部F/C拟合。ρ1、D=.02取原二进制值，恒D/恒Γ在此同一模型，不重复两套计算。

原T1 N32真实m/c0/ID按字节沿用，m=1/1024。c0仍是有限物质单元质量平均，不从变形后点位置重采样。所有28张新world欧氏图先封存，再加载真实m/c0、参考或旧状态；图不读真实内部物质面、标量或预制g。原矩形几何、正交适配、普通两半阻力和时间代码字节不改；s0软件检查精确调用原矩形分支，返回原系数包逐项一致。

非零s仅用逆基坐标作基本胞代表和空间桶，实际双中垂线、面长/中心距全部在world欧氏度量中计算。八个近约束只是临时围界；之后根据world半径保守补齐全部可能切割点/镜像，不截断k近邻。所有单元满足远镜像排除半径证据，独立审计从world双中垂线重建顶点/互反面/阻力/面积；另用不同桶细度和重建最终半径逐单元检查全部可能切割半平面，固定五探针再枚举全部点九镜像。

本均匀晶格三组无向位移h(1,0)、h(s,1)、h(s−1,1)，封存后逐面核对g1=D(1−s+s²)、g2=D(1−s)、g3=Ds，各N²条；每单元二阶矩为2Dh²I、面积h²。解析系数未给构造器。正面不删除，原始面积、密度、质量、行、权重和状态不归一化或修补。

## 两模态物质参考与误差

原sin(x)cos(y)包含w±=sin[2π(a_x±a_y)]，质量平均振幅A=.1sinc(π/N)²。两个物理时钟J±=D[2t∓5t²+25t³/3]，参考为.5+AΣ exp(−4π²J±)w±，保留混合导数造成的不同衰减。原c0与双模态浮点评价的初态桥绝对范数{r['initial_two_mode_bridge_absolute']:.17g}保留，未将原状态投影成理想双模态，也未删除其他舍入模态。

独立80/100位参考从有限单元积分/两个物理时钟及独立离散符号核对，固定评价余量64eps√M max(1,max|c0|)。高精度正弦对有理相位的数学零和±1精确取值；原NumPy权重、模态和实际状态仍保留其原浮点值，不把它们裁成零。两精度结果逐位一致，所有保存评价在原余量内。没有新PDE数值求解。

误差分母是原初质量波动范数{r['initial_mass_fluctuation_norm']:.17g}。K16实际结果：

| t | 相对物质PDE误差 | 实际H | 参考H | 原始库存漂移 |
|---|---:|---:|---:|---:|
{table}

初始参考H的微小负数原样保留。终时空间符号项/σ0={last['spatial_symbol_error_relative']:.12g}，时间项/σ0={last['midpoint_time_error_relative']:.12g}；其余原机器/1e−16尾项、完整序列差、两模态/常量残差积分、评价与初态桥逐项正加。模态残差先满足原1e−10代数尺度，再用于预算。实际H用原全状态质量方差和原H扰动指标核验，经验机器项不冒充认证PDE总误差界。

## 方向响应与时间控制

记录原始I±=Σm(c−μ0)w±，以各自初值定义变化。下表方向差为ΔI+−ΔI−；不重标定观测，不修改或投影实际状态。指标来自两观测的全状态Cauchy–Schwarz正项和及实际8/16观测差，不能用有符号误差抵消放行。

| t | 实际方向变化差 | 物理参考变化差 | 固定正项指标 | 判定 |
|---|---:|---:|---:|---|
{dt}

每个原始观测变化也分别与各自参考核对。三时间档实际差：

| t | 状态4/8差/σ0 | 状态8/16差/σ0 | H4/8差 | H8/16差 |
|---|---:|---:|---:|---:|
{tt}

两非零时刻的状态/H差下降均超过预定数值/参考/残差/初态桥指标。只记录本例时间趋势，不用Richardson或拟合状态，不认证一般时间阶。

![滑移周期域、原始模态观测、实际H及方向差](../../runs/reconnect-affine-shear/affine-shear.png)

## 报告序列化故障与不重算恢复

主计算至JSON输出故障{ex['elapsed_main_to_reporting_failure_seconds']:.3f}秒/{ex['peak_main_RSS_MiB']:.2f}MiB，28图和3条实际/独立路径及参考已全部保存，仍在900秒/1024MiB限额内。NumPy布尔值不能由原JSON编码器直接写出，导致第一次results.json不完整，原进程退出1；不称主计算首次全流程成功。

原失败日志、failed-execution.json和不完整results.json均保留。[独立补全入口](affine_shear_completion.py)只加载保存结果，把NumPy标量转换为对应Python标量，写新的**results-completed.json**及状态；没有重构图、重算传播、改公式或阈值。197个补全前文件及冻结实现的哈希全部保持。补全{ex['elapsed_completion_seconds']:.3f}秒/{ex['peak_completion_RSS_MiB']:.2f}MiB，不覆盖不完整文件；原权威数值路径和失败记录均可审查。新的报告/图/重放显式使用有效的completed文件，不能将不完整旧JSON作为当前结果入口。

## 检查、审计与保全

四项新增软件检查首次通过：边界/原s0分支与非法输入；独立world面/矩/阻力及损坏拒绝；独立积分时钟/有限均值/初态桥；原始时间/方向观测和异常完整序列拒绝。旧483项未重跑，累计记录487；旧v0.8.0/246未恢复。

28图纯world审计{geo['elapsed_including_wait_seconds']:.3f}秒（含等待）/{geo['peak_auditor_RSS_MiB']:.2f}MiB，最终独立数值审计首次{ar['elapsed_audit_seconds']:.3f}秒/{ar['peak_auditor_RSS_MiB']:.2f}MiB。全部局部精确率、最小Poisson截断、未归一权重和三条完整独立序列逐位核对；共享面散射/对称作用、常量、极值、方差、耗散与原始漂移按原尺度通过。主/控制最大相对差{audit['maximum_primary_control_relative_difference']:.17g}，原始库存最大漂移{audit['maximum_native_raw_mass_drift']:.17g}。所有28图还通过逐单元完整半平面排除审计。

782原文件、15附件、97份既有正式科学清单及旧覆盖交付核验，4份历史诊断既有差异保留。原README/METHOD/CODEX_TASK/PHASE、附件和旧代码/数据/报告不改，仅更新RESEARCH_STATUS。没有PDF、版本、提交推送、环境配置更改或发布。

## 下一项及Stage3边界

本次补上给定非正交边界的N32仿射剪切执行记录，不能因此宣布一般非正交域、一般T3或新空间收敛通过。终时空间项明显大于时间项，下一项只[登记同一仿射剪切的N16/32/64空间控制](../../runs/reconnect-affine-shear/next-gate-spec.json)：复用当前N32三路径，只补N16/N64、原K4/8/16，共56新图、6实际/6完整独立路径，固定N16共同物质观察与原生浓度/H并查；原质量、初态和所有桥保留，细档方向响应仍使用原正项门槛。本轮未生成这些新空间档的点、图、参考或状态。

下一项1500秒/1024MiB资源封顶，依据现有N32耗时作节点规模粗规划，不是精度门槛变化。失败或未分辨保留，不自动N128/K32/角度/模型扫描。原已通过的压缩与正弦剪切不重跑。最终同一后端T4、三密度来源以及研究指数作用与原生产默认隐式Euler交付范围仍按原标准区分；不新增DNS/动态ORC或恢复v080的等待条件。Stage3 incomplete，fluid ORC关闭，kappa未计算。

[有效结果与原失败记录](../../runs/reconnect-affine-shear/) · [可重放数据包](../../exports/ltn-affine-shear.zip) · [交付核验](../../exports/ltn-affine-shear-delivery.json)。CRC/SHA及全新解压重放以交付JSON为准。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/affine_shear_completion.py --directory runs/reconnect-affine-shear --replay
```

该入口从28个封存缓存重放本轮3条实际/完整独立路径，核对半步、原CSR/权重、双模态参考、初态桥和所有判别，同时验证补全前文件仍未改变。它不再调用仍保留原报告故障的旧生成器重建几何，不求PDE或旧候选，也不丢弃原失败记录。完整历史保全审计仍需原工作区，累计清单引用未打包历史。
'''
    with (ROOT/'research/reconnect/AFFINE_SHEAR_COMPARISON.md').open('x') as f:f.write(report)
    index=ROOT/'docs/RESEARCH_STATUS.md';text=index.read_text();marker='执行上一轮只读覆盖复核。'
    assert marker in text and '\n98. ' not in text
    text=text.split(marker)[0].replace('AFFINE_SHEAR_DOMAIN_CONTRACT.md)（本轮）。','AFFINE_SHEAR_DOMAIN_CONTRACT.md)。')
    text+=f'''98. [T3仿射剪切：给定滑移周期域的普通欧氏FV判别](../research/reconnect/AFFINE_SHEAR_COMPARISON.md)（本轮）。

执行上轮唯一N32/rho1/D=.02/T=.05/s=5t/K4/8/16仿射剪切登记项，新增可选精确滑移周期world欧氏Voronoi，旧矩形/正交适配/普通阻力/时间代码字节不改，s0调用原矩形分支。28图先封存再读原m/c0/参考，3实际及3完整独立路径，两个倾斜模态参考和原始观测不拟合、不归一化，原初态桥保留。

数值判别{r['comparison_decision']}。终时相对物质参考误差{last['reference_error_relative']:.12g}，实际H{last['H']:.15g}、参考H{last['reference_H']:.15g}；两非零时刻原始方向变化差均超过固定正项指标且与PDE相容，4/8到8/16状态/H差下降可分辨。单N32不声明新空间曲线或一般非正交/T3接纳。28图独立world顶点/面/阻力/面积/探针与逐单元全半平面检查、精确率/尾项/权重/完整独立序列及80/100位参考审计通过。

四项新软件检查首次通过，记录487，旧483未重跑。主计算{ex['elapsed_main_to_reporting_failure_seconds']:.3f}秒/{ex['peak_main_RSS_MiB']:.2f}MiB至结果写JSON因NumPy布尔值序列化失败，原失败日志与不完整results.json保留；追加补全器仅从保存数据转换JSON标量类型，写results-completed.json，197个原文件和冻结实现保持，未重算传播或改判据。报告/图/重放均使用有效completed结果，不能称主流程首次全成功。782原文件、15附件、97份旧正式清单及旧交付保持，4份历史诊断差异保留，旧v0.8.0/246未恢复。

下一项只登记同一仿射剪切N16/32/64必要空间控制：复用N32三路径，只补N16/N64及原K4/8/16，56图、6实际/6完整独立路径，原N16共同区域与原生浓度/H、细档方向门槛均预定正项比较，1500秒/1024MiB；本轮未执行，不自动扩矩阵。Stage3 incomplete，fluid ORC关闭，kappa未计算，一般T3/光滑流/非正交/有限节点接纳false。报告：research/reconnect/AFFINE_SHEAR_COMPARISON.md；入口：runs/reconnect-affine-shear/stage3-status.json；下载：exports/ltn-affine-shear-downloads.md。原README/METHOD/CODEX_TASK/PHASE、附件与旧代码/数据/报告不改，仅更新本入口。无PDF、版本、提交推送、环境配置更改或发布，CRC/SHA和全新解压重放以交付JSON为准。
'''
    index.write_text(text)


if __name__=='__main__':main()
