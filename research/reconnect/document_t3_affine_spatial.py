"""Additive spatial-control report and source-linked, unexecuted coverage-review gate."""
from pathlib import Path
import shutil
from audit_route_decision import ROOT,read,save,digest


def main():
    out=ROOT/'runs/reconnect-t3-affine-spatial';r=read(out/'results.json');ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'numerical-audit.json')
    assert audit['all112_independent_world_geometries_exact_rates_and_full_raw_sequences_verified'] and audit['prior_formal_manifests_verified']==95
    assert ar['audit_exit_code']==0 and not r['per_grid_controls']['32']['T3_registered_affine_case_passed']
    for old,name in [('/tmp/ltn-t3-spatial-main.log','main.log'),('/tmp/ltn-t3-spatial-geometry.log','geometry-audit.log'),('/tmp/ltn-t3-spatial-audit.log','audit.log')]:
        assert not (out/name).exists();shutil.copyfile(old,out/name)
    sources={
      'runs/reconnect-t3-affine-review/sha256.json':'previous cumulative seal,4545 entries',
      'runs/reconnect-t3-affine-review/next-gate-spec.json':'immutable three-space-level registration; historical already_executed=false preserved',
      'research/reconnect/T3_AFFINE_BUDGET_REVIEW.md':'saved N32 spatial-budget reason for this controlled experiment',
      'research/reconnect/T3_AFFINE_COMPARISON.md':'original N32 pair and retained inconclusive model H decision',
      'exports/ltn-t3-affine-review.zip':'previous budget-review delivery unchanged, omitted from new ZIP',
      'exports/ltn-t3-affine.zip':'original complete N32 delivery unchanged, omitted from new ZIP',
      'docs/METHOD.md':'original PDE/material/T1-T4 and three-space/three-time-level criteria',
      'docs/CODEX_TASK.md':'original stage and density-source acceptance requirements',
      'research/reconnect/periodic_voronoi_fv.py':'unchanged geometry and ordinary center resistance',
      'research/reconnect/oriented_periodic_boundary.py':'unchanged exact orthogonal boundary representation, E=I here',
      'research/reconnect/voronoi_time_action.py':'unchanged actions and original material I/H',
      'research/reconnect/voronoi_time16_core.py':'unchanged chronological primary/full independent propagation',
      'research/reconnect/t3_affine_core.py':'unchanged per-grid frozen-symbol/input/time/reference and full model H criterion',
      'research/reconnect/t1_uniform_core.py':'original spatial comparison structure and unchanged quadratic H indicator'}
    save(out/'evidence-index.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai',
      'sources':[{'path':name,'sha256':digest(ROOT/name),'role':role} for name,role in sources.items()],
      'source_copies':read(out/'source-copies.json'),'old_N32_paths_rerun':False,'original_N32_decision_rewritten':False,
      'new_reference_PDE_solves':0,'original_mass_c0_identity_or_physical_models_changed':False})
    passed=r['T3_registered_three_space_level_affine_control_passed']
    targets=['docs/METHOD.md','docs/CODEX_TASK.md','docs/PHASE3_CURRENT.md','docs/PHASE3_REBUILD.md',
      'research/reconnect/T1_UNIFORM_COMPARISON.md','research/reconnect/T2_TRANSLATION_COMPARISON.md',
      'research/reconnect/T2_D0_COMPARISON.md','research/reconnect/T2_ROTATION_COMPARISON.md',
      'research/reconnect/SHEAR_UNION64_COMPARISON.md','runs/reconnect-shear-union64/results.json',
      'research/reconnect/COMPRESSION_UNION64_COMPARISON.md','research/reconnect/CAPACITY_INPUT_AND_STAGE3.md']
    current=['results.json','numerical-audit.json','implementation-protocol.json','stage3-status.json']
    next_sources=[{'path':name,'sha256':digest(ROOT/name)} for name in targets]
    next_sources.extend({'path':str((out/name).relative_to(ROOT)),'sha256':digest(out/name)} for name in current)
    save(out/'next-gate-spec.json',{'already_executed':False,
      'type':'read_only_same_backend_T1_T4_and_T3_direction_shear_coverage_review' if passed else 'read_only_saved_affine_spatial_control_budget_review',
      'reason':'registered affine three-grid control passed; connect existing same-backend shear/direction evidence to original criteria before any next experiment' if passed else 'preserve adverse/unresolved spatial result and inspect saved controls before expansion',
      'sources':next_sources,'scope':'quote original acceptance requirements and link already sealed evidence; distinguish affine compression, actual shear, direction excitation,boundary mode,model law and density sources',
      'decision_required':'mark already satisfied criteria and exact remaining gaps; select only one minimal discriminating next task if needed, with no repeated passed affine compression or new acceptance demands',
      'prohibited':'no new points/graphs/actions/paths/references/PDE/ORC or old candidate replay; no retroactive N32 pass,H-indicator change,new DNS/dynamicORC prerequisite or automatic newN/K/flow/model scan',
      'new_points':0,'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_complete_independent_uniformization_sequences':0,
      'new_reference_PDE_solves':0,'old_candidate_reruns':0,'planned_new_software_checks':0,
      'main_wall_seconds_cap':120,'peak_RSS_MiB_cap':1024,'next_task_executed_this_round':False,
      'general_T3_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    levels=(16,32,64);models=('constant_D','constant_Gamma')
    final=lambda N,M:r['per_grid_observations'][f'n{N}_{M}'][-1]
    table='\n'.join(f"| {N} | {final(N,'constant_D')['native_error_native_denominator']:.12g} | {final(N,'constant_Gamma')['native_error_native_denominator']:.12g} | {final(N,'constant_D')['common_error']:.12g} | {final(N,'constant_Gamma')['common_error']:.12g} |" for N in levels)
    spatial=[]
    for row in r['adjacent_spatial_comparisons']:
        if row['time']==0:continue
        values=[f"{row['gains'][name]:.8g} / {row['fixed_indicators'][name]:.8g}" for name in ('native','common','native_H','common_H')]
        spatial.append(f"| {row['model']} | {row['coarse_N']}→{row['fine_N']} | {row['time']:g} | "+' | '.join(values)+' |')
    fine=r['per_grid_controls']['64']['model_comparisons'][1:]
    model_table='\n'.join(f"| {v['time']:g} | {v['model_state_difference_relative']:.12g} / {v['fixed_state_indicator_relative']:.12g} | {v['model_H_difference']:.12g} / {v['fixed_H_indicator']:.12g} | {'分辨' if v['model_response_resolved'] else '未决'} |" for v in fine)
    maxmass=max(v['maximum_native_raw_mass_drift'] for v in audit['per_new_grid_controls'].values())
    maxseq=max(v['maximum_primary_control_relative_difference'] for v in audit['per_new_grid_controls'].values())
    status_word='通过' if passed else '失败或未决'
    report=f'''# T3仿射压缩：同一普通FV后端的三档空间对照

2026-11-02（Asia/Shanghai）。执行[上轮登记项](../../runs/reconnect-t3-affine-review/next-gate-spec.json)，[本轮固定口径](t3_affine_spatial_protocol.json)及实现锁先于新科学输入生效。

**本次登记的三档空间对照{status_word}：{r['comparison_decision']}。** 两模型的16→32、32→64原生与共同浓度及H误差收益，在两个非零时刻均与先定正项指标比较；细档N64模型响应仍使用原含空间项的H门槛。原N32单例继续保持未决，不作追认。此结论限本仿射压缩判别例，Stage3仍incomplete，fluid ORC关闭。

## 输入、状态及复用

同一N16/32/64原T1物质单元质量与初始质量平均。原m为1/N²，原c0是.5+.2sin(2πlabel_x)cos(2πlabel_y)的有限单元积分平均，不换成轨迹点值。N32六条保存路径按字节复用；只新增N16/N64两模型各K4/8/16，共112图、12条实际及12条完整独立路径。原N32不重新构图或求解。

a=1−5t、T=.05，world点(a·label_x,label_y)、周期(a,1)、ρ_exact=1/a。恒D=.02和恒Γ下D_exact(t)=Γ0a是两个不同PDE；.02仍取原float的精确二进制值。给定ρ/D各自一次转float，真实乘积Γ_wire及其与名义模型的差原样保存，以原Duhamel正项计入；不配平质量、密度或通量。

112张局部/世界图全部封存之后，才读取真实m/c0、新参考或旧N32传播状态。图只收实际点、ID、给定ρ/D与全局边界；无内部真实物质面、局部F/C或参考响应拟合。原周期Voronoi、普通中心面阻力、时间后端及正交边界适配层字节不变。各新图gx=Γ_wire/a、gy=Γ_wire·a逐面精确核对；均匀正交例中辅助分区与真实物质分区恰好一致，不推广为一般流动性质。

参考分别沿用两本构的解析物质扩散时钟：恒D Ix=D0t/a、Iy=D0t；恒Γ Ix=Γ0ln(1/a)/5、Iy=Γ0(t−2.5t²)。旧T1静止正D参考只作历史来源。新N16/N64的初始有限单元积分、两个PDE参考及冻结/连续网格模态由独立80/100位评价核对；没有新参考PDE数值求解。

## 原生与同物质区域误差

共同观测固定N16原物质区域。子组分质量积分直接相加，再除以原N16质量；不经过质量修正的中间投影。原生和共同误差同时保留，防止只在粗观察区域隐藏细尺度误差。原生误差按本档σ0归一化，共同误差按N16原初波动范数**{r['common_initial_fluctuation_norm']:.17g}**归一化；用于比较不同原生网格的收益另外统一使用该共同分母，CSV列名明确区分。

T=.05、K16，各列均为相对各自物理PDE参考的质量L²误差：

| N | 恒D原生/本档σ0 | 恒Γ原生/本档σ0 | 恒D共同/原N16σ0 | 恒Γ共同/原N16σ0 |
|---|---:|---:|---:|---:|
{table}

两个模型是不同PDE，上表不是模型或算法优劣排名。三档误差下降不是一般空间阶证明，未作拟合、Richardson外推或替代实际状态。

## 收益与固定不确定度

下表每格为“实际误差减少量 / 固定指标”。登记结论要求全部非零时刻、相邻空间对的原生/共同浓度和原生/共同H均可分辨。

| 模型 | 相邻档 | t | 原生浓度 | 共同浓度 | 原生H | 共同H |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(spatial)}

浓度预算保留双方实际8/16差、理想中点16到连续网格的时间差、8/16原数值/输入/参考/模态残差正项；共同观测另保留原容量比导出的范数因子和所有参考/初态投影桥。原始投影桥及浮点差不归零。

空间H收益比较沿用原T1结构：实际H8/16绝对差、解析时间态H绝对差、参考/初态H投影桥逐项正加，仅数值/输入/参考/残差余量用原二次H扰动指标转换。**模型响应判别**仍完整调用未修改的t3_affine_core，双方空间/时间/输入/数值/参考及实际8/16状态差一起经原H形式比较，未用空间收益门槛替换模型门槛。两套比较口径都在科学输入前锁定。

N16/32/64两模型各4/8/16时间控制、冻结模态/完整序列及PDE误差分解全部保留。完整逐项字段在[空间CSV](../../runs/reconnect-t3-affine-spatial/spatial-comparisons.csv)、[时间CSV](../../runs/reconnect-t3-affine-spatial/time-controls.csv)与[JSON](../../runs/reconnect-t3-affine-spatial/results.json)，不扣除时间差或利用相关抵消放行。指标含经验机器项，不是连续PDE总误差认证界。

## N64模型响应与原N32结论

每格为“实际模型差 / 原固定指标”，状态差按N64原初波动归一化。

| t | 状态模型差/指标 | H_D−H_Γ / 原H指标 | 判定 |
|---|---:|---:|---|
{model_table}

原N32 t=.025的H差.00497045808719仍小于其原指标.00521984579759；原单例结果及判定逐值保持，未改报告、数据或H公式。新N64有了独立的细空间证据，不能把它写成旧N32当时已经通过。

![三档原生/共同误差、H和细档模型判别](../../runs/reconnect-t3-affine-spatial/t3-affine-spatial.png)

## 检查、独立审计与资源

三项新软件检查首次通过：多N方向/模态及损坏作用拒绝；固定父质量的直接子积分及容量不匹配不归一化；小合成三档预算与异常完整序列拒绝。合成输入不计新物理算例。旧480项未重跑，累计记录483；旧v0.8.0/246未恢复。

科学主计算首次{ex['elapsed_main_seconds']:.3f}秒/{ex['peak_RSS_MiB']:.2f}MiB，满足900秒/1024MiB；此前封存校验另计{ex['preflight_manifest_verification_seconds']:.3f}秒。112图独立世界几何审计388.346秒（含等待）、196.57MiB。最终数值/来源审计首次{ar['elapsed_audit_seconds']:.3f}秒/{ar['peak_auditor_RSS_MiB']:.2f}MiB通过，各进程资源分别记录。

世界审计不读质量/标量/参考，直接重建世界双中垂线、互反面、两半阻力、面积与固定全镜像探针。数值审计逐项核对112组精确率、最小1e−16尾项、原始权重及12条完整独立序列，守恒/常量/最大值/方差和共享面能量作用按原尺度核验。新主/独立路径最大相对差{maxseq:.17g}，新原始库存最大漂移{maxmass:.17g}，均不修补。

三档标签次序与嵌套物质中心精确核对；共同积分审计独立生成父ID分组、逐快照fsum求组分，全部固定父探针另用Fractions核对。所有共同快照、容量与原生结果一致，原生产投影与直接积分最大差{audit['common_material_audit']['maximum_original_vs_direct_fsum_component_mean_difference']:.17g}，按原代数尺度核验。未使用参考梯度、真实内部面或拟合状态。

782原文件、15附件、95份既有正式科学清单及旧两份T3交付哈希核验，4份历史诊断既有差异保留。原README/METHOD/CODEX_TASK/PHASE、附件、代码、报告和数据不改，仅更新RESEARCH_STATUS。无PDF、版本、提交推送、环境配置更改或发布。

## 当前验收位置与下一项

本次是原T3仿射压缩在同一普通FV后端上的受控三空间档/三时间档、恒D/恒Γ正交基准，结论不覆盖一般方向激发、任意边界、全部剪切或重构密度。下一项仅[登记已存T1–T4/T3方向与剪切覆盖复核](../../runs/reconnect-t3-affine-spatial/next-gate-spec.json)：对照原METHOD/CODEX_TASK/PHASE及已存剪切三档证据，列明已满足项与精确缺口，然后只为真实缺项选择一个最小判别任务；本轮未执行，不反复重跑已通过的仿射压缩，不新增DNS或动态ORC等待条件。

最终同一后端T4移动激波/独立接触及oracle/重构/ledger密度来源对照仍须按原要求完成。一般T3、一般光滑流、有限节点候选不宣布接纳；Stage3 incomplete，fluid ORC关闭，kappa未计算。

[数据、CSV与完整审计](../../runs/reconnect-t3-affine-spatial/) · [可重放包](../../exports/ltn-t3-affine-spatial.zip) · [下载核验](../../exports/ltn-t3-affine-spatial-delivery.json)。最终CRC/SHA及全新解压重放以交付JSON为准。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/t3_affine_spatial_study.py --replay runs/reconnect-t3-affine-spatial
```

重放只从112张新封存图重建N16/N64十二条实际/完整独立路径及全部半步状态、原CSR/权重，旧N32六条路径仅加载；核对两模型参考、原物质并集积分、全部正项指标及旧N32未决结论。不重新构图、求PDE或重跑旧候选。累计科学清单引用未打包历史，完整历史保全审计仍需原工作区。
'''
    with (ROOT/'research/reconnect/T3_AFFINE_SPATIAL_COMPARISON.md').open('x') as f:f.write(report)
    index=ROOT/'docs/RESEARCH_STATUS.md';old=index.read_text();marker='执行上轮只读复核登记项。'
    assert marker in old and '\n96. ' not in old
    old=old.split(marker)[0].replace('T3_AFFINE_BUDGET_REVIEW.md)（本轮）。','T3_AFFINE_BUDGET_REVIEW.md)。')
    old+=f'''96. [T3仿射压缩：同一后端三档空间对照](../research/reconnect/T3_AFFINE_SPATIAL_COMPARISON.md)（本轮）。

执行上一轮N16/32/64空间登记项，原m/c0/ID及物理D/Gamma定义不变，N32六条旧路径仅加载复用；只补N16/N64各两模型K4/8/16，共112新图、12实际/12完整独立路径。全部系数先封存再读m/c0/参考或旧状态，原几何/普通阻力/时间/边界后端字节不变。

本次登记对照{status_word}：{r['comparison_decision']}。两个模型16→32及32→64在t=.025/.05的原生/共同浓度和H误差收益均超过先定正项指标，三档时间/模态/参考控制通过；N64两时刻状态及H模型响应通过原含空间项门槛。终点原生相对误差：恒D N16 {final(16,'constant_D')['native_error_native_denominator']:.12g}、N32 {final(32,'constant_D')['native_error_native_denominator']:.12g}、N64 {final(64,'constant_D')['native_error_native_denominator']:.12g}；恒Gamma分别{final(16,'constant_Gamma')['native_error_native_denominator']:.12g}、{final(32,'constant_Gamma')['native_error_native_denominator']:.12g}、{final(64,'constant_Gamma')['native_error_native_denominator']:.12g}。不同PDE不作算法优劣排名。N32原t=.025模型H未决判定逐值保持，不追认旧例通过；不拟合、裁剪、归一化或放松H公式/阈值。

三项新软件检查首次通过，记录483，旧480未重跑；旧v0.8.0/246仍未恢复。112图世界几何、精确率/尾项/权重/12条完整统一化序列及80/100位新参考独立审计通过，共同N16原物质积分和所有容量/投影桥保留。主计算首次{ex['elapsed_main_seconds']:.3f}秒/{ex['peak_RSS_MiB']:.2f}MiB，满足900秒/1024MiB；新原始库存最大漂移{maxmass:.12g}。782原文件、15附件、95份旧正式清单和旧交付保持，4份历史诊断既有差异保留。

下一项只登记同一后端已存T1–T4/T3方向与剪切覆盖复核，核对原标准与旧剪切三档证据，确认真正缺口后才选一个最小任务；本轮未执行，不再自动加密/扫描或重跑已通过仿射压缩。一般T3/光滑流/有限节点接纳false，Stage3 incomplete，fluid ORC关闭，kappa未计算。报告：research/reconnect/T3_AFFINE_SPATIAL_COMPARISON.md；入口：runs/reconnect-t3-affine-spatial/stage3-status.json；下载：exports/ltn-t3-affine-spatial-downloads.md。原README/METHOD/CODEX_TASK/PHASE、附件与旧代码/数据/报告不改，仅更新本入口。无PDF、版本、提交推送、环境配置更改或发布；交付CRC/SHA及全新解压重放以交付JSON为准。
'''
    index.write_text(old)


if __name__=='__main__':main()
