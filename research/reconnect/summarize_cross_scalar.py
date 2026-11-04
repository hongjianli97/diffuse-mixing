"""Close the restricted scalar repair route; preserve all previous rows."""
from pathlib import Path
import argparse,json
import numpy as np
from package_compression_midpoint import dump,digest
from summarize_shear_transient_spatial import csv_exact


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve()
    assert not (out/'sha256.json').exists(),'never rewrite sealed evidence'
    r=json.loads((out/'proof-certificates.json').read_text());audit=json.loads((out/'proof-audit.json').read_text())
    assert audit['seven_exact_rational_certificate_groups_recomputed']
    csvs=[]
    with np.load(out/'cross-scalar.npz') as z:
        for side in ['left','right']:
            keys=['row_columns','normal_centers','original_mass','original_rho0_point','saved_L_row','probe_quadratic_mass_means',
                'centered_action_terms_per_s','same_phase_mask','cross_phase_mask','raw_CSR_action_terms_per_s']
            values=np.column_stack([z[side+'_'+key] for key in keys])
            csvs.append(csv_exact(out/(side+'-phase-decomposition.csv'),z[side+'_original_ids'],values,
                ['material_id',*keys]))
    dump(out/'observations-audit.json',{'CSVs':csvs,'raw_NPZ_sha256':digest(out/'cross-scalar.npz')})
    previous=json.loads((root/'runs/reconnect-interface-pair/results.json').read_text())
    controls=json.loads((out/'raw-controls.json').read_text())
    next_plan={'scope':'theory-only candidate-route decision using already sealed evidence; no repeated proofs or new experiments',
        'source_reports':['research/reconnect/CROSS_PHASE_SCALAR_COMPATIBILITY.md','research/reconnect/CONTACT_CONSTITUTIVE_REVIEW.md',
            'research/rebuild/MASS_MOMENT_CONE.md','research/rebuild/SHARED_EXCHANGE.md','research/rebuild/DENSITY_GEOMETRY.md',
            'research/rebuild/HYBRID_MATERIAL_FACE.md','research/reconnect/MAINLINE_RECONNECTION.md'],
        'tasks':['map the scope of each obstruction: one cross-phase scalar, fixed stencil/moments, or finite-input density identifiability',
            'state the fixed-mass-average and allowed point-input contract for any next candidate, with explicit closure assumptions',
            'separate the ordinary central-resistance FV baseline, extra-information oracles, blocked candidate families and genuinely unresolved alternatives'],
        'no_repeated_parameter_scan_LP_or_physical_case':True,'no_new_candidate_before_explicit_model_and_input_assumptions':True,
        'no_true_faces_F_C_or_velocity_gradients_added_as_required_candidate_inputs':True,
        'ordinary_central_resistance_FV_baseline_retained':True,
        'new_candidate_graphs':0,'new_edge_weights':0,'new_operator_quadrature_evaluations':0,
        'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'fluid_orc_enabled':False}
    for rel in next_plan['source_reports']:assert (root/rel).exists() or rel.endswith('CROSS_PHASE_SCALAR_COMPATIBILITY.md')
    p=r['compatibility'];left=r['rows']['left'];right=r['rows']['right']
    status={'updated':'2026-11-01','timezone':'Etc/UTC','stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'scope':'completed predeclared theory-only shared cross-phase scalar compatibility review',
        'restricted_family':'one common nonnegative cross-phase multiplier, fixed within-phase exchange, stencil, original masses and specified material-mean quadratic probe',
        'same_phase_coefficients_exactly_equal':True,'cross_phase_coefficient_ratio_exact':'4',
        'shared_scalar_compatible':False,'universal_max_relative_error_lower_percent':r['universal_max_relative_error_lower_percent'],
        'original_mass_weighted_relative_L2_lower_percent_same_bound':r['universal_max_relative_error_lower_percent'],
        'source_diagonal_defects_retained_no_go':True,'independent_exact_material_quadratic_no_go':True,
        'restricted_scalar_repair_route_closed':True,'all_positive_exchange_methods_ruled_out':False,
        'constraint_ratios_not_used_as_parameters':True,'parameter_values_evaluated_on_a_graph':0,
        'original_raw_CSR_rows_mass_ids_c0_PDE_and_candidate_unchanged':True,
        'new_numerical_or_flow_tests':0,'exact_algebra_certificate_groups':7,'recorded_checks_total':363,
        'old_v080_or_246_tests_recovered':False,'new_candidate_graphs':0,'new_edge_weights':0,'new_operator_quadrature_evaluations':0,
        'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'finite_node_candidate_admitted':False,'contact_transient_admitted':False,
        'next_sealed_evidence_route_decision_predeclared':next_plan,'next_task':'reuse sealed evidence for a candidate-route and explicit input-closure decision',
        'remaining':['admissible non-scalar material-average interface closure','finite-input density identifiability','general physical and transient acceptance'],
        'report':'research/reconnect/CROSS_PHASE_SCALAR_COMPATIBILITY.md','pdf_generated':False,'software_version_created':False,'published':False}
    dump(out/'stage3-status.json',status)
    A=left['A_same_phase']['decimal_for_reading_only'];CL=left['C_cross_phase']['decimal_for_reading_only'];CR=right['C_cross_phase']['decimal_for_reading_only']
    report=f'''# 同一界面两行的共享跨相标量相容性

2026-11-01（UTC）。Stage 3 **incomplete**；fluid ORC **关闭**。

预定理论审查完成：**一个共同跨相标量不能同时修复两侧响应。** 在固定同相交换、跨相模板形状、原质量及同一二次质量平均探针下，对所有共同非负标量t，至少一侧相对响应误差不小于 **{r['universal_max_relative_error_lower_percent']:.8f}%**。这给出停止该类单标量小修的明确依据，没有运行任何新参数或修改边权。

结论只针对下述受限模型类，不排除边依赖交换、更丰富闭合假设或其他正交换方法。普通中心阻力物质FV基线保留。

## 1. 固定输入与精确算术

执行 `runs/reconnect-interface-pair/stage3-status.json` 的 `next_cross_phase_scalar_compatibility_review_predeclared`，先核验1118项科学清单。只用上一轮缓存的行63/64、原质量/密度/身份、原L与辅助cbar；快照哈希为 `b4eb6d7e3cd599c2a537b02cd10cbc817dd6af47b0e4750cbb85c5f4fb9103b0`。未读取另一个物理算例、构图或重新积分核。

辅助场仍为c=(a−0.5)²/2，按原方格物质质量平均；两侧正确PDE响应D=1/50。上一轮原始CSR响应{previous['rows']['left']['row']['raw_CSR_quadratic_response_per_s']:.14f}、{previous['rows']['right']['row']['raw_CSR_quadratic_response_per_s']:.14f} s⁻¹保持其原主值。

Fraction.from_float把保存的二进制L、原质量和缓存cbar精确解释为有理数，用于代数证书；NumPy仅负责载入快照与保留浮点归约记录。另用原二进制坐标、ell=1/128与精确二次平均独立形成物理多项式证书。**精确算术没有覆盖原始CSR浮点主值，也不是新的物理系数。**

## 2. 原中心差作用的同相/跨相分解

对目标i，按原节点rho是否与目标相同，将原非对角项分为S_i、X_i。定义

A_i=sum_(j in S_i) (−L_ij/m_i)(cbar_j−cbar_i)，

C_i=sum_(j in X_i) (−L_ij/m_i)(cbar_j−cbar_i)。

每行同相非对角项{left['same_phase_offdiagonal_entries']}个、跨相项{left['cross_phase_offdiagonal_entries']}个，分区完整且不重叠。L、m、状态均未裁剪或重新归一化。按保存二进制值的精确求和得到：

| 系数 | 行63，rho=1 | 行64，rho=4 |
|---|---:|---:|
| A，同相贡献，s⁻¹ | {A:.16g} | {right['A_same_phase']['decimal_for_reading_only']:.16g} |
| C，原跨相贡献，s⁻¹ | {CL:.16g} | {CR:.16g} |
| A+C，原中心差精确值，s⁻¹ | {left['exact_centered_original_response']['decimal_for_reading_only']:.16g} | {right['exact_centered_original_response']['decimal_for_reading_only']:.16g} |

精确恒等关系为 **A_L=A_R=A<D，C_L=4C_R>0**，不是根据显示小数近似判定相等。镜像几何和1:4原质量下，这个关系由当前保存数据逐项核验。

## 3. 两个响应约束不相容

只在符号层面令所有原跨相交换共同乘t≥0，同相交换不变。保守中心差形式应满足

R_L(t)=A+4tC_R，R_R(t)=A+tC_R。

若两侧都正确，则R_L=R_R=D。两式相减得3tC_R=0；C_R>0迫使t=0，但此时R_L=R_R=A<D，矛盾。因此不存在这样的共享标量。

同一结论也可从两条等式的必要比值得到：

| 单侧等式要求的比值 | 数值，仅显示 |
|---|---:|
| (D−A_L)/C_L | {p['necessary_left_constraint_ratio']['decimal_for_reading_only']:.12f} |
| (D−A_R)/C_R | {p['necessary_right_constraint_ratio']['decimal_for_reading_only']:.12f} |

这些是互不相容约束的代数商，不是拟合、选定或执行的新参数。程序没有生成t对应的边权、对角、矩阵或传播结果。通用消去行列式(D−A_L)C_R−(D−A_R)C_L精确非零，显示值为{p['determinant']['decimal_for_reading_only']:.12e} s⁻²；完整有理数保存在JSON中。

## 4. 对所有共同标量的误差下界

令e_L=R_L−D、e_R=R_R−D。任意t均有

e_L−4e_R=3(D−A)>0。

三角不等式给出

3(D−A) ≤ |e_L|+4|e_R| ≤ 5 max(|e_L|,|e_R|)，

所以

max(|e_L|,|e_R|) ≥ (3/5)(D−A)
= **{p['universal_max_absolute_error_lower_per_s']['decimal_for_reading_only']:.14f} s⁻¹**，

max(|e_L|,|e_R|)/D ≥ **{r['universal_max_relative_error_lower_percent']:.8f}%**。

这是受限类的解析下界，不是新验收阈值、两次求积差或参数扫描结果；无需选择任何最优t即可得到。它足以关闭“只改一个共同跨相标量就消除两侧约20%缺陷”的路线，但没有宣称原参数是某个全局最优。

原质量比1:4还给出相同的局部质量加权L2下界。令d=D−A、z=tC_R，则直接配方可得

(e_L²+4e_R²)/5 = 4(z−2d/5)²+9d²/25 ≥ 9d²/25。

因此按原质量定义的两格局部相对L2误差也不小于{r['universal_max_relative_error_lower_percent']:.8f}%。这是基于已精确核验质量比与系数关系的解析推论，不是改变观测权重或尝试一个新参数。

## 5. 原对角与浮点余差没有被修复

令sigma_i=−sum_j L_ij/m_i，使用同一保存数据的精确求和。原始CSR作用严格分解为

R_raw,i(1)=A_i+C_i+cbar_i*sigma_i。

原cbar_i*sigma_i分别为{left['exact_target_times_diagonal_defect_per_s']['decimal_for_reading_only']:.9e}、{right['exact_target_times_diagonal_defect_per_s']['decimal_for_reading_only']:.9e} s⁻¹；没有置零或改原对角。

即便在符号约束中把这些原常数缺陷作为固定加项保留，新的消去行列式仍精确非零，至少一侧相对误差的下界显示为{100*p['same_saved_diagonal_defects_retained_relative_lower']['decimal_for_reading_only']:.8f}%。因此不相容性不能归于省略原行和缺陷。

| 与上一轮浮点主值的控制差 | 行63 | 行64 |
|---|---:|---:|
| 原CSR浮点主值−精确二进制数据作用，s⁻¹ | {controls['left']['previous_raw_minus_exact_binary_raw_per_s']:.9e} | {controls['right']['previous_raw_minus_exact_binary_raw_per_s']:.9e} |
| 原中心差浮点作用−精确A+C，s⁻¹ | {controls['left']['previous_centered_minus_exact_A_plus_C_per_s']:.9e} | {controls['right']['previous_centered_minus_exact_A_plus_C_per_s']:.9e} |
| 分相浮点归约和−原中心差作用，s⁻¹ | {controls['left']['phase_float_sum_minus_previous_centered_per_s']:.9e} | {controls['right']['phase_float_sum_minus_previous_centered_per_s']:.9e} |

此外，完全采用精确物理二次单元平均，而非缓存中已舍入的cbar，仍有A_L=A_R、C_L=4C_R及相同的不相容性；其下界显示值亦为{100*r['physical_polynomial_compatibility']['universal_max_relative_error_lower']['decimal_for_reading_only']:.8f}%。两套结果分开记录，未挑选较好者替换主值。

## 6. 路线决定与边界

停止对此固定模板、固定同相交换类继续尝试共同跨相标量微调。它不能满足当前两个已知质量平均响应，且其误差有上述严格正下界。当前接触缺陷、Stage 3 incomplete和fluid ORC关闭状态保持，普通中心阻力物质FV基线保留。

该证书没有固定所有可能的边依赖系数、模板变化或额外闭合先验，因而不是全部正交换方法的不可能定理。它也没有解决有限点rho能否识别内部密度的问题，更没有解释原余弦场全域误差或证明瞬态收敛。

## 7. 证书、复现与完整性

7组精确代数证书分别核对：原分相与质量比、原始/中心差/对角缺陷恒等式、镜像仿射系数、两约束不相容、普遍误差下界、保留原行缺陷仍不相容、独立精确物质二次场证书。证明逻辑写在本文，程序不是完整自动定理证明器。

来源核验执行与独立进程复算结果完全一致。没有新增数值或流动测试，累计重建检查仍 **363项**；7组代数证书另列。旧v0.8.0/246项未恢复。782原始文件、15历史附件及52份先前正式科学清单保持；4份历史诊断清单既有路径差异另核。原README/METHOD/CODEX_TASK、原图、质量/身份/c0、PDE及旧结果均未改，仅更新可变索引。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/cross_scalar_certificate.py --output /workspace/cross-scalar-reproduction
python research/reconnect/cross_scalar_certificate.py --certificates-only --snapshot runs/reconnect-cross-scalar/cross-scalar.npz
```

第一条要求完整历史来源且输出目录尚不存在；第二条只需当前快照、代码和NumPy，Fraction运算来自标准库。`runs/reconnect-cross-scalar/` 保存不变缓存、分相掩码、全部精确分数、浮点控制、CSV、独立进程结果及审计。

本轮未执行任何t参数、创建或替换边权/对角/图、积分新算子、传播状态、扫描参数、拟合F/C、重归一化、改阈值、生成PDF/版本、提交/推送、修改环境配置或发布。

## 8. 下一步回收既有证据，决定候选路线

下一步只回查已封存的[本构与输入可辨识性](CONTACT_CONSTITUTIVE_REVIEW.md)、[质量平均矩锥](../rebuild/MASS_MOMENT_CONE.md)、[共享交换](../rebuild/SHARED_EXCHANGE.md)、[密度与几何](../rebuild/DENSITY_GEOMETRY.md)、[物质FV基线](../rebuild/HYBRID_MATERIAL_FACE.md)与[主线回接](MAINLINE_RECONNECTION.md)。把“受限标量/固定模板障碍”“有限输入不可辨识”“仍未证明的替代路线”分开，明确下一候选的状态、允许输入与闭合先验。

这项路线决策尚未完成。不重复原证明、LP或物理算例，不继续扫描旧共同标量，也不把真实面、F/C或速度梯度加入候选必要输入。先形成可审查的模型与输入假设，再考虑新的数值实现。

Stage 3 **incomplete**；fluid ORC **继续关闭**。

- [报告、精确证书与缓存审阅包](../../exports/ltn-cross-scalar-review.zip)。
- [完整增量包及十六份历史依赖](../../exports/ltn-cross-scalar-downloads.md)。
- [交付哈希](../../exports/ltn-cross-scalar-delivery.json)。
'''
    (root/'research/reconnect/CROSS_PHASE_SCALAR_COMPATIBILITY.md').write_text(report)
    p=root/'docs/RESEARCH_STATUS.md';text=p.read_text().replace(
        '52. [已存接触两侧的零通量二次物质平均探针](../research/reconnect/SAVED_INTERFACE_PAIR_PROBE.md)（本轮）。',
        '52. [已存接触两侧的零通量二次物质平均探针](../research/reconnect/SAVED_INTERFACE_PAIR_PROBE.md)。\n53. [共享跨相单标量不相容证书与误差下界](../research/reconnect/CROSS_PHASE_SCALAR_COMPATIBILITY.md)（本轮）。')
    text=text[:text.index('本轮关键进展：')]+f'''本轮关键进展：完成预定共同跨相标量理论审查。原缓存两行同相A精确相同，跨相C_L=4C_R>0，且A<D；两侧约束分别要求t约0.625003245266、2.500012981064，不能共用。对任意共同t>=0，至少一侧相对误差不小于{r['universal_max_relative_error_lower_percent']:.8f}%。下界由消去与三角不等式得到，没有参数扫描或调权实验。

精确二进制有理数证书、保留原对角常数缺陷的证书、独立精确物质二次均值证书均不相容；原CSR主值和浮点控制另行保留，未修对角、矩或状态。明确停止该受限共同跨相标量小修路线，不推广为所有边依赖正交换不可能。普通中心阻力FV基线保持。

新增7组精确代数证书，独立进程复算一致；无新增数值/流动测试，重建检查累计仍363项，旧v0.8.0/246项未恢复。782原文件、15历史附件和52份先前正式科学清单保持，4份历史诊断清单既有路径差异另核。

下一步只回收已封存的本构/可辨识性、质量平均矩锥、共享交换、密度几何、物质FV与主线回接证据，明确候选路线、输入与闭合先验；不重复证明/LP/算例，不继续扫共同标量，不新增真实面、F/C或速度梯度为候选必要输入。

数据入口：runs/reconnect-cross-scalar/stage3-status.json。Stage 3 incomplete，fluid ORC关闭；无PDF/新版本、提交/推送、环境配置修改或发布。
'''
    p.write_text(text)
    print(json.dumps({'CSVs':csvs,'universal_max_error_lower_percent':r['universal_max_relative_error_lower_percent'],
        'route_closed':'one common cross-phase scalar only','next_task':status['next_task']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
