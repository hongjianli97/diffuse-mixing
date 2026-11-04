"""Keep the adverse interface result; no scan or coefficient adjustment."""
from pathlib import Path
import argparse,json
import numpy as np
from package_compression_midpoint import dump,digest
from summarize_shear_transient_spatial import csv_exact


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve()
    assert not (out/'sha256.json').exists(),'never rewrite sealed evidence'
    r=json.loads((out/'results.json').read_text());cfg=json.loads((out/'config.json').read_text())
    audit=json.loads((out/'numerical-audit.json').read_text());assert audit['mass_weighted_region_and_arithmetic_diagnostic_kept_distinct']
    csvs=[]
    with np.load(out/'interface-pair.npz') as z:
        for side in ['left','right']:
            def get(key):return z[side+'_'+key]
            values=np.column_stack([get('row_columns'),get('original_material_label_centers'),get('original_mass'),get('original_rho0_point'),
                get('original_c0'),get('original_quadrature_area'),get('saved_L_row'),get('derived_Q_row_per_s'),get('normal_displacement'),
                get('probe_quadratic_mass_means'),get('independent_integrated_quadratic_mass_means'),get('raw_CSR_action_terms_per_s'),get('centered_action_terms_per_s')])
            csvs.append(csv_exact(out/(side+'-saved-row.csv'),get('original_ids'),values,['material_id','original_column_index','a_m','b_m',
                'original_mass_kg','original_rho_node','original_cosine_c0','saved_area_m2','saved_L_kg_per_s','derived_Q_per_s',
                'normal_displacement_m','probe_quadratic_mass_mean','independent_integrated_mass_mean','raw_action_term_per_s','centered_action_term_per_s']))
    dump(out/'observations-audit.json',{'CSVs':csvs,'raw_NPZ_sha256':digest(out/'interface-pair.npz')})
    left=r['rows']['left'];right=r['rows']['right'];region=r['two_cell_region']
    next_plan={'scope':'theory-only compatibility of one shared cross-phase scalar on the same two cached rows; no tuning experiment',
        'source_snapshot':'runs/reconnect-interface-pair/interface-pair.npz','source_snapshot_sha256':digest(out/'interface-pair.npz'),
        'target_indices':[63,64],'probe_origin_a_m':.5,'D_m2_per_s':.02,'fixed_original_masses':True,
        'restricted_family':'same within-phase exchange and original cross-phase stencil shape; a single shared nonnegative scalar multiplier of cross-phase exchange',
        'tasks':['decompose original centered action into same-phase and cross-phase contributions, retaining raw CSR and float residual separately',
            'derive the two material-average quadratic-response constraints and test symbolic scalar compatibility without a parameter scan',
            'separate a no-go for this restricted scalar class from all positive exchange models'],
        'no_parameter_fitted_or_used_to_replace_edges':True,'no_diagonal_or_moment_repair':True,
        'no_F_C_fitting_density_reconstruction_or_oracle_interface_backend':True,
        'ordinary_central_resistance_FV_baseline_retained':True,'new_candidate_graphs':0,'new_edge_weights':0,
        'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'new_operator_quadrature_evaluations':0,
        'parameter_or_mesh_matrix_scan':False,'fluid_orc_enabled':False,
        'stop_condition':'retain incompatible constraints; do not run scalar changes as passing candidates'}
    status={'updated':'2026-11-01','timezone':'Etc/UTC','stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'scope':'completed predeclared two-row zero-normal-flux quadratic interface probe',
        'left_raw_response_per_s':left['row']['raw_CSR_quadratic_response_per_s'],'left_relative_error_percent':left['relative_error_percent'],
        'right_raw_response_per_s':right['row']['raw_CSR_quadratic_response_per_s'],'right_relative_error_percent':right['relative_error_percent'],
        'correct_reference_response_both_sides_per_s':.02,'two_cell_mass_average_response_per_s':region['mass_average_rate_per_s'],
        'two_cell_mass_average_error_percent':100*region['mass_average_relative_error'],
        'two_cell_local_mass_weighted_relative_L2_percent':100*region['mass_weighted_local_relative_L2_error'],
        'arithmetic_mean_is_material_observation':False,'current_saved_candidate_contact_defect_retained':True,
        'old_cosine_errors_updated_or_reattributed':False,'original_mass_ids_c0_saved_PDE_and_candidate_unchanged':True,
        'new_checks_passed':3,'previous_recorded_checks':360,'recorded_checks_total':363,'old_v080_or_246_tests_recovered':False,
        'saved_row_evaluations':2,'new_candidate_graphs':0,'new_edge_weights':0,'new_operator_quadrature_evaluations':0,
        'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'finite_node_candidate_admitted':False,'contact_transient_admitted':False,
        'next_cross_phase_scalar_compatibility_review_predeclared':next_plan,'next_task':'same cached interface pair, theory-only shared cross-phase scalar compatibility',
        'remaining':['admissible interface constitutive closure','finite-input density identifiability','transient, shocks, walls, ledger, nonplanar/DNS physical acceptance'],
        'report':'research/reconnect/SAVED_INTERFACE_PAIR_PROBE.md','pdf_generated':False,'software_version_created':False,'published':False}
    dump(out/'stage3-status.json',status)
    rows=['| 原行 | rho | 原质量，kg | 原始CSR响应，s⁻¹ | 正确PDE响应，s⁻¹ | 相对误差 |','|---|---:|---:|---:|---:|---:|']
    for v in [left,right]:
        q=v['row'];rows.append(f"| {v['target_index']} | {v['target_rho']:g} | {q['original_mass_kg']:.12g} | {q['raw_CSR_quadratic_response_per_s']:.14f} | 0.02 | {v['relative_error_percent']:+.8f}% |")
    report=f'''# 已存密度接触两侧的零法向通量二次探针

2026-11-01（UTC）。Stage 3 **incomplete**；fluid ORC **关闭**。

预定只读界面探针完成，暴露明确接触缺陷。同一N128、h=0.04、D=0.02下，原行63、64对同一质量平均二次场的响应分别为 **{left['row']['raw_CSR_quadratic_response_per_s']:.11f}、{right['row']['raw_CSR_quadratic_response_per_s']:.11f} s⁻¹**，正确值均为0.02；误差分别为 **{left['relative_error_percent']:+.6f}%、{right['relative_error_percent']:+.6f}%**。

两格原质量比为1:4。它们组成的物质区域平均响应为 **{region['mass_average_rate_per_s']:.11f} s⁻¹，偏低{abs(100*region['mass_average_relative_error']):.6f}%**；局部质量加权相对L2误差为{100*region['mass_weighted_local_relative_L2_error']:.6f}%。接近正确值的算术平均不能代表这个物质区域，不能用异号抵消制造通过。

## 1. 固定来源与独立物理参考

执行 `runs/reconnect-point-row/stage3-status.json` 已封存的 `next_saved_interface_pair_predeclared`，先核验1090项科学清单。仍读取同一原图，SHA-256为 `{cfg['source_graph_sha256']}`；未构图或改任何行。只对原行63、64作一次局部二次探针，原m/c0/身份、密度、面积、PDE和余弦场结果均保持。

接触a*=0.5，局部辅助场c=(a−a*)²/2。它在接触处c=0、c_a=0，所以浓度及两侧rho D c_a均连续，没有界面Dirac源。二次片覆盖两行全部核支撑，其他区域可平滑周期延拓；没有将这一解析辅助场写回原c0。

每个原方格完全在其本相内，真实质量平均为(a_center−a*)²/2+ell²/24，ell=1/128。独立参考以Fraction精确积分rho*c及左右物质面扩散通量，并除以**原保存质量**：

q_left=−rho D(a_left−a*)，q_right=−rho D(a_right−a*)，

PDE平均变化率=ell*(q_left−q_right)/原m=D。

左格右面与右格左面的q恰为零；两格精确cbar均为1/98304。独立积分得到的全部邻居单元平均与预定中心公式逐项相同，两个原始CSR作用也完全一致。这排除了将点值当平均、错质量分母或错误参考界面源造成此结果。

'''+'\n'.join(rows)+f'''

## 2. 保留原始CSR作用和控制矩

每行保留原333项（332个非对角项），其中156项连接另一相；原m/rho、已存求积面积和原方格参考面积逐项相同。所有行邻居仍在原2h支撑内；没有补边、剥皮、裁剪或改变阈值。

以原始CSR dot为事先指定的主值，不改为更接近PDE的控制形式：

| 原始控制 | 行63，rho=1 | 行64，rho=4 |
|---|---:|---:|
| 常数作用，s⁻¹ | {left['row']['raw_constant_response_per_s']:.9e} | {right['row']['raw_constant_response_per_s']:.9e} |
| 一阶法向矩，m/s | {left['row']['first_normal_moment_m_per_s']:.12f} | {right['row']['first_normal_moment_m_per_s']:.12f} |
| 半二阶法向矩，m²/s | {left['row']['half_second_normal_moment_m2_per_s']:.14f} | {right['row']['half_second_normal_moment_m2_per_s']:.14f} |
| 原始作用−中心差形式，s⁻¹ | {left['row']['raw_minus_centered_per_s']:.9e} | {right['row']['raw_minus_centered_per_s']:.9e} |
| 原始作用−矩重构，s⁻¹ | {left['row']['raw_minus_moment_reconstruction_per_s']:.9e} | {right['row']['raw_minus_moment_reconstruction_per_s']:.9e} |

原始作用可按cbar_target*常数作用+(a_center−a*)*一阶矩+半二阶矩重构。原始浮点余差原样保存，不强制为零。约±0.00404 s⁻¹的偏差远大于这些数值控制，不能归为舍入误差。

相邻两行之间的同一已存L非对角项逐字相同，为{r['saved_shared_offdiagonal_L_left_to_right_kg_per_s']:.12e} kg/s。两目标cbar相同，故这条直接相邻边的交换差为零；其他非局部邻居仍会影响两行。不能从一条边的零交换或共享对称性推断局部PDE正确。

## 3. 两格物质观测与算术诊断分开

物质区域K63 union K64的平均变化率定义为(m63*r63+m64*r64)/(m63+m64)，使用原质量，不改变任何m、状态或边权。

| 两格观测/诊断 | 数值 |
|---|---:|
| 原总质量，kg | {region['total_original_mass_kg']:.12g} |
| 正确物质平均响应，s⁻¹ | {region['mass_average_reference_per_s']:.12f} |
| 原图物质平均响应，s⁻¹ | {region['mass_average_rate_per_s']:.14f} |
| 物质平均相对误差 | {100*region['mass_average_relative_error']:+.8f}% |
| 两格局部质量加权相对L2误差 | {100*region['mass_weighted_local_relative_L2_error']:.8f}% |
| 算术平均，仅诊断，s⁻¹ | {region['arithmetic_mean_diagnostic_only_per_s']:.14f} |
| 算术平均相对误差，仅诊断 | {100*region['arithmetic_mean_relative_error_diagnostic_only']:+.10f}% |

L2使用本辅助探针在这两格的局部质量范数，不是原余弦场全域误差定义的更新，不能与原14.16%或3.12%直接比较。两格也不是封闭交换系统，局部误差库存不是全图守恒残差。

![两侧响应与正确物质区域观测](../../runs/reconnect-interface-pair/interface-pair.png)

## 4. 当前路线判定

上一轮纯相行31的法向二次响应接近正确值，不能推广为接触一致性。本轮在无界面源、原质量平均和独立面通量参考下仍有约20%的两侧缺陷，当前点候选不能据此推进接触瞬态或fluid ORC。

保留全部不利结果，不用纯相理想闭合的2.86%修正当前两行，不调核宽或跨相系数，不重新归一化矩，也不以算术平均抵消替代物质观测。普通中心阻力物质FV基线继续保留。

本轮只确认同一接触与同一零通量二次探针的局部问题，没有给出新收敛阶、全场掺混预测或所有正交换方法的不可能结论。

## 5. 检查、复现与完整性

新增3项检查首次通过：两相独立精确状态/面通量参考、异号响应在1:4原质量下与算术平均的区别、错误质量输入不会被参考过程悄悄修复。累计360+3=**363项重建检查**；旧v0.8.0/246项未恢复，未重跑全部历史测试。

审计核验源图哈希、两行原始字节/列索引及邻居原质量/c0/身份/密度/面积/余弦场参照，逐项复算CSR主值、矩控制、独立Fraction参考与物质区域观测。两个CSV按原身份和全部浮点解析后与NPZ完全一致。782原始文件、15历史附件及51份先前正式科学清单保持；4份历史诊断清单既有路径差异另核。原README/METHOD/CODEX_TASK、旧后端与结果不改，只更新可变索引。峰值RSS约{r['peak_process_RSS_MiB_linux']:.1f} MiB。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/test_interface_pair_probe.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/interface_pair_study.py --output /workspace/interface-pair-reproduction
```

`runs/reconnect-interface-pair/` 保存两行完整快照、独立参考、逐项作用、CSV/图、测试与审计。无新边权/矩阵/图、算子求积、流动/参考瞬态、参数扫描、拟合F/C、裁剪、归一化修正、阈值更改、PDF、版本、提交/推送、环境配置修改或发布。

## 6. 下一步只审查受限共享标量的相容性

复用本轮缓存的两行，在固定同相交换和跨相模板形状下，将“跨相交换共同乘一个非负标量”只写成符号约束。分解原中心差作用的同相/跨相贡献，原始CSR与浮点余差另外保留，检查两个质量平均响应目标是否相容。

这项理论审查尚未完成。它不扫描或拟合一个参数去替换边权，不修对角或矩，不构造新的候选，也不把某个受限标量类的结论推广到所有共享正交换。其目的是判断是否应彻底停止这类单标量小修路线，而不是继续做调参通过实验。

Stage 3 **incomplete**；fluid ORC **继续关闭**。

- [报告、代码与完整两行记录审阅包](../../exports/ltn-interface-pair-review.zip)。
- [完整增量包及十五份历史依赖](../../exports/ltn-interface-pair-downloads.md)。
- [交付哈希](../../exports/ltn-interface-pair-delivery.json)。
'''
    (root/'research/reconnect/SAVED_INTERFACE_PAIR_PROBE.md').write_text(report)
    p=root/'docs/RESEARCH_STATUS.md';text=p.read_text().replace(
        '51. [同一二次探针的已存点图行比较](../research/reconnect/SAVED_POINT_ROW_COMPARISON.md)（本轮）。',
        '51. [同一二次探针的已存点图行比较](../research/reconnect/SAVED_POINT_ROW_COMPARISON.md)。\n52. [已存接触两侧的零通量二次物质平均探针](../research/reconnect/SAVED_INTERFACE_PAIR_PROBE.md)（本轮）。')
    text=text[:text.index('本轮关键进展：')]+f'''本轮关键进展：完成预定只读界面行63/64探针。c=(a-0.5)^2/2的两侧原始CSR响应为{left['row']['raw_CSR_quadratic_response_per_s']:.11f}、{right['row']['raw_CSR_quadratic_response_per_s']:.11f} s^-1，独立质量积分/面通量参考均为0.02，误差分别{left['relative_error_percent']:+.6f}%、{right['relative_error_percent']:+.6f}%。原质量比1:4下两格物质平均偏低{abs(100*region['mass_average_relative_error']):.6f}%，局部质量加权相对L2误差{100*region['mass_weighted_local_relative_L2_error']:.6f}%；接近正确值的算术平均仅为诊断，不能用于制造通过。

辅助场在界面浓度和法向rho D c_a连续、无Dirac源，独立精确单元均值和预定均值逐项一致；原常数/一二阶矩与浮点余量均保留，无行和或2.86%纯相修正。原m/c0/身份、密度、面积、原余弦PDE和候选结果保持。该约20%局部缺陷阻止接触瞬态/ORC接纳，但不外推为全部正交换不可能。

新增3项检查首次通过，累计363项；旧v0.8.0/246项未恢复。782原文件、15历史附件和51份先前正式科学清单保持，4份历史诊断清单既有路径差异另核。下一步只复用缓存两行作受限共享跨相标量的符号相容性审查，不扫描、拟合参数替换边权、修对角或构新候选。普通中心阻力FV基线保持。

数据入口：runs/reconnect-interface-pair/stage3-status.json。Stage 3 incomplete，fluid ORC关闭；无PDF/新版本、提交/推送、环境配置修改或发布。
'''
    p.write_text(text)
    print(json.dumps({'CSVs':csvs,'left_error_percent':left['relative_error_percent'],'right_error_percent':right['relative_error_percent'],
        'mass_average_error_percent':100*region['mass_average_relative_error'],'next_task':status['next_task']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
