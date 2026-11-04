"""Keep the pure-phase row result local; predeclare a contact-pair probe."""
from pathlib import Path
import argparse,json
import numpy as np
from package_compression_midpoint import dump,digest
from summarize_shear_transient_spatial import csv_exact


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve()
    assert not (out/'sha256.json').exists(),'never rewrite sealed evidence'
    r=json.loads((out/'results.json').read_text());cfg=json.loads((out/'config.json').read_text())
    audit=json.loads((out/'numerical-audit.json').read_text());assert audit['all_signed_terms_and_float_identity_residual_recomputed']
    with np.load(out/'point-row.npz') as z:
        ids=z['original_ids'];values=np.column_stack([z['row_columns'],z['original_material_label_centers'],z['original_mass'],z['original_rho0_point'],
            z['original_c0'],z['original_quadrature_area'],z['saved_L_row'],z['derived_Q_row_per_s'],z['normal_displacement'],
            z['probe_quadratic_mass_means'],z['raw_CSR_action_terms_per_s'],z['centered_action_terms_per_s']])
        csv=csv_exact(out/'saved-row.csv',ids,values,['material_id','original_column_index','a_m','b_m','original_mass_kg','original_rho_node',
            'original_cosine_c0','saved_quadrature_area_m2','saved_L_kg_per_s','derived_Q_per_s','normal_displacement_m',
            'auxiliary_quadratic_mass_mean','raw_CSR_action_term_per_s','centered_action_term_per_s'])
    dump(out/'observations-audit.json',{'CSV':csv,'raw_NPZ_sha256':digest(out/'point-row.npz')})
    row=r['row'];budget=r['signed_budget'];graph=cfg['protocol']['source_graph']
    next_plan={'scope':'one read-only interface-adjacent two-row quadratic transmission probe on the same saved graph',
        'source_graph':graph,'source_graph_sha256':cfg['source_graph_sha256'],'target_indices':[63,64],'target_ids':[63,64],
        'N_per_direction':128,'h_m':.04,'D_m2_per_s':.02,'cell_side_m':'1/128','contact_origin_a_m':.5,
        'probe':'local c=(a-0.5)^2/2; aligned square mass average (a_center-0.5)^2/2+ell^2/24',
        'transmission_reference':'c and rho*D*c_a continuous at contact because c_a=0 there; local PDE mass-average response D on both sides, no Dirac source',
        'global_extension':'analytic quadratic patch covers both row supports; smooth periodic extension elsewhere, no replacement original c0',
        'main_action_preselected':'raw original CSR action / saved original row mass; centered form and raw 0/1/2 moments retained as controls',
        'no_pure_phase_ideal_excess_correction':True,'no_new_ideal_interface_integrals_or_density_reconstruction':True,
        'original_mass_ids_cosine_c0_and_saved_PDE_and_candidate_unchanged':True,
        'no_new_edge_weights_or_graph_or_matrix_scan':True,'no_fitting_clipping_renormalization_or_threshold_changes':True,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'fluid_orc_enabled':False,
        'stop_condition':'record either side defect without tuning; no transient or ORC admission'}
    status={'updated':'2026-11-01','timezone':'Etc/UTC','stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'scope':'completed predeclared read-only saved pure-phase point row comparison',
        'raw_point_response_per_s':r['raw_point_response_per_s'],'PDE_unit_curvature_response_per_s':r['PDE_unit_curvature_response_per_s'],
        'point_relative_error_percent':r['point_relative_error_percent'],'point_relative_error_ppm':r['point_relative_error_ppm'],
        'saved_ideal_direct_response_per_s':r['saved_ideal_direct_response_per_s'],'signed_budget':budget,
        'row_sum_repair_or_better_value_selection':False,'one_row_quadratic_is_physical_acceptance':False,
        'old_cosine_errors_recomputed_or_attributed_to_this_budget':False,
        'new_checks_passed':4,'previous_recorded_checks':356,'recorded_checks_total':360,'old_v080_or_246_tests_recovered':False,
        'saved_point_row_evaluations':1,'new_candidate_graphs':0,'new_edge_weights':0,'new_operator_quadrature_evaluations':0,
        'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'finite_node_candidate_admitted':False,'contact_transient_admitted':False,
        'next_saved_interface_pair_predeclared':next_plan,'next_task':'same saved graph interface rows 63/64 on a zero-normal-flux quadratic mass-mean probe',
        'remaining':['interface transmission and general density/geometry consistency','finite-input subcell density closure','transient, shocks, walls, ledger, nonplanar/DNS acceptance'],
        'report':'research/reconnect/SAVED_POINT_ROW_COMPARISON.md','pdf_generated':False,'software_version_created':False,'published':False}
    dump(out/'stage3-status.json',status)
    report=f'''# 同一辅助二次场的已存点图行比较

2026-11-01（UTC）。Stage 3 **incomplete**；fluid ORC **关闭**。

预定只读比较完成。原N128/h0.04/D0.02的纯相行31，按事先指定的**原始CSR作用**得到 **{r['raw_point_response_per_s']:.14f} s⁻¹**，相对正确值0.02的误差为 **{r['point_relative_error_percent']:.10f}%（{r['point_relative_error_ppm']:.6f} ppm）**。

同一探针的已存理想质量平均闭合响应为{r['saved_ideal_direct_response_per_s']:.14f} s⁻¹。预定中间参照下，点−理想与理想−PDE分别约−2.8611871%、+2.8610123%，几乎抵消。**更准确地求积到该理想分片常数闭合，并不自动等于更准确的局部PDE离散。** 本轮未修改原算法，也不把这个特定预算当作唯一因果解释。

## 1. 同一行、原始质量和质量平均探针

执行 `runs/reconnect-ideal-row/stage3-status.json` 的 `next_saved_point_row_review_predeclared`，先核验1063项科学清单。原图文件SHA-256为 `{cfg['source_graph_sha256']}`，与预定值逐字一致。

只对原CSR第31行作数值分析，保留其{row['row_entries']}项（对角1项，非对角{row['offdiagonal_entries']}项）原始值、列索引和数据类型。NPZ存储需要解压已有数组，但没有组装新的稀疏/稠密矩阵、重新计算边权或构图。原全图、初场、PDE及原余弦场结果保持。

目标a=0.24609375、rho=1、原质量{row['original_mass_kg']:.12g} kg；全部行邻居均处于同一纯相。原m/rho、已存求积面积和原方格参考面积逐项相同，均为(1/128)²；最大行邻居距离{r['maximum_saved_row_neighbor_distance_m']:.12f} m<2h。

辅助二次场c=a²/2仍使用正确单元平均a_center²/2+ell²/24，未将其称作点值或覆盖原c0。该局部二次片可在远离本行支撑的区域光滑延拓，不需要在周期接触处引入额外源。原始CSR主值在计算前已固定为−dot(L_row, cbar_at_columns)/原m，未改成更接近PDE的中心差值。

## 2. 带符号结果，保留总量

| 同一辅助探针的量 | s⁻¹ | 除以PDE响应后的百分比 |
|---|---:|---:|
| 点图−已存理想直接响应 | {budget['point_minus_ideal_per_s']:.12e} | {100*budget['point_minus_ideal_relative_to_PDE']:+.10f}% |
| 已存理想直接响应−PDE | {budget['ideal_minus_PDE_per_s']:.12e} | {100*budget['ideal_minus_PDE_relative_to_PDE']:+.10f}% |
| 点图−PDE | {budget['point_minus_PDE_per_s']:.12e} | {r['point_relative_error_percent']:+.10f}% |

前两项之和减最后一项的原始浮点余量为{budget['signed_identity_residual_per_s']:.9e} s⁻¹，本次恰为零，程序未强制置零。采用的是上一轮预先指定的细规则**原始直接理想响应**，没有换成解析矩预测或较接近PDE的比较量。

本行纯相中端点谐和系数与路径系数同为rho D，体积权重也已知准确。因此这里可以把点−理想视为该特定分片常数中间参照的点对求积差，把理想−PDE视为平均闭合差；两者不是相互独立的因果份额，也不能由范数大小推断原算法普遍依赖偶然抵消。它们的定义依赖已固定的中间参照。

![同探针响应与带符号预算](../../runs/reconnect-point-row/point-row.png)

## 3. 未修复的行和与矩控制

记d_a为列节点相对目标的法向位移，Q=-M⁻¹L。以下全部保留原值：

| 只读控制 | 数值 |
|---|---:|
| 原始CSR二次作用，s⁻¹ | {row['raw_CSR_quadratic_response_per_s']:.17g} |
| 中心差形式作用，s⁻¹ | {row['centered_difference_response_per_s']:.17g} |
| 两者差，s⁻¹ | {row['raw_minus_centered_per_s']:.9e} |
| 原始CSR对常数的作用，s⁻¹ | {row['raw_constant_response_per_s']:.9e} |
| 原始L行逐项sum，kg/s | {row['raw_L_row_sum_kg_per_s']:.9e} |
| 法向一阶矩sum Q*d_a，m/s | {row['first_normal_moment_m_per_s']:.9e} |
| 半法向二阶矩sum Q*d_a²/2，m²/s | {row['half_second_normal_moment_m2_per_s']:.17g} |
| 二次作用−矩重构，s⁻¹ | {row['raw_minus_moment_reconstruction_per_s']:.9e} |
| 逐项作用sum−原dot主值，s⁻¹ | {row['term_sum_minus_dot_response_per_s']:.9e} |

dot与逐项sum在对角/非对角抵消中的浮点归约次序不同；原始常数作用与−sum(L)/m因而可能有舍入尺度差异。没有选更小的残差替换主值，也没有修对角、补行和或归一化二阶矩。约3.50e−8 s⁻¹的PDE响应差明显大于这些浮点控制差，不能写成精确二次一致。

## 4. 当前结论与适用边界

原点图在这个纯相、一个位置、一个法向二次质量平均探针上接近正确响应；这不是空间收敛阶、完整各向同性、密度接触或有限时间掺混验收。原保存余弦态的约14.16%点图误差、3.12%oracle连续核误差均保持其原定义，本轮不重新解释它们。

尤其不能把理想平均闭合的2.86%当作原点图必须具有的误差，或拿它去校正现有边权。普通中心阻力物质FV基线保留。后续要检验的仍是界面传输及更一般密度/几何。

## 5. 检查、复现与完整性

新增4项检查首次通过：已知单行的原质量/二次平均作用、故意带行和缺陷时保留原始响应、有符号预算及非零浮点闭合余差保留、非法质量/缺失对角拒绝。仅用行向量代数夹具，没有创建新图。重建检查累计356+4=**360项**；旧v0.8.0/246项未恢复，也不是本轮重跑全部历史测试。

审计核验源图哈希、原行字节/列索引、所有邻居原质量/身份/c0/密度/参考数组，逐项复算主值、控制矩和带符号预算；CSV按原身份类型与全部浮点解析后与NPZ完全一致。782个原始文件、15份历史附件和50份先前正式科学清单保持；4份历史诊断清单既有路径差异另核。原README/METHOD/CODEX_TASK、后端和旧结果均不改，只更新可变索引。峰值RSS约{r['peak_process_RSS_MiB_linux']:.1f} MiB。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/test_point_row_response.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/point_row_study.py --output /workspace/point-row-reproduction
```

`runs/reconnect-point-row/` 保存原始行/邻居快照、辅助探针、逐项作用、CSV、图、测试和审计。无新边权、矩阵/图重建、算子求积、流动/参考瞬态、参数矩阵扫描、裁剪、拟合F/C、重归一化、阈值修改、PDF、版本、提交/推送、环境配置修改或发布。

## 6. 下一步只读同一接触两侧相邻行

预定下一步固定原行63、64，接触位于a=0.5；使用局部c=(a−0.5)²/2及其精确方格质量平均。该场在界面浓度连续、c_a=0，故法向rho D c_a连续，没有Dirac源；两侧局部PDE平均变化率均应为D。二次片覆盖两行及其核支撑，其他区域可平滑周期延拓，不替换原c0。

仍以原始CSR作用为主值，保存两侧原质量、常数/一二阶矩与中心差控制。只读既有两行，不建新图、不作新界面理想积分、不用纯相2.86%校正，不因任何一侧偏差调权或改阈值。这项界面探针尚未运行。

Stage 3 **incomplete**；fluid ORC **继续关闭**。

- [报告、代码与完整单行记录审阅包](../../exports/ltn-point-row-review.zip)。
- [完整增量包及十四份历史依赖](../../exports/ltn-point-row-downloads.md)。
- [交付哈希](../../exports/ltn-point-row-delivery.json)。
'''
    (root/'research/reconnect/SAVED_POINT_ROW_COMPARISON.md').write_text(report)
    p=root/'docs/RESEARCH_STATUS.md';text=p.read_text().replace(
        '50. [原比例的一行理想质量平均二次响应](../research/reconnect/IDEAL_MATERIAL_ROW_RESPONSE.md)（本轮）。',
        '50. [原比例的一行理想质量平均二次响应](../research/reconnect/IDEAL_MATERIAL_ROW_RESPONSE.md)。\n51. [同一二次探针的已存点图行比较](../research/reconnect/SAVED_POINT_ROW_COMPARISON.md)（本轮）。')
    text=text[:text.index('本轮关键进展：')]+f'''本轮关键进展：完成预定只读原点图行31比较。原始CSR二次质量平均响应{r['raw_point_response_per_s']:.14f} s^-1，正确值0.02，相对误差{r['point_relative_error_percent']:.10f}%（{r['point_relative_error_ppm']:.6f} ppm）。固定中间理想参照下，点−理想约−2.8611871%、理想−PDE约+2.8610123%，几乎抵消；保留有符号账本，不作唯一因果份额或原余弦态归因。

333项原始行及332个邻接项、原质量/c0/身份、保存面积与纯相假设均核验；主值事先固定为原CSR dot，中心差/常数/一二阶矩和浮点余量保留，无行和或矩修正。只读取既有矩阵，未重建或求积。新增4项检查首次通过，累计360项；旧v0.8.0/246项未恢复。

782原文件、15历史附件和50份先前正式科学清单保持，4份历史诊断清单既有路径差异另核。单个纯相法向二次矩接近正确值不是空间/界面/瞬态通过。下一步只读同一接触相邻行63/64，用以界面0.5为原点、零界面法向通量的局部二次质量平均检验两侧响应，参考均为D；不建图、调边、用纯相2.86%校正或改阈值。普通中心阻力FV基线保持。

数据入口：runs/reconnect-point-row/stage3-status.json。Stage 3 incomplete，fluid ORC关闭；无PDF/新版本、提交/推送、环境配置修改或发布。
'''
    p.write_text(text)
    print(json.dumps({'CSV':csv,'point_relative_error_ppm':r['point_relative_error_ppm'],'signed_budget':budget,'next_task':status['next_task']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
