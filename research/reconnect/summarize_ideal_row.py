"""Record one original-ratio row; no refitting or candidate promotion."""
from pathlib import Path
import argparse,json
import numpy as np
from package_compression_midpoint import dump,digest
from summarize_shear_transient_spatial import csv_exact


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve()
    assert not (out/'sha256.json').exists(),'never rewrite sealed evidence'
    r=json.loads((out/'results.json').read_text());cfg=json.loads((out/'config.json').read_text())
    a=json.loads((out/'numerical-audit.json').read_text())
    assert a['all_response_metrics_rule_differences_and_strict_bounds_recomputed']
    csvs=[]
    with np.load(out/'ideal-row.npz') as z:
        for name in ['coarse','fine']:
            keys=['s','normal_weights','panel','kernel_transverse_b0','cell_shift_integer','fractional_phase','eta_integrand',
                'direct_phase_mean_jump','direct_phase_mean_quadratic_delta','phase_variance_identity_residual']
            values=np.column_stack([z[name+'_'+key] for key in keys])
            csvs.append(csv_exact(out/(name+'-quadrature.csv'),np.arange(len(values),dtype=np.int64),values,
                ['quadrature_node_id',*keys]))
        target=cfg['target_index'];center=float(z['material_label_centers'][target,0]);ell=1/128
        assert min(center,.5-center)>2*.04+1.5*ell
    dump(out/'observations-audit.json',{'quadrature_CSVs':csvs,'raw_NPZ_sha256':digest(out/'ideal-row.npz'),
        'all_neighbor_material_cells_pure_phase_margin_m':min(center,.5-center)-2*.04-1.5*ell})
    fine=r['rules']['fine'];delta=r['coarse_fine_absolute_differences']
    graph='runs/reconnect-contact-frozen/contact-frozen.npz'
    next_plan={'scope':'read-only quadratic response of the same one existing point-candidate CSR row; no graph rebuild',
        'source_graph':graph,'source_graph_sha256':digest(root/graph),'target_index':target,'target_id':cfg['target_id'],
        'N_per_direction':128,'h_m':.04,'D_m2_per_s':.02,'cell_side_m':'1/128',
        'probe':'same auxiliary c=a^2/2 and exact square-cell mass means; original cosine c0 unchanged',
        'read_only_keys':['mass','ids','material_label_centers','rho0_point','L_data','L_indices','L_indptr','L_shape'],
        'main_action_preselected':'raw CSR action -dot(L_row, quadratic_cell_means_at_columns)/saved_mass; no row-sum repair',
        'controls':['raw constant and first/second normal moments','centered-difference action retained separately from raw CSR action',
            'all nonzero neighbors in same pure phase; original point-density quadrature area checked against existing square reference'],
        'signed_comparison':'point-minus-saved-ideal-direct + saved-ideal-direct-minus-PDE = point-minus-PDE, preserving floating residual',
        'saved_ideal_response_source':'runs/reconnect-ideal-row/results.json',
        'saved_ideal_response_key':'rules.fine.raw_direct_aggregate_rate_per_s',
        'interpretation':'same probe only; no attribution of auxiliary row terms to saved cosine errors or unique causal shares',
        'no_new_edge_weights_or_full_matrix':True,'no_fitting_clipping_renormalization_or_threshold_changes':True,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'parameter_or_mesh_matrix_scan':False,'fluid_orc_enabled':False}
    status={'updated':'2026-11-01','timezone':'Etc/UTC','stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'scope':'completed one predeclared original-ratio ideal aggregate quadratic row integral',
        'ideal_reference_only':True,'fine_eta':fine['eta'],'fine_eta_percent':r['fine_eta_percent'],
        'PDE_unit_curvature_rate_per_s':r['auxiliary_PDE_rate_per_s'],
        'fine_exact_moment_prediction_rate_per_s':fine['effective_rate_exact_moment_prediction_per_s'],
        'fine_raw_direct_rate_per_s':fine['raw_direct_aggregate_rate_per_s'],'rule_differences':delta,
        'direct_phase_and_variance_checks_retained':True,'quadrature_difference_is_rigorous_bound':False,
        'old_cosine_errors_recomputed_or_refitted':False,'candidate_and_original_mass_ids_c0_and_PDE_unchanged':True,
        'new_checks_passed':4,'previous_recorded_checks':352,'recorded_checks_total':356,'old_v080_or_246_tests_recovered':False,
        'aggregate_row_rule_evaluations':2,'edge_weights_created':0,'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'finite_node_candidate_admitted':False,'contact_transient_admitted':False,
        'next_saved_point_row_review_predeclared':next_plan,'next_task':'read-only same-probe action of existing point-candidate row 31',
        'remaining':['same-probe point-versus-ideal signed comparison','finite-node subcell density closure','transient and general physical acceptance'],
        'report':'research/reconnect/IDEAL_MATERIAL_ROW_RESPONSE.md','pdf_generated':False,'software_version_created':False,'published':False}
    dump(out/'stage3-status.json',status)
    table=['| 原规则 | eta | 单位曲率原始直接响应，s⁻¹ | D(1+eta)，s⁻¹ |','|---|---:|---:|---:|']
    for name,label in [('coarse','32/32'),('fine','64/64')]:
        v=r['rules'][name];table.append(f"| {label} | {v['eta']:.17g} | {v['raw_direct_aggregate_rate_per_s']:.17g} | {v['effective_rate_exact_moment_prediction_per_s']:.17g} |")
    report=f'''# 原比例的一行理想质量平均二次响应

2026-11-01（UTC）。Stage 3 **incomplete**；fluid ORC **关闭**。

完成预定单行控制。同一N128、h=0.04、D=0.02、ell/h=25/128下，理想单元对积分闭合的 **eta={fine['eta']:.12f}，相对扩散率偏大{r['fine_eta_percent']:.8f}%**。单位二次曲率的正确变化率为0.02 s⁻¹，原始直接子单元积分为 **{fine['raw_direct_aggregate_rate_per_s']:.14f} s⁻¹**，对应有效扩散率约0.020572202464 m²/s。

该结果量化的是上一轮已证明的**理想质量平均闭合偏差**，不是原保存点候选的误差，也不是原余弦态3.1229%或其方向误差的解释。本轮未实现新后端或提高算法准确率。

## 1. 来源与同一原质量

执行 `runs/reconnect-mass-projection/stage3-status.json` 的 `next_single_row_response_predeclared`，先核验1035项科学清单。原 `runs/reconnect-contact-frozen/contact-observations.npz` 中m、ids、c0、节点rho、物质标签、原PDE和点候选作用逐项原样复制并核验。

按固定几何规则选择原第0行、第N/4−1列，索引/身份{target}，a中心={center} m，rho=1，原质量={cfg['saved_original_mass_kg']:.12g} kg。该单元及其核配对可能涉及的完整邻居单元均在原低密度纯相，最小剩余几何裕量约{min(center,.5-center)-.08-1.5*ell:.6f} m；没有按误差选点。

原余弦初场的c0[{target}]={cfg['unchanged_original_c0_mass_average']:.15g}保持。辅助解析二次场c=a²/2的正确单元平均是a_center²/2+ell²/24={fine['analytic_quadratic_target_mass_average']:.15g}；它单独记录，**没有覆盖原c0，也未当成轨迹点值**。下述变化率针对单位二次曲率。

## 2. 只执行两组预定求积

使用原始B_h及其既有横向偶对称积分。法向范围[−2h,2h]在0、所有内部整数单元宽度s=k*ell处拆分，共22个面板；两规则法向节点分别704、1408。法向/横向阶数仅为预定32/32、64/64，最细原始规则事先指定，没有外推或按接近PDE挑选。

eta=ell² integral b0(s)*frac(s/ell)*(1−frac(s/ell)) ds。

原核精确二阶矩为1，但另外保留求积原值M2。直接子单元积分在每个目标单元相位区间上以二点Gauss积分**邻居二次单元平均**之差，所有几何跳点显式拆开；该分片常数积分在各段是精确的。这是同一行内独立的几何核对，没有生成单个边权。

'''+'\n'.join(table)+f'''

细规则原始方差响应按Gamma*ell²*(M2+eta)/原m计算，为{fine['raw_variance_aggregate_rate_per_s']:.17g} s⁻¹；原始直接响应与之差为{fine['raw_direct_minus_raw_variance_per_s']:.6e} s⁻¹。两项均保留，不强制相等。

## 3. 必要数值控制与未修正余差

| 控制量 | 原始值 |
|---|---:|
| 两规则eta差 | {delta['eta']:.9e} |
| 两规则D(1+eta)差，s⁻¹ | {delta['effective_rate_exact_moment_prediction_per_s']:.9e} |
| 两规则原始直接响应差，s⁻¹ | {delta['raw_direct_aggregate_rate_per_s']:.9e} |
| 细规则M2−1 | {fine['raw_second_moment_minus_exact_one']:.9e} |
| 细规则mu0相对精确9/(2h²)之差 | {fine['raw_zero_moment_relative_difference_from_exact']:.9e} |
| 粗规则mu0相对差 | {r['rules']['coarse']['raw_zero_moment_relative_difference_from_exact']:.9e} |
| 细规则原始方差响应−精确矩预测，s⁻¹ | {fine['raw_variance_minus_exact_moment_prediction_per_s']:.9e} |
| 细规则直接几何相位/方差恒等式最大余差 | {fine['direct_quadratic_phase_identity_max_error']:.9e} |

两档差已经处于浮点舍入尺度，不是严格误差上界或16位有效数字保证。已测控制量明显小于约2.86%的闭合偏差；本轮只是单行数值量化，不是新的物理验收。eta还落在上一轮严格解析区间(2.278904e−6,0.042915345)内。

**D(1+eta)使用解析矩恒等式作独立预测，原始响应则保留数值M2。没有除以M2或1+eta修正核/响应，没有裁剪分数相位、重归一化或修改阈值。** 峰值RSS约{r['peak_process_RSS_MiB_linux']:.1f} MiB。

![原比例单行响应与闭合余量积分](../../runs/reconnect-ideal-row/ideal-row.png)

## 4. 结论边界

该值确认理想分片常数质量平均闭合在原比例下的有效扩散率偏差。它没有量化端点系数、有限点求积或原余弦场残差，也没有证明原点候选的同一偏差为2.86%。额外真实单元几何仍只用于此oracle参照。普通中心阻力物质FV基线保留。

只保存两规则的完整求积节点/正权重、相位、原b0、面板贡献、独立直接相位均值及数值余差；没有构造全图、传播标量或模拟流动。因此不作新的全图守恒、最大值原理或有限时间掺混精度声明。

## 5. 检查与完整性

新增4项检查首次运行全部通过：有符号位移的独立二次单元相位积分、原核解析0/2阶矩、直接aggregate行与方差恒等式、非法质量/几何输入拒绝。重建检查累计352+4=**356项**；不是本轮重跑全部旧测试，旧v0.8.0/246项仍未恢复。

审计逐项复算两规则全部原始记录、直接相位积分、原质量分母、指标和规则差；CSV按原节点序号类型/全部浮点解析后与NPZ完全一致。782个原始文件、15份历史附件和49份先前正式科学清单保持；4份历史诊断清单已有路径差异另核。原README/METHOD/CODEX_TASK、旧后端和结果不改，仅更新可变进度入口。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/test_ideal_row_response.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/ideal_row_study.py --output /workspace/ideal-row-reproduction
```

`runs/reconnect-ideal-row/` 保存原输入/比较量、两规则原始数组、CSV、图、测试与审计。没有新边权、候选图、流动或参考瞬态、全矩阵扫描、PDF、版本、提交/推送、环境配置更改或发布。

## 6. 下一步只读取同一个已存点图行

已确认原 `contact-frozen.npz` 保存CSR数组。下一步固定同一索引{target}、原质量和同一辅助二次平均，直接读取这一行计算原始CSR作用与常数/一二阶矩；不重新计算边权或构图。原始CSR作用事先指定为主值，中心差形式单独保留，不能修复行和或挑更接近PDE的值。

再用本轮细规则**原始直接理想响应**作同一探针的带符号比较：点−理想，加上理想−PDE，应等于点−PDE，保留浮点余量。它检验同一场下求积项和平均闭合项是否抵消，不能当成原余弦误差的唯一因果份额。该只读行比较尚未运行。

Stage 3 **incomplete**；fluid ORC **继续关闭**。

- [报告、代码与本轮完整观测审阅包](../../exports/ltn-ideal-row-review.zip)。
- [完整增量包及十三份历史依赖](../../exports/ltn-ideal-row-downloads.md)。
- [交付哈希](../../exports/ltn-ideal-row-delivery.json)。
'''
    (root/'research/reconnect/IDEAL_MATERIAL_ROW_RESPONSE.md').write_text(report)
    p=root/'docs/RESEARCH_STATUS.md';text=p.read_text().replace(
        '49. [质量平均投影弱预算与固定比例闭合反例](../research/reconnect/MASS_AVERAGE_PROJECTION_WEAK_BUDGET.md)（本轮）。',
        '49. [质量平均投影弱预算与固定比例闭合反例](../research/reconnect/MASS_AVERAGE_PROJECTION_WEAK_BUDGET.md)。\n50. [原比例的一行理想质量平均二次响应](../research/reconnect/IDEAL_MATERIAL_ROW_RESPONSE.md)（本轮）。')
    text=text[:text.index('本轮关键进展：')]+f'''本轮关键进展：完成预定N128/h0.04/D0.02的一行理想二次响应。eta={fine['eta']:.12f}，有效扩散率偏大{r['fine_eta_percent']:.8f}%；正确单位曲率响应0.02 s^-1，细规则原始直接响应{fine['raw_direct_aggregate_rate_per_s']:.14f} s^-1。两档eta差{delta['eta']:.3e}，直接子单元相位积分与方差响应差{fine['raw_direct_minus_raw_variance_per_s']:.3e} s^-1；原始核矩余差保留，没有归一化修正。

该值只量化理想质量平均闭合偏差，不是原点候选或保存余弦场3.12%误差的归因。原m/c0/身份、PDE和点候选作用保持。两规则仅32/32、64/64，22个法向面板；无边权/全图或瞬态。新增4项检查首次通过，累计356项；旧v0.8.0/246项未恢复。

782原文件、15历史附件和49份先前正式科学清单保持，4份历史诊断清单既有路径差异另核。下一步只读同一个已存点图CSR行31，原始矩阵作用为预定主值，常数/一二阶矩及中心差形式为控制；与本轮原始直接理想响应作同探针带符号预算，不重建图、调边权或改阈值。普通中心阻力FV基线保持。

数据入口：runs/reconnect-ideal-row/stage3-status.json。Stage 3 incomplete，fluid ORC关闭；无PDF/新版本、提交/推送、环境配置修改或发布。
'''
    p.write_text(text)
    print(json.dumps({'quadrature_CSVs':csvs,'fine_eta_percent':r['fine_eta_percent'],'next_task':status['next_task']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
