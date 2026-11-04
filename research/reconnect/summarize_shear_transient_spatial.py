"""Render audited fine/coarse shear observations without new physical solves."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from package_compression_midpoint import digest,dump


def csv_exact(path,ids,numbers,header):
    with path.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(header)
        for identity,row in zip(ids,numbers):writer.writerow([identity,*row])
    with path.open(newline='') as f:rows=list(csv.reader(f))
    assert len(rows)==len(ids)+1
    np.testing.assert_array_equal(np.asarray([row[0] for row in rows[1:]],dtype=ids.dtype),ids)
    np.testing.assert_array_equal(np.asarray([row[1:] for row in rows[1:]],dtype=float),numbers)
    return {'file':path.name,'rows':len(ids),'sha256':digest(path),'IDs_and_all_numeric_values_exact_after_parsing':True}


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve()
    assert not (out/'sha256.json').exists(),'never rewrite sealed scientific evidence'
    r=json.loads((out/'results.json').read_text());audit=json.loads((out/'numerical-audit.json').read_text())
    assert audit['raw_common_observation_and_paired_improvement_recomputed']
    comparisons=r['comparison_common_original_cells'];middle,final=comparisons[1:]
    distinguishable=all(v['improvement_exceeds_empirical_indicator'] for v in comparisons[1:])
    rows=[row for case in r['cases'].values() for row in case['steps']]
    summary={
        'error_reduction_distinguishable_at_both_nonzero_times':distinguishable,
        'minimum_conductance_kg_per_s':min(v['operator_diagnostics']['minimum_raw_conductance_kg_per_s'] for v in rows),
        'edge_count_range':[min(v['operator_diagnostics']['geometric_support_pairs'] for v in rows),max(v['operator_diagnostics']['geometric_support_pairs'] for v in rows)],
        'maximum_constant_residual_per_s':max(v['sparse_graph_audit']['constant_residual_per_s'] for v in rows),
        'maximum_mass_residual_kg_per_s':max(v['sparse_graph_audit']['mass_residual_kg_per_s'] for v in rows),
        'maximum_weighted_symmetry_residual_kg_per_s':max(v['sparse_graph_audit']['weighted_symmetry_residual_kg_per_s'] for v in rows),
        'maximum_shared_edge_assembly_difference':max(v['shared_edge_assembly_max_difference'] for v in rows),
        'maximum_pair_chi_vs_CSR_difference_kg_per_s':max(v['pair_chi_start_vs_CSR_absolute_difference'] for v in rows),
        'chi_range_kg_per_s':[min(min(v['chi_start_mid_end_kg_per_s']) for v in rows),max(max(v['chi_start_mid_end_kg_per_s']) for v in rows)],
        'maximum_absolute_Simpson_energy_difference_kg':max(abs(v['Simpson_energy_balance_difference_kg']) for v in rows),
        'raw_graph_bytes':sum(p.stat().st_size for p in out.glob('graph-*.npz')),
        'maximum_component_mass_drift_kg':audit['maximum_raw_component_mass_drift_kg'],
        'maximum_exponential_control_relative_difference':audit['maximum_exponential_control_relative_difference']}
    with np.load(out/'shear-transient-spatial.npz') as z:
        # Cache each NPZ array once before writing/reading the row-wise CSVs.
        arrays={key:z[key] for key in ['ids','mass','c0','parent_id_per_child','parent_ids','parent_mass','parent_c0','parent_reference_states',
            'reference_M127-k256_states','coarse_k4_observations','coarse_k8_observations','k4_parent_states','k8_parent_states','k4_states','k8_states']}
    common=np.column_stack([arrays['parent_mass'],arrays['parent_c0'],arrays['parent_reference_states'].T,
        arrays['coarse_k4_observations'][[1,2]].T,arrays['coarse_k8_observations'][[1,2]].T,
        arrays['k4_parent_states'][[2,4]].T,arrays['k8_parent_states'][[4,8]].T])
    common_csv=csv_exact(out/'common-observations.csv',arrays['parent_ids'],common,
        ['id','mass_kg','c0','reference_t0','reference_t0.025','reference_t0.05','coarse4_t0.025','coarse4_t0.05','coarse8_t0.025','coarse8_t0.05','fine4_observed_t0.025','fine4_observed_t0.05','fine8_observed_t0.025','fine8_observed_t0.05'])
    own=np.column_stack([arrays['parent_id_per_child'],arrays['mass'],arrays['c0'],arrays['reference_M127-k256_states'].T,
        arrays['k4_states'][[2,4]].T,arrays['k8_states'][[4,8]].T])
    own_csv=csv_exact(out/'fine-observations.csv',arrays['ids'],own,
        ['id','parent_id','mass_kg','c0','reference_t0','reference_t0.025','reference_t0.05','fine4_t0.025','fine4_t0.05','fine8_t0.025','fine8_t0.05'])
    dump(out/'observations-audit.json',{'common':common_csv,'fine':own_csv,'raw_NPZ_sha256':digest(out/'shear-transient-spatial.npz')})
    dump(out/'summary.json',summary)

    next_contact={
        'scope':'one independent stationary periodic density-contact frozen-rate probe; no transient or ORC',
        'prior_interface_evidence':['research/rebuild/KERNEL_JOINT_LIMIT.md','research/rebuild/INNER_MATCHING.md'],
        'physical_cases':1,'N_per_direction':128,'h_m':.04,'D_m2_per_s':.02,'periods_m':[1.,1.],
        'positions':'fixed regular material-square centers; zero velocity',
        'density_kg_per_m3':'rho=1 for 0<=a<0.5, rho=4 for 0.5<=a<1; periodic contacts at 0 and 0.5',
        'state':'c(a,b)=0.5+0.2*cos(2*pi*xi(a)+pi/5)*cos(2*pi*b)',
        'resistance_coordinate':'xi(a)=(integral_0^a dx/(rho(x)*D))/(integral_0^1 dx/(rho(x)*D))',
        'purpose':'continuous c and continuous rho*D*c_a at both contacts, with nonzero normal flux; isolate interface conductance from m/rho volume ambiguity',
        'mass_and_c0':'exact fixed-cell rho and rho*c integrals; cell mass averages, never point scalar values',
        'references':['analytic four-face integrated conservative rate','independent piecewise full-PDE volume Gauss8/16'],
        'new_frozen_graphs':1,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'measure':['global original-mass-weighted rate error','within distance 2h of either contact, with a fixed global reference norm','remaining cells with the same global norm','separate contribution norms and raw mass/energy/graph diagnostics'],
        'candidate_backend_unchanged':True,'candidate_inputs':'positions, fixed mass and given positive point rho only; no faces, derivatives or F/C',
        'parameter_or_mesh_matrix_scan':False,'full_P_constructed':False,'fluid_orc_enabled':False,
        'stop_condition':'preserve any interface defect; no clipping, renormalization, fitted interface weights or threshold changes, no automatic transient admission'}
    status={
        'updated':'2026-11-01','timezone':'Etc/UTC','stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'scope':'one predeclared N256/h.03 actual shear joint control on common original N128 material regions',
        'new_midpoint_graphs':12,'new_candidate_transient_runs':2,'physical_cases':1,'new_reference_runs':0,
        'coarse_candidate_rerun':False,'reference_rerun':False,'comparison_common_original_cells':comparisons,
        'error_reduction_distinguishable_at_both_nonzero_times':distinguishable,
        'indicator_is_a_rigorous_error_bound':False,'two_joint_levels_prove_spatial_order':False,
        'reused_checks_passed':10,'new_checks_passed':0,'recorded_checks_total':330,'old_v080_or_246_tests_recovered':False,
        'next_contact_frozen_predeclared':next_contact if distinguishable else None,
        'next_task':'one stationary density-contact frozen-rate probe' if distinguishable else 'review unresolved spatial/time control; do not enlarge tests',
        'remaining':['general geometry and density closure','density-contact/shock, solid walls, ledger/nonplanar/DNS physical acceptance'],
        'report':'research/reconnect/SHEAR_TRANSIENT_SPATIAL_CONTROL.md','pdf_generated':False,'software_version_created':False,'published':False}
    dump(out/'stage3-status.json',status)
    def e(v):return f'{v:.6e}'
    tables=['| t，s | 粗4区间 | 粗8区间 | 细4投影 | 细8投影 |','|---|---:|---:|---:|---:|']
    old=json.loads((root/'runs/reconnect-ordered-shear/results.json').read_text())
    for index in [1,2]:
        a=old['cases'];b=r['cases']
        tables.append(f"| {comparisons[index]['time_s']} | {e(a['4']['observations'][index]['relative_concentration_error_initial_fluctuation_norm'])} | {e(a['8']['observations'][index]['relative_concentration_error_initial_fluctuation_norm'])} | {e(b['4']['observations'][index]['common_parent_concentration_error'])} | **{e(b['8']['observations'][index]['common_parent_concentration_error'])}** |")
    controls=['| 共同原区域的对照 | t=0.025 s | T=0.05 s |','|---|---:|---:|']
    for title,key in [('误差减少量','absolute_error_reduction'),('合并经验差异指标','combined_empirical_indicator'),('减少量 / 合并指标','improvement_over_indicator')]:
        controls.append(f'| {title} | {e(middle[key])} | {e(final[key])} |')
    for key,title in [('coarse_4_vs_8','粗4/8状态差'),('fine_4_vs_8','细4/8状态差，投影后'),('reference_63_vs_127','参考空间差'),('reference_fine_time','参考最大步减半差'),('reference_projection_roundoff','参考重复投影差')]:
        controls.append(f"| {title} | {e(middle['empirical_difference_components'][key])} | {e(final['empirical_difference_components'][key])} |")
    mixing=['| 同原区域的8区间H | t=0.025 s | T=0.05 s |','|---|---:|---:|']
    for title,key in [('粗档','coarse_H'),('细档投影','fine_H_common_cells'),('独立参考','reference_H')]:
        mixing.append(f'| {title} | {middle[key]:.10f} | {final[key]:.10f} |')
    mixing.append(f"| 细H−参考，百分点 | {100*middle['fine_H_minus_reference']:.7f} | {100*final['fine_H_minus_reference']:.7f} |")
    mass=r['parent_mass_roundoff_audit'];own_final=r['cases']['8']['observations'][-1]
    conclusion='两次非零观测时刻的减少量均超过预定经验差异指标，支持继续一个新的物理判别。' if distinguishable else '未能在两次非零观测时刻同时区分改善与经验差异；不扩大验证，保留不确定或不利结果。'
    report=f'''# 同一二维剪切流的细档实际传播与共同物质区域比较

2026-11-01（UTC）。Stage 3 **incomplete**；fluid ORC **关闭**。

完成预定N256/h0.03细档的4/8区间实际传播，原N128/h0.04结果和独立参考均复用。投影回同一原物质区域，8区间终时浓度误差由 **{100*final['coarse_error_common_cells']:.7f}%** 变为 **{100*final['fine_error_common_cells']:.7f}%**（相对原初始波动），减少 **{100*final['relative_error_reduction_fraction']:.4f}%**。减少量为预定合并经验差异指标的 **{final['improvement_over_indicator']:.3f}倍**。{conclusion}这不是物理验收通过或一般收敛证明。

## 1. 原样复用输入与后端

执行上轮 `runs/reconnect-ordered-shear/stage3-status.json` 的 `next_shear_spatial_transient_predeclared`，运行前核验811项科学清单。唯一细档N=256、h=0.03，原档N=128、h=0.04；D=0.02、T=0.05、rho0=1+0.25cos(2pi a)、s(t)=0.25t/0.05。运动仍为Xx=a+s(t)sin(2pi b)/(2pi)+0.2t、Xy=b−0.1t（周期1）。

细m/c0/身份/点rho/父子映射直接读取已存 `runs/reconnect-compression-spatial/compression-spatial.npz`，不是重新生成或拟合。各档质量和身份全时段固定，状态为固定物质区域的rho加权质量平均，未用点值替换。

候选仅接收位置、固定质量、给定点rho；真实物质面、解析运动导数、J和F/C不进入候选。原正核与 `frozen_exponential_step` 均未修改。两档各自4/8区间中点图按时间顺序稀疏指数向量作用，本轮只新增12张细图，不构造全P或生成元矩阵乘积。

## 2. 同一原物质区域和固定的比较方式

每个原父区域由2×2细物质区域组成。观测Rc_f=sum(m_f*c_f)/m_parent_saved，分母始终是原保存父质量。父子质量差最大相对值 **{mass['maximum_relative_child_sum_discrepancy']:.6e}**，最大绝对值{mass['maximum_absolute_child_sum_discrepancy_kg']:.6e} kg，{mass['non_bitwise_equal_parent_count']}个父区域的求和不逐位相等，均原样保留。常数投影与1最大差{mass['constant_projection_max_difference_from_one']:.6e}，未重归一化。

共同误差用原父m、原父c0波动范数和原父参考：sqrt(sum m_parent*(Rc_f−c_ref)^2)/sqrt(sum m_parent*(c0_parent−mean0)^2)。均值、方差、I和H=1−E/E0也用原父输入。细区域自身指标单列，不能用观察尺度改变宣称改善。

独立参考复用已封存完整二维Fourier/DOP853的N128/N256质量平均快照，最细原始参考为M127-k256；本轮没有重求PDE、外推或修正参考。细参考再次投影与原父参考最大绝对差 **{r['reference_projected_twice_vs_saved_parent_max_difference']:.6e}**。参考空间差、最大步减半差和重复投影差原样进入比较。

## 3. 质量平均浓度和掺混结果

'''+'\n'.join(tables)+f'''

主比较预定为8区间，两次非零时刻均报告。终时细区域自身浓度误差为{own_final['fine_own_cell_concentration_error']:.6e}，共同区域误差为{final['fine_error_common_cells']:.6e}，两种尺度没有混用。

'''+'\n'.join(controls)+f'''

合并指标在计算前固定为上述五个非负范数之和，没有事后删除较大项。它是经验差异指标，不是严格PDE误差界；两个时间档不足以证明时间阶，两档N/h联合细化也不足以证明空间阶或分离有限核宽、点求积和质量平均闭合误差。

'''+'\n'.join(mixing)+f'''

细区域自身终时H={own_final['fine_own_metrics']['H']:.10f}，与投影后H的差保留，不替换共同观测。初始细c0投影与原父c0最大差{r['initial_parent_projection_vs_saved_c0_max_difference']:.6e}；细参考初值与原细c0最大差{r['fine_initial_reference_vs_exact_c0_max_difference']:.6e}，未强令为零。

![共同物质区域的剪切误差和经验差异](../../runs/reconnect-shear-transient-spatial/shear-transient-spatial.png)

## 4. 原始图、守恒与耗散审计

12张细图各有65536个节点，共享支持边数{summary['edge_count_range'][0]}至{summary['edge_count_range'][1]}，最小原始导通率{summary['minimum_conductance_kg_per_s']:.6e} kg/s，小权重未裁剪。常数、质量和加权对称性残差最大值分别为{summary['maximum_constant_residual_per_s']:.6e} /s、{summary['maximum_mass_residual_kg_per_s']:.6e} kg/s、{summary['maximum_weighted_symmetry_residual_kg_per_s']:.6e} kg/s。代数容差沿用原1e-10，不作为物理验收阈值。

全部保存端点保持初值范围、组分质量和方差下降；最大组分质量漂移{summary['maximum_component_mass_drift_kg']:.6e} kg。每段首、冻结指数中点及末的原始chi=cᵀLc为非负，范围[{summary['chi_range_kg_per_s'][0]:.8f},{summary['chi_range_kg_per_s'][1]:.8f}] kg/s；共享边/CSR耗散差最大{summary['maximum_pair_chi_vs_CSR_difference_kg_per_s']:.6e} kg/s。

同一冻结图的直接指数、两个半步与主采样终点最大相对差{summary['maximum_exponential_control_relative_difference']:.6e}，未挑选检查状态替换主状态。独立共享边组装与L最大差{summary['maximum_shared_edge_assembly_difference']:.6e}。三点Simpson耗散积分与半方差减少量最大差{summary['maximum_absolute_Simpson_energy_difference_kg']:.6e} kg，仅为经验求积核对，没有修正传播。峰值RSS约{r['peak_process_RSS_MiB_linux']:.1f} MiB。

原始12张CSR共{summary['raw_graph_bytes']:,}字节。NPZ同时保留原细/父输入、三份细参考、粗档检查点、全部细边界/指数中点/直接及两个半步状态、点历史、父投影和全部差向量。两个CSV的身份和全部数值在按原类型解析后与NPZ逐项精确相等。

## 5. 检查与完整性

本轮重跑10项相关既有检查（剪切接口4项、物质嵌套6项），全部通过，不重复计数；累计仍为**330项重建检查**。旧v0.8.0/246项未恢复。782个原始文件、15份历史附件和40份先前正式科学清单字节保持；4份历史诊断清单的已有路径差异单独按原记录核验。旧README/METHOD/CODEX_TASK、原始附件、后端和旧结果未改，仅更新可变进度入口 `docs/RESEARCH_STATUS.md`。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/shear_transient_spatial_study.py --output /workspace/shear-transient-spatial-reproduction
```

数据入口为 `runs/reconnect-shear-transient-spatial/stage3-status.json`。没有PDF、版本、提交、推送、环境配置修改或发布；fluid ORC关闭。

## 6. 下一步的停止条件

'''
    if distinguishable:
        report+='''光滑压缩和剪切的限定证据不能覆盖密度跳跃。此前[一维界面反例](../rebuild/KERNEL_JOINT_LIMIT.md)和[界面内层分析](../rebuild/INNER_MATCHING.md)已显示谐和系数核的限制，不能用当前光滑结果覆盖这些不利证据。下一步只将界面判别接回当前二维点输入后端：一个静止周期密度接触的冻结变化率检验，N128、h0.04、D0.02，rho在a<0.5为1、其余为4，周期接触位于0与0.5。质量和初值使用精确固定物质积分。

定义xi(a)为归一化积分阻力坐标integral dx/(rho D)，c=0.5+0.2cos(2pi xi+pi/5)cos(2pi b)。其c与rho D c_a在接触两侧连续，法向通量非零；本例m/rho恰等于单元面积，专门隔离界面电导问题。候选仍不读真实面、梯度或F/C。

只建一张原始图，与解析四面通量及独立完整PDE体积分Gauss8/16比较；报告全域、距任一接触不超过2h区域及剩余区域，使用同一全域参考范数。若出现界面缺陷就保留，不拟合界面权重、不调阈值、不增加扫描，不自动进入该接触算例的瞬态。局部强残差也不能单独作为弱极限错误的证明；普通中心阻力FV基线和既有失败结果继续保留。本轮没有运行该新问题。
'''
    else:
        report+='改善未能同时与两个时刻的差异指标区分，不启动新的物理算例；先审查当前空间/时间证据，不扩展参数矩阵。\n'
    report+='''
一般几何/密度闭合、激波/接触、固壁、ledger、非平面/DNS仍未验收。**Stage 3 incomplete，fluid ORC继续关闭。**

## 7. 交付

- [代码、报告与完整实际标量审阅包](../../exports/ltn-shear-transient-spatial-review.zip)。
- [12张原始细CSR与完整数据分卷](../../exports/ltn-shear-transient-spatial-downloads.md)。
- [交付哈希清单](../../exports/ltn-shear-transient-spatial-delivery.json)。

本轮为增量交付，历史完整审计/CLI复现需按下载说明叠加四份历史基包；782个原始文件审计还需原交接材料。
'''
    (root/'research/reconnect/SHEAR_TRANSIENT_SPATIAL_CONTROL.md').write_text(report)
    index=root/'docs/RESEARCH_STATUS.md';text=index.read_text()
    text=text.replace('40. [同一二维剪切流的实际标量顺序传播与质量掺混](../research/reconnect/ORDERED_SHEAR_TRANSIENT.md)（本轮）。',
        '40. [同一二维剪切流的实际标量顺序传播与质量掺混](../research/reconnect/ORDERED_SHEAR_TRANSIENT.md)。\n41. [同一二维剪切流的细档实际传播与共同物质区域比较](../research/reconnect/SHEAR_TRANSIENT_SPATIAL_CONTROL.md)（本轮）。')
    text=text[:text.index('本轮关键进展：')]+f'''本轮关键进展：完成同一剪切流已有N256/h0.03细档4/8区间实际传播，原粗档和独立参考均未重算。按原父质量观察回共同N128区域，8区间终时误差从{100*final['coarse_error_common_cells']:.7f}%变为{100*final['fine_error_common_cells']:.7f}%（原初始波动），减少{100*final['relative_error_reduction_fraction']:.4f}%，减少量/预定经验指标为{final['improvement_over_indicator']:.3f}。{conclusion}两档不证明一般空间阶，经验差异不是严格界。

相关10项既有检查全部通过，不重复计数，累计330项；旧v0.8.0/246项未恢复。782原文件、15历史附件及40份先前正式科学清单保持，4份历史诊断清单已有路径差异另核。候选后端、传播器和旧结果未改。

待推进：{status['next_task']}；具体单例输入、独立参考、固定分区和停止条件保存在本轮stage3-status.json，不扩大参数矩阵，不裁剪、拟合或重归一化。一般几何/密度、激波/接触、固壁、ledger、非平面/DNS未验收。Stage 3 incomplete，fluid ORC关闭。

当前数据入口：runs/reconnect-shear-transient-spatial/stage3-status.json。未生成PDF或新版本，未提交/推送、修改环境配置或发布。
'''
    index.write_text(text)
    print(json.dumps({'comparison':comparisons[1:],'summary':summary,'CSVs_exactly_verified':True,'recorded_checks_total':330,'fluid_orc_enabled':False},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
