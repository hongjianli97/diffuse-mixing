"""Record the audited directional budget without promoting an oracle model."""
from pathlib import Path
import argparse,json
import numpy as np
from package_compression_midpoint import dump,digest
from summarize_shear_transient_spatial import csv_exact


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve()
    assert not (out/'sha256.json').exists(),'never rewrite sealed evidence'
    r=json.loads((out/'results.json').read_text());audit=json.loads((out/'numerical-audit.json').read_text())
    assert audit['all_region_signed_Gram_cross_terms_and_vector_identities_recomputed']
    with np.load(out/'oracle-directional.npz') as z:
        a={key:z[key] for key in ['ids','material_label_centers','mass','c0','rho0_point','PDE_rate_a','PDE_rate_b','reference_rate',
            'fine_rate_a','fine_rate_b','fine_saved_total_rate','fine_error_vectors','fine_total_error','near_contact','distance_to_contact']}
    values=np.column_stack([a['material_label_centers'],a['mass'],a['rho0_point'],a['c0'],a['PDE_rate_a'],a['PDE_rate_b'],a['reference_rate'],
        a['fine_rate_a'],a['fine_rate_b'],a['fine_saved_total_rate'],a['fine_error_vectors'].T,a['fine_total_error'],a['near_contact'],a['distance_to_contact']])
    csv=csv_exact(out/'oracle-directional.csv',a['ids'],values,['id','a_m','b_m','mass_kg','rho_node','c0_mass_average','PDE_rate_a','PDE_rate_b','PDE_total_rate',
        'oracle_rate_a','oracle_rate_b','unchanged_saved_oracle_total','normal_error','transverse_error','evaluation_residual','saved_total_error','near_contact_0_or_1','distance_to_contact_m'])
    dump(out/'observations-audit.json',{'CSV':csv,'raw_NPZ_sha256':digest(out/'oracle-directional.npz')})
    fine=r['directional_runs']['fine'];coarse=r['directional_runs']['coarse'];b=fine['budget'];g=b['all'];delta=r['component_quadrature_differences']
    next_plan={'scope':'theory-only second-order interface consistency review of this unchanged isotropic oracle path kernel',
        'source_operator':'research/reconnect/oracle_path_reference.py','source_report':'research/reconnect/ORACLE_DIRECTIONAL_RESIDUAL_BUDGET.md',
        'density_contact':[1,4],'D_m2_per_s':.02,'h_m':.04,'original_radial_kernel_and_second_moments_unchanged':True,
        'review_questions':['derive one-sided interface traces for locally flux-compatible normal and tangential quadratic fields',
            'separate preservation of a constant normal flux from second-order curvature consistency',
            'state the order of interface/h limits and distinguish pointwise trace from material-average or weak consistency'],
        'must_not_claim':['unique causal attribution of current cosine-state directional norms','failure of every positive graph method','transient accuracy or physical acceptance'],
        'no_new_kernel_coefficient_or_density_reconstruction':True,'no_fitting_clipping_normalization_or_threshold_change':True,
        'ordinary_central_resistance_FV_baseline_retained':True,'new_candidate_graphs':0,'new_candidate_transient_runs':0,
        'parameter_or_mesh_matrix_scan':False,'fluid_orc_enabled':False}
    status={'updated':'2026-11-01','timezone':'Etc/UTC','stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'scope':'completed predeclared normal-first directional residual budget of saved oracle path operator',
        'reference_only':True,'uses_additional_oracle_density_information':True,'candidate_and_saved_total_operator_unchanged':True,
        'fine_budget':b,'component_quadrature_differences':delta,'component_norms_are_additive_causal_fractions':False,
        'evaluation_residual_forced_to_zero':False,'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'finite_node_candidate_admitted':False,'contact_transient_admitted':False,'new_checks_passed':5,'previous_recorded_checks':347,
        'recorded_checks_total':352,'old_v080_or_246_tests_recovered':False,'quadrature_differences_are_rigorous_bounds':False,
        'next_second_order_interface_review_predeclared':next_plan,'next_task':'theory-only second-order interface consistency review',
        'remaining':['remaining oracle constitutive discrepancy','finite-node density identifiability','general geometry/density, shocks, walls, ledger, nonplanar/DNS acceptance'],
        'report':'research/reconnect/ORACLE_DIRECTIONAL_RESIDUAL_BUDGET.md','pdf_generated':False,'software_version_created':False,'published':False}
    dump(out/'stage3-status.json',status)
    rows=['| 区域；均用原全域PDE范数 | 法向误差 | 横向误差 | 保存总误差 | 法/横相关系数 | 2×带符号交叉项 |','|---|---:|---:|---:|---:|---:|']
    for key,label in [('all','全域'),('near_contact','原2h邻域'),('far_from_contact','其余区域')]:
        v=b[key];rows.append(f"| {label} | {100*v['normal_error_norm']:.7f}% | {100*v['transverse_error_norm']:.7f}% | {100*v['saved_total_error_norm']:.7f}% | {v['normal_transverse_correlation']:.7f} | {v['signed_twice_cross_terms']['normal_transverse']:.9e} |")
    controls=['| 原规则差/参考控制，全域 | 相对原全域PDE范数 |','|---|---:|',
        f"| 法向两档向量差 | {delta['rate_a']['all']:.9e} |",f"| 横向两档向量差 | {delta['rate_b']['all']:.9e} |"]
    for key,label in [('PDE_a_face_volume16_difference','PDE法向面/体16差'),('PDE_a_volume8_16_difference','PDE法向体8/16差'),
        ('PDE_b_face_volume16_difference','PDE横向面/体16差'),('PDE_b_volume8_16_difference','PDE横向体8/16差')]:
        controls.append(f"| {label} | {r['saved_PDE_component_controls'][key]['all']:.9e} |")
    gram=np.asarray(g['gram_over_global_reference_norm_squared'])
    report=f'''# 同一oracle沿段阻力核的方向残差预算

2026-11-01（UTC）。Stage 3 **incomplete**；fluid ORC **关闭**。

完成上一轮预定的法向优先分解。原oracle总误差保持 **{100*g['saved_total_error_norm']:.7f}%**；法向误差 **{100*g['normal_error_norm']:.7f}%**，横向误差 **{100*g['transverse_error_norm']:.7f}%**，两项相关系数 **{g['normal_transverse_correlation']:.7f}**。较小总误差包含明显抵消，不能解释为只剩某个方向的小误差。本轮没有修改算子或提高算法准确率。

## 1. 沿用的输入与预定约定

执行已封存 `runs/reconnect-oracle-path/stage3-status.json` 的 `next_oracle_directional_budget_predeclared`，运行前核验其954项科学清单。保留同一N128、h=0.04、D=0.02、周期1:4密度接触、原连续初场、原物质身份、质量、c0、原2h分区、候选节点图与原PDE参考。

状态仍是固定物质单元的质量平均：m_i=∫K_i rho dV，cbar_i=∫K_i rho*c dV/m_i。旧 `coarse_rate`、`fine_rate` 总作用逐项原样复制；不以本轮分量和覆盖它们。完整解析rho/接触位置仍是额外oracle输入，未从有限节点数据识别。

对c=0.5+A f(a)cos(kb)，使用同一正对称K_path、原B_h和原横向积分b0、bk，固定先法向后横向：

Ja(a)=∫ K_path(a,a+s)b0(s)[f(a+s)−f(a)] ds，

Jb(a)=∫ K_path(a,a+s)f(a+s)[bk(s)−b0(s)] ds。

r_a,b=2A cos(kb)J_a,b/rho；两者分别积分rho*r再除以**原保存质量**。外层节点、权重、物质列身份及横向观察积分直接复用旧NPZ；两组阶数仍为(8,32,32)、(16,64,64)，最细原始规则预先指定。

参考分量复用 `runs/reconnect-contact-frozen/contact-observations.npz` 中的 `face_rate_a`、`face_rate_b`，并保留其与原体积求积的差向量。两个参考分量之和逐项等于原PDE总作用。

## 2. 保存总量的精确误差账本

定义E_a=r_a−r_PDE,a，E_b=r_b−r_PDE,b，以及**未强制置零**的求值余量E_c=r_saved−r_a−r_b，则

E_saved=E_a+E_b+E_c。

所有区域使用同一分母R²=∑_全域 m_i*r_PDE,i²；Gram项G_uv=∑_区域 m_i*E_u,i*E_v,i/R²，平方误差为各对角项与所有2G_uv之和。分量范数不可相加，也不是唯一因果份额。分解顺序是预定分析约定，横向项还含f(a+s)，不代表两个可独立实现的物理生成元。

'''+'\n'.join(rows)+f'''

细规则全域：G_aa={gram[0,0]:.12e}，G_bb={gram[1,1]:.12e}，2G_ab={2*gram[0,1]:.12e}。负交叉项抵消了相当部分对角项；其余区域相关系数为正，不能将全域抵消概括成每个位置均相反。

求值余量范数为 **{g['evaluation_residual_norm']:.9e}**；误差向量恒等式余差范数为{g['vector_identity_residual_norm']:.9e}，平方范数账本余差为{g['squared_norm_identity_residual']:.9e}。这些原始浮点余差保留，没有通过裁剪、补偿或重归一化消除。

![原方向误差、带符号交叉项与数值控制](../../runs/reconnect-oracle-directional/oracle-directional.png)

## 3. 必要数值控制

粗规则法向/横向/总误差为{100*coarse['budget']['all']['normal_error_norm']:.7f}% / {100*coarse['budget']['all']['transverse_error_norm']:.7f}% / {100*coarse['budget']['all']['saved_total_error_norm']:.7f}%。以下均为**向量差范数**，不是两个误差范数的差。

'''+'\n'.join(controls)+f'''

原总作用的两档差约4.210e-06，本轮几乎全部定位到横向积分；法向分量在这两个规则间稳定到约6.083e-14。5.42%的法向偏差明显大于已测求积与参考控制差，不能仅归于本轮求积变化。两个规则差不是严格误差上界，也不排除共同偏差；没有因此宣布物理通过或追加参数/网格矩阵扫描。

## 4. 库存、能量与适用范围

细规则原始法向、横向、保存总量的组分质量变化率分别为 {fine['raw_component_mass_rates_kg_per_s'][0]:.9e}、{fine['raw_component_mass_rates_kg_per_s'][1]:.9e}、{fine['raw_component_mass_rates_kg_per_s'][2]:.9e} kg/s。

投影−∑m*cbar*r的法向、横向、求值余量贡献分别为 {fine['projected_chi_components_kg_per_s'][0]:.12e}、{fine['projected_chi_components_kg_per_s'][1]:.12e}、{fine['projected_chi_components_kg_per_s'][2]:.12e} kg/s，保存总投影为 {fine['saved_total_projected_chi_kg_per_s']:.12e} kg/s。这些只是该有序分解的投影账本，不能当成独立非负物理耗散通道；原连续物理耗散沿用上一轮记录，本轮未重新定义或传播状态。

这是一个固定h、固定接触态、含完整密度oracle的冻结诊断。没有新的候选图、标量或参考瞬态，没有新I/H或最大值原理实验，也未验证一般有限节点密度闭合。普通中心阻力物质FV基线保持。

## 5. 检查、完整性与复现

5项新增检查首次运行全部通过：独立直接二维方向积分、分量和与旧总算子、原质量分母/库存、含非正交误差的带符号Gram账本、非法输入拒绝。累计347+5=**352项重建检查**；旧v0.8.0及其246项测试仍未恢复。峰值RSS约{r['peak_process_RSS_MiB_linux']:.1f} MiB。

审计复算原始分量、求积样本、质量观察、所有区域Gram/交叉项/向量恒等式与控制量；CSV按原身份类型及全部浮点数解析后与NPZ完全一致。782个原始文件、15份历史附件与45份先前正式科学清单保持字节一致；4份历史诊断清单已有路径差异另核，不计作当前匹配。原README/METHOD/CODEX_TASK与旧结果均未改，仅更新可变进度入口。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/test_oracle_directional.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/oracle_directional_study.py --output /workspace/oracle-directional-reproduction
```

`runs/reconnect-oracle-directional/` 包含原输入/比较量、两档全部方向原始记录、差向量、CSV、测试与审计。未拟合F/C，未裁剪、重归一化或改变阈值；未生成PDF/版本、提交/推送、修改环境配置或发布。

## 6. 下一步：只作同一核的二阶界面一致性理论审查

恒定法向通量保真不足以说明当前含曲率/横向模式的误差来源。下一步保留同一K_path、原径向核及二阶矩，推导满足界面浓度/法向通量连续条件的局部法向与横向二次场的一侧作用极限；明确先趋近界面与h趋零的顺序，并区分点态迹、固定物质平均与弱一致性。

这项理论审查尚未完成。它用于判断该oracle核是否还值得向候选后端推进，不用新系数或密度重构修饰当前结果，不把局部二次场结论直接当作当前余弦态5.42%/3.99%的定量因果解释，也不外推为所有正图方法不可能。

预定下一步不建图、不传播、不扫描，不改FV基线；Stage 3 **incomplete**，fluid ORC **继续关闭**。

## 7. 交付

- [报告、代码与当前完整观测审阅包](../../exports/ltn-oracle-directional-review.zip)。
- [完整数据与九份历史依赖入口](../../exports/ltn-oracle-directional-downloads.md)。
- [交付哈希](../../exports/ltn-oracle-directional-delivery.json)。

本轮为增量交付；完整历史审计/CLI复现依说明叠加九份历史基包，782个原始文件审计仍需原交接材料。
'''
    (root/'research/reconnect/ORACLE_DIRECTIONAL_RESIDUAL_BUDGET.md').write_text(report)
    p=root/'docs/RESEARCH_STATUS.md';text=p.read_text().replace('45. [同一接触oracle沿段串联阻力连续核对照](../research/reconnect/ORACLE_PATH_RESISTANCE_CONTROL.md)（本轮）。',
        '45. [同一接触oracle沿段串联阻力连续核对照](../research/reconnect/ORACLE_PATH_RESISTANCE_CONTROL.md)。\n46. [同一oracle核的法向优先方向残差预算](../research/reconnect/ORACLE_DIRECTIONAL_RESIDUAL_BUDGET.md)（本轮）。')
    text=text[:text.index('本轮关键进展：')]+f'''本轮关键进展：完成同一oracle核的预定方向残差预算。原总误差仍{100*g['saved_total_error_norm']:.7f}%，法向误差{100*g['normal_error_norm']:.7f}%，横向误差{100*g['transverse_error_norm']:.7f}%，相关系数{g['normal_transverse_correlation']:.7f}，存在明显抵消。求值余量范数{g['evaluation_residual_norm']:.3e}；这是误差诊断，没有修改算子或改善算法。

法向/横向两档向量求积差分别{delta['rate_a']['all']:.3e}、{delta['rate_b']['all']:.3e}，原总求积差几乎全在横向。原质量平均、m/c0/身份、PDE分量、旧总作用及候选图均保持。完整解析密度仍为额外oracle输入；有序分解及其范数不作唯一因果份额。

新增5项检查首次运行通过，累计352项重建检查；旧v0.8.0/246项未恢复。782原文件、15历史附件和45份先前正式科学清单保持，4份历史诊断清单已有路径差异另核。

下一步只作同一核的二阶界面一致性理论审查：局部通量相容的法向/横向二次场、一侧迹及界面/h极限顺序，区分恒通量保真与曲率一致性；不直接归因当前余弦态误差，不改系数或密度闭合，不建图、不作瞬态/扫描。普通中心阻力FV基线保持。

数据入口：runs/reconnect-oracle-directional/stage3-status.json。Stage 3 incomplete，fluid ORC关闭；无PDF/新版本、提交/推送、环境配置修改或发布。
'''
    p.write_text(text)
    print(json.dumps({'CSV':csv,'fine_all_budget':g,'next_task':status['next_task']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
