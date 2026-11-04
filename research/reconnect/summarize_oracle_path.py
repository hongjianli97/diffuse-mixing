"""Report an extra-information constitutive control without backend promotion."""
from pathlib import Path
import argparse,json
import numpy as np
from package_compression_midpoint import dump,digest
from summarize_shear_transient_spatial import csv_exact
from compression_midpoint import weighted_norm


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve()
    assert not (out/'sha256.json').exists(),'never rewrite sealed evidence'
    r=json.loads((out/'results.json').read_text());audit=json.loads((out/'numerical-audit.json').read_text())
    assert audit['all_region_errors_operator_differences_and_fixed_indicator_sums_recomputed']
    with np.load(out/'oracle-path.npz') as z:
        keys=['ids','material_label_centers','mass','rho0_point','c0','reference_rate','kernel_rate','harmonic_fine_rate','coarse_rate','fine_rate','oracle_rule_difference','oracle_error','near_contact','distance_to_contact']
        a={key:z[key] for key in keys}
    values=np.column_stack([a['material_label_centers'],a['mass'],a['rho0_point'],a['c0'],a['reference_rate'],a['kernel_rate'],a['harmonic_fine_rate'],a['coarse_rate'],a['fine_rate'],a['oracle_rule_difference'],a['oracle_error'],a['near_contact'],a['distance_to_contact']])
    csv=csv_exact(out/'oracle-path.csv',a['ids'],values,['id','a_m','b_m','mass_kg','rho_node','c0_mass_average','saved_PDE_rate','unchanged_node_candidate_rate','saved_harmonic_continuous_rate','oracle_coarse_rate','oracle_fine_rate','oracle_rule_difference','oracle_PDE_error','near_contact_0_or_1','distance_to_contact_m'])
    dump(out/'observations-audit.json',{'CSV':csv,'raw_NPZ_sha256':digest(out/'oracle-path.npz')})
    c=r['comparison'];g=c['all'];fine=r['oracle_runs']['fine'];coarse=r['oracle_runs']['coarse']
    harmonic=json.loads((root/'runs/reconnect-contact-continuous/results.json').read_text())['continuous_runs']['fine']
    original=json.loads((root/'runs/reconnect-contact-frozen/results.json').read_text())
    candidate_error=weighted_norm(a['mass'],a['kernel_rate']-a['reference_rate'])/weighted_norm(a['mass'],a['reference_rate'])
    next_plan={
        'scope':'normal-first directional residual budget of this saved oracle operator; no new model or candidate',
        'source':'runs/reconnect-oracle-path/oracle-path.npz','saved_total_keys':['coarse_rate','fine_rate'],
        'PDE_component_source':'runs/reconnect-contact-frozen/contact-observations.npz',
        'PDE_component_keys':['face_rate_a','face_rate_b'],'N_per_direction':128,'h_m':.04,'D_m2_per_s':.02,
        'same_original_mass_c0_ids_and_full_oracle_density':True,
        'decomposition_order_fixed_before_run':'normal-first: Ja=integral K_path*b0*(f_neighbor-f_target) ds; Jb=integral K_path*f_neighbor*(bk-b0) ds',
        'observe_components':'same original rho-weighted material means, saved mass denominators',
        'preserve_raw_total':'saved total rate unchanged; independently computed component sum minus saved total retained as evaluation residual',
        'error_identity':'E_saved = (ra_oracle-ra_PDE) + (rb_oracle-rb_PDE) + (total_saved-ra_oracle-rb_oracle)',
        'measure':'all/original within-2h/complement norms and signed Gram/cross terms under same full PDE norm; no additive causal shares',
        'fixed_quadrature_levels':[{'cell_x_order':8,'normal_displacement_order':32,'transverse_order':32},{'cell_x_order':16,'normal_displacement_order':64,'transverse_order':64}],
        'reference_only':True,'interpretation':'ordered algebraic direction budget, not unique causal attribution; full density remains extra oracle information',
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'no_coefficient_state_weight_or_threshold_changes':True,'parameter_or_mesh_matrix_scan':False,'fluid_orc_enabled':False,
        'stop_condition':'retain unresolved/adverse residual; no general-node backend or transient admission'}
    status={'updated':'2026-11-01','timezone':'Etc/UTC','stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'scope':'completed predeclared same-contact oracle path-resistance continuous control',
        'comparison':c,'reference_only':True,'uses_additional_oracle_density_information':True,
        'candidate_backend_and_saved_graph_unchanged':True,'unchanged_discrete_candidate_error':candidate_error,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'finite_node_candidate_admitted':False,'contact_transient_admitted':False,
        'new_checks_passed':6,'previous_recorded_checks':341,'recorded_checks_total':347,'old_v080_or_246_tests_recovered':False,
        'quadrature_indicator_is_rigorous_bound':False,'single_frozen_problem_proves_transient_accuracy':False,
        'next_oracle_directional_budget_predeclared':next_plan,
        'next_task':'same-operator normal-first directional residual budget only',
        'remaining':['remaining oracle constitutive discrepancy','density closure assumptions for finite-node candidates','general geometry/density, shocks, walls, ledger, nonplanar/DNS physical acceptance'],
        'report':'research/reconnect/ORACLE_PATH_RESISTANCE_CONTROL.md','pdf_generated':False,'software_version_created':False,'published':False}
    dump(out/'stage3-status.json',status)
    table=['| 区域，同一全域PDE范数 | 原谐和连续核 | oracle沿段阻力 | 误差减少量 | 原误差的减少比例 |','|---|---:|---:|---:|---:|']
    for key,label in [('all','全域'),('near_contact','原2h邻域'),('far_from_contact','其余区域')]:
        v=c[key];table.append(f"| {label} | {100*v['saved_harmonic_error']:.7f}% | {100*v['oracle_path_error']:.7f}% | {v['absolute_error_reduction']:.6e} | {100*v['relative_error_reduction_fraction']:.4f}% |")
    controls=['| 全域已预定核对量 | 相对原PDE范数 |','|---|---:|']
    for key,label in [('oracle_rule_difference','oracle两档求积差'),('harmonic_rule_difference','已存谐和两档求积差'),('PDE_face_volume16_difference','已存PDE面/体16差'),('PDE_volume8_16_difference','已存PDE体8/16差')]:
        controls.append(f"| {label} | {g['empirical_indicator_components'][key]:.6e} |")
    report=f'''# 同一接触的oracle沿段串联阻力连续核对照

2026-11-01（UTC）。Stage 3 **incomplete**；fluid ORC **关闭**。

预定oracle对照完成。同一原物质区域、连续场和h=0.04下，冻结变化率误差由原谐和连续核的 **{100*g['saved_harmonic_error']:.7f}%降为{100*g['oracle_path_error']:.7f}%**，减少 **{100*g['relative_error_reduction_fraction']:.4f}%**。减少量为预定合并经验差异指标的{g['reduction_over_indicator']:.1f}倍；剩余误差约3.12%，仍需残差诊断；Stage 3验收未完成。

这是**使用完整解析密度信息的连续本构控制**。有限节点候选未改，已存候选冻结误差仍为 **{100*candidate_error:.7f}%**，没有以oracle结果替换原图或状态，也不据此宣称节点算法或有限时间掺混预测改善。

## 1. 固定输入和额外信息

运行前核验 `runs/reconnect-contact-constitutive/sha256.json` 的925项科学清单，执行 `next_oracle_path_control_predeclared`。保持N=128、h=0.04、D=0.02，静止周期密度比1:4的两相接触（rho分别为1与4），原m/c0/身份、连续初场、PDE平均变化率及2h分区全部不变。原谐和连续核两规则的结果直接复用，没有重算比较对象。

额外oracle信息是**整条线段上的原解析rho分布及其接触位置**。上一轮[输入可辨识性审查](CONTACT_CONSTITUTIVE_REVIEW.md)已经说明，它不能无条件地从有限节点rho、固定质量和cbar唯一确定。本轮只选择原已知分段常量背景作为本构参照；并未解决一般点输入的密度闭合问题。

普通中心阻力物质FV基线继续保留。本轮不是替换FV通量，也不是将新的oracle公式直接升级为面向质量平均未知量的通用图后端。

## 2. 正对称的沿段系数与稳定积分

定义Gamma=rho D，使用理论审查中的

K_path(x,y) = [integral_0^1 1/Gamma(x+theta(y−x)) dtheta]⁻¹。

当前Gamma只随法向a变化。令法向位移s，若路径不跨接触，K_path=Gamma_target；若跨接触，令alpha为已知原解析背景中起点一相所占的路径比例，则

K_path = [alpha/Gamma_start + (1−alpha)/Gamma_other]⁻¹。

alpha由原解析接触位置与位移直接计算，不按PDE结果拟合。系数正且对换端点不变。实现用位移与接触距离比较确定穿越，再算比例，避免对接近的阻力原函数数值作差。没有裁剪alpha、状态或权重，也没有重归一化。

原径向核B_h和因子2不变。对原连续场c=0.5+A f(a)cos(kb)，A=0.2、k=2pi，使用同一横向Fourier精确约化：

J_path(a)=integral_(−2h to 2h) K_path(a,a+s)[f(a+s)bk(s)−f(a)b0(s)] ds，
r_path(a,b)=2A cos(kb)J_path(a)/rho(a)。

b0、bk仍为原核横向积分，沿用已通过极坐标/二阶矩自检的正半轴偶对称求积。再积分rho*r_path并除以**原保存m**观察到原固定物质区域，没有用点值替换c0。

## 3. 只做两个预定求积控制

两组(单元a阶,s阶,z阶)固定为(8,32,32)、(16,64,64)。其全部外层节点、权重、列身份和横向观察积分与原谐和参照逐项相同；只改变oracle本构系数。最细原始规则预先指定，不按接近PDE程度挑选或外推。

粗/细规则全域误差分别为{100*coarse['errors']['all']:.7f}%、{100*fine['errors']['all']:.7f}%；**向量**求积差为{g['empirical_indicator_components']['oracle_rule_difference']:.6e}，它不是两个误差范数之差。

'''+'\n'.join(controls)+f'''

四个非负范数在计算前固定相加，合并指标为 **{g['combined_empirical_indicator']:.6e}**。本次误差减少量为{g['absolute_error_reduction']:.6e}，明显大于已测差异。这是经验数值分离，不是严格误差界、时间精度证明或新的物理验收阈值。沿段系数在接触邻域的积分结构不同，本轮两档差也明显大于旧谐和规则差，原样保留，没有追加扫描使其消失。

## 4. 同一原物质区域的结果

'''+'\n'.join(table)+f'''

每个量均以原全域sqrt(sum m*r_PDE²)归一化，没有用各区域自身范数改变比例。原2h邻域外，两算子作用差范数约{c['far_from_contact']['oracle_vs_harmonic_difference']:.6e}，误差约0.785%基本不变；不将其极小差异当作新的实质远场改善。

同一oracle算子相对原谐和算子的全域作用差为{g['oracle_vs_harmonic_difference']:.6e}。它含额外密度信息带来的本构改变，不是原有限节点算法实现优化，也不是将离散误差与本构误差范数直接相加或按份额分配。

上一轮的局部恒通量理论条件在本参考中保持，但当前完整二维初场还含法向曲率和横向模式，仍有约3.12%的冻结PDE偏差。通过局部恒通量检验，不能直接保证完整瞬态优于普通FV或节点候选。

![共同区域的误差和同色标残差](../../runs/reconnect-oracle-path/oracle-path.png)

## 5. 库存和耗散并未给出同样的改善

细规则总组分质量变化率为{fine['raw_mass_rate_kg_per_s']:.6e} kg/s，保留未经修正的原值。

| 冻结耗散/投影量 | kg/s |
|---|---:|
| 原谐和连续核物理耗散 | {harmonic['physical_continuous_nonlocal_chi_kg_per_s']:.12f} |
| oracle沿段核物理耗散 | {fine['physical_nonlocal_chi_kg_per_s']:.12f} |
| 原局部PDE连续物理耗散 | {original['physical_continuous_chi_kg_per_s']:.12f} |
| oracle投影−sum m*cbar*r_path | {fine['projected_chi_kg_per_s']:.12f} |

连续物理耗散与单元投影量分开保留。oracle的局部变化率误差大幅下降，但整体物理耗散没有同步更接近PDE。不能据此宣布有限时间掺混更准确；本轮没有传播c，没有新的I/H或最大值原理数值实验。

## 6. 检查与完整性

6项新增检查最终全部通过：路径系数正性/互易/纯相、独立阻力原函数核对、局部恒通量配对恒等式、直接二维积分/模式约化、原质量分母/库存/正耗散、非法输入拒绝。累计341+6=**347项重建检查**，旧v0.8.0/246项未恢复。

首次运行有3项因NumPy标量不支持索引赋值而报错，另3项通过。正式对照前仅将gamma保持为零维数组，系数公式、求积规则和断言容差未改；首次日志及修正前源码保留，审计确认源码差异只有该数组转换，最终检查计数一次。

782个原始文件、15个历史附件和44份先前正式科学清单字节保持；4份历史诊断清单已有路径差异另核。原README/METHOD/CODEX_TASK、节点后端、原图、旧参考和普通FV均未改，只更新 `docs/RESEARCH_STATUS.md`。原观察数组、旧谐和作用和候选作用逐项核验，CSV按原身份/浮点类型解析后与NPZ完全一致。峰值RSS约{r['peak_process_RSS_MiB_linux']:.1f} MiB。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/test_oracle_path_reference.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/oracle_path_study.py --output /workspace/oracle-path-reproduction
```

`runs/reconnect-oracle-path/` 保存原输入/旧参照、两组全部外层求积记录、oracle作用、差向量、预定指标、CSV、测试日志和审计。没有新候选图、流体或参考瞬态、全P、拟合/裁剪/重归一化、阈值修改、PDF、版本、提交、推送、环境配置更改或发布。

## 7. 下一步只分析剩余方向残差

固定“先法向、后横向”的代数约定，对本轮同一个oracle核定义

Ja = integral K_path*b0*(f_neighbor−f_target) ds，
Jb = integral K_path*f_neighbor*(bk−b0) ds。

两项相加为原总J。各自观察到同一原物质区域，与原解析面PDE的a、b分量比较；原总作用数组不改，用独立计算的分量和与保存总量之差作额外数值余量。

只用原两组求积记录和规则，保存方向误差、原2h分区、带符号Gram/交叉项及求积差。这个约定分解不作唯一因果归因，范数不当作可相加的误差百分比。完整密度仍为额外oracle信息；不新增模型、候选图、瞬态或密度重构，不根据方向结果调权重。

该方向分析尚未运行。一般有限节点密度闭合、几何/激波/固壁/ledger/非平面/DNS仍未验收。**Stage 3 incomplete，当前接触算例不推进候选瞬态，fluid ORC继续关闭。**

## 8. 交付

- [报告、代码与完整观测审阅包](../../exports/ltn-oracle-path-review.zip)。
- [完整数据及历史依赖](../../exports/ltn-oracle-path-downloads.md)。
- [交付哈希](../../exports/ltn-oracle-path-delivery.json)。

本轮为增量交付，历史完整来源审计与CLI复现需按下载说明叠加八份历史基包；782个原始文件审计仍需原交接材料。
'''
    (root/'research/reconnect/ORACLE_PATH_RESISTANCE_CONTROL.md').write_text(report)
    p=root/'docs/RESEARCH_STATUS.md';text=p.read_text().replace('44. [平面接触共享系数必要条件与有限输入可辨识性](../research/reconnect/CONTACT_CONSTITUTIVE_REVIEW.md)（本轮）。',
        '44. [平面接触共享系数必要条件与有限输入可辨识性](../research/reconnect/CONTACT_CONSTITUTIVE_REVIEW.md)。\n45. [同一接触oracle沿段串联阻力连续核对照](../research/reconnect/ORACLE_PATH_RESISTANCE_CONTROL.md)（本轮）。')
    text=text[:text.index('本轮关键进展：')]+f'''本轮关键进展：完成预定oracle沿段阻力连续核对照，同一N128/h0.04/D0.02和原物质区域，误差由原谐和连续核{100*g['saved_harmonic_error']:.7f}%降至{100*g['oracle_path_error']:.7f}%，减少{100*g['relative_error_reduction_fraction']:.4f}%。两档oracle向量求积差{g['empirical_indicator_components']['oracle_rule_difference']:.3e}，合并经验指标{g['combined_empirical_indicator']:.3e}，非严格界。改善主要位于原2h界面邻域，远场基本不变；整体物理耗散未同步更接近PDE。

该对照使用完整解析密度/接触位置，是额外oracle信息，未解决有限节点密度闭合。原节点候选、原图、原m/c0/身份与PDE参考保持，已存节点误差仍{100*candidate_error:.7f}%；不以oracle结果替换原状态或宣布算法/掺混预测改善。

新增6项检查最终通过，累计347项；首次标量数组类型错误日志及修正前源码保留，只改零维数组转换，公式/阶数/容差未变。旧v0.8.0/246项未恢复。782原文件、15历史附件和44份先前正式科学清单保持，4份历史诊断清单已有路径差异另核。

下一步只作同一oracle核的预定法向优先方向残差预算，复用保存总作用与原PDE a/b面通量分量，原两组求积不变；保存原始方向向量、闭合余量与带符号Gram，不作唯一因果份额，不改系数/状态或引入节点密度重构，不建图、不作瞬态。普通中心阻力FV基线保留。

数据入口：runs/reconnect-oracle-path/stage3-status.json。Stage 3 incomplete，fluid ORC关闭；无PDF/新版本、提交/推送、环境配置修改或发布。
'''
    p.write_text(text)
    print(json.dumps({'comparison':c,'CSV_exactly_verified':True,'recorded_total':347,'reference_only':True,'finite_node_candidate_admitted':False,'fluid_orc_enabled':False},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
