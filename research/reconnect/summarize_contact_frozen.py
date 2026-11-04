"""Render the audited contact defect; do not admit a transient or repair g."""
from pathlib import Path
import argparse,json
import numpy as np
from package_compression_midpoint import dump,digest
from summarize_shear_transient_spatial import csv_exact


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve()
    assert not (out/'sha256.json').exists(),'never overwrite sealed science'
    r=json.loads((out/'results.json').read_text());audit=json.loads((out/'numerical-audit.json').read_text())
    assert audit['all_region_errors_and_contributions_recomputed']
    with np.load(out/'contact-observations.npz') as z:
        a={key:z[key] for key in ['ids','material_label_centers','mass','rho0_point','c0','kernel_rate','reference_rate','rate_error','near_contact','distance_to_contact','face_rate_a','face_rate_b']}
    numbers=np.column_stack([a['material_label_centers'],a['mass'],a['rho0_point'],a['c0'],a['kernel_rate'],a['reference_rate'],a['rate_error'],a['near_contact'],a['distance_to_contact'],a['face_rate_a'],a['face_rate_b']])
    csv=csv_exact(out/'contact-observations.csv',a['ids'],numbers,['id','a_m','b_m','mass_kg','rho_kg_per_m3','c0_mass_average','candidate_rate_per_s','PDE_rate_per_s','error_per_s','near_contact_0_or_1','distance_to_contact_m','PDE_a_rate_per_s','PDE_b_rate_per_s'])
    dump(out/'observations-audit.json',{'CSV':csv,'raw_NPZ_sha256':digest(out/'contact-frozen.npz'),'small_NPZ_sha256':digest(out/'contact-observations.npz')})
    next_plan={
        'scope':'fixed-h continuous nonlocal reference on this same contact state, mass-averaged on original cells; no new candidate graph or transient',
        'physical_cases':1,'N_per_direction':128,'h_m':.04,'D_m2_per_s':.02,
        'source':'runs/reconnect-contact-frozen/contact-frozen.npz','same_mass_c0_ids_and_PDE_reference':True,
        'continuous_physical_field':'same c(a,b)=0.5+0.2*f(a)*cos(2*pi*b), f(a)=cos(2*pi*xi(a)+pi/5)',
        'method':'exact transverse Fourier reduction of the unchanged 2D radial kernel; independent quadrature of its continuous integral, then exact-y and Gauss-x mass observation',
        'kernel':'unchanged positive_kernel_2d.slope_over_radius; raw harmonic rho*D coefficient',
        'fixed_quadrature_levels':[{'cell_x_order':8,'normal_displacement_order':32,'transverse_order':32},{'cell_x_order':16,'normal_displacement_order':64,'transverse_order':64}],
        'splitting':'normal displacement at 0 and density-contact crossings; cell x at support-entry points contact +/-2h; periodic wrap in reference only',
        'observation':'rho-weighted material cell means using original saved masses, no normalization',
        'decomposition':'discrete(cbar)-PDEbar = [discrete(cbar)-R continuous_nonlocal(c)] + [R continuous_nonlocal(c)-PDEbar]',
        'interpretation':'first remainder combines point quadrature and mean-state closure, not a unique quadrature attribution; second measures this fixed-width constitutive deviation',
        'measure':'all/within-2h/complement norms with the original global PDE norm; vector identity and quadrature differences',
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'parameter_or_mesh_matrix_scan':False,'full_P_constructed':False,'fluid_orc_enabled':False,
        'stop_condition':'retain adverse or unresolved result; no state/weight replacement, moment fit, renormalization, threshold change, or transient admission'}
    q=r['reference_controls'];g=r['regional_comparison'];raw=r['raw_graph_controls'];diag=r['operator_diagnostics'];alg=r['graph_algebra']
    status={'updated':'2026-11-01','timezone':'Etc/UTC','stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'scope':'one predeclared stationary periodic 1:4 density-contact frozen graph',
        'new_frozen_graphs':1,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'regional_comparison':g,'reference_controls':q,'candidate_backend_unchanged':True,
        'contact_transient_admitted':False,'diagnosis':'clear interface-localized frozen-rate defect at fixed N/h; positive conservative graph is insufficient',
        'strong_defect_proves_weak_limit_failure':False,'fixed_width_and_quadrature_and_state_closure_uniquely_separated':False,
        'new_checks_passed':6,'previous_recorded_checks':330,'recorded_checks_total':336,'old_v080_or_246_tests_recovered':False,
        'next_contact_continuous_control_predeclared':next_plan,
        'next_task':'same fixed-h continuous-kernel material-average reference; no new graph or transient',
        'remaining':['contact/interface constitutive closure','general geometry/density, shocks, walls, ledger, nonplanar/DNS physical acceptance'],
        'report':'research/reconnect/PERIODIC_DENSITY_CONTACT_FROZEN.md','pdf_generated':False,'software_version_created':False,'published':False}
    dump(out/'stage3-status.json',status)
    table=['| 区域 | 单元数 | 误差 / 全域参考范数 | 全部误差平方的份额 | 最大绝对误差，/s |','|---|---:|---:|---:|---:|']
    for key,title in [('all','全域'),('near_contact','距任一接触<=2h'),('far_from_contact','其余区域')]:
        v=g[key];table.append(f"| {title} | {v['cells']} | {100*v['error_over_global_reference_norm']:.6f}% | {100*v['error_squared_fraction_of_global']:.6f}% | {v['maximum_absolute_rate_error_per_s']:.8f} |")
    energy_bias=(raw['chi_CSR_kg_per_s']-r['projected_reference_chi_kg_per_s'])/r['projected_reference_chi_kg_per_s']
    report=f'''# 二维周期密度接触：一张冻结图的界面通量检验

2026-11-01（UTC）。Stage 3 **incomplete**；fluid ORC **关闭**。

上一轮预定的N128/h0.04、D0.02、密度比1:4静止周期接触已完成。未经修改的点核相对完整PDE质量平均变化率的全域误差为 **{100*g['all']['error_over_global_reference_norm']:.6f}%**。预定2h界面邻域包含 **{100*g['near_contact']['error_squared_fraction_of_global']:.6f}%的误差平方**。两条参考路线差仅{q['analytic_faces_vs_volume16_relative']:.6e}，共享边/CSR作用差{raw['pair_scatter_vs_CSR_relative']:.6e}，均远小于该缺陷。正性、守恒和能量代数不能替代这一物理比较；**不接纳该接触算例的新瞬态传播**。

## 1. 问题定义与真实质量平均

执行 `runs/reconnect-shear-transient-spatial/stage3-status.json` 的 `next_contact_frozen_predeclared`，运行前核验851项科学清单。只构建一张新图，无参数/网格矩阵扫描。周期单位方形上u=0，rho=1（0<=a<0.5）、rho=4（0.5<=a<1），接触位于a=0与0.5，D=0.02。128×128固定物质区域与接触面对齐，单位厚度1 m。

定义总阻力R=integral_0^1 da/(rho D)=31.25，xi(a)=integral_0^a dx/(rho D)/R，theta=2pi xi+pi/5。连续标量为

c(a,b)=0.5+0.2 cos(theta) cos(2pi b)。

两个接触面均满足[c]=0和[rho D c_a]=0，法向通量并非恒为零。按q=+rho D grad(c)记梯度通量，rho c_t=div(q)，物理Fick通量为−q：

q_a=−0.2(2pi)/R sin(theta)cos(2pi b)，
q_b=−0.2(2pi)rho D cos(theta)sin(2pi b)。

q_a在密度跳跃处连续，故PDE分布散度没有额外的界面delta项。不能把本初场当作没有扩散的稳态，它只是符合透射条件的独立冻结状态。

每个单元的rho为常数，m_i=rho_i/N²，令kappa_i=1/(rho_i D R)，有精确物质平均

c0_i=0.5+0.2 sinc(kappa_i/N)sinc(1/N)cos(theta_i)cos(2pi b_i)。

sinc(z)=sin(pi z)/(pi z)。原质量和初场直接使用上述积分；点值与质量平均最大差 **{r['mass_average_vs_point_scalar_max_difference']:.6e}**，未互相替代。本例m/(rho·单位厚度)与真实面积恰好逐位相等，差为 **0**，因此不能将本轮界面缺陷解释为m/rho的体积误差。候选仍仅接收点位置、固定质量和点rho，解析面、梯度、J、F/C没有进入候选。

共享轨迹数据结构保存两个完全相同的静止帧，仅为满足现有接口；没有进行时间积分或计算有限时间H。

## 2. 两条独立积分路线

第一条参考解析积分四个面的q，并按(right−left+top−bottom)/m得到原物质单元平均变化率。第二条在每个纯相内使用完整PDE：

c_t=−0.2(2pi)²D(kappa_i²+1)cos(theta)cos(2pi b)，

分别以Gauss8和Gauss16积分rho*c_t，再除以同一原m。不在接触处对rho求普通导数；通量透射条件已经消除了界面分布项。参考分量沿a、b方向分别保留，不将它们称作候选误差的唯一来源。

| 参考核对 | 数值 |
|---|---:|
| 解析面 / Gauss16体积分，相对全域变化率范数 | {q['analytic_faces_vs_volume16_relative']:.6e} |
| Gauss8 / Gauss16，同范数 | {q['volume8_vs_volume16_relative']:.6e} |
| 精确m / Gauss16最大相对差 | {q['exact_mass_vs_Gauss16_max_relative']:.6e} |
| 精确c0 / Gauss16最大绝对差 | {q['exact_c0_vs_Gauss16_max_absolute']:.6e} |
| 面通量总库存变化率，kg/s | {q['physical_inventory_rate_faces_kg_per_s']:.6e} |
| 完整连续能量恒等余差，kg/s | {q['physical_energy_identity_volume16_residual_kg_per_s']:.6e} |

原始舍入量全部保留，不做质量或状态重归一化。上述差是数值核对量，不是严格连续问题误差上界。

## 3. 未修改点核的区域误差

原 `positive_kernel_2d.build_periodic_exchange` 使用固定h=0.04、完整2h支撑、m/rho求积面积和原谐和rho D系数，没有修改界面边。令r_disc=−M^-1 L c0，r_ref为上述解析面参考。**所有区域均除以同一个全域sqrt(sum m*r_ref²)**；区域掩码只用于统计，不参与构图或改权重。

'''+'\n'.join(table)+f'''

近、远区域误差范数的平方之和等于全域误差平方（保留舍入），不是两个百分比直接相加。预定近界面区域含5120个单元，占31.25%单元及质量；其余11264单元。不能把远场的0.785%全域归一化量误称为远场自身参考范数下的误差。

该固定N/h结果说明当前点核不能凭光滑压缩/剪切改善而自动接入密度接触传播。它与[已有一维界面反例](../rebuild/KERNEL_JOINT_LIMIT.md)及[界面内层证据](../rebuild/INNER_MATCHING.md)相容，但不构成二维弱极限错误证明，也没有仅凭本图分离有限核宽、节点求积和质量平均闭合的各自影响。

![原始变化率、界面残差及数值核对](../../runs/reconnect-contact-frozen/contact-frozen.png)

## 4. 图代数和积分能量为何不能替代局部检验

图有{diag['nodes']}个节点、{diag['positive_exchange_pairs']}条共享正边，连通，最小原始导通率{diag['minimum_raw_conductance_kg_per_s']:.6e} kg/s；原代数检查通过。常数残差{alg['constant_residual_per_s']:.6e} /s、质量残差{alg['mass_residual_kg_per_s']:.6e} kg/s、加权对称性残差{alg['weighted_symmetry_residual_kg_per_s']:.6e} kg/s，沿用原1e-10代数容差，不将其用作物理验收阈值。

CSR作用与独立共享边散射相对差{raw['pair_scatter_vs_CSR_relative']:.6e}，最大绝对差{raw['pair_scatter_vs_CSR_max_absolute_per_s']:.6e} /s。候选总组分质量变化率为{raw['raw_component_mass_rate_kg_per_s']:.6e} kg/s，未强制清零。共享边重新组装与原L最大差{raw['shared_edge_assembly_max_difference']:.6e}。

| 冻结耗散量 | kg/s |
|---|---:|
| 图cbarᵀL cbar | {raw['chi_CSR_kg_per_s']:.12f} |
| 原共享边sum g(cbar_j−cbar_i)² | {raw['chi_pair_kg_per_s']:.12f} |
| 投影参考−sum m*cbar*r_ref | {r['projected_reference_chi_kg_per_s']:.12f} |
| 连续物理integral rho D grad(c)² | {r['physical_continuous_chi_kg_per_s']:.12f} |

图耗散与投影参考的偏差约{100*energy_bias:.4f}%，而完整变化率误差约14.16%。一个较小的整体耗散偏差不能保证界面局部作用正确。投影参考与连续物理能量本身也不完全相等，包含单元内协方差，不混用这两种观测量。没有把这些数用于修正状态或边权，没有传播c或运行最大值原理数值实验。

## 5. 检查、保存与原始文件

新增6项检查全部通过：精确质量平均/独立Gauss、两个接触的浓度/非零通量透射、解析面/完整PDE积分、连续库存/能量恒等式、预定周期2h分区、非法参考输入拒绝。累计330+6=**336项重建检查**，旧v0.8.0/246项仍未恢复。

782个原始文件、15个历史附件和41份先前正式科学清单字节保持；4份历史诊断清单的已有路径差异另核。原README/METHOD/CODEX_TASK、附件、原后端与旧结果未改，只更新 `docs/RESEARCH_STATUS.md`。内容哈希按路径/大小/修改时间缓存已读取字节，逐项比较所有正式清单，没有跳过已知历史差异。

数据目录 `runs/reconnect-contact-frozen/` 保存原始共享边/电导/CSR、全部点输入、真实平均初场、面通量、Gauss8/16、掩码、原始变化率/误差、CSV和审计。CSV按原类型解析后与NPZ逐项一致。峰值RSS约{r['peak_process_RSS_MiB_linux']:.1f} MiB。没有PDF、新版本、提交、推送、环境配置更改或发布。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/test_contact_frozen_probe.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/contact_frozen_study.py --output /workspace/contact-frozen-reproduction
```

## 6. 下一步只做同一固定h的连续核参照

不运行这个接触问题的瞬态，不扩展N/h/密度比，不修正任何界面边。下一轮只对当前同一连续物理场评估**原核的连续积分**，再观察到同一原物质区域，用于区分固定核宽本构偏差与剩余离散/状态闭合差异。

令f(a)=cos(2pi xi(a)+pi/5)，B_h=−W'_h/r，Gamma=rho D，H为原谐和Gamma。定义

b0(s)=integral B_h(sqrt(s²+z²)) dz，
bk(s)=integral B_h(sqrt(s²+z²))cos(2pi z) dz，

z积分区间为±sqrt(4h²−s²)。利用横向周期模式的精确消去，连续核变化率为

r_h(a,b)=2(0.2)cos(2pi b)/rho(a) * integral_(s from −2h to 2h) H(a,a+s)[f(a+s)bk(s)−f(a)b0(s)] ds。

a+s按周期取值。再对rho*r_h做原区域积分并除以原保存m；候选图不接收这些几何输入。只使用两组固定求积控制：(单元a阶, s阶, z阶)=(8,32,32)、(16,64,64)，在密度接触、s=0和接触±2h支撑进入位置分段。不是新的核宽/参数矩阵扫描。

按向量恒等式保留

r_disc(cbar)−r_PDE_bar = [r_disc(cbar)−R r_h(c)] + [R r_h(c)−r_PDE_bar]。

第一项仍合并节点求积与质量平均状态闭合，不将其唯一归因为求积；第二项是该固定核宽在当前连续物理场上的本构偏差。两档求积差不是严格界。若缺陷持续或结果未分清，保留结论，不做校正、不拟合，不自动接纳瞬态。本轮没有执行这一新积分参照。

普通中心阻力FV基线与既有不利结果继续保留；一般几何/密度、激波、固壁、ledger、非平面/DNS未验收。**Stage 3 incomplete，fluid ORC继续关闭。**

## 7. 交付

- [代码、报告与完整观测审阅包](../../exports/ltn-contact-frozen-review.zip)。
- [原始图及完整数据下载](../../exports/ltn-contact-frozen-downloads.md)。
- [交付哈希清单](../../exports/ltn-contact-frozen-delivery.json)。

本轮为增量交付，完整历史审计/CLI复现需按下载说明叠加五份历史基包；782个原始文件审计还需原交接材料。
'''
    (root/'research/reconnect/PERIODIC_DENSITY_CONTACT_FROZEN.md').write_text(report)
    p=root/'docs/RESEARCH_STATUS.md';text=p.read_text()
    text=text.replace('41. [同一二维剪切流的细档实际传播与共同物质区域比较](../research/reconnect/SHEAR_TRANSIENT_SPATIAL_CONTROL.md)（本轮）。',
        '41. [同一二维剪切流的细档实际传播与共同物质区域比较](../research/reconnect/SHEAR_TRANSIENT_SPATIAL_CONTROL.md)。\n42. [二维周期密度接触的一张冻结图检验](../research/reconnect/PERIODIC_DENSITY_CONTACT_FROZEN.md)（本轮）。')
    text=text[:text.index('本轮关键进展：')]+f'''本轮关键进展：执行预定N128/h0.04/D0.02、rho比1:4的静止周期密度接触一张图。全域质量加权变化率误差{100*g['all']['error_over_global_reference_norm']:.6f}%，{100*g['near_contact']['error_squared_fraction_of_global']:.6f}%的误差平方位于预定2h接触邻域；m/rho与真实面积精确相等。解析面/完整PDE Gauss16差{q['analytic_faces_vs_volume16_relative']:.3e}、共享边/CSR差{raw['pair_scatter_vs_CSR_relative']:.3e}。图代数通过不能替代通量精度，不接纳该接触算例的新瞬态，也不以局部强缺陷证明弱极限错误。

新增6项检查通过，累计336项重建检查；旧v0.8.0/246项未恢复。782原文件、15历史附件及41份先前正式科学清单保持，4份历史诊断清单已有路径差异另核。原后端、旧结果与附件未改。

下一步只做同一h/物理场的连续二维核积分，经横向Fourier精确约化并观察到原物质区域。固定两组求积(8,32,32)/(16,64,64)，分开保留固定宽本构偏差与离散/平均闭合组合余差；不建新图、不做瞬态、不修权重/状态或阈值，不扩展扫描。一般几何/密度、激波、固壁、ledger、非平面/DNS未验收。Stage 3 incomplete，fluid ORC关闭。

数据入口：runs/reconnect-contact-frozen/stage3-status.json。未生成PDF或新版本，未提交/推送、修改环境配置或发布。
'''
    p.write_text(text)
    print(json.dumps({'regional_comparison':g,'CSV_exactly_verified':True,'new_checks_passed':6,'recorded_total':336,'contact_transient_admitted':False,'fluid_orc_enabled':False},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
