"""Seal-ready report of a fixed-h model defect; no numerical correction."""
from pathlib import Path
import argparse,json
import numpy as np
from package_compression_midpoint import dump,digest
from summarize_shear_transient_spatial import csv_exact


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve()
    assert not (out/'sha256.json').exists(),'never rewrite sealed evidence'
    r=json.loads((out/'results.json').read_text());audit=json.loads((out/'numerical-audit.json').read_text())
    assert audit['all_regional_vectors_norms_signed_cross_terms_and_identities_recomputed']
    with np.load(out/'contact-continuous.npz') as z:
        keys=['ids','material_label_centers','mass','c0','kernel_rate','reference_rate','coarse_rate','fine_rate',
            'fixed_width_deviation','discrete_mean_closure_remainder','quadrature_difference','near_contact','distance_to_contact']
        a={key:z[key] for key in keys}
    numbers=np.column_stack([a['material_label_centers'],a['mass'],a['c0'],a['kernel_rate'],a['reference_rate'],a['coarse_rate'],a['fine_rate'],
        a['fixed_width_deviation'],a['discrete_mean_closure_remainder'],a['quadrature_difference'],a['near_contact'],a['distance_to_contact']])
    csv=csv_exact(out/'contact-continuous.csv',a['ids'],numbers,['id','a_m','b_m','mass_kg','c0','saved_discrete_rate','saved_PDE_rate','continuous_coarse_rate','continuous_fine_rate','C_fixed_width_deviation','R_combined_remainder','quadrature_difference','near_contact_0_or_1','distance_to_contact_m'])
    dump(out/'observations-audit.json',{'CSV':csv,'raw_NPZ_sha256':digest(out/'contact-continuous.npz')})
    b=r['continuous_runs']['fine']['budget'];v=b['all'];delta=r['quadrature_difference_controls']['all']
    source=root/'runs/reconnect-contact-frozen';old=json.loads((source/'results.json').read_text())
    next_review={
        'scope':'theory-only planar contact constitutive consistency and finite-input identifiability review',
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'parameter_scan':False,
        'same_kernel_and_density_contrast':True,'baseline':'ordinary central-resistance material FV remains a control branch',
        'tasks':['derive two one-sided continuous-kernel traces for a local constant normal-flux transmission field',
            'check necessary conditions on a shared cross-phase coefficient depending only on phase diffusivities',
            'review whether fixed finite point/mass/density data identify the needed along-segment resistance',
            'state the limited scope: finite-h local trace consistency and finite-input information, not a universal impossibility or weak-limit theorem'],
        'do_not_introduce_as_required_candidate_inputs':['true interface or material faces','F/C or fitted gradients','unavailable along-edge density integrals'],
        'no_implementation_before_review':True,'no_clipping_or_moment_fit_or_normalization':True,
        'no_automatic_physical_acceptance_or_ORC':True,'fluid_orc_enabled':False}
    status={'updated':'2026-11-01','timezone':'Etc/UTC','stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'scope':'same fixed-h contact continuous integral, two predeclared quadrature rules, material mass-average observation',
        'best_raw_rule':'fine','budget':b,'quadrature_difference_controls':r['quadrature_difference_controls'],
        'norms_are_additive_error_fractions':False,'quadrature_difference_is_rigorous_bound':False,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'contact_transient_admitted':False,
        'conclusion':'the fixed-h continuous constitutive deviation persists and dominates the measured combined discrete/mean-state remainder; quadrature alone does not remove it',
        'weak_limit_failure_proven':False,'unique_point_quadrature_vs_mean_state_split_obtained':False,
        'new_checks_passed':5,'previous_recorded_checks':336,'recorded_checks_total':341,'old_v080_or_246_tests_recovered':False,
        'initial_reference_selfcheck_failure_preserved':True,'assertion_tolerances_unchanged_after_exact_parity_fix':True,
        'unchanged_contact_candidate_numerical_refinement':'paused pending constitutive review',
        'next_contact_constitutive_review_predeclared':next_review,'next_task':'theory-only shared-coefficient interface consistency and input-identifiability review',
        'remaining':['interface constitutive closure','general geometry/density, shocks, walls, ledger, nonplanar/DNS physical acceptance'],
        'report':'research/reconnect/CONTACT_CONTINUOUS_KERNEL_CONTROL.md','pdf_generated':False,'software_version_created':False,'published':False}
    dump(out/'stage3-status.json',status)
    table=['| 区域，同一全域PDE范数 | 原离散总误差 E | 连续固定宽偏差 C | 离散/平均闭合组合余差 R | 两档求积差 |','|---|---:|---:|---:|---:|']
    for key,label in [('all','全域'),('near_contact','原2h邻域'),('far_from_contact','其余区域')]:
        row=b[key];table.append(f"| {label} | {100*row['total_error']:.7f}% | {100*row['continuous_fixed_width_error']:.7f}% | {100*row['discrete_mean_closure_remainder']:.7f}% | {r['quadrature_difference_controls'][key]:.6e} |")
    coarse=r['continuous_runs']['coarse'];fine=r['continuous_runs']['fine']
    report=f'''# 接触界面固定h连续核积分与同物质区域误差预算

2026-11-01（UTC）。Stage 3 **incomplete**；fluid ORC **关闭**。

完成上轮预定的同一h=0.04连续核积分。原离散冻结变化率误差为 **{100*v['total_error']:.7f}%**；将原核作用于连续物理场并观察到同一原物质区域，仍有 **{100*v['continuous_fixed_width_error']:.7f}%** 的PDE偏差。离散/质量平均闭合组合余差为 **{100*v['discrete_mean_closure_remainder']:.7f}%**，两组预定求积的相对差为 **{delta:.6e}**。当前固定核宽下的主要偏差已存在于连续本构层面，不能仅靠提高节点求积精度消除。

三个范数不是可相加的误差份额：交叉项为负，原离散结果与连续固定宽偏差有部分抵消。结果不证明h趋零的弱极限错误；本轮没有新图、瞬态、权重修正或物理验收通过。

## 1. 原问题与原输入保持

运行前核验 `runs/reconnect-contact-frozen/sha256.json` 的879项科学清单，严格执行 `next_contact_continuous_control_predeclared`。N=128、h=0.04、D=0.02、静止周期密度比1:4、初始连续场、原m/c0/身份、PDE参考和离散变化率全部沿用上轮。当前rho在每个原区域内为常数，m/(rho·单位厚度)与真实面积精确相等，未重新估计体积。

连续物理场为c(a,b)=0.5+A f(a)cos(kb)，A=0.2、k=2pi，f(a)=cos(2pi xi(a)+pi/5)，xi为归一化积分阻力坐标。该场满足两个接触面的浓度与rho D c_a透射条件。连续积分参考直接作用于c，再作质量平均；它不是将点值替换为候选状态，原离散候选始终使用原cbar。

## 2. 未改核的二维积分约化

B_h(r)=−W'_h(r)/r沿用原 `positive_kernel_2d.slope_over_radius`，Gamma=rho D，H为原谐和Gamma。连续算子为

r_h(x)=2/rho(x) * integral H(x,x+r) B_h(|r|)[c(x+r)−c(x)] dr。

记法向位移s、横向位移z、L(s)=sqrt(4h²−s²)，令

b0(s)=integral_(−L to L) B_h(sqrt(s²+z²)) dz，
bk(s)=integral_(−L to L) B_h(sqrt(s²+z²))cos(kz) dz。

横向正弦项因核的偶对称性精确消去，因此

J(a)=integral_(−2h to 2h) H(a,a+s)[f(a+s)bk(s)−f(a)b0(s)] ds，
r_h(a,b)=2A cos(kb)J(a)/rho(a)。

周期取值仅属于参考积分，没有修改候选坐标。对原区域K_i观察：

(𝓡 r_h)_i = [2A * integral_cell-a J(a)da * integral_cell-b cos(kb)db] / m_i_saved。

分母保持原质量，横向余弦积分解析计算，法向单元积分按既定Gauss求积，不重归一化组分量、状态或核矩。保存全部原始外层节点、权重、所属列和J响应，便于复核积分到质量平均的闭合。

## 3. 两组固定求积与自检修正

仅运行预定的(单元a阶,s阶,z阶)=(8,32,32)和(16,64,64)。s在0和密度接触穿越点分段，单元a在接触±2h支撑进入处分段；原始分段和浮点舍入不作修正。两组外层节点数分别为{coarse['outer_quadrature_points']}和{fine['outer_quadrature_points']}。

初次自检使用跨整个横向区间的Gauss规则，在径向核中心附近精度不足：原始二阶矩差2.0154e-9、均匀Fourier/独立极坐标Bessel特征值差7.9829e-9，未通过原2e-11容差。正式计算前改用**精确偶对称性，在[0,L]求积并乘2**；原核、两组阶数和全部断言容差不变。直接二维核对也对正负半轴分别积分，未使用模式消去。

最终5项检查全部通过：未经归一化的核二阶矩、与独立极坐标Bessel积分的Fourier特征值、接触近/远点的直接二维积分、原质量分母观察、不合法输入。首次失败日志和修正前参考/测试源码均保存；审计逐项确认断言容差没有放宽。没有通过矩归一化制造通过。

两档实际质量平均作用差为{delta:.6e}（同一原全域PDE范数），不是严格误差界。最细原始规则在运行前选定，没有按接近PDE程度挑选或外推。当前固定宽偏差远大于已测求积差。

## 4. 向量预算与反向抵消

以𝓡表示原单元质量平均观察。定义E=r_disc(cbar)−r_PDE_bar，C=𝓡 r_h(c)−r_PDE_bar，R=r_disc(cbar)−𝓡 r_h(c)，则E=C+R。第一余差R仍合并节点求积与质量平均状态闭合，不能唯一归因为求积误差。C是在此连续物理场上的固定宽本构偏差。

'''+'\n'.join(table)+f'''

所有区域、分量与求积差使用同一个原全域sqrt(sum m*r_PDE_bar²)，原2h掩码不变，不影响积分或图。预算的平方恒等式为

||E||² = ||C||² + ||R||² + 2<C,R>。

全域带符号交叉项2<C,R>/||r_PDE_bar||²为 **{v['signed_twice_cross_term_over_reference_norm_squared']:.12e}**，平方恒等余差为 **{v['squared_norm_identity_residual']:.6e}**；向量恒等余差范数为{v['vector_identity_residual']:.6e}。不能把14.47%和0.46%解释为合计14.16%的两个正份额，也不能按范数比宣布严格的因果百分比。

本结果支持一个限定判断：在此h、密度接触与连续物理场上，原谐和系数核的连续作用本身已偏离目标局部PDE。只加密节点求积或改进该积分不会消除这个固定宽偏差。这里没有研究h趋零路径，也没有由局部强残差宣称弱极限失败。

![同一物质区域的连续核偏差、组合余差与带符号交叉项](../../runs/reconnect-contact-continuous/contact-continuous.png)

## 5. 库存与耗散保持原始数值

细规则质量平均变化率的总库存残差为{fine['raw_mass_rate_kg_per_s']:.6e} kg/s，未强制清零。

| 耗散/投影量 | kg/s |
|---|---:|
| 原离散图cbarᵀL cbar | {old['raw_graph_controls']['chi_CSR_kg_per_s']:.12f} |
| 连续核投影−sum m*cbar*(𝓡r_h) | {fine['projected_chi_kg_per_s']:.12f} |
| 连续非局部物理−A² integral fJ da | {fine['physical_continuous_nonlocal_chi_kg_per_s']:.12f} |
| 原局部PDE连续物理耗散 | {old['physical_continuous_chi_kg_per_s']:.12f} |

投影和连续物理能量是不同观察量，包含单元内协方差，未混用它们修正状态。整体耗散接近也不能覆盖14%左右的局部变化率偏差。

## 6. 检查、完整性与复现

5项新增检查最终通过，计数一次，累计336+5=**341项重建检查**；不是旧v0.8.0/246项的恢复。782个原始文件、15个历史附件和42份先前正式科学清单保持；4份历史诊断清单已有路径差异单独核验。原README/METHOD/CODEX_TASK、后端、原图和旧结果未改，只更新可变进度入口 `docs/RESEARCH_STATUS.md`。

数据目录 `runs/reconnect-contact-continuous/` 保存原输入/参考/离散作用、两规则的所有原始外层求积记录、质量平均作用、误差分量/交叉项、CSV、完整自检与审计。CSV按原身份/浮点类型解析后与NPZ逐项一致。峰值RSS约{r['peak_process_RSS_MiB_linux']:.1f} MiB，包含读取历史证据的开销。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/test_contact_continuous.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/contact_continuous_study.py --output /workspace/contact-continuous-reproduction
```

没有新候选图、瞬态、全P、裁剪、归一化、F/C拟合、阈值调整、PDF、软件版本、提交、推送或发布。

## 7. 下一步转为本构必要条件与输入可辨识性审查

暂停继续加密当前原样接触候选。下一轮只作理论审查：对平面接触局部恒法向通量场，推导原连续核的双侧迹；检查只依赖两相Gamma的共享跨相系数要满足哪些必要条件，并审查固定有限的点位置/质量/节点rho能否辨识所需沿段阻力。

审查范围限定为有限h局部迹一致性和有限输入信息，不预设一般正交换不可能，也不以此直接否定弱极限。若涉及沿段密度积分或界面位置，要明确附加信息要求，不能将真实界面/物质面、F/C、拟合梯度或未提供的密度积分直接升级为候选必要输入。此轮不实现新后端、不建图、不扫描、不作瞬态，不通过拟合或归一化消去已保存缺陷。

普通中心阻力物质FV基线和[旧一维界面证据](../rebuild/KERNEL_JOINT_LIMIT.md)继续保留；一般几何/密度、激波、固壁、ledger、非平面/DNS未验收。**Stage 3 incomplete，该接触算例不接纳瞬态，fluid ORC继续关闭。**

## 8. 交付

- [代码、报告与完整观测审阅包](../../exports/ltn-contact-continuous-review.zip)。
- [完整数据及历史依赖下载说明](../../exports/ltn-contact-continuous-downloads.md)。
- [交付哈希清单](../../exports/ltn-contact-continuous-delivery.json)。

本轮为增量交付，历史完整审计与CLI复现需按下载说明叠加六份历史基包；原782文件审计仍需原交接材料。
'''
    (root/'research/reconnect/CONTACT_CONTINUOUS_KERNEL_CONTROL.md').write_text(report)
    p=root/'docs/RESEARCH_STATUS.md';text=p.read_text().replace('42. [二维周期密度接触的一张冻结图检验](../research/reconnect/PERIODIC_DENSITY_CONTACT_FROZEN.md)（本轮）。',
        '42. [二维周期密度接触的一张冻结图检验](../research/reconnect/PERIODIC_DENSITY_CONTACT_FROZEN.md)。\n43. [同一接触固定h连续核与物质平均误差预算](../research/reconnect/CONTACT_CONTINUOUS_KERNEL_CONTROL.md)（本轮）。')
    text=text[:text.index('本轮关键进展：')]+f'''本轮关键进展：同一h0.04接触的连续核物质平均参照完成。原离散误差{100*v['total_error']:.7f}%，连续固定宽偏差{100*v['continuous_fixed_width_error']:.7f}%，离散/平均闭合组合余差{100*v['discrete_mean_closure_remainder']:.7f}%；两档预定求积差{delta:.3e}。负交叉项使总误差小于连续偏差，范数不作正份额相加。该固定宽缺陷已存在于连续本构，单独改进求积不能消除；没有证明h->0弱极限错误。

新增5项自检最终通过，累计341项；首次横向全区间求积的两项精度失败与修正前源码保留，精确偶对称半轴求积修正后容差/阶数/原核未变。旧v0.8.0/246项未恢复。782原文件、15历史附件和42份先前正式科学清单保持，4份历史诊断清单已有路径差异另核。

下一步只作平面接触共享跨相系数的必要条件和有限点/质量/密度输入可辨识性审查，暂停继续细化原样接触候选；不实现新后端、不建图、不扫描、不作瞬态。真实面、F/C、拟合梯度和未提供的沿段密度积分不能自动变成候选输入。普通中心阻力FV基线与旧不利结果保留。

数据入口：runs/reconnect-contact-continuous/stage3-status.json。Stage 3 incomplete，fluid ORC关闭；未生成PDF或新版本，未提交/推送、修改环境配置或发布。
'''
    p.write_text(text)
    print(json.dumps({'budget':b,'quadrature_difference':r['quadrature_difference_controls'],'CSV_exactly_verified':True,'recorded_total':341,'contact_transient_admitted':False,'fluid_orc_enabled':False},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
