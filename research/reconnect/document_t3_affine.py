"""Additive report, provenance and a read-only next gate; no scientific generation."""
from pathlib import Path
import csv,shutil
from audit_route_decision import ROOT,read,save,digest


def main():
    out=ROOT/'runs/reconnect-t3-affine';source=ROOT/'runs/reconnect-t2-rotation'
    r=read(out/'results.json');audit=read(out/'numerical-audit.json');ex=read(out/'execution.json');ar=read(out/'audit-execution.json')
    assert r['comparison_decision']=='T3_affine_model_response_inconclusive' and not r['T3_registered_affine_case_passed']
    assert audit['all56_independent_world_geometries_exact_rates_and_full_raw_sequences_verified'] and ar['final_audit_passed']
    assert digest(ROOT/'exports/ltn-t2-rotation.zip')=='007d1037de1eac12ccfe43b8a6202825d0751abcf52682b6a78183e7240e011d'
    for old,new in [('/tmp/ltn-t3-affine-main.log','main.log'),('/tmp/ltn-t3-affine-geometry.log','geometry-audit.log'),('/tmp/ltn-t3-affine-audit-final.log','audit-final.log')]:
        assert not (out/new).exists();shutil.copyfile(old,out/new)
    sources={
      'runs/reconnect-t2-rotation/sha256.json':'previous cumulative scientific seal,4015 entries',
      'runs/reconnect-t2-rotation/next-gate-spec.json':'immutable original T3 registration; already_executed=false retained',
      'exports/ltn-t2-rotation.zip':'previous delivery unchanged; omitted from new ZIP',
      'research/reconnect/T2_ROTATION_COMPARISON.md':'previous supported rotation evidence and T3 plan',
      'docs/METHOD.md':'original physical PDE, constitutive models, material means and T1-T4 requirements',
      'docs/CODEX_TASK.md':'original staged task, no clipping or acceptance fabrication',
      'research/reconnect/periodic_voronoi_fv.py':'unchanged geometry and ordinary resistance backend',
      'research/reconnect/oriented_periodic_boundary.py':'unchanged exact orthogonal boundary adapter; E=I here',
      'research/reconnect/voronoi_time_action.py':'unchanged raw sparse actions and original I/H definitions',
      'research/reconnect/voronoi_time16_core.py':'unchanged ordered action and whole independent uniformization',
      'runs/reconnect-t2-rotation/original-pure-inputs.npz':'original N32 labels and IDs; byte-identical local copy',
      'runs/reconnect-t2-rotation/original-material-reference.npz':'original m/c0 reused unchanged; old T1 reference is history only'}
    save(out/'evidence-index.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai',
      'sources':[{'path':name,'sha256':digest(ROOT/name),'role':role} for name,role in sources.items()],
      'source_original_gate_rewritten':False,'new_reference_PDE_solves':0,'old_candidate_reruns':0,
      'original_material_mass_c0_or_identity_changed':False,'old_T1_reference_used_as_affine_target':False})
    next_sources=['results.json','analytic-parameters.json','analytic-reference-audit.json','numerical-audit.json','implementation-protocol.json']
    save(out/'next-gate-spec.json',{'already_executed':False,'type':'read_only_saved_T3_affine_control_and_error_budget_review',
      'reason':'t=.025 H modeldifference .004970458087194651 doesnot exceed fixedindicator .005219845797588792; preserve inconclusive and stop expansion',
      'sources':[{'path':str((out/name).relative_to(ROOT)),'sha256':digest(out/name)} for name in next_sources],
      'allowed':'read saved controls,locate positive numerical/time/input/spatial budget contributions and document limits; no retroactive acceptance or new scientific solve',
      'prohibited':'no replacement H indicator or tolerance to turn this case into passed,no new N/K/model,graphs,paths,PDE,clipping,normalization,F/Cfit or oldcandidate rerun',
      'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_independent_uniformization_sequences':0,'new_reference_PDE_solves':0,
      'old_candidate_reruns':0,'planned_new_software_checks':0,'main_wall_seconds_cap':120,'peak_RSS_MiB_cap':1024,
      'automatic_space_or_time_expansion':False,'stage3_physical_acceptance':'incomplete','general_T3_admitted':False,'fluid_orc_enabled':False})
    rows=[]
    for case,values in r['reference_error_components'].items():
        for v in values:rows.append({'case':case,'time':v['time'],**v['components_absolute'],'total_absolute':v['total_absolute']})
    with (out/'error-components.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    final={M:r['per_model_observations'][M][-1] for M in ('constant_D','constant_Gamma')}
    reference_max=max(max(v['mass_norm_differences']) for v in read(out/'analytic-reference-audit.json')['checks'])
    report=f'''# T3平面仿射压缩：恒D/恒Γ成对模型对照

2026-11-02（Asia/Shanghai）。执行[上轮登记项](../../runs/reconnect-t2-rotation/next-gate-spec.json)，[本轮实现前约定](T3_AFFINE_CONTROL.md)和[固定协议](t3_affine_protocol.json)先于N32科学输入锁定。

**本例结论：T3_affine_model_response_inconclusive，尚未通过。** 方向系数、两个模型各自的解析参考/离散模态和4/8/16时间控制通过；两个非零时刻的状态模型差已分辨，终点H模型差已分辨，但t=0.025的H差没有超过预定指标。所有原始结果和指标保留，不调阈值、不扩网格/时间步。Stage3 incomplete，fluid ORC关闭。

## 物质状态与两个不同PDE

复用原T1的N32×32标签/ID、固定质量m=1/N²、初始有限物质单元质量平均c0；旧正D T1参考只作历史来源，不能作本轮压缩参考。m和c0按字节沿用，点轨迹不用来重新采样标量。两个模型均用同一普通中心阻力Voronoi后端、同一边界表示及固定m。

T=1/20，a(t)=1−t/(4T)=1−5t，x=(a·label_x,label_y)，周期(a,1)，ρ_exact=1/a。均匀矩形物质单元在该流动下仍为矩形，其真实质量固定。图输入只有轨迹点、ID、给定ρ/D与边界，不读m/c0、参考或真实内部物质面。

D0=Γ0取原float .02的精确二进制值。恒D：D(t)=D0、Γ_nominal=D0/a；恒Γ：Γ_nominal=Γ0、D_exact(t)=Γ0a。这是两个不同物理PDE，误差不能用来给同一算法的优劣排名。生产输入ρ和D各自只转换一次float，实际Γ_wire=Fraction(ρ_float)Fraction(D_float)，全部乘积缺陷原样保存；不重标定ρ、质量或通量。

原有向矩形适配层在本例E=I，物理压缩保留在实际点和周期长度中。采样时刻j/(40K)，a=1−j/(8K)，坐标/周期无损可表示。两模型各K4/8/16，共56张新局部/世界图全部先封存，再加载真实m/c0或参考。原几何、阻力、时间后端及适配层均未修改。

## 方向与独立解析参考

普通中心阻力给出gx=Γ_wire/a、gy=Γ_wire·a，每图x/y各N²条，gx/gy=a⁻²。56图逐面精确成立。独立审计从世界双中垂线重建全部顶点、共享面、两半阻力、面积和原固定镜像探针，不调用原构造器。本均匀正交算例的真实物质分区与辅助Voronoi分区重合；不能推广为一般流动中二者等同。

物质坐标PDE为∂t c=D(t)[a⁻²∂²_label_x c+∂²_label_y c]。原正弦/余弦有限单元质量平均的衰减因子为exp[−4π²(Ix+Iy)]，其中

- 恒D：Ix=D0·t/a，Iy=D0·t；
- 恒Γ：Ix=4TΓ0 ln(1/a)，Iy=Γ0[t−t²/(8T)]。

独立80/100位评价从原标签和精确有限单元积分重建初场及两个解析时钟，不调用生产时钟或参考评价器。两精度转float结果逐位一致；与保存评价的最大质量范数差{reference_max:.17g}，小于先定64eps√M max(1,max|c0|)=1.4210854715202004e−14。没有新的参考PDE数值求解。

离散空间模态用qh=4N²sin²(π/N)，冻结中点时钟直接由给定物理ρ/D/几何积分得到，未从实际状态或g拟合。原c0模态及常量残差先满足原1e−10代数尺度，才将其时间积分计入控制；最大残差和{audit['maximum_mode_plus_constant_residue_absolute']:.17g}。

输入转换Duhamel项保留精确积分dt·4N²|δΓ|(a+a⁻¹)，再乘原初始波动向上界。数值指标保留原机器项、每区间精确1e−16尾项、完整独立序列差、初态桥和两倍评价余量。PDE对照另外保留理想冻结中点到连续时间网格的时间项、网格到连续PDE的空间项；不能把这些参考态替换成实际状态。经验机器指标使这里的总指标不构成连续PDE总误差认证定理。

## 实际结果与时间控制

误差统一为||c−c_ref||_m/σ0，σ0={r['initial_mass_fluctuation_norm']:.17g}；H=1−E(t)/E(0)，E按原质量均值中心化。没有用参考标准差作分母。

| T=0.05，K16 | 恒D | 恒Γ |
|---|---:|---:|
| 相对各自PDE参考误差 | {final['constant_D']['reference_error_relative']:.12g} | {final['constant_Gamma']['reference_error_relative']:.12g} |
| 连续网格到PDE空间项/σ0 | {final['constant_D']['spatial_symbol_error_relative']:.12g} | {final['constant_Gamma']['spatial_symbol_error_relative']:.12g} |
| 中点到连续网格时间项/σ0 | {final['constant_D']['midpoint_time_error_relative']:.12g} | {final['constant_Gamma']['midpoint_time_error_relative']:.12g} |
| 实际H | {final['constant_D']['H']:.15g} | {final['constant_Gamma']['H']:.15g} |
| 各自PDE参考H | {final['constant_D']['reference_H']:.15g} | {final['constant_Gamma']['reference_H']:.15g} |
| 原始库存漂移 | {final['constant_D']['raw_mass_change']:.12g} | {final['constant_Gamma']['raw_mass_change']:.12g} |

终时恒D状态4/8差约4.78532e−5、8/16差约1.19987e−5；恒Γ分别1.37673e−5、3.44789e−6，均按σ0归一化。两个非零时刻的状态/H步间差下降均超过先定正项指标，完整逐项值见[时间控制CSV](../../runs/reconnect-t3-affine/time-controls.csv)。约四倍下降是本模态本次三时间档的观测，不认证一般时间阶或总误差。

| t | 模型状态差/σ0 | 固定状态指标/σ0 | H_D−H_Γ | 固定H指标 | 模型响应判定 |
|---|---:|---:|---:|---:|---|
| 0.025 | 0.00258875386706 | 0.000260151487552 | 0.00497045808719 | 0.00521984579759 | 状态分辨，H未决 |
| 0.05 | 0.0111070787974 | 0.000527108003151 | 0.0203887360795 | 0.0105763727442 | 状态/H均分辨 |

模型差指标包含双方全部数值/参考/输入/时间/空间项和实际8/16差；H沿用原二次质量范数扰动公式，不事后收紧。t=0.025的实际H差是固定指标的约95.22%，因此不能声称本登记例通过。两时刻实际状态差与解析PDE状态差相容；未决并不否定方向系数或解析参考的核验，但它是本次停止扩展的条件。

![本次H、参考误差及未决模型差](../../runs/reconnect-t3-affine/t3-affine.png)

## 检查、审计与保全

三项新增软件检查最终通过，旧477项未重跑，累计记录480项。首轮合成N16例把H未决误写为应通过，仅修正测试预期；判别公式、科学参数与余量不变，首轮失败日志/JSON保留，修正在N32科学点/图/状态生成前完成。旧v0.8.0及其246项测试仍未恢复，不能宣称本轮重现了它们。

主科学计算首次{ex['elapsed_main_seconds']:.3f}秒/{ex['peak_RSS_MiB']:.2f}MiB，满足600秒/1024MiB限制。56图独立世界审计102.284秒（含等待）、66.32MiB；最终数值审计{ar['elapsed_final_audit_seconds']:.3f}秒/{ar['peak_auditor_RSS_MiB']:.2f}MiB。审计首轮在所有参考/56条区间序列均已核完后，因Python3.12补偿sum与登记模型预算顺序累加差1ulp而停止；只修正审计的浮点重算顺序，保留首轮日志和既有逐项审计文件，最终逐字核对后通过。科学数据、结果、协议及阈值未改。

全部56组精确率、最小Poisson截断、原始未归一权重和6条完整独立统一化序列逐步复核通过；主/独立路径最大相对差{audit['maximum_primary_control_relative_difference']:.17g}，原始库存最大漂移{audit['maximum_native_raw_mass_drift']:.17g}，守恒/常量/最大值与方差检查按原尺度通过。保留全部实际端点、独立端点和指数半步状态，不作裁剪或修补。

782原始文件、15附件和93份旧正式科学清单哈希核验，4份历史诊断既有差异原样保留；旧旋转ZIP SHA 007d1037de1eac12ccfe43b8a6202825d0751abcf52682b6a78183e7240e011d保持。原README/METHOD/CODEX_TASK/PHASE及旧代码/结果/报告/附件不改，仅更新RESEARCH_STATUS入口。

## 下一项与Stage3位置

本轮只完成一个N32成对仿射压缩判别例；对称双方向初始模态不等于所有方向激发，单档N也不证明一般空间收敛。一般T3、一般光滑流和有限节点候选均不宣布接纳。原T3还需完整方向/模型证据；最终同一后端的T4移动激波/独立接触及三密度来源对照仍缺，不能给出靠检查数推算的Stage3完成日期。DNS/动态ORC不是新增等待条件。

遵守未决即停止扩展：下一项仅[登记保存控制/误差预算只读复核](../../runs/reconnect-t3-affine/next-gate-spec.json)，定位各正项及解释适用范围；不得事后替换H指标把本例改判通过，不自动N64/K32、额外模型或全矩阵扫描。本轮未执行下一项。fluid ORC关闭，未计算kappa，无PDF、新版本、提交推送、环境配置更改或发布。

[数据与审计](../../runs/reconnect-t3-affine/) · [完整可重放包](../../exports/ltn-t3-affine.zip) · [下载目录](../../exports/ltn-t3-affine-downloads.md) · [交付核验](../../exports/ltn-t3-affine-delivery.json)。最终CRC/SHA和全新解压逐位重放以交付JSON为准。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/t3_affine_study.py --replay runs/reconnect-t3-affine
```

重放从56个封存缓存重建本轮6条实际与完整独立传播路径，核对原始CSR/权重、实际/独立/半步状态、模态残差、两个解析参考、全部指标及未决判定；不重新构图、求PDE或重跑旧候选。世界坐标几何输入和审计收据一并包含；完整历史保全审计仍需原工作区，累计科学清单引用未打包历史。
'''
    target=ROOT/'research/reconnect/T3_AFFINE_COMPARISON.md'
    with target.open('x') as f:f.write(report)
    index=ROOT/'docs/RESEARCH_STATUS.md';text=index.read_text();marker='执行上一轮适配层/连续旋转登记项。'
    assert marker in text and '94. ' not in text
    text=text.split(marker)[0].replace('T2_ROTATION_COMPARISON.md)（本轮）。','T2_ROTATION_COMPARISON.md)。')
    text+='''94. [T3平面仿射压缩：恒D/恒Γ成对模型对照](../research/reconnect/T3_AFFINE_COMPARISON.md)（本轮）。

执行上一轮登记的N32、T=.05平面仿射压缩成对模型，原m/c0/ID不变，a=1−5t、周期(a,1)、rho=1/a。恒D=.02，恒Gamma时D=Gamma0*a；给定rho/D各自一次转float，实际Gamma乘积缺陷保留，不配平质量或伪造密度。原几何/普通阻力/时间后端和边界适配层均未修改；56图先封存再加载m/c0，6条实际和6条完整独立路径已执行。

方向gx=Gamma_wire/a、gy=Gamma_wire*a逐面精确核验，两模型各自解析物质参考由80/100位独立核对；全部世界几何、原精确率/尾项/权重/完整统一化序列与数值控制通过。终点各自参考相对误差恒D 2.736000460014463e−4、恒Gamma 2.380609670659642e−4，不作不同PDE的算法优劣排名。两模型4/8到8/16的状态/H差下降在两个非零时刻均已分辨。

本例正式结论T3_affine_model_response_inconclusive：状态模型差两个时刻均已分辨；t=.05 H差.0203887360795大于固定指标.0105763727442，但t=.025 H差.00497045808719小于原指标.00521984579759。保留未决，不调整H指标/阈值、不扩N/K。单档N及对称模态不认证一般空间/方向收敛或完整T3。

三项新软件检查最终通过，累计记录480，旧477未重跑；首轮合成例错误通过预期已修正，失败记录保留，科学判别公式不改。独立数值审计首轮因顺序浮点求和重算相差1ulp停止，仅修正审计重算顺序后通过，两次日志保留。主计算首次88.784秒/132.73MiB；最大原始库存漂移2.59015899006787e−16。782原文件、15附件、93份旧正式清单及旧交付保持，4份历史诊断既有差异保留；旧v0.8.0/246仍未恢复。

下一项只登记已存控制/误差预算只读复核，禁止事后改判或自动加密/扫描，本轮未执行。T3完整方向/模型、最终同一后端T4及三密度来源证据仍缺；一般光滑流/有限节点接纳false，Stage3 incomplete，fluid ORC关闭，kappa未计算。报告：research/reconnect/T3_AFFINE_COMPARISON.md；入口：runs/reconnect-t3-affine/stage3-status.json；下载：exports/ltn-t3-affine-downloads.md。原README/METHOD/CODEX_TASK/PHASE及旧代码/结果/附件不改，仅更新本入口。无PDF、版本、提交推送、环境配置更改或发布；CRC/SHA及全新解压重放以交付JSON为准。
'''
    index.write_text(text)


if __name__=='__main__':main()
