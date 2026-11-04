"""Additive read-only review report and mutable status entry."""
from pathlib import Path
import shutil
from audit_route_decision import ROOT,read,save,digest


def main():
    out=ROOT/'runs/reconnect-t3-affine-review';source=ROOT/'runs/reconnect-t3-affine'
    r=read(out/'results.json');ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'review-audit.json')
    assert audit['receipt_consistent'] and audit['prior_formal_manifests_verified']==94
    shutil.copyfile('/tmp/ltn-t3-affine-review-final.log',out/'review-final.log');shutil.copyfile('/tmp/ltn-t3-affine-review-audit.log',out/'audit.log')
    sources={
      'runs/reconnect-t3-affine/sha256.json':'previous cumulative science seal,4483 entries',
      'runs/reconnect-t3-affine/next-gate-spec.json':'immutable saved-budget review registration; historical already_executed=false retained',
      'research/reconnect/T3_AFFINE_COMPARISON.md':'original N32 experiment and unchanged inconclusive decision',
      'research/reconnect/T3_AFFINE_CONTROL.md':'original H indicator and stop rule',
      'research/reconnect/t3_affine_core.py':'original grouping/evaluation order, source inspection only; not imported or executed',
      'runs/reconnect-t3-affine/t3-affine.png':'original experiment plot retained for report context',
      'exports/ltn-t3-affine-delivery.json':'previous fresh full-path replay receipt',
      'exports/ltn-t3-affine.zip':'previous full delivery, byte-identical, omitted from this archive',
      'docs/METHOD.md':'original material state, T3 and three-space/three-time-level requirements',
      'docs/CODEX_TASK.md':'original physical acceptance scope and density-source comparison requirements'}
    save(out/'evidence-index.json',{'sources':[{'path':name,'sha256':digest(ROOT/name),'role':role} for name,role in sources.items()],
      'source_copies':read(out/'source-copies.json'),'next_material_N16_N64_hashed_only_not_parsed':True,
      'new_scientific_experiments':0,'original_results_or_acceptance_changed':False})
    a,b=r['checkpoints'][1:];share=lambda row,key:100*row['state_indicator_fractions'][key]
    text=f'''# T3仿射压缩：保存误差预算复核与空间对照方案

2026-11-02（Asia/Shanghai）。执行[上轮只读登记项](../../runs/reconnect-t3-affine/next-gate-spec.json)。[复核协议](t3_affine_budget_review_protocol.json)限定于保存的N32控制量和旧数组观测，不是新物理试验的事前注册。

**决定：不继续单独增加时间步，下一项登记必要的三档空间对照。** t=0.025的原绝对状态指标中，空间项占{share(a,'spatial_symbol_error'):.5f}%；仅这个空间项经原H公式得到的诊断值约{a['spatial_only_original_formula_diagnostic_lower']['float']:.12g}，已经高于实际H模型差{a['source_actual_H_gap']:.12g}和保存连续时间网格H差{a['saved_continuous_grid_H_gap']['float']:.12g}。原N32算例仍为T3_affine_model_response_inconclusive，没有改判。Stage3 incomplete，fluid ORC关闭。

## 读取了什么，保留了什么

只读取前轮已封存的results/解析参数/参考审计，以及原m/c0和已保存的PDE、连续时间离散网格状态。NPZ按标准库解析，仅求旧状态的观测量；没有生成新点、图、系数、作用、传播路径或参考状态，没有新时钟/指数评价或PDE求解。N16/N64候选来源只做整文件哈希，没有在本轮解析其质量/初场。

原N32的56图、6条实际/独立路径、原始质量漂移、误差及H指标不改。前轮源门槛文件的already_executed=false作为历史注册保留，本轮执行凭本轮新收据记录，不重写它。旧交付包SHA仍为f58a50adc6c9b71e42351655330e5dc87670e425ce0564d44fab42bc1d213205。

## 正项预算归属

下面比例的分母是原**绝对状态比较指标**，不是实际H误差，也不是可相加的H指标份额。

| 分量 | t=0.025 | t=0.05 |
|---|---:|---:|
| 连续网格到PDE的空间项 | {share(a,'spatial_symbol_error'):.8f}% | {share(b,'spatial_symbol_error'):.8f}% |
| 理想中点到连续网格的时间项 | {share(a,'midpoint_time_error_symbol'):.8f}% | {share(b,'midpoint_time_error_symbol'):.8f}% |
| 实际8/16差 | {share(a,'observed8_16'):.8f}% | {share(b,'observed8_16'):.8f}% |
| 原机器/尾项 | {share(a,'machine_tail'):.9f}% | {share(b,'machine_tail'):.9f}% |
| 输入舍入Duhamel项 | {share(a,'input_rounding_Duhamel'):.12g}% | {share(b,'input_rounding_Duhamel'):.12g}% |

其余完整序列差、模态残差、两倍评价余量和初态桥均保留，详见[CSV](../../runs/reconnect-t3-affine-review/budget-components.csv)及[精确分数账目](../../runs/reconnect-t3-affine-review/results.json)。原每模型总量和模型间顺序累加逐字复算；按分量重组后的精确总和与原浮点累加间的末位残差单列，不伪装为物理误差，也不用作负项抵扣。

## 为什么现有证据不支持只细化时间

设b_s为双方已存空间状态项之和。沿用原形式B_H(b)=[2√M max(1,max|c0|)b+b²]/E0。这里M=1、max(1,max|c0|)=1，未引入更紧公式。对原保存数值方差与精确质量方差取分母包络，仅用于诊断的保守下值；原浮点H指标保持原样。

| t | 原实际H模型差 | 已存连续时间网格H差 | 仅空间项的原形式诊断下值 | 原完整H指标 |
|---|---:|---:|---:|---:|
| 0.025 | {a['source_actual_H_gap']:.12g} | {a['saved_continuous_grid_H_gap']['float']:.12g} | {a['spatial_only_original_formula_diagnostic_lower']['float']:.12g} | {a['source_fixed_H_indicator']:.12g} |
| 0.05 | {b['source_actual_H_gap']:.12g} | {b['saved_continuous_grid_H_gap']['float']:.12g} | {b['spatial_only_original_formula_diagnostic_lower']['float']:.12g} | {b['source_fixed_H_indicator']:.12g} |

t=0.025时，即使理想地去掉时间、数值、输入和评价项，仅空间项的原形式诊断仍比保存的连续时间网格H差高{a['spatial_floor_minus_saved_grid_gap']['float']:.12g}。因此继续只增加K不是当前证据支持的下一项。这不是运行了K→∞，也不是对任意未来舍入序列或新K实际状态的预测；连续时间网格数组已在上一轮保存，本轮仅重新计算其质量H。

空间份额占主导，说明的是**现有保守指标的组成**。t=.025时，两个模型各自已存空间H偏差绝对值之和约{a['sum_absolute_individual_H_spatial_bias']['float']:.12g}，其有符号模型差约{a['signed_model_H_spatial_bias']['float']:.12g}；原范数到H的转换及正项相加较保守。有符号抵消仅作解释，不替代固定正项门槛，不据此把N32改判通过。指标含经验机器项，不能称连续PDE总误差认证界。

![只读预算归属与原H判定](../../runs/reconnect-t3-affine-review/t3-affine-budget.png)

## 复核、失败记录与保全

主复核和收据审计均只依赖Python标准库。原NPY二进制读入后，用精确分数计算既有m/c0及既有网格/PDE状态的观测；独立审计采用另一套array模块解码和80/100位Decimal的原始二阶矩方差公式，与主流程的中心化精确方差不同。两个精度的诊断值一致，空间诊断高于两个旧H差的关系均复核通过。没有把这些来源/算术核对计为新的物理算例、软件检查或代数证书组。

首次主复核失败：错误假定保存标准差的平方与精确方差最多相差1ulp，实际为1.0441499899528233ulp。标准差本身来自浮点方差的开平方，再平方不满足该无依据假设。已保留完整首轮代码、锁、来源副本和日志；最终改为核对已存精确方差及已有认证向上界，不引入新的误差容差。原实验指标、结果、阈值和未决判定未改。

最终主复核{ex['elapsed_main_seconds']:.3f}秒/{ex['peak_RSS_MiB']:.2f}MiB，满足120秒/1024MiB；独立来源/算术审计首次{ar['elapsed_audit_seconds']:.3f}秒/{ar['peak_auditor_RSS_MiB']:.2f}MiB通过。新软件检查0，累计记录仍480，旧检查未重跑，旧v0.8.0/246仍未恢复。

782原文件、15附件及94份既有正式科学清单核验，4份历史诊断既有差异保留。旧原README/METHOD/CODEX_TASK/PHASE、附件、代码、结果和报告不改，只更新RESEARCH_STATUS入口。没有新数值传播重放、PDF、版本、提交推送、环境配置更改或发布。

## 下一项：同一仿射对的三档空间控制，只登记未执行

空间项已经定位，因此[下一项机器方案](../../runs/reconnect-t3-affine-review/next-gate-spec.json)单独登记N16/32/64共同物质观测：复用原N32的6条保存路径，不重跑它们；只补N16和N64，两模型各K4/8/16。原METHOD要求至少三空间档、三时间档；物理模型、时窗及时间档均不新增，这不是流动/厚度/核宽参数矩阵扫描。本轮没有生成该项点、图、初场或参考。

原T1已有各档纯标签、真实质量和初始有限单元平均可复用；旧T1正D参考仍仅作历史，不是压缩目标。新112图、12条实际和12条完整独立路径全部在下一轮完成，所有新系数必须在读取真实m/c0/新参考和旧N32状态前封存；原几何/阻力/时间后端及边界适配层继续不改。精确ρ/D输入舍入缺陷、两物理时钟、80/100位参考检查、原尾项/代数尺度及原H形式全部保留。

共同观测为原N16固定物质单元，子质量组分积分直接相加并除以原父质量；保留初态/参考投影桥，不重配质量或将桥强制归零。固定K16比较每个模型的16→32、32→64原生与共同参考误差/H误差收益，在两个非零时刻均须超过预先保留的时间/输入/数值/参考/桥正项；每档4/8到8/16时间下降须分辨；细档N64模型状态/H差仍须通过原含空间项的模型判别。粗档结果原样报告，不要求粗档都分辨模型差，不事后改变N32结论。

主资源上限登记900秒/1024MiB，依据已有56张N32图约85.285秒按节点数作(.25+4)比例的粗规划，非实测性能或精度保证。三类必要新软件检查仅在下一轮实现后执行，本轮0项。若失败或未决立即保留，不自动N128/K32、新流动/模型或额外参数扫描，不放松H公式或阈值。

这一步即使完成也不自动等于Stage3整体通过：原T3完整方向/剪切证据、最终同一后端T4移动激波/独立接触和三密度来源对照仍须按原标准核对。一般光滑流/有限节点候选不接纳，kappa未计算，fluid ORC关闭。DNS/动态ORC不是新增等待条件。

[报告数据与收据](../../runs/reconnect-t3-affine-review/) · [只读复核包](../../exports/ltn-t3-affine-review.zip) · [交付核验](../../exports/ltn-t3-affine-review-delivery.json)。该包用标准库核对预算及已存H观测，不重放旧数值路径；完整历史保全审计仍需原工作区。

```sh
PYTHONDONTWRITEBYTECODE=1 python -S research/reconnect/audit_t3_affine_budget_review.py \\
  --directory runs/reconnect-t3-affine-review --receipt-only
```
'''
    with (ROOT/'research/reconnect/T3_AFFINE_BUDGET_REVIEW.md').open('x') as f:f.write(text)
    index=ROOT/'docs/RESEARCH_STATUS.md';old=index.read_text();marker='执行上一轮登记的N32、T=.05平面仿射压缩成对模型，'
    assert marker in old and '\n95. ' not in old
    old=old.split(marker)[0].replace('T3_AFFINE_COMPARISON.md)（本轮）。','T3_AFFINE_COMPARISON.md)。')
    old+='''95. [T3仿射压缩：保存误差预算复核与空间对照方案](../research/reconnect/T3_AFFINE_BUDGET_REVIEW.md)（本轮）。

执行上轮只读复核登记项。t=.025的原绝对状态指标中，空间项96.95129%，中点时间项0.76250%，实际8/16差2.28614%，机器/尾项0.0000722%；份额不是实际H误差归因。仅空间项经原H形式的保守诊断约.00506070588758，高于实际模型H差.00497045808719和旧连续时间网格H差.00497245233399。故不再选只增加时间步；原N32结论T3_affine_model_response_inconclusive保持，不替换原H指标或改判。

新点/图/系数/作用/路径/参考状态/PDE/软件检查/代数证书组均0，记录仍480；旧数值路径未重放。精确分数账目和独立80/100位Decimal旧状态H观测复核通过，正项重组的浮点末位残差单列，不用有符号抵消放行。首轮复核错误假设保存标准差平方与精确方差最多差1ulp，实为1.04415ulp；保留代码/锁/来源/日志，最终以旧精确方差及既有向上界核对，原科学指标/结果/阈值不变。

下一项单独登记同一仿射对N16/32/64空间控制，复用N32六路径，只补N16/N64及原K4/8/16，共112图、12实际/12完整独立路径；共同N16固定物质观测，原质量/初值及桥保留，三档空间与三档时间按原METHOD，细档模型响应仍用原H门槛。900秒/1024MiB资源封顶，仅登记未执行；不自动N128/K32或新流动/模型扫描，不改变旧N32未决结论。

782原文件、15附件、94份旧正式清单和旧T3交付保持，4份历史诊断既有差异保留；旧v0.8.0/246未恢复。一般T3/光滑流/有限节点接纳false，Stage3 incomplete，fluid ORC关闭，kappa未计算。报告：research/reconnect/T3_AFFINE_BUDGET_REVIEW.md；入口：runs/reconnect-t3-affine-review/stage3-status.json；下载：exports/ltn-t3-affine-review-downloads.md。原README/METHOD/CODEX_TASK/PHASE、附件及旧代码/报告/结果不改，仅更新本入口。无PDF、版本、提交推送、环境配置更改或发布；CRC/SHA和全新解压标准库收据以交付JSON为准。
'''
    index.write_text(old)


if __name__=='__main__':main()
