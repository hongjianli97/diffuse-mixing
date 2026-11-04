"""Additive source-linked coverage report; no scientific recomputation."""
from pathlib import Path
import shutil
from audit_route_decision import ROOT,read,save


def main():
    out=ROOT/'runs/reconnect-t3-coverage';r=read(out/'coverage-review.json');ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'review-audit.json')
    assert audit['receipt_consistent'] and audit['prior_formal_manifests_verified']==96
    assert not r['T3_affine_shear_executed'] and not r['general_T3_admitted']
    shutil.copyfile('/tmp/ltn-t3-coverage-final.log',out/'review-final.log');shutil.copyfile('/tmp/ltn-t3-coverage-audit.log',out/'audit.log')
    quote_lines='\n'.join(f"- {q['id']}：[{q['path']} 第{q['line']}行](../../{q['path']})：`{q['text']}`" for q in r['original_requirement_quotes'])
    report=f'''# 同一普通FV研究后端：T1–T4与T3剪切覆盖复核

2026-11-02（Asia/Shanghai）。执行[上一轮只读登记项](../../runs/reconnect-t3-affine-spatial/next-gate-spec.json)，[本轮协议](t3_coverage_protocol.json)约束只读来源/代码/保存结论；附件和旧文档作为资料，不自动成为新的用户指令。

**决定：保留已通过的登记基准，下一项只补一个仿射剪切的滑移周期域控制。** 已有正弦剪切确实属于同一普通FV实现链，但它不是仿射映射。原METHOD表T3中的仿射剪切在本次核对的注册证据链中没有执行记录，当前正交周期接口也不能直接表达它。T3完整覆盖尚不能宣告，Stage3 incomplete，fluid ORC关闭。

## 已有证据与实际缺口

| 原要求/项目 | 当前可记入的证据 | 适用范围与剩余项 |
|---|---|---|
| T1静止均匀正弦 | N16/32/64、K4/8/16登记基准通过，原生/共同浓度和H控制保留 | 研究指数作用，不冒称生产默认隐式Euler路径已交付 |
| T2平移、D0、刚体旋转 | 三个登记控制通过；D0逐位恒等、精确旋转边界协变分别留档 | 限声明的合法输入/边界模式；未运行fluid κ，不据此接纳任意几何 |
| T3仿射压缩与正交几何 | 原m/c0下N16/32/64三档、两模型、三时间档对照通过 | 不重跑；N32原中间时刻H未决保留，细档通过不追认粗档 |
| T3方向与恒D/恒Γ | gx=Γ/a、gy=Γa逐面核验；N64模型状态/H差超过原指标 | 正交方向与本构区别已有证据；不再要求任意角度或新增方向矩阵扫描 |
| 同一后端的光滑剪切 | 周期正弦剪切N64/128/256原生/共同浓度、H收益可分辨 | 非仿射；N128/256有4/8/16，N64为必要8/16；保留为正面证据，不更名或重跑 |
| 原T3仿射剪切 | 缺少注册执行；现接口只接矩形/正交旋转矩形 | 给定滑移晶格的world欧氏几何是具体缺口，下一项处理 |
| T4 Becker | 固定δ=.08、D=.02、T=.1的真实物质FV三档空间/时间/参考控制存在 | 是1D物质面/中心和移动端点基线，不等于最终点几何/边界/传播链的T4闭合 |
| T4独立接触 | 静止对齐Cartesian有限单元参考、普通阻力及N128/256对照存在 | 在对齐几何下与普通面阻力条件相容；尚非最终点构造全链三档T4记录 |
| 三密度来源 | oracle隔离与原密度/界面理论保留 | 同一最终后端、m/c0/D/边界下oracle、重构、ledger实验来源尚未联合验收 |

该表不是完成百分比，也不把软件检查数当物理通过数。完整逐项来源见[验收账本JSON](../../runs/reconnect-t3-coverage/acceptance-ledger.json)与[CSV](../../runs/reconnect-t3-coverage/acceptance-ledger.csv)。

## 同一实现链与时间格式声明

逐项核对七份不可变implementation-lock：正弦剪切、光滑压缩、T1、T2平移、T2 D0、T2精确旋转和最新T3空间对照。三份核心源码periodic_voronoi_fv.py（包含普通面阻力）、voronoi_time_action.py和voronoi_time16_core.py，在七份锁中的SHA均与当前文件一致。因此不能把已有正面剪切证据当作一个无关或已经被替换的后端；也不能用不同真实几何先验的旧FV基线直接补齐当前完整输入管线。

当前研究实验明确使用中点冻结指数作用和完整独立统一化；原METHOD的生产默认是隐式Euler。两者是不同的时间离散声明，当前科学证据不等于原默认CLI/schema/版本迁移已交付。这里记录实施范围，不把包装或恢复旧v0.8.0变成运行下一解析基准的新前提。

PHASE3_REBUILD/CURRENT保留2026-10-31时点状态，不能按其中“尚无新辅助面作用”等旧文字否认后续已封存的曲线；也不能据旧材料中同名T3标签继承METHOD T3的通过。最新来源链与原始定义分别列出。

## 为什么不能把现有剪切算作仿射剪切

原剪切映射源码是

`x=a+s sin(2πb)/(2π)+.2t, y=b−.1t`。

它的剪切量随b变化，是合法的单位周期域光滑剪切，已有混合导数/密度漂移及实际传播证据；但它不等于`x=a+s b, y=b`。前者的成功不能作为后者的执行记录。原T3要求的对象来自METHOD，未新增“所有方向/所有剪切”的验收项。

仿射映射要求周期晶格同步成为B(t)Z²，B=[[1,s],[0,1]]。当前矩形接口只接两个周期长度，旋转适配层又明确检查EᵀE=I，因此不能把这个给定基直接传入。逆剪切用于搜索索引可以，但若随后使用普通label欧氏距离构图，会漏掉材料PDE的混合导数和度量项，改变物理问题。

因此本轮新增[书面滑移周期域契约](AFFINE_SHEAR_DOMAIN_CONTRACT.md)：保持world欧氏双中垂线及普通面阻力，另设可选精确滑移基模式；s0调用原矩形分支，所有旧代码字节不动。B是用户算例给定的全局边界几何，不是局部F/C拟合。此模式只声明0≤s≤1/4，不扩大成通用非正交域要求或能力声明。

契约给出本均匀Bravais族的三组邻接、普通阻力、二阶矩身份、远镜像排除条件和两个相反倾斜模态的独立材料参考。全部是书面条件推导/设计；本轮没有实现几何、计算新g或执行机器代数证书。解析系数只能在下一轮构图封存后用来核对，不能给构造器替代欧氏面计算。

## 选定的唯一下一项

[机器方案](../../runs/reconnect-t3-coverage/next-gate-spec.json)仅登记，尚未执行：

- 新文件实现给定滑移晶格的点定义欧氏Voronoi几何及独立world审计，旧矩形、正交适配和时间后端不改。
- 固定N32、T=.05、s(t)=5t、ρ1、D=.02，复用原质量/ID和原c0；ρ1下恒D/恒Γ为同一模型，不重复两套等价计算。
- 只做K4/8/16，共28张新图、3条实际及3条完整独立路径；全部图/系数先封存，再读取实际m/c0、参考或旧状态。
- 参考含原sin(x)cos(y)的两个相反倾斜模态，保留原乘积初场与解析分解的桥，不替换或投影实际c0。记录原始线性模态观测、全状态和实际H，判别混合导数的方向响应。
- 原尾项1e−16、代数1e−10、种子20261101、H公式和固定正项预算保持；900秒/1024MiB封顶，四类必要软件检查只在下一轮实现后执行。本轮新检查0，记录仍483。

这是进入新边界/方向模式的一次判别例，不宣布新空间收敛曲线。实现/资源/物理比较失败或未分辨则保留停止，不自动扩N/K/角度/流动/模型，不改变边界、初场、物理D或H门槛制造通过。

后续必须完成的仍是原标准中的最终同一后端T4和三密度来源闭合，不能用本次覆盖审查或未来单例代替。T5/T6、DNS及动态ORC不新增为当前Stage3等待条件；D0/等距下的κ条件关系与未运行fluid κ区分。Stage3未通过前fluid ORC保持关闭。

## 来源核验、失败记录与保全

首轮来源核对错误地要求每个历史文件都有父累计清单的直接内容条目；PHASE3_REBUILD实际由上一轮不可变next-gate源SHA绑定。首轮代码、锁、日志和已复制来源原样保留；最终将这唯一历史文件的来源链记为“父清单→已封存门槛→该文档SHA”，其余44份仍是直接内容条目。没有修改文档、绕过SHA或放宽科学阈值。

最终主复核{ex['elapsed_main_seconds']:.3f}秒/{ex['peak_RSS_MiB']:.2f}MiB，满足120秒/1024MiB；独立来源/结构审计首次{ar['elapsed_audit_seconds']:.3f}秒/{ar['peak_auditor_RSS_MiB']:.2f}MiB通过。两者只依赖Python标准库，不导入/运行数值后端，旧传播不重放。

782原文件、15附件、96份先前正式科学清单及上一轮三档T3交付哈希核验，4份历史诊断既有差异保留。新点、图、系数、作用、路径、参考状态、PDE、软件检查、机器代数证书组和LP均0；记录仍483，旧v0.8.0/246未恢复。原README/METHOD/CODEX_TASK/PHASE、附件及旧代码/数据/报告不改，仅更新RESEARCH_STATUS入口。无PDF、版本、提交推送、环境配置更改或发布。

## 原要求逐行来源

{quote_lines}

[覆盖数据、账本与收据](../../runs/reconnect-t3-coverage/) · [只读复核包](../../exports/ltn-t3-coverage.zip) · [交付核验](../../exports/ltn-t3-coverage-delivery.json)。全新解压后只用标准库核对来源和覆盖收据，不重放任何数值路径；完整历史保全审计仍需原工作区。

```sh
PYTHONDONTWRITEBYTECODE=1 python -S research/reconnect/audit_t3_coverage.py \\
  --directory runs/reconnect-t3-coverage --receipt-only
```
'''
    with (ROOT/'research/reconnect/T3_COVERAGE_REVIEW.md').open('x') as f:f.write(report)
    index=ROOT/'docs/RESEARCH_STATUS.md';text=index.read_text();marker='执行上一轮N16/32/64空间登记项，'
    assert marker in text and '\n97. ' not in text
    text=text.split(marker)[0].replace('T3_AFFINE_SPATIAL_COMPARISON.md)（本轮）。','T3_AFFINE_SPATIAL_COMPARISON.md)。')
    text+='''97. [同一普通FV后端：T1–T4与T3剪切覆盖复核](../research/reconnect/T3_COVERAGE_REVIEW.md)，配套[仿射剪切滑移周期域书面契约](../research/reconnect/AFFINE_SHEAR_DOMAIN_CONTRACT.md)（本轮）。

执行上一轮只读覆盖复核。七份运行锁中的原几何/普通阻力/时间三核心SHA一致，T1与T2平移/D0/精确旋转、T3仿射压缩三档及周期正弦剪切三档证据均保留，不重跑。原N32 H未决不追认。现有剪切映射含sin(2πb)，是非仿射周期剪切；按原METHOD表T3，仿射剪切在该注册链中仍缺执行。原矩形/正交旋转周期接口不能直接表达给定滑移晶格；逆剪切后沿用label欧氏距离会改变PDE，不采用该捷径。

已有正交方向gx/gy和恒D/恒Gamma区别记为已覆盖的登记证据，不新增任意方向/角度扫描。T4 Becker/独立接触物质FV基线有用但并非最终点几何全链验收；同一后端oracle/重构/ledger三来源对照仍缺。研究中点指数传播与原生产默认隐式Euler/CLI/schema交付区分，不新增恢复v080、DNS或动态ORC等待条件。

下一项仅登记可选精确滑移周期world欧氏Voronoi模式及一个N32/T=.05/s=5t/rho1/D=.02的K4/8/16仿射剪切判别：28图、3实际/3完整独立路径，原m/c0不改，旧后端字节不改，先封存图再读标量；两个相反倾斜模态参考和原始观测检验混合导数方向响应，保留初态桥/原H/正项控制。四类必要新软件检查只在下一轮执行，900秒/1024MiB；失败/未决保留，不自动扩矩阵。该几何实现和算例本轮均未执行，书面公式不是机器核验证书。

本轮新点/图/作用/路径/参考/PDE/软件检查/代数证书/LP均0，记录仍483。首轮误将一份历史PHASE的间接门槛SHA绑定当作缺直接内容条目，保留代码/锁/日志；最终核对唯一间接链，未改源文档或科学指标。782原文件、15附件、96份旧正式清单和旧交付保持，4份历史诊断既有差异保留。Stage3 incomplete，一般T3/光滑流/有限节点接纳false，fluid ORC关闭，kappa未计算。原README/METHOD/CODEX_TASK/PHASE、附件及旧代码/数据/报告不改，仅更新本入口。报告：research/reconnect/T3_COVERAGE_REVIEW.md；入口：runs/reconnect-t3-coverage/stage3-status.json；下载：exports/ltn-t3-coverage-downloads.md。无PDF、版本、提交推送、环境配置更改或发布，交付CRC/SHA和全新解压标准库收据以交付JSON为准。
'''
    index.write_text(text)


if __name__=='__main__':main()
