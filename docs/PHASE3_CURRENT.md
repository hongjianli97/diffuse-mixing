# 第三阶段重建：当前证据索引

更新：2026-10-31。该文件是新的当前索引，旧 PHASE3_REBUILD.md 和各轮报告保留其当时状态，不覆盖历史哈希。

当前已经完成：固定物质质量平均和 Becker 剖面重建；单物理算例的三通量对照；共同状态通量诊断；全过程离散误差预算；同一 Becker 问题 32/64/128 物质列的空间细化及必要时间/参考分离。

| 证据 | 文档 | 数据 |
|---|---|---|
| 物理定义、基线及初次判别 | [重建报告](../research/rebuild/REPORT.md) | runs/rebuild-becker-final/ |
| 为什么恒通量精确不保证瞬态优势 | [通量诊断](../research/rebuild/FLUX_DIAGNOSIS.md) | runs/rebuild-flux-diagnosis/ |
| 初始、空间、中心和时间误差的有序传播 | [全过程预算](../research/rebuild/TIME_ERROR_BUDGET.md) | runs/rebuild-stage3-budget/ |
| 三档空间细化与不确定度 | [本轮报告](../research/rebuild/SPATIAL_CONVERGENCE.md) | runs/rebuild-stage3-space/ |

本轮三种方案误差均下降，64→128 列观察阶约 1.93～1.95；预设经验误差分离条件全部满足。但最细网格的方法间差距接近合并经验不确定度，不能宣布普适优劣。普通中心阻力仍为基线，质量平均阻力仍为实验对照。

第三阶段全面物理验收保持 incomplete；fluid ORC 保持关闭。当前只有一个固定的光滑 Becker 物理问题的这部分证据。旧 v0.8.0、核结果和 246 项测试仍未恢复；密度重构/ledger、独立接触间断、二维非平面和 DNS 验收尚未完成。

重建测试按轮次记录为 15+5+6+4 项；不代表旧测试恢复，也不代表本轮重跑全部测试。D=0 的旧 experiment.candidate 相对求积诊断存在未定义 0/0，已在本轮报告说明；本轮 D=.02 空间研究不受影响。

下一项宜定义并验证独立接触间断的物质平均和标量共享通量约定；不先扩大物理参数矩阵、不启用 ORC。本轮没有新版本、PDF、Git 提交/推送、环境配置修改或发布。
