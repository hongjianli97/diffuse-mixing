# diffuse-mixing / LTN：报告、代码与下一步入口

2026-11-07。按用户要求交付完整 PDF、整理研究代码、更新 GitHub 并制定下一步计划。科学状态仍为 **Stage3 incomplete，fluid ORC 关闭，κ 未计算**；累计 508 项检查记录。本次只重跑 15 项已有软件检查，不新增物理实验。

## 立即阅读与下载

- [当前入口 START_HERE](START_HERE.md)
- [完整 PDF](exports/LTN_project_complete_20261107.pdf)
- [小型代码 ZIP](exports/LTN_project_code_20261107.zip)
- [交付哈希与状态](exports/LTN_project_delivery_20261107.json)
- [项目总览](docs/PROJECT_OVERVIEW.md)、[代码指南](docs/CODE_GUIDE.md)、[下一步计划](docs/NEXT_STEPS.md)

PDF 包含理论主线、已有推导、验证与未决、保存数据选表、代码使用和后续安排。原 README/METHOD/CODEX_TASK/PHASE、782 原文件、历史算法和结果保持不变；新增操作入口在 tools/，没有把原代码整体改名或改成一个已验收的新版本。

## 当前科学主线

固定物质质量平均 → 当前局部物理交换 → 时间有序传播 → 实际掺混及经物理验收后解释 ORC。真物质面的一维 FV 基线与从点轨迹构造的辅助面候选各有范围，不能把一个的验证直接算到另一个上。

已有 oracle T1–T3、静态接触三档和 Becker 共同 K32 空间证据保留。三密度来源的瞬态空间/接触/激波覆盖，以及层厚度与部分 H 时间控制仍有缺口。ledger 必须胜过 oracle、无限后移 H 时间档、DNS、T5/T6 或旧 v080 恢复都没有新增为当前解析验收条件。

下一项拟做一个平滑压缩流的 ledger 瞬态空间对照，复用原 N64/128/256 资料；先在同一任务中核对最小缺项，再按固定预算执行或明确停止。细节及后续接触/激波顺序见计划。本次仅规划，没有运行。旧固定 h=1/32 卷积提案继续暂停，旧 `next-gate-spec.json` 不自动恢复执行。

## 下载与恢复边界

代码 ZIP 包含当前研究源码、原始代码材料、文档和排版源，含 782 原文件的校验清单。它不含 `runs/` 结果、15 附件、Git 历史、聊天或未取回的 v080。解包后运行 `python tools/ltn_project.py verify`，再按代码指南运行有限检查。

GitHub 本次只新增报告、代码和说明，保留远端此前已有数据，不上传新运行数据。实际文件由 `backup/github-backup-manifest.json` 覆盖，范围见 `backup/BACKUP_SCOPE.json`。本地完整历史与 GitHub 发布检出分别保存，两处清单因范围不同而不同。初次包含约 1.2 GB 新对象及结果的大批量推送被自动审批拒绝，原因是缺少明确公开这些数据的授权；当前缩小后的范围不包含这些新增数据。

本次 GitHub 更新已由用户明确授权，不需要再次询问。实际远端提交、下载链接和传输核验以交付回执为准；环境发布与 GitHub 提交不是同一操作，本次不发布云环境。

## 新聊天交接文字

> 请接手 diffuse-mixing，先读 START_HERE.md、docs/PROJECT_OVERVIEW.md、docs/CODE_GUIDE.md 和 docs/NEXT_STEPS.md。按当前检出的清单核验恢复范围。当前报告与代码交付已获 GitHub 更新授权；具体远端状态查交付回执。新物理实验尚未执行，下一项拟做一个平滑压缩流的 ledger 瞬态空间对照，禁止从历史 next-gate-spec 自动续跑。Stage3 incomplete，fluid ORC OFF，κ 未算；508 是累计检查记录，此次重跑的 15 项不重复计数。保留原物质质量平均、普通阻力基线、全部误差预算和失败/未决。原 782 文件与附件不改；v080 未恢复，源码 ZIP 和 GitHub 精选检出均不是完整工作区。
