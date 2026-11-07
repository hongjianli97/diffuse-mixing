# 代码导航与复现说明

2026-11-07。当前交付是研究代码整理，不冒用丢失的 ltnmix 0.8.0 版本。原算法文件和目录保持原样；新增统一入口负责导航、校验和有限的软件检查。

## 目录职责

| 路径 | 内容 | 使用方式 |
|---|---|---|
| `docs/PROJECT_OVERVIEW.md` | 理论—代码—证据总览 | 首先阅读 |
| `docs/NEXT_STEPS.md` | 当前工作计划 | 新实验尚未执行 |
| `research/material_fv/` | 真物质面的一维 FV 基线 | 真实移动面是额外基准输入 |
| `research/rebuild/` | Becker/接触重建与离散候选 | 各研究报告限定适用范围 |
| `research/reconnect/` | 图层接口、点轨迹候选、参考与审计 | 不将全部脚本当一个成熟生产后端 |
| `reference_baseline/`、`prototypes/`、`legacy_sources/` | 782 个原文件所在的历史材料 | 保留字节，不整体导入旧运行脚本 |
| `runs/` | 各轮保存证据 | 本次不新增上传；GitHub 只保留此前已有部分，源码 ZIP 不包含该目录 |
| `tools/ltn_project.py` | 本次新增操作入口 | 小图检查、FV 检查、包完整性验证 |
| `tools/build_project_report.py` | 总报告生成器 | 合并总览、理论附录、代码指南及下一步计划 |
| `reports/project-20261107/` | 排版源、图、构建及检查记录 | 是整理交付，不是新的科学算例 |
| `exports/` | PDF、代码 ZIP 与交付清单 | 下载及哈希核验 |

没有搬迁或批量改名旧 Python 文件。研究脚本之间存在相对导入与封存源码哈希，目录整理通过导航和统一入口实现，避免改变既有数值行为。

## 安装与快速检查

建议 Python 3.12。固定依赖见 `backup/requirements-replay.txt`；它是已使用的研究环境版本清单，不代表旧 v080 环境恢复。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r backup/requirements-replay.txt
.venv/bin/python tools/ltn_project.py list
.venv/bin/python tools/ltn_project.py verify
.venv/bin/python tools/ltn_project.py checks --output /tmp/ltn-checks-new
```

输出目录必须不存在。`checks` 只运行两个已有软件检查组：小图链 10 项、一维 FV 5 项。它们验证导入、算子、时间顺序、守恒和受限解析控制，不运行 Becker 或大规模流体实验，不启用 ORC。既有 508 项记录不因重跑这 15 项而增加。

`verify` 优先使用代码包内的 `delivery-manifest.json`；完整 Git 检出则使用 `backup/github-backup-manifest.json`。同时校验 `backup/original-handoff-sha256.json` 的全部 782 个原文件。代码包的校验范围与 Git 精选恢复范围各自明确，不混称全量恢复。

## 核心接口及边界

| 对象 | 文件/函数 | 注意点 |
|---|---|---|
| 轨迹、固定质量与图层传播 | `trajectory_chain.py`：`TrajectorySamples`、`propagate_supplied` | 共享电导由调用者提供；位置本身不证明电导物理正确 |
| 真实物质面基线 | `material_fv.py`：`center_resistance`、`step`、`propagate` | 单位横截面、一维、无通量端点、普通几何中点 |
| 三种通量 | `rebuild/experiment.py`：`conductance` | 普通几何中心、质量中心、质量平均阻力是不同定义 |
| 辅助面 | `periodic_voronoi_fv.py`：`build_periodic_faces` | 不把 Voronoi 面积当成已知物质体积，不重算质量 |
| 时间作用 | `voronoi_time_action.py`：`operators`、`exponential`、`uniformization` | 指数作用与独立递推；保持时间有序，不平均整个时间窗 |
| 混合统计 | 同上：`metrics`、`dissipation` | `variance` 为未除总质量的平方和；H 使用比值，绝对耗散须核对因子 |
| 密度来源 | `t1_density_sources.py`、`compression_density_sources.py` | 原 kernel 和固定邻居环 ledger 均保持原定义；无 F 拟合、裁剪或离散重归一化 |

## 如何复核一个旧结果

先读该案例的报告、配置、源码锁及哈希，再运行对应只读审计。不要直接执行名称含 `study`、`deliver`、`package` 的脚本来“快速测试”：有的依赖未打包的历史数据，有的会进行长计算或生成交付文件。

例如完整本地工作区包含所需来源时，可按报告中的命令核验 `runs/reconnect-density-error-structure/`。本次 GitHub 更新没有上传这组新数据，代码 ZIP 也不含 `runs/`，所以不能声称只靠小包或当前远端就能重算所有保存物理结果。缺失输入应明确报告，不能伪造或更换。

## 重建 PDF

需要 Pandoc、XeLaTeX、Noto CJK 字体和常用 LaTeX 包，数值依赖不参与排版。

```bash
python tools/build_project_report.py
```

图表使用随报告保存的既有结果图；生成器不运行物理求解器。排版源和 PDF 的文件哈希记录在交付清单。原始 README、METHOD、CODEX_TASK 及 PHASE 文档不为新交付而改写。
