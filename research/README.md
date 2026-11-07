# 当前研究代码入口

先读 [项目总览](../docs/PROJECT_OVERVIEW.md) 和 [代码指南](../docs/CODE_GUIDE.md)。

- `material_fv/`：真实物质面的一维普通中心阻力基线。
- `rebuild/`：独立 Becker/接触重建、通量和几何对照。
- `reconnect/`：图层接口、点轨迹交换候选、密度适配和保存数据核验。

统一入口为 `python tools/ltn_project.py list`（在仓库根目录运行）。原目录和数值源码保持字节不变；历史报告里的“下一步”不自动覆盖 [当前计划](../docs/NEXT_STEPS.md)。

Stage3 incomplete，fluid ORC 关闭。当前交付不是恢复的 ltnmix 0.8.0。
