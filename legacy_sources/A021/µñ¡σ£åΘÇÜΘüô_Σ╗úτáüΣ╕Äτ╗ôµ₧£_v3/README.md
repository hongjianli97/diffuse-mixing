# 椭圆通道 第 3 版：参数、强度场与定义的适定性

本包含有 LHD 验证代码包 `bench/` 的全部脚本（含第 1–3 版改动），以及本轮新增的脚本。

## 本轮新增
- `sweep.py`、`sweep_sum.py`：平滑核宽度、平滑规则与 δ 扫描（9 个参照涡 × 3 个噪声种子）
- `neff.py`：单步占比与参与比
- `sync_lib.py`、`sync_main.py`、`sync2.py`：频率簇层级（合并树）强度场与集合检验
- `sheet.py`：涡片上逐点转角与环量的适定性检验
- `circ_strength.py`：相干环路上的环量强度及客观性检验
- `afe_tables3.py`、`afe_tables4.py`：报告中的表

## 代码改动
- `tcx.py` 的 `[rev3]` 段：`n_smooth` 的尺寸上限改为 w(n) = 0.55·√n·h ≤ r_eq/6
- `tcx_run.py`：J_max 改为 0.35

## 运行
`bash run_all_rev3.sh`（六算例，单核约 15 分钟）；其余脚本见报告末尾“复现”一节。
