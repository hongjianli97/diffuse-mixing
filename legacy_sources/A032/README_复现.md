# 阶段 0 · 结论登记表与本地复现

本包四样东西：

| 文件 | 作用 |
|---|---|
| `结论登记表.xlsx` | 140 条数值结论的登记表（6 个工作表：说明、登记表、更正与冲突、复现清单、待验证、文档与脚本）。“登记值”列由脚本直接从沙盒产出的 `res_*.json` / `tables_auto.tex` 抽取，没有手工转录。 |
| `tools/fix_paths.py` | 把代码包里 20 处硬编码的沙盒路径（`/home/claude/bench/`、`/home/claude/pkg`、`/tmp/work/`）改成相对路径。 |
| `tools/run_bench.sh` | 按依赖顺序跑完 bench，写日志、计时、产物已存在就跳过。 |
| `tools/check_register.py` | 把本机重跑得到的数字写回登记表并逐行判定（通过 / 不一致）。 |
| `patches/` | 阶段 0 修正：4 个新脚本（`ablation_complex_dg.py`、`struct_stats.py`、`wp3_common.py`、`stretch_extra.py`）、3 个改过的脚本（`run_sb.py`、`figs.py`、`figs2.py`，另附 `.diff`）、6 个出图脚本（`fig_register.py`、`fig_d06.py`、`fig_o04.py`、`fig_v18.py`、`fig_s16.py`、`fig_t03.py`）和共用样式 `figstyle.py`。全部复制进 `bench/` 覆盖即可；`results/` 里是它们在沙盒里的产出。 |
| `修正说明.pdf` / `.md`、`figs/` | 五个修正项各一张图加登记表概览（图 0–5），以及按 V-18 修改后重出的报告图 `fig_sb_window.png`。 |

## 1. 目录布局

把三个代码包解开，放成下面的样子（名字可改，但 `bench/` 与上一级的关系要保持）：

```
work/
├── bench/                     ← LHD_validation_code.zip 里的 bench/ 全部内容（含 res_*.json 原值）
│   ├── patches/ 里的 7 个脚本覆盖进来，results/ 里的 5 个结果文件也放进来（tables_extra.tex 放 report/）
│   ├── stretch_est.py evalcore.py cases_mix.py dns_markers.py run_all.py make_tables.py figs_new.py diag.py
│   │                          ← 从 界面拉伸估计_源文件与代码.zip 的 code/ 复制进来（它们 import bench 里的 cases/lhd/lhdx/euler2d）
│   ├── fix_paths.py run_bench.sh check_register.py      ← 本包 tools/
│   └── 结论登记表.xlsx
├── lhd2.py  lhd_window.py  smoke_window.py             ← LHD_validation_code.zip 顶层三个文件
├── B1_classic_cases/                                    ← B1_classic_cases_scripts.zip
└── lagorc/  (lagorc.py, compare.py)                     ← 动态 Ollivier（只有库，没有实验脚本）
```

建议先把原始 `res_*.json` 另存一份（`cp -r bench bench_sandbox_原值`），重跑会覆盖它们；登记表里已经保存了全部原值，但留一份方便 diff。

## 2. 环境

Python ≥ 3.10；`numpy scipy matplotlib scikit-image openpyxl`（scikit-image 只有 `paper_ell.py` 用，openpyxl 只有比对脚本用）。
沙盒版本：numpy 2.4.4、scipy 1.17.1、matplotlib 3.10.8、scikit-image 0.26.0。版本不同可能让 1e-3 量级的数字变化，所以登记表给每行都设了容差。
所有随机数都用 `np.random.default_rng(seed)` 固定了种子；多线程 BLAS 可能改变最后几位，比对不一致时先试 `OMP_NUM_THREADS=1`。

## 3. 步骤

```bash
cd work/bench
python fix_paths.py . . ..          # ① 改路径（幂等；原文件备份为 *.py.orig）
bash run_bench.sh data core         # ② 先跑最短的两组（约 25 min），看环境是否正常
python check_register.py 结论登记表.xlsx --bench . --stretch ./report   # ③ 立刻比对一次
bash run_bench.sh afrc unified dirac transport evo stretch                # ④ 其余各组（约 1.5 h）
python check_register.py 结论登记表.xlsx --bench . --stretch ./report   # ⑤ 全量比对
```

`run_bench.sh` 的阶段与耗时（单核估计，来自 README 与脚本内记录）：

| 阶段 | 内容 | 耗时 |
|---|---|---|
| data | euler2d（激波–气泡 DNS）、sv_case（激波–涡）、sb_diff_dns（带扩散标量） | ~11 min |
| core | 双环流、Sod、爆炸波、激波–气泡、激波–涡的核心验证 | ~15 min |
| afrc | AFRC_E/H 复查、穹顶、论文图 3 复现 | ~20 min |
| unified | 统一网络 | ~5 min |
| dirac | Dirac 算子、Kelvin 链、WP3 | ~10 min |
| transport / evo | 质量输运、扩散、演化、离线追踪理论 | ~25 min |
| stretch | 带标记曲线的 DNS 重跑、七组估计、制表、diag 打印 | ~30 min |
| figs（可选） | 出图 | ~5 min |

另外三项不在 `run_bench.sh` 里，手动跑：

```bash
# B1 §13–14 算例（约 20 min）
cd work/B1_classic_cases && python run_cases.py R B D S Rn Bn && python afrc_tri_input.py R B D S && python dg_absrel.py && python run_h.py B D S
# Part-1 驱动冒烟（约 1 min；预期 RH 检验中位 1.000，末帧密度误差 0.66%）
cd work && python smoke_window.py
# 修订版自测（可选，旧算子，约 3.5 min）：files(7) 包里的 lhd_selftest.py
```

出图（需要 `fonts-noto-cjk`；`fig_v18.py` 依赖 `run_sb.py` 与 `run_sb_phys.py` 的产出，`fig_register.py` 依赖上一步生成的 `结论登记表_复现.xlsx`）：

```bash
for f in fig_d06 fig_o04 fig_v18 fig_s16 fig_t03; do python $f.py; done
python fig_register.py 结论登记表_复现.xlsx          # 图 0，不计动态 Ollivier
python run_sb_afrc.py && python figs2.py merge_sb sb  # 重出报告里的激波–气泡图（含 fig_sb_window.png）
```

图都写到 `figs_stage0/`。

## 4. 怎么读比对结果

- `check_register.py` 把本机值写进“本地复现值”（G 列），单值行由 H 列公式按“相对容差 / 绝对容差”判定，多值行看 S 列“脚本判定”。
- 终端同时打印汇总和所有“不一致”“未找到”的行。
- “无自动比对”的行（42 条）是没有数值定位的结论：描述性结论、只在对话记录里的数字、脚本缺失的结论。它们要么人工对照终端打印（`diag.py`、B1、冒烟），要么等脚本补齐。
- 一行“不一致”时按顺序排查：BLAS 线程数 → numpy/scipy 版本 → 脚本是否被改过（`*.py.orig` 可 diff）→ 产物是否来自旧版脚本（比如 `res_wp3.json` 与 `wp3_core.py` 就不是一套）。确认后把登记值更新为本机值，并在备注写明原因。

在当前沙盒里已做的自检：重跑了 `run_1d.py sod/blast`、`run_dg.py + run_dg2.py + run_dg_phys.py`、`sv_case.py + offline_theory.py + run_sv.py`、`euler2d.py + run_sb.py`、`dns_markers.py sb/sv`，再加 patches 里的四个新脚本，101 条可比对行全部通过（登记表的登记值已换成这次重跑的产出，与随附 json 相比除 O-04 外没有超过容差的差异）。

## 5. 复现不了的部分（登记表里标为“脚本缺失”）

- B1 v4 的恒等式、噪声、激波、源驱动势脚本（`check_identities.py`、`exp_*.py`）——只上传了 §13/§14 的算例包。
- 动态 Ollivier 曲率报告的 7 组实验脚本——只有 `lagorc.py` 库和 `compare.py`。
- `res_fig3_stage.json` 的生成脚本（A-02）。
（修订版 §2 的 struct 统计和 WP3 表 7 已由 `struct_stats.py`、`wp3_common.py` 解决。）

这些数字在拿到或重写脚本之前不进论文。

## 6. 登记表的维护规则

1. 新结论先登记再写进报告；改一条结论时不覆盖旧行，旧行状态改“已更正”，关联列指向新行。
2. 只有参照类型为 精确解 / 独立求解器 / 独立参照 / 代数恒等式 的结论能在论文里写成定量验证；DNS同源 的要说明参照与方法共用同一 DNS。
3. “定位”列是结论与数字之间唯一的锚；新脚本一律把数字写进 `res_*.json`，不要只打印。
