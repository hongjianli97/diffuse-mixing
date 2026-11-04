# LHD：质量度量下的拉格朗日 Hodge–Dirac 计算（代码包）

只用示踪点位置（加一个参考时刻的密度）计算可压缩流中的物质账本、环量、变密度扩散、相干集与掺混归因。

## 目录
- `lhd/` 核心模块
  - `lhdp.py`：三点映射 (p,q)、复位移上链（环量 + 面积变化）、位置形式 Kelvin、Kabsch 对齐
  - `lhdx.py`, `lhd.py`：剥离 Delaunay 复形、物质账本（`chained_lnJ_v2`）、Galerkin Hodge、DTFE/SPH 密度
  - `bcom.py`：账本变体（抗噪、自适应锚定 `chained_lnJ_adapt`、窗口化断轨）、SPH 传导、误差统计
  - `stream.py`：断轨下的逐轨账本（`tracks_from_particles`, `ledger_stream`, `ledger_windowed`）
  - `netdiff.py`：质量 / 体积度量的网络对流—扩散
  - `stretch_est.py`：链式张量与 Nanson 余向量的界面拉伸估计
  - `classify.py`, `afrc_h_engine.py`：AFRC_H 与脊分类（后续工作）
  - `cases.py`：双环流、Lamb–Oseen、Sod、柱面爆炸波（解析 / 精确）
  - `euler2d.py`, `euler2d_visc.py`, `sv_case.py`：二维 Euler / Navier–Stokes DNS（激波–重气泡、激波–等熵涡）
- `scripts/`：报告中每一项结果的脚本（`n*` 理论与经典算例，`b*` 基线 / 敏感性 / 实际数据，`run_*`/`dns_ref.py` DNS 与加密，`g*` 缺口补齐，`s*` 断轨，`r1_rm.py` RM）
- `results/`：全部数值结果 `res_*.json`
- `figures/`：各算例的场图（`viz_*.pdf`，中文；`viz_en_*.pdf`，英文，脚本 `scripts/viz_fields.py`）与论文插图
- `LHD_demo.ipynb`：可运行演示（已含输出）；`build_notebook.py` 为其生成与执行脚本（无需 Jupyter）

## 运行
Python 3.10+，numpy、scipy、matplotlib。在本目录下：
```
python build_notebook.py                 # 重新执行 notebook（约 2 分钟）
python scripts/run_sb_any.py 0.005       # 激波–重气泡 DNS（单核约 3 分钟）
python scripts/run_sb_visc.py 0.005      # 粘性激波–重气泡
python scripts/run_rm.py 0.005           # 单模 Richtmyer–Meshkov
python scripts/s1_exp.py sod             # 断轨检验
```
脚本默认在含 `lhd/` 模块的目录下运行（可 `export PYTHONPATH=lhd:scripts`）。
