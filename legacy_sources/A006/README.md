# 轨迹网络掺混：可行性原型（2026-10-28）

依赖：LHD_v3.2 代码包中的 `lhd/`（路径在脚本开头 `sys.path` 里改）。单核，各脚本 < 5 分钟。

1. `python LHD_v3.2/scripts/run_sb_any.py 0.005`  生成激波–重气泡 DNS（约 2.5 分钟）
2. `t2_build.py` → `t2_build_m.py`  建网络（后者加对称线镜像鬼点，必须用它）
3. `t1_sod_filter.py`, `t1b.py`  Sod 精确运动学上的压缩滤网
4. `t2_series.py 3e-5,1e-4 a` 与 `t2_series.py 3e-4,1e-3 b`  接触/交换势时间历程与 Pe 扫描
5. `t3_spec.py`  吸收随机游走（λ1、首达时间）；`t3chk*.py` 对称线薄片三角形诊断
6. `t4_afrc.py`  AFRC 屏障强度（对照）
7. `t5_maps.py`, `t5b.py`, `t5c.py`  场图数据与"哪个网络量解释掺混增量"
8. `fig1.py`, `fig2.py`  两张图

核心模块 `netmix.py`：点权 = 质量，边权 = 余切电导（截零），生成元 M⁻¹L(t)；
扩散用平均 K 的隐式步；`node_dirichlet` 给节点交换率密度，时间积分即网络掺混时钟；
`mfpt` 首达时间；`far_contacts` 参考距离过滤的远程接触。
全部数值在 `res_*.json`。
