# 轨迹网络掺混原型 v2（激波–重气泡 + RM 单次激波/再激波）

依赖 LHD_v3.2 的 `lhd/`；`netmix.py` 在 `shock_bubble/` 下（RM 脚本按 `/home/claude/w` 的目录结构 import，换环境时改 sys.path）。
单核，每条命令 < 5 分钟；DNS 可断点续算。

## RM（rm/）
1. `python rm_run.py single 250`（重复到 DONE，约 5 分钟）；`python rm_run.py reshock 250`（约 12 分钟）
   Ms=1.5，At=0.5，a0=0.1，dx=h=0.005；单次激波在随界面运动的参考系（1D Riemann 接触速度），两端透射；
   再激波为实验室系、x=1.3 反射壁。求解器内推进自适应标记曲线作参照。
2. `python rm_build.py single 1`；`python rm_build.py reshock 2`  物质账本（只用位置）+ 对称线/壁面镜像鬼点的网络
3. `python rm_series.py <case> 1e-4 ser`  交换势分解、割长、标记曲线参照、反事实网络扩散；
   `python rm_series.py <case> 3e-5`、`... 3e-4`  只跑反事实扩散（Pe 扫描）
4. `python rm_take2.py <case>`  各判据接管时刻（含割长分箱的 ⟨λ²⟩）；`python rm_mixtake.py`  混合质量判据随 Γ
5. `fig_rm_fields.py <case>`、`fig_rm_quant.py`  图 3–5

反事实算子：物质电导张量 J C⁻¹ = (1/J)·(J² C⁻¹)。只压缩 = 冻结余切权 / J_e；只拉伸 = 当前余切权 × J_e；
α=0（ρκ）时完整算子再除以 J_e，只压缩除以 J_e²。

已知问题：标记曲线加点用"按下标"的 4 点插值，标签不均匀时会插出 <1e-4 的错位点；总长度不受影响，
逐点 λ 改为按参考标签分箱（箱宽 0.0125）计算。下次重跑应改为在标签空间插值。

## 激波–重气泡（shock_bubble/）：见该目录 README。
