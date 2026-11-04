# 统一建立在一个 LTN 之上的运动学–动力学通道
ltn_dyn.py：LTN 类继承 LHD 的 trajnet.TrajectoryNetwork（G = 各帧互 kNN 对 ∪ K(t) 的 Delaunay 边；K(t) ⊂ G 由断言检查）。
  对称延拓：mirror_y（激波–重气泡）或 mirror_band（RM，y=0 与 y=0.5 两侧镜像）。
  channels(s)：速度/加速度/力密度余链，Hodge 分解，压力（ρa 梯度势），Kelvin 面元力矩，μ_E，三点映射的 ∇u 与 Φ/μ。
  pair_strain(s)：动力学激活——由 G 中短边的成对伸长率逐节点最小二乘拟合应变率。
  edge_ages(s)：G 的激活统计（边的出生/死亡/年龄）。
run_ltn_sb.py：粘性激波–重气泡上全部通道的回归检验（对照此前各脚本）与两种激活的耗散。
rmv_run.py / rmv2_run.py：粘性单模 RM（μ = 5e-4 / 2e-3），由 ../rm/rm_run.py 改为 ViscSolver。
run_ltn_rm.py <inv|visc|visc2>：RM 上的 μ_E、两种激活的耗散、激活统计。
接管判据沿用 v2 包中的 rm_build/rm_series/rm_take2/rm_mixtake（K(t) 与 LTN 的几何激活相同）。
fig12.py、fig13.py：图 12、图 13。
