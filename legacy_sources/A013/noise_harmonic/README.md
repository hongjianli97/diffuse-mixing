# 正定型的噪声偏置 与 调和通道
- gcv_ell.py：图低通 (M + ℓ²L)^-1 M 的 GCV 选 ℓ（Hutchinson 迹估计），导热项；依赖 ../entropy_channel 与 ../viscous_channel 的输出
- cross_dirichlet.py：分样交叉估计（棋盘分成 A、B 两组示踪点，各自建复形、账本、压力、温度），D_cross = kΣw (ΔlnT_A)(ΔlnT_B→A)
- harm_lo.py：扩散 Lamb–Oseen 环形区（核为洞），精确运动学；β₁、速度/加速度余链调和部分的绕洞周期 vs 理论
  孔边缘按外缘同样的方式剥离长三角形（只剥与孔相邻的），避免伪孔
- fig11.py：图 11
