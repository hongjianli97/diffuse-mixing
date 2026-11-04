# 耗散闭合（三点映射）与 Dirac 声学通道
- dissip_tpm.py：粘性激波–重气泡（先跑 ../viscous_channel/sbv_run.py），逐物质三角形 F = D(t+Δt)D(t−Δt)^-1，
  ∇u = log F/2Δt（2×2 精确矩阵对数），Φ = μ(2S:S − 2/3θ²)，对照 DNS ∫Φ；结果 res_dissip_tpm.json
- ac_run.py：静止气体中的线性声脉冲（四壁反射，ε=0.01，σ=0.06，c0=1），示踪点 h=0.005，帧间隔 0.005（约 15 s）
- ac_dirac.py：账本 ρ'、由位置估计 c²（以加速度为自变量回归）、速度余链旋度份额、t=0 复形上的 Dirac 前推模型
- ac_diag.py：回归偏差诊断；fig_ac.py：图 9
