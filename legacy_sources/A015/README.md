# 精确解验证套件（全部经 ltn_dyn.LTN，无 DNS）
python suite.py <case>：linear | lamb [dt] [h] | tg | vortex | acoustic | sod | lamella
- linear：五种线性流（转动、纯应变、简单剪切、各向同性膨胀、一般），一阶通道应精确到舍入或 O(Δt²)
- lamb：扩散 Lamb–Oseen（精确 NS），涡量、压力、μ_E、耗散、粘性力矩；可给 dt、h 做收敛
- tg：衰减 Taylor–Green（精确 NS，周期，3×3 平铺）
- vortex：平移的非均熵涡（精确定常 Euler，变密度），压力、熵、"应为零"的旋度
- acoustic：线性声驻波（周期），账本密度、c²、Dirac 前推、能量
- sod：Sod 激波管精确运动学（cases.case_sod），密度、稀疏波膨胀率、压力、熵、激波 η 与 RH Δs
- lamella：纯应变下的片层扩散，与一维压缩下 α=1/α=0 两种本构（质量度量 vs 面积度量）
fig15.py：图 15（验证矩阵、Sod 剖面、Lamb–Oseen 加密收敛）；调和通道结果取自 ../noise_harmonic/res_harm_lo.json
