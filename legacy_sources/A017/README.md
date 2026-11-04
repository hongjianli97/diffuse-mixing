# LHD 报告九个算例的 LTN 重算
python allcases.py <dg|lo|sod|blast|sv|sb|sbv|rm|rmv>
  轨迹：dg/lo/sod/blast 由 LHD cases.py 生成；sv 由 LHD sv_case.run() 跑 DNS（另存 sv_dns.pkl）；sb/sbv/rm/rmv 用前几轮的 DNS
  LTN 通道：账本密度、面元涡量、ρa 的 Hodge 分解（压力、旋度份额、μ_E）、熵、两种激活的应变、AFRC、网络 LAVD
  基线（同一组示踪点）：DTFE 密度；compare.py 中的 FTLE、DBS（最小二乘 F）、相遇数
viz_recalc.py：图 19（通道场图）、图 20（输运结构对比）；fig21.py：图 21（定量对比，含报告表 7 的 SPH/DTFE/账本值）
注意：报告的 RM 算例设置与这里不同（这里为 a0=0.1、随界面运动参考系、界面带示踪点），RM 数值不可与报告直接对照。
