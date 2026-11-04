# 轨迹复形上的运动学 Hodge 分解与动力学对应（激波–重气泡原型）

依赖：LHD_v3.2 的 lhd/（lhdx）；激波–重气泡 DNS（run_sb_any.py 0.005）与镜像网络 sb_net_m.pkl、sb_lite_m.npz（见 v2 包 shock_bubble/）。
1. hodge_dyn.py：物质账本（只用位置）；速度/加速度/力密度余链；质量度量 Hodge 分解；压力 = ρa 余链梯度势；
   Kelvin（加速度余链旋度）对照 DNS 斜压力矩；结果 res_hodge_dyn.json
2. hd_check2.py：余链 Galerkin 散度、∫ω²（上 Hodge 能量）对照 DNS
3. hd_check3.py：散度用物质面积率（账本/三点映射形式）对照 DNS
4. fig_hd.py：图 6、图 7
