# 粘性通道：由位置反演粘度、耗散与动能收支（激波–重气泡）
1. python sbv_run.py 255   粘性 DNS（LHD euler2d_visc，μ=5e-4，Pr=0.72），与无粘算例同一组示踪点；可断点续算（约 3 分钟）
2. python visc_hodge.py visc ; python visc_hodge.py inv   （inv 需要无粘 DNS 与 ../hodge_dynamics 的账本缓存）
   ρa 余链的 Hodge 分解；μ_E = −<(ρa)_c, u>/∫ω²；力矩形式 μ_T；压力；耗散；DNS 粘性力与 Φ 仅作参照
3. python fig_visc.py   图 8
