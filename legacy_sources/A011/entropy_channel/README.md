# 熵通道（激波–重气泡，无粘与粘性）
依赖：../viscous_channel（visc_hodge.py、sbv_run.py 及其输出）与无粘 DNS、账本缓存。
1. python entropy.py inv ; python entropy.py visc   每帧由位置重建 p（ρa 梯度势，每帧一个参考常数），账本 ρ，三点映射 Φ，L0 导热项
2. python entropy_an.py   熵场、光滑区守恒、激波熵跳变（RH(η_账本) / 网络 p,ρ / DNS）、熵收支
3. python entropy_cond.py  对 T 做图低通 (M + ℓ²L)^-1 M 后的导热项与熵产生；T 的噪声水平
4. python fig_entropy.py  图 10
