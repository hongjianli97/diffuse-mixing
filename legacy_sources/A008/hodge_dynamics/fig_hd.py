import numpy as np, json, sys
src = open('hodge_dyn.py').read(); exec(src[:src.index('res = dict(series')])
sys.path.insert(0, '/home/claude/w'); import cjk
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
mp = np.load('maps_hodge_dyn.npz'); R = json.load(open('res_hodge_dyn.json'))['frames']
C2 = json.load(open('res_hd_check2.json')); C3 = json.load(open('res_hd_check3.json'))
def tri2node(vT, tri, A):
    v = np.zeros(N); c = np.zeros(N)
    for k in range(3): np.add.at(v, tri[:, k], vT * A); np.add.at(c, tri[:, k], A)
    return v / np.maximum(c, 1e-30)
fig, axs = plt.subplots(2, 6, figsize=(25, 7.6), constrained_layout=True)
for r, s in enumerate((30, 50)):
    P, tri, A, E, w, B1, B2, Ast = ops(s)
    Am, Ap = X._star(fr[s - 1], tri, N), X._star(fr[s + 1], tri, N)
    with np.errstate(all='ignore'): thL = np.where((Am > 0) & (Ap > 0), (np.log(Ap) - np.log(Am)) / (2 * dt), 0.0)
    tr = Triangulation(P[:, 0], P[:, 1], tri); lab = np.hypot(fr[0][:, 0] - 0.75, fr[0][:, 1]) < 0.25
    xl = (P[:N0][lab[:N0], 0].min() - 0.15, P[:N0][lab[:N0], 0].max() + 0.2); yl = (0, 0.36)
    pd, pr = mp['p_dns_%d' % s], mp['p_rec_%d' % s]; lim = (np.percentile(pd[:N0], 1), np.percentile(pd[:N0], 99))
    bT = tri2node(mp['bar_dns_%d' % s], tri, A); nT = tri2node(mp['tq_net_%d' % s], tri, A)
    tl = np.percentile(np.abs(bT[:N0]), 99.5)
    chi = mp['chi_%d' % s]; cl = np.percentile(np.abs(chi), 99)
    panels = [(pd, '压力 p（DNS，参照）', 'viridis', lim, 'g'), (pr, '压力（只由位置：ρa 余链的梯度势）', 'viridis', lim, 'g'),
              (bT, '斜压力矩 ∇ρ×∇p/ρ²（DNS，参照）', 'RdBu_r', (-tl, tl), 'g'), (nT, '加速度余链的旋度 = 物质面元 dΓ/dt', 'RdBu_r', (-tl, tl), 'g'),
              (chi, '流函数 χ（速度余链的涡量部分）', 'PuOr', (-cl, cl), 'f'), (thL, '膨胀率 θ（物质面积率，账本形式）', 'BrBG', (-3, 3), 'g')]
    for c, (v, title, cmap, lm, kind) in enumerate(panels):
        ax = axs[r, c]
        if kind == 'g': ax.tripcolor(tr, v, cmap=cmap, vmin=lm[0], vmax=lm[1], shading='gouraud', rasterized=True)
        else: ax.tripcolor(tr, facecolors=v, cmap=cmap, vmin=lm[0], vmax=lm[1], rasterized=True)
        b = np.abs(np.hypot(fr[0][:, 0] - 0.75, fr[0][:, 1]) - 0.25) < 0.003
        ax.plot(P[:N0][b[:N0], 0], P[:N0][b[:N0], 1], 'k.', ms=0.3)
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(*lm)), ax=ax, shrink=0.8, pad=0.01)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(('t = %.1f   ' % t[s] if c == 0 else '') + title, fontsize=10)
fig.suptitle('激波–重气泡（无粘 DNS 运动学）：轨迹复形上的 Hodge 分解与动力学对应（黑点为初始界面示踪点）', fontsize=13)
fig.savefig('/home/claude/w/out/图6_运动学Hodge分解与动力学对应_场图.png', dpi=170)
# ---------------- quantitative
ks = sorted(R, key=lambda k: R[k]['t']); tt = [R[k]['t'] for k in ks]
fig, ax = plt.subplots(2, 3, figsize=(17, 9.4), constrained_layout=True)
a = ax[0, 0]; a.plot(tt, [R[k]['p_relrms'] for k in ks], 'o-', label='全部示踪点'); a.plot(tt, [R[k]['p_relrms_offshock'] for k in ks], 's-', label='去掉激波块内示踪点')
a.plot(tt, [1 - R[k]['p_corr'] for k in ks], '^-', label='1 − 相关系数'); a.set_yscale('log'); a.set_xlabel('t'); a.legend(fontsize=9)
a.set_title('(a) 压力：Hodge 梯度势 vs DNS（RMS 误差 / std(p)）')
a = ax[0, 1]; a.stackplot(tt, [R[k]['frac_u_grad'] for k in ks], [R[k]['frac_u_curl'] for k in ks], [max(R[k]['frac_u_harm'], 0) for k in ks],
                          labels=['梯度（膨胀/声学 + 激波诱导整体运动）', '旋度（涡量）', '调和'], colors=['#9ecae1', '#fc9272', '#bdbdbd'])
a.set_ylim(0.8, 1.0); a.set_xlabel('t'); a.legend(fontsize=9, loc='lower right'); a.set_title('(b) 质量度量下动能的 Hodge 分解（Kovasznay 型）')
a = ax[0, 2]; a.plot(tt, [R[k]['frac_a_curl'] for k in ks], 'o-', label='加速度 a 的旋度份额（斜压）'); a.plot(tt, [R[k]['frac_f_curl'] for k in ks], 's-', label='ρa 的旋度份额（应为 0：粘性/数值残差）')
a.set_xlabel('t'); a.legend(fontsize=9); a.set_title('(c) 动量方程的 Hodge 检验：ρa = −∇p 是否成立')
a = ax[1, 0]; tk = ['15', '30', '50']; x = np.arange(3)
a.bar(x - 0.3, [R[k]['kelvin_cloud_net'] for k in tk], 0.2, label='整团：网络 Σ dΓ/dt'); a.bar(x - 0.1, [R[k]['kelvin_cloud_dns'] for k in tk], 0.2, label='整团：DNS ∫斜压力矩')
a.bar(x + 0.1, [R[k]['kelvin_heavyband_net'] for k in tk], 0.2, label='重气体+界面带：网络'); a.bar(x + 0.3, [R[k]['kelvin_heavyband_dns'] for k in tk], 0.2, label='重气体+界面带：DNS')
a.set_xticks(x); a.set_xticklabels(['t=0.3', 't=0.6', 't=1.0']); a.axhline(0, color='k', lw=0.5); a.legend(fontsize=8)
a.set_title('(d) Kelvin：物质区域环量变化率（分箱相关 %.2f/%.2f/%.2f）' % tuple(R[k]['torque_bin_corr'] for k in tk))
a = ax[1, 1]
a.bar(x - 0.3, [C3[k]['enstrophy_net'] for k in tk], 0.2, label='∫ω²：网络（上 Hodge 能量）'); a.bar(x - 0.1, [C3[k]['enstrophy_dns'] for k in tk], 0.2, label='∫ω²：DNS')
a.bar(x + 0.1, [C3[k]['dil2_ledger_tri_offshock'] * 10 for k in tk], 0.2, label='∫θ² 光滑区 ×10：网络（账本面积率）'); a.bar(x + 0.3, [C3[k]['dil2_dns_offshock'] * 10 for k in tk], 0.2, label='∫θ² 光滑区 ×10：DNS')
a.set_xticks(x); a.set_xticklabels(['t=0.3', 't=0.6', 't=1.0']); a.legend(fontsize=8); a.set_title('(e) 耗散恒等式两项：Φ = μ∫(ω² + 4/3 θ²) + 边界项')
a = ax[1, 2]
a.bar(x - 0.2, [C2[k]['spearman_theta'] for k in tk], 0.4, label='余链 Galerkin 散度（速度差分）'); a.bar(x + 0.2, [C3[k]['spearman_ledger'] for k in tk], 0.4, label='物质面积率（三点映射/账本）')
a.set_xticks(x); a.set_xticklabels(['t=0.3', 't=0.6', 't=1.0']); a.set_ylim(0, 1); a.legend(fontsize=9)
a.set_title('(f) 膨胀率与 DNS 的 Spearman 相关：散度必须用物质形式')
fig.savefig('/home/claude/w/out/图7_运动学–动力学对应_定量.png', dpi=160); print('ok')
