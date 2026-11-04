import numpy as np, sys, pickle
sys.path.insert(0, '/home/claude/w'); import cjk
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
case = sys.argv[1]
b = np.load('base_%s.npz' % case); net = pickle.load(open('net_%s.pkl' % case, 'rb')); mk = pickle.load(open('marker_%s.pkl' % case, 'rb'))
ms = np.load('maps_series_%s.npz' % case); mc = np.load('maps_c_%s_G0.0001.npz' % case)
N = int(b['N']); sel = b['sel']; t = b['t']; phi0 = b['phi0']; lab = (phi0 > 0).astype(float); lnJ = b['lnJ']
keep = {'single': (40, 80, 120), 'reshock': (20, 40, 80, 120)}[case]
rep = lambda v: np.r_[v, v, v]
cols = [('lab', '物质标签 + 标记曲线（黑）', 'coolwarm', (-0.3, 1.3)),
        ('comp', '压缩 −ln J（物质账本，只用位置）', 'Purples', (0, 1.6)),
        ('st', '网络拉伸 ln λ²（4h 带交换势，会饱和）', 'RdBu_r', (-3, 3)),
        ('clock', '网络掺混时钟 ln(τ/τ冻结)', 'RdBu_r', (-2.5, 2.5)),
        ('mix', '局部混合度 4c(1−c)（Γ=1e-4）', 'magma', (0, 1)),
        ('dmix', '流动带来的掺混增量（有流动 − 冻结）', 'PiYG', (-0.4, 0.4))]
fig, axs = plt.subplots(len(keep), 6, figsize=(21, 4.1 * len(keep)), constrained_layout=True)
for r, s in enumerate(keep):
    P = np.r_[b['fr'][sel[s]], b['fr'][sel[s]] * [1, -1], np.c_[b['fr'][sel[s]][:, 0], 1 - b['fr'][sel[s]][:, 1]]]
    T = net['tri'][s]
    cf, cz = mc['c_full_%d' % s], mc['c_frozen_%d' % s]
    vals = dict(lab=lab, comp=-lnJ[sel[s]], st=ms['lnstretch2_%d' % s], clock=ms['clock_%d' % s], mix=4 * cf * (1 - cf), dmix=4 * cf * (1 - cf) - 4 * cz * (1 - cz))
    xs = b['fr'][sel[s]][np.abs(phi0) < 0.03, 0]; xl = (xs.min() - 0.06, xs.max() + 0.06)
    for c, (key, title, cmap, lim) in enumerate(cols):
        ax = axs[r, c]; v = rep(np.asarray(vals[key], float)); ok = np.isfinite(v)[T].all(1)
        if key in ('st', 'clock'): ax.set_facecolor('#e6e6e6')
        ax.tripcolor(Triangulation(P[:, 0], P[:, 1], T[ok]), np.nan_to_num(v), cmap=cmap, vmin=lim[0], vmax=lim[1], shading='gouraud', rasterized=True)
        if key == 'lab':
            M, _ = mk[sel[s]]; ax.plot(M[:, 0], M[:, 1], 'k-', lw=0.6)
        else:
            fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(*lim)), ax=ax, shrink=0.85, pad=0.01)
        ax.set_xlim(*xl); ax.set_ylim(0, 0.5); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(('t = %g   ' % round(t[sel[s]], 2) if c == 0 else '') + title, fontsize=9.5)
name = {'single': '单次激波 RM（a₀=0.1，随界面运动的参考系，h=0.005）', 'reshock': '再激波 RM（a₀=0.1，反射壁 x=1.3，h=0.005）'}[case]
fig.suptitle(name + '：只由示踪点位置 + 一时刻质量得到的网络场', fontsize=13)
fig.savefig('/home/claude/w/out/图%s_%s_网络场.png' % ({'single': '3', 'reshock': '4'}[case], {'single': 'RM单次激波', 'reshock': 'RM再激波'}[case]), dpi=170)
print('ok')
