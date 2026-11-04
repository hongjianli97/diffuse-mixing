import numpy as np, sys, pickle
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd'); import cjk
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import netmix as NM
net, z = NM.load_sb(); N0 = int(z['N0']); fr = net.fr; t = z['t']; mp = dict(np.load('out/maps_all.npz')); af = np.load('out/maps_afrc.npz')
P0 = fr[0]; phi0 = np.hypot(P0[:, 0] - 0.75, P0[:, 1]) - 0.25; lab = phi0 < 0
w0 = 0.02; c0 = 0.5 * (1 - np.tanh(phi0 / (w0 / 2))); q0 = net.node_dirichlet(0, c0); Q = np.zeros(net.N); clock = {}
for s in range(1, 71):
    Q += 0.5 * net.dt * (net.node_dirichlet(s - 1, c0) + net.node_dirichlet(s, c0))
    if s in (30, 70): clock[s] = np.where(np.abs(phi0) < w0, np.log(np.maximum(Q, 1e-300) / (s * net.dt * np.maximum(q0, 1e-300))), np.nan)
np.savez_compressed('out/maps_clock.npz', clock30=clock[30][:N0], clock70=clock[70][:N0])
def dup(v): v = np.asarray(v, float)[:N0]; return np.r_[v, v]
cols = [('lab', '物质标签（红：重气体）', 'coolwarm', (-0.3, 1.3)),
        ('clock', '网络掺混时钟 ln(τ/τ冻结)\n（余切电导对时间积分，界面带）', 'RdBu_r', (-2, 2)),
        ('mix', '局部混合度 4c(1−c)\n（网络扩散 Γ=1e-4）', 'magma', (0, 1)),
        ('dmix', '流动带来的掺混增量\n（有流动 − 冻结网络）', 'PiYG', (-0.3, 0.3)),
        ('farx', '跨物质远程接触数\n（参考距离 > 4h 的新邻居）', 'Purples', (0, 12)),
        ('afrc', 'AFRC 屏障强度（输运，对照）', 'inferno', None)]
fig, axs = plt.subplots(2, 6, figsize=(24, 7.4), constrained_layout=True)
for r, s in enumerate((30, 70)):
    P = fr[s]; T = net.tri[s]; tr = Triangulation(P[:, 0], P[:, 1], T)
    xl = P[:N0][lab[:N0], 0].min() - 0.12, P[:N0][lab[:N0], 0].max() + 0.15; yl = 0, max(P[:N0][lab[:N0], 1].max() + 0.08, 0.32)
    for c, (key, title, cmap, lim) in enumerate(cols):
        ax = axs[r, c]
        if key == 'lab': v = dup(lab)
        elif key == 'clock': v = dup(clock[s][:N0])
        elif key == 'afrc': v = dup(af['barrier_s%d' % s]); lim = (0, float(np.nanpercentile(v, 99.5)))
        else: v = dup(mp['%s_%d' % (key, s)])
        ok = np.isfinite(v)[T].all(1)
        ax.set_facecolor('#e8e8e8' if key == 'clock' else 'white')
        ax.tripcolor(Triangulation(P[:, 0], P[:, 1], T[ok]), np.nan_to_num(v), cmap=cmap, vmin=lim[0], vmax=lim[1], shading='gouraud', rasterized=True)
        if key == 'lab':
            b = np.abs(phi0[:N0]) < 0.004; ax.plot(P[:N0][b, 0], P[:N0][b, 1], 'k.', ms=0.5)
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(*lim)); fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.01)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(('t = %.1f   ' % t[s] if c == 0 else '') + title, fontsize=10.5)
fig.suptitle('激波–重气泡（Ms=1.5，无粘 DNS 运动学，h=0.005）：只由示踪点位置 + 一时刻质量得到的网络掺混场（窗口 [0, t]）', fontsize=13)
fig.savefig('out/图1_激波气泡_网络掺混场.png', dpi=200)
print('ok')
