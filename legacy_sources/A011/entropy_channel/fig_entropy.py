import numpy as np, json, pickle, sys
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd'); import cjk, lhdx as X
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
G, CV = 1.4, 2.5
R = json.load(open('res_entropy.json')); C = json.load(open('res_entropy_cond.json')); J = np.load('jumps_inv.npz')
z = np.load('entropy_inv.npz'); o = pickle.load(open('/home/claude/w/LHD_v3.2/sb_dns_dx0.005.pkl', 'rb'))
s = 50; P = o['frames'][s]; tri = X.complex_peeled(P)['tri']; tr = Triangulation(P[:, 0], P[:, 1], tri)
s_rec = CV * (np.log(z['pr'][s]) - G * np.log(z['rl'][s])); s_dns = CV * (np.log(o['E']['p'][s]) - G * np.log(o['E']['rho'][s]))
ref = np.median(s_dns[np.hypot(o['frames'][0][:, 0] - 0.75, o['frames'][0][:, 1]) > 0.3])
fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
for c, (v, title) in enumerate(((s_rec - ref, '(a) t = 1.0  熵 s = c_v ln(p/ρ^γ)：只由位置（p 来自 ρa 梯度势，ρ 来自账本）'), (s_dns - ref, '(b) t = 1.0  熵：DNS（参照）'))):
    a = ax[0, c]; a.tripcolor(tr, np.nan_to_num(v), cmap='cividis', vmin=-4.2, vmax=0.6, shading='gouraud', rasterized=True)
    a.set_xlim(0.45, 1.3); a.set_ylim(0, 0.42); a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(cmap='cividis', norm=plt.Normalize(-4.2, 0.6)), ax=a, shrink=0.8)
a = ax[0, 2]; e = J['eta']; ee = np.linspace(1.2, 3.0, 50); p21 = ((G + 1) * ee - (G - 1)) / ((G + 1) - (G - 1) * ee)
a.plot(ee, CV * (np.log(p21) - G * np.log(ee)), 'k-', lw=2, label='RH：Δs(η)，η 来自账本（纯运动学）')
for tag, mk, col, name in (('light', J['light'], 'C0', '轻气体（入射激波）'), ('hv', J['hv'], 'C3', '重气体（透射激波）')):
    bins = np.linspace(np.percentile(e[mk], 5), np.percentile(e[mk], 95), 9); bi = np.digitize(e[mk], bins)
    cen = [e[mk][bi == k].mean() for k in range(1, 9) if (bi == k).sum() > 20]
    a.plot(cen, [np.median(J['dsR'][mk][bi == k]) for k in range(1, 9) if (bi == k).sum() > 20], 'o', color=col, label=name + '：网络 p, ρ')
    a.plot(cen, [np.median(J['dsD'][mk][bi == k]) for k in range(1, 9) if (bi == k).sum() > 20], 'x', color=col, ms=9, label=name + '：DNS')
a.set_xlabel('账本密度比 η'); a.set_ylabel('熵跳变 Δs'); a.legend(fontsize=8.5); a.set_title('(c) 激波熵跳变：三条独立途径（无粘）')
a = ax[1, 0]; pairs = [('20_40', 't 0.4→0.8'), ('40_60', 't 0.8→1.2'), ('50_69', 't 1.0→1.38')]; x = np.arange(3)
a.bar(x - 0.2, [R['inv_rms_ds_rec_' + p] for p, _ in pairs], 0.4, label='网络'); a.bar(x + 0.2, [R['inv_rms_ds_dns_' + p] for p, _ in pairs], 0.4, label='DNS')
a.set_xticks(x); a.set_xticklabels([l for _, l in pairs]); a.set_ylabel('沿轨迹 Δs 的 RMS')
a.set_title('(d) 光滑区熵守恒（重/轻气体熵差 %.2f，网络 %.2f）' % (R['contrast_theory'], R['inv_heavy_minus_light_rec_50'])); a.legend(fontsize=9)
a = ax[1, 1]; M = C['meas']; tt = np.array(M['t'])
a.plot(tt, M['dns_heavy'], 'k--', lw=2, label='DNS：Σ m Ds/Dt（重气体）'); a.plot(tt, M['meas_heavy'], 'k-', alpha=0.6, label='网络测得：沿轨迹的 Ds/Dt')
for l, col in (('0.0', 'C3'), ('0.01', 'C0'), ('0.02', 'C2')):
    a.plot(tt, C['smooth'][l]['pred_heavy'], '-', color=col, label='网络预测 (Φ + ∇·k∇T)/ρT，T 低通 ℓ=%s' % l)
a.set_ylim(0, 0.2); a.set_xlabel('t'); a.legend(fontsize=8); a.set_title('(e) 重气体的熵收支（粘性算例，μ、k 已知）')
a = ax[1, 2]
for l, col in (('0.0', 'C3'), ('0.01', 'C0'), ('0.02', 'C2')): a.plot(tt, C['smooth'][l]['prod_cond'], '-', color=col, label='导热熵产生 kΣw(Δln T)²，ℓ=%s' % l)
a.plot(tt, M['dns_all'], 'k--', label='DNS：整团 Σ m Ds/Dt（含边界通量）'); a.plot(tt, M['meas_all'], 'k-', alpha=0.6, label='网络测得：整团')
a.set_ylim(-0.02, 0.2); a.set_xlabel('t'); a.legend(fontsize=8); a.set_title('(f) 第二定律 = 拉普拉斯 Dirichlet 型 ≥ 0：噪声会被当成熵产生')
fig.savefig('/home/claude/w/out/图10_熵通道.png', dpi=160); print('ok')
