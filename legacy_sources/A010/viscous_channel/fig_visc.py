import numpy as np, json, sys
sys.path.insert(0, '/home/claude/w'); import cjk
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
V = json.load(open('res_visc_hodge_visc.json')); I = json.load(open('res_visc_hodge_inv.json'))
kv = sorted(V, key=lambda k: V[k]['t']); ki = sorted(I, key=lambda k: I[k]['t'])
tv = np.array([V[k]['t'] for k in kv]); ti = np.array([I[k]['t'] for k in ki]); g = lambda D, ks, f: np.array([D[k][f] for k in ks])
mp = np.load('maps_visc_visc.npz'); s = 50
tri, P, chi, om = mp['tri_%d' % s], mp['P_%d' % s], mp['fcurlpot_%d' % s], mp['om_%d' % s]
A = 0.5 * np.abs((P[tri[:, 1], 0] - P[tri[:, 0], 0]) * (P[tri[:, 2], 1] - P[tri[:, 0], 1]) - (P[tri[:, 1], 1] - P[tri[:, 0], 1]) * (P[tri[:, 2], 0] - P[tri[:, 0], 0]))
mu_chi = float((chi * om * A).sum() / (om * om * A).sum()); cc = float(np.corrcoef(chi, om)[0, 1])
fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
a = ax[0, 0]
a.plot(tv, g(V, kv, 'mu_E') * 1e4, 'o-', label='粘性 DNS（μ = 5×10⁻⁴）：网络估计')
a.plot(tv, -g(V, kv, 'dns_visc_power_curl') / g(V, kv, 'enstrophy') * 1e4, 'k--', label='同一比值，分子换成 DNS 物理粘性力')
a.plot(ti, g(I, ki, 'mu_E') * 1e4, 's-', label='无粘 DNS：网络估计（= 数值有效粘度）')
a.axhline(5, color='C3', lw=1); a.set_ylim(-1, 10); a.set_xlabel('t'); a.set_ylabel('μ ×10⁴')
a.set_title('(a) 只由位置反演粘度：μ_E = −⟨(ρa)_旋度, u⟩ / ∫ω²'); a.legend(fontsize=8.5)
a = ax[0, 1]
a.plot(tv, g(V, kv, 'P_total'), 'k-', label='⟨ρa, u⟩ 总功率'); a.plot(tv, g(V, kv, 'P_grad'), 'C0-', label='梯度通道（压力功）')
a.plot(tv, g(V, kv, 'P_curl') * 50, 'C3-', label='旋度通道 ×50（涡动能的粘性耗散）'); a.plot(tv, g(V, kv, 'dns_visc_power_curl') * 50, 'C3--', label='DNS 物理粘性力的旋度通道 ×50')
a.plot(tv, np.abs(g(V, kv, 'P_cross')) * 1e5, 'C7:', label='|交叉项| ×10⁵（正交性）'); a.axhline(0, color='grey', lw=0.5)
a.set_xlabel('t'); a.legend(fontsize=8.5); a.set_title('(b) 动能收支按 Hodge 通道分解（粘性算例）')
a = ax[0, 2]
a.plot(tv, g(V, kv, 'net_Phi_true_mu') * 1e3, 'o-', label='网络：μ(∫ω² + 4/3∫θ²)，θ 用账本面积率')
a.plot(tv, g(V, kv, 'dns_Phi_int_smooth') * 1e3, 'k--', label='DNS ∫Φ（去掉激波块）'); a.plot(tv, g(V, kv, 'dns_Phi_int') * 1e3, 'k:', label='DNS ∫Φ（含激波，数值激波层）')
a.set_ylim(0, 6); a.set_xlabel('t'); a.set_ylabel('×10⁻³'); a.legend(fontsize=8.5); a.set_title('(c) 粘性耗散：Hodge 能量形式 vs DNS（未计边界项）')
a = ax[1, 0]
a.plot(tv, g(V, kv, 'dns_visc_curlpart_corr_with_fc'), 'o-', label='网络 (ρa) 旋度部分 与 DNS ∇·τ 旋度部分 的相关')
a.plot(tv, g(V, kv, 'torque_bin_corr'), 's-', label='力矩形式：curl(ρa) 与 ∇²ω 分箱相关（失败）')
a.axhline(0, color='grey', lw=0.5); a.set_ylim(-1, 1); a.set_xlabel('t'); a.legend(fontsize=8.5); a.set_title('(d) 粘性力的空间结构')
a = ax[1, 1]; tr = Triangulation(P[:, 0], P[:, 1], tri); lim = np.percentile(np.abs(chi), 99)
a.tripcolor(tr, facecolors=chi, cmap='RdBu_r', vmin=-lim, vmax=lim, rasterized=True); a.set_xlim(0.55, 1.35); a.set_ylim(0, 0.38); a.set_aspect('equal')
fig.colorbar(plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(-lim, lim)), ax=a, shrink=0.8)
a.set_title('(e) t = 1.0：ρa 旋度通道的面势 χ（粘性"流函数"）'); a.set_xticks([]); a.set_yticks([])
a = ax[1, 2]; k = np.random.default_rng(0).choice(len(om), 6000, replace=False)
a.plot(om[k], chi[k], '.', ms=1.5, alpha=0.4); xx = np.linspace(np.percentile(om, 0.5), np.percentile(om, 99.5), 10); a.plot(xx, mu_chi * xx, 'r-', label='斜率 %.2e（相关 %.2f）' % (mu_chi, cc))
a.plot(xx, -5e-4 * xx, 'k--', label='χ = −μω，μ = 5×10⁻⁴（真值）'); a.set_xlabel('面元涡量 ω（只由位置）'); a.set_ylabel('χ'); a.legend(fontsize=9)
a.set_title('(f) 逐面元：χ = −μ·ω（只需一阶导数，不需要 ∇²ω）')
fig.savefig('/home/claude/w/out/图8_粘性通道_由位置反演粘度与耗散.png', dpi=160)
print('mu_chi', mu_chi, 'corr', cc)
