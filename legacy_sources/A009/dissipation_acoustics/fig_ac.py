import numpy as np, json, sys
src = open('ac_dirac.py').read(); exec(src[:src.index('res = dict(frames={})')])
sys.path.insert(0, '/home/claude/w'); import cjk
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
R = json.load(open('res_ac_dirac.json')); D = json.load(open('/home/claude/w/visc/res_dissip_tpm.json')); V = json.load(open('/home/claude/w/visc/res_visc_hodge_visc.json'))
fwd, rev, ts = [], [], []
for s in range(2, S - 1):
    A_ = (fr[s + 1] - 2 * fr[s] + fr[s - 1]) / dt ** 2; ae = coch(fr[s], A_); g = B1.T @ np.log(rho_led[s])
    fwd.append(-(w * ae * g).sum() / (w * g * g).sum()); rev.append(-(w * ae * ae).sum() / (w * ae * g).sum()); ts.append(t[s])
fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
a = ax[0, 0]; kd = sorted(D, key=lambda k: D[k]['t']); td = [D[k]['t'] for k in kd]
a.plot(td, [D[k]['Phi_net_smooth'] * 1e3 for k in kd], 'o-', label='网络：三点映射 ∇u = log F / 2Δt，μ(2S:S − 2/3 θ²)')
a.plot(td, [D[k]['Phi_dns_smooth'] * 1e3 for k in kd], 'k--', label='DNS ∫Φ（去掉激波块）')
kv = sorted(V, key=lambda k: V[k]['t']); a.plot([V[k]['t'] for k in kv], [V[k]['net_Phi_true_mu'] * 1e3 for k in kv], ':', color='C1', label='上一轮：μ(∫ω² + 4/3∫θ²)，未计 −4μ∫det∇u')
a.set_ylim(0, 3.2); a.set_xlabel('t'); a.set_ylabel('×10⁻³'); a.legend(fontsize=8.5); a.set_title('(a) 粘性耗散闭合（粘性激波–重气泡，μ 已知）')
a = ax[0, 1]; a.plot(ts, fwd, 's-', ms=3, label='把账本密度梯度当自变量回归（有偏：账本梯度含噪声）'); a.plot(ts, rev, 'o-', ms=3, label='把加速度当自变量回归（误差在账本一侧）')
a.axhline(1.0, color='k', lw=1, ls='--', label='真值 c² = γp/ρ = 1'); a.set_ylim(0.5, 1.2); a.set_xlabel('t'); a.set_ylabel('c²'); a.legend(fontsize=8.5)
a.set_title('(b) 只由位置得到声速：a = −c²∇lnρ（全程 c² = %.3f）' % R['c2_all'])
a = ax[0, 2]; rb = np.array(R['rbins'])
for i, s in enumerate((20, 40, 60)):
    a.plot(rb, np.array(R['prof']['dns_%d' % s]) * 1e3, 'k-', lw=2.2, alpha=0.5, label='DNS' if i == 0 else None)
    a.plot(rb, np.array(R['prof']['led_%d' % s]) * 1e3, '-', color='C%d' % i, label='t=%.1f 账本（位置）' % t[s])
    a.plot(rb, np.array(R['prof']['dirac_%d' % s]) * 1e3, '--', color='C%d' % i, label='t=%.1f Dirac 前推' % t[s])
a.axvline(0.35, color='grey', lw=0.6, ls=':'); a.set_xlabel('r'); a.set_ylabel("ρ' ×10³"); a.legend(fontsize=8, ncol=2); a.set_title("(c) 声脉冲的径向剖面（虚竖线：示踪点区域边缘）")
a = ax[1, 0]; tt = t[:len(R['energy_dirac'])]
a.plot(tt, np.array(R['energy_dirac']) * 1e7, '-', label='Dirac 前推（离散能量守恒）'); a.plot(t[1:S - 1], np.array(R['energy_meas']) * 1e7, 'o', ms=3, label='由位置测得：½c²ρ\'²/ρ + ½ρ|u|²')
a.set_xlabel('t'); a.set_ylabel('×10⁻⁷'); a.legend(fontsize=9); a.set_title('(d) 声学能量')
mp = np.load('maps_ac.npz'); tr = Triangulation(mp['P0'][:, 0], mp['P0'][:, 1], mp['tri']); lim = np.abs(mp['rp_dns40']).max()
for c, (key, title) in enumerate((('rp_dns40', "(e) t = 0.2  DNS ρ'（参照）"), ('rp_dirac40', "(f) t = 0.2  网络 Dirac 前推 ρ'（初值 + c² + t=0 复形）"))):
    a = ax[1, 1 + c]; a.tripcolor(tr, mp[key], cmap='RdBu_r', vmin=-lim, vmax=lim, shading='gouraud', rasterized=True)
    a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([]); a.set_title(title)
    fig.colorbar(plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(-lim, lim)), ax=a, shrink=0.8)
fig.savefig('/home/claude/w/out/图9_耗散闭合与Dirac声学通道.png', dpi=160); print('ok')
