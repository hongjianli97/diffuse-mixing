import numpy as np, json, sys
sys.path.insert(0, '/home/claude/w'); import cjk
import matplotlib.pyplot as plt
cases = [('inv', '/home/claude/w/rm', '无粘', 0.0, 'k'), ('visc', '/home/claude/w/rmv', 'μ = 5×10⁻⁴', 5e-4, 'C0'), ('visc2', '/home/claude/w/rmv2', 'μ = 2×10⁻³', 2e-3, 'C3')]
L = {c: json.load(open('res_ltn_rm_%s.json' % c)) for c, *_ in cases}
T = {c: json.load(open(d + '/res_single_take.json')) for c, d, *_ in cases}
Sr = {c: json.load(open(d + '/res_single_G0.0001_ser.json')) for c, d, *_ in cases}
MT = {c: json.load(open(d + '/res_mixtake.json'))['single_G0.0001']['a1'] for c, d, *_ in cases}
fig, ax = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
a = ax[0, 0]
for c, d, lab, mu, col in cases:
    ks = sorted(L[c], key=lambda k: L[c][k]['t']); a.plot([L[c][k]['t'] for k in ks], [L[c][k]['mu_E'] * 1e4 for k in ks], 'o-', color=col, label=lab + '：μ_E（只由位置）')
    if mu: a.axhline(mu * 1e4, color=col, ls='--', lw=1)
a.set_xlabel('t'); a.set_ylabel('μ ×10⁴'); a.legend(fontsize=9); a.set_title('(a) 单模 RM：由 LTN 反演粘度（虚线为真值；无粘时即数值粘度）')
a = ax[0, 1]
for c, d, lab, mu, col in cases[1:]:
    ks = sorted(L[c], key=lambda k: L[c][k]['t']); tt = [L[c][k]['t'] for k in ks]
    a.plot(tt, [mu * L[c][k]['phi_tpm'] * 1e4 for k in ks], 'o-', color=col, label=lab + '：几何激活（三点映射）')
    a.plot(tt, [mu * L[c][k]['phi_pair'] * 1e4 for k in ks], 's--', color=col, label=lab + '：动力学激活（成对伸长率）')
a.set_xlabel('t'); a.set_ylabel('∫Φ ×10⁴'); a.set_yscale('log'); a.legend(fontsize=8.5); a.set_title('(b) 耗散：两种激活一致')
a = ax[1, 0]
crit = [('几何·界面长度：标记曲线', '几何 ln L'), ('时钟α=1·界面长度：标记曲线', '时钟 α=1'), ('速率α=1·算术<λ²>：标记曲线', '速率 α=1 ⟨λ²⟩'), ('速率α=0·算术<λ²>：标记曲线', '速率 α=0 ⟨λ²⟩'), ('MM', '混合质量 α=1（Γ=10⁻⁴）')]
x = np.arange(len(crit)); wd = 0.26
for i, (c, d, lab, mu, col) in enumerate(cases):
    vals = [MT[c] if k == 'MM' else T[c]['takeover'][k] for k, _ in crit]
    a.bar(x + (i - 1) * wd, [v if v is not None else np.nan for v in vals], wd, color=col if col != 'k' else 'grey', label=lab)
    for xi, v in zip(x + (i - 1) * wd, vals): a.text(xi, (v or 0) + 0.05, '%.2f' % v if v else '>6', ha='center', fontsize=7.5)
a.set_xticks(x); a.set_xticklabels([n for _, n in crit], fontsize=9); a.set_ylabel('拉伸接管时刻 t'); a.legend(fontsize=9); a.set_title('(c) 接管时刻随雷诺数（标记曲线参照；混合质量为 LTN 反事实扩散）')
a = ax[1, 1]
for c, d, lab, mu, col in cases:
    tt = np.array(T[c]['t']); a.plot(tt, T[c]['marker_lnL'], '-', color=col, label=lab + '：ln L/L₀（标记曲线）')
    a.plot(tt, Sr[c]['series']['logmean_comp'], ':', color=col, label=lab + '：⟨−ln J⟩（账本）')
a.set_xlabel('t'); a.legend(fontsize=8.5, ncol=2); a.set_title('(d) 粘性削弱拉伸，导热削弱压缩（冷重气体被加热膨胀）')
fig.savefig('/home/claude/w/out/图13_有限雷诺数RM.png', dpi=160)
for c, *_ in cases: print(c, 'comp@6', Sr[c]['series']['logmean_comp'][-1], 'lnL@6', T[c]['marker_lnL'][-1])
