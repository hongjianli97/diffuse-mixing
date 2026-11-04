import numpy as np, json, sys
sys.path.insert(0, '/home/claude/w'); import cjk
import matplotlib.pyplot as plt
V = {v: json.load(open('res_verify_%s.json' % v)) for v in ('base', 'dt2', 'subA', 'noise1', 'noise5', 'noise20')}
names = {'base': '基准\nΔt=0.02, h', 'dt2': 'Δt 加倍', 'subA': '间距 √2h', 'noise1': '噪声 0.2%h', 'noise5': '噪声 1%h', 'noise20': '噪声 4%h'}
fig, ax = plt.subplots(1, 4, figsize=(21, 4.8), constrained_layout=True); x = np.arange(len(V)); ts = ['0.6', '1.0', '1.2']
mets = [('mu_E', 'μ_E / μ', lambda r: r['mu_E'] / 5e-4), ('tpm_over_dns', 'Φ（三点映射）/ DNS', lambda r: r['tpm_over_dns']),
        ('pair_over_dns', 'Φ（成对伸长率）/ DNS', lambda r: r['pair_over_dns']), ('p_relrms_smooth', '压力 RMS 误差 / std(p)', lambda r: r['p_relrms_smooth'])]
for a, (k, lab, f) in zip(ax, mets):
    for i, tq in enumerate(ts):
        a.bar(x + (i - 1) * 0.27, [f(V[v][tq]) for v in V], 0.27, label='t = %s' % tq)
    if k != 'p_relrms_smooth': a.axhline(1, color='k', lw=0.8)
    else: a.set_yscale('log')
    a.set_xticks(x); a.set_xticklabels([names[v] for v in V], fontsize=8.5); a.set_title(lab); a.legend(fontsize=8)
    if k == 'tpm_over_dns': a.set_ylim(0, 3)
    if k == 'mu_E': a.set_ylim(0, 2)
fig.suptitle('验证：粘性激波–重气泡上的帧间隔、示踪点间距与位置噪声敏感性（全部经同一 LTN 计算）', fontsize=12)
fig.savefig('/home/claude/w/out/图14_验证_敏感性.png', dpi=160); print('ok')
