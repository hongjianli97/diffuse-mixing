import numpy as np, sys, json
sys.path.insert(0, '/home/claude/w'); import cjk
import matplotlib.pyplot as plt
t1 = json.load(open('out/res_t1_sod.json')); t1b = json.load(open('out/res_t1b_sod_energy.json'))
A = json.load(open('out/res_t2_series_a.json')); B = json.load(open('out/res_t2_series_b.json')); sw = {**A['sweep'], **B['sweep']}
sp_ = json.load(open('out/res_t3_spec.json')); c5 = json.load(open('out/res_t5c.json'))
t = np.array(A['t'])
fig, ax = plt.subplots(2, 3, figsize=(17, 9.6), constrained_layout=True)
# (a) Sod filter
f = t1['field']; rows = []
rows.append(('ε-图 平均度（Padberg-Gehle）', f['eps_degree'][1] / f['eps_degree'][0] - 1, 'comb'))
rows.append(('ε-图 聚类系数', f['eps_clustering'][1] / f['eps_clustering'][0] - 1, 'comb'))
rows.append(('Delaunay 边更替比例（应为 0）', f['delaunay_edges_changed_frac'], 'comb'))
for ang in ('0', '45', '90'):
    rows.append(('组合 Dirichlet 能量 %s°' % ang, t1b[ang]['comb'] / t1b[ang]['pred_GammaConst'] - 1, 'comb'))
rows.append(('组合同配系数（条纹）', t1['stripes45']['assort_comb'][1] / t1['stripes45']['assort_comb'][0] - 1, 'comb'))
rows.append(('余切同配系数（条纹）', t1['stripes45']['assort_cot'][1] / t1['stripes45']['assort_cot'][0] - 1, 'cot'))
rows.append(('余切割长 vs Nanson（圆）', (t1['circle']['L_net'][1] / t1['circle']['L_net'][0]) / t1['circle']['L_pred_ratio'] - 1, 'cot'))
rows.append(('余切割长 vs Nanson（45°条纹）', (t1['stripes45']['L_net'][1] / t1['stripes45']['L_net'][0]) / t1['stripes45']['L_pred_ratio'] - 1, 'cot'))
for ang in ('0', '45', '90'):
    rows.append(('余切 Dirichlet 能量 %s°' % ang, t1b[ang]['cot'] / t1b[ang]['pred_GammaConst'] - 1, 'cot'))
rows.append(('远程接触（参考距离>3h）', 0.0, 'cot'))
y = np.arange(len(rows))[::-1]; v = np.array([abs(r[1]) * 100 for r in rows])
a = ax[0, 0]; a.barh(y, np.maximum(v, 0.05), color=['#c0392b' if r[2] == 'comb' else '#2471a3' for r in rows])
a.set_yticks(y); a.set_yticklabels([r[0] for r in rows], fontsize=9); a.set_xscale('log'); a.set_xlim(0.05, 300)
for yi, vi, r in zip(y, v, rows): a.text(max(vi, 0.05) * 1.15, yi, ('%+.1f%%' % (r[1] * 100)) if r[0] != '远程接触（参考距离>3h）' else '0 个', va='center', fontsize=8)
a.axvline(5, color='k', lw=0.6, ls=':')
a.set_xlabel('相对精确响应的偏差 |%|（Sod 精确运动学，κ=0，激波扫过前后）')
a.set_title('(a) 压缩滤网：红 = 组合（无度量）量，蓝 = 余切电导 + 质量量')
# (b) series
a = ax[0, 1]
a.plot(t, A['lnL'], 'k', label='接触层：余切割长 ln L/L₀')
a.plot(t, A['lnP_Gconst'], 'C0', label='交换势 ln P（Γ = 常数，α=1）')
a.plot(t, A['lnP_rhokappa'], 'C3', label='交换势 ln P（ρκ，α=0，DNS 密度）')
a.axhline(0, color='grey', lw=0.5); a.set_xlabel('t'); a.legend(fontsize=9); a.set_title('(b) 激波–重气泡：网络接触与交换势的时间历程')
# (c) mixed fraction
a = ax[0, 2]
for i, G in enumerate(sorted(sw, key=float)):
    a.plot(t, sw[G]['mix_flow'], color='C%d' % i, label='Γ=%s 有流动' % G)
    a.plot(t, sw[G]['mix_frozen'], color='C%d' % i, ls='--', lw=1)
a.plot(t, A['mix_DNS_Y1'], color='grey', ls=':', lw=2, label='DNS 标量（只有数值掺混）')
a.set_xlabel('t'); a.set_ylabel('混合质量分数 Σm·4c(1−c) / Σm·c₀'); a.legend(fontsize=8); a.set_title('(c) 马尔可夫时间（Pe）扫描：实线有流动，虚线冻结网络')
# (d) enhancement
a = ax[1, 0]
for i, G in enumerate(sorted(sw, key=float)):
    a.plot(t[5:], np.array(sw[G]['mix_flow'][5:]) / np.array(sw[G]['mix_frozen'][5:]), color='C%d' % i, label='Γ=%s' % G)
for tt, lo, hi in ((0.6, 1.040, 1.051), (1.4, 1.337, 1.420)):
    a.plot([tt, tt], [lo, hi], 'k-', lw=4, alpha=0.5)
a.plot([], [], 'k-', lw=4, alpha=0.5, label='网络掺混时钟预测（高 Pe 极限）')
a.set_xlabel('t'); a.set_ylabel('掺混增强 = 有流动 / 冻结'); a.legend(fontsize=8); a.set_title('(d) 流动对掺混的增强：由余切电导的时间积分预测')
# (e) killed walk
a = ax[1, 1]
wins = ['0-0.6', '0.6-1.0', '1.0-1.4', '0-1.4']; lz = sp_['frozen']['lam'][0]; hz = sp_['frozen']['mfpt_mean']
x = np.arange(len(wins))
a.bar(x - 0.2, [sp_[w]['lam'][0] / lz for w in wins], 0.4, label='λ₁ / λ₁冻结（最慢未混模态衰减率）')
a.bar(x + 0.2, [hz / sp_[w]['mfpt_mean'] for w in wins], 0.4, label='T冻结 / T（平均首达时间）')
a.axhline(1, color='k', lw=0.6); a.set_xticks(x); a.set_xticklabels(['[0,0.6]', '[0.6,1.0]', '[1.0,1.4]', '[0,1.4]'])
a.text(0.02, 0.95, '冻结网络 λ₁ = %.1f，解析圆盘 %.1f（−1.6%%）' % (lz, sp_['analytic_frozen_lam1']), transform=a.transAxes, fontsize=9, va='top')
a.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0, 0.9)); a.set_title('(e) 吸收随机游走：重气体逃到轻气体的时间尺度（时均生成元）')
# (f) which network quantity explains where mixing was enhanced
a = ax[1, 2]
keys = [('dist', '到界面距离（对照）'), ('net_clock', '网络掺混时钟'), ('afrc_barrier', 'AFRC 屏障强度'), ('far_cross', '跨物质远程接触'), ('raw_contacts', '原始接触数（不扣仿射）')]
x = np.arange(len(keys))
a.bar(x - 0.2, [c5['30'][k] for k, _ in keys], 0.4, label='t = 0.6'); a.bar(x + 0.2, [c5['70'][k] for k, _ in keys], 0.4, label='t = 1.4')
a.axhline(0, color='k', lw=0.6); a.set_xticks(x); a.set_xticklabels([n for _, n in keys], rotation=20, fontsize=9)
a.set_ylabel('Spearman ρ（与掺混增量，界面带 |φ₀|<0.03）'); a.legend(fontsize=9); a.set_title('(f) 哪个网络量解释"流动在哪里加速了掺混"')
fig.savefig('out/图2_定量检验.png', dpi=170); print('ok')
