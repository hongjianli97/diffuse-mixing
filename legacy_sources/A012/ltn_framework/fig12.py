import numpy as np, json, sys
sys.path.insert(0, '/home/claude/w'); import cjk
import matplotlib.pyplot as plt
R = json.load(open('res_ltn_sb.json')); F = R['frames']; ks = sorted(F, key=lambda k: F[k]['t']); t = [F[k]['t'] for k in ks]
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
a = ax[0]; x = np.arange(len(ks)); wd = 0.25
a.bar(x - wd, [F[k]['mu_E'] / F[k]['old_mu_E'] for k in ks], wd, label='μ_E')
a.bar(x, [F[k]['p_relrms_smooth'] / F[k]['old_p_relrms_smooth'] for k in ks], wd, label='压力误差')
a.bar(x + wd, [(F[k]['Phi_tpm_smooth'] / F[k]['Phi_dns_smooth']) / (F[k]['old_Phi_tpm_smooth'] / F[k]['old_Phi_dns_smooth']) if 'old_Phi_tpm_smooth' in F[k] else np.nan for k in ks], wd, label='Φ / DNS')
a.axhline(1, color='k', lw=0.8); a.set_ylim(0.9, 1.1); a.set_xticks(x); a.set_xticklabels(['t=%.2f' % v for v in t])
a.set_ylabel('LTN 统一实现 / 此前各脚本'); a.legend(fontsize=9); a.set_title('(a) 回归：所有通道改由同一个 LTN 计算')
a = ax[1]
a.plot(t, [F[k]['Phi_dns_smooth'] * 1e3 for k in ks], 'k--', lw=2, label='DNS ∫Φ（参照）')
a.plot(t, [F[k]['Phi_tpm_smooth'] * 1e3 for k in ks], 'o-', label='几何激活：K(t) 面元三点映射')
a.plot(t, [F[k]['Phi_pair_smooth'] * 1e3 for k in ks], 's-', label='动力学激活：G 中成对伸长率拟合应变')
for k, tt in zip(ks, t): a.annotate('%.3f' % F[k]['strain_corr_pair_vs_tpm'], (tt, F[k]['Phi_pair_smooth'] * 1e3), textcoords='offset points', xytext=(4, -12), fontsize=8)
a.set_xlabel('t'); a.set_ylabel('×10⁻³'); a.legend(fontsize=9); a.set_title('(b) 耗散的两条 LTN 途径（数字：两者应变张量的逐点相关）')
a = ax[2]
a.plot(t, [F[k]['births_per_node_per_time'] for k in ks], 'o-', label='每节点每单位时间新激活的边'); a2 = a.twinx()
a2.plot(t, [F[k]['births_far_frac'] for k in ks], 's--', color='C3', label='其中参考距离 > 3h 的比例（新接触）')
a.set_xlabel('t'); a.set_ylabel('边出生率'); a2.set_ylabel('新接触比例', color='C3'); a2.set_ylim(0, 0.6)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, fontsize=9, loc='upper left')
a.set_title('(c) G 的激活统计：早期是仿射翻转，后期是搅拌造成的新接触')
fig.savefig('/home/claude/w/out/图12_统一建立在LTN之上.png', dpi=160); print('ok')
