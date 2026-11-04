import numpy as np, sys, json
sys.path.insert(0, '/home/claude/w'); import cjk
import matplotlib.pyplot as plt
T = {c: json.load(open('res_%s_take.json' % c)) for c in ('single', 'reshock')}
R = {c: json.load(open('res_%s_G0.0001_ser.json' % c)) for c in ('single', 'reshock')}
MT = json.load(open('res_mixtake.json'))
doc = {'single': {'geomL': 3.43, 'clockL': 2.19, 'mm1': 3.6, 'mm0': 5.3, 'geomlog': 5.0},
       'reshock': {'geomL': 8.45, 'clockL': 5.9, 'mm1': 8.6, 'mm0': None, 'geomlog': None}}
fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
for j, c in enumerate(('single', 'reshock')):
    a = ax[0, j]; t = np.array(T[c]['t']); s = R[c]['series']
    a.plot(t, T[c]['marker_lnlam2'], 'k', lw=2, label='标记曲线 ln⟨λ²⟩（参照）')
    a.plot(t, T[c]['cut_lnlam2'], 'C0', label='网络：尖锐标签割长分箱')
    a.plot(t, s['lnS'], 'C1', ls='--', label='网络：4h 带交换势（信息极限下饱和）')
    a.plot(t, s['lnC1'], 'C4', label='压缩因子 ln⟨1/J⟩（α=1）'); a.plot(t, s['lnC0'], 'C4', ls=':', label='压缩因子 ln⟨1/J²⟩（α=0）')
    a.set_xlabel('t'); a.legend(fontsize=8.5); a.set_title('(%s) %s：掺混速率的拉伸因子与压缩因子（算术平均）' % ('ab'[j], '单次激波' if c == 'single' else '再激波'))
a = ax[0, 2]
for c, ls in (('single', '-'), ('reshock', '--')):
    t = np.array(T[c]['t']); s = R[c]['series']
    a.plot(t, T[c]['marker_lnL'], 'k', ls=ls, lw=2, label='标记曲线 ln L/L₀（%s）' % ('单次' if c == 'single' else '再激波'))
    a.plot(t, s['lnLnet'], 'C0', ls=ls, label='网络割长（%s）' % ('单次' if c == 'single' else '再激波'))
    a.plot(t, s['logmean_comp'], 'C4', ls=ls, lw=1, label='⟨−ln J⟩ 账本（%s）' % ('单次' if c == 'single' else '再激波'))
a.set_xlabel('t'); a.legend(fontsize=8); a.set_title('(c) 界面长度：网络割长 vs 标记曲线')
rows = [('几何：ln L > ⟨−ln J⟩', '几何·界面长度：标记曲线', '几何·界面长度：网络割长', None, 'geomL'),
        ('时钟 α=1：2 ln L > ⟨−ln J⟩', '时钟α=1·界面长度：标记曲线', '时钟α=1·界面长度：网络割长', None, 'clockL'),
        ('速率 α=1：⟨λ²⟩ > ⟨1/J⟩', '速率α=1·算术<λ²>：标记曲线', '速率α=1·算术<λ²>：网络割长分箱', '速率α=1：网络交换势(4h带)', None),
        ('速率 α=0：⟨λ²⟩ > ⟨1/J²⟩', '速率α=0·算术<λ²>：标记曲线', '速率α=0·算术<λ²>：网络割长分箱', '速率α=0：网络交换势(4h带)', None),
        ('混合质量 α=1（网络反事实，Γ=3e-5）', None, 'MM1', None, 'mm1'),
        ('几何：⟨ln λ⟩ > ⟨−ln J⟩', '几何·均值<lnλ>：标记曲线', None, None, 'geomlog')]
for j, c in enumerate(('single', 'reshock')):
    a = ax[1, j]; tk = T[c]['takeover']; tend = 6 if c == 'single' else 12
    for i, (lbl, km, kn, ke, kd) in enumerate(rows):
        y = len(rows) - 1 - i
        vm = tk.get(km) if km else None
        vn = (MT['%s_G3e-05' % c]['a1'] if kn == 'MM1' else tk.get(kn)) if kn else None
        ve = tk.get(ke) if ke else None; vd = doc[c].get(kd) if kd else None
        for v, mkr, col, off, lab in ((vm, 'o', 'k', 0.15, '标记曲线'), (vn, 's', 'C0', 0.0, '网络'), (ve, 'D', 'C1', -0.15, '网络交换势(4h)'), (vd, 'x', 'grey', -0.3, '掺混文档')):
            if v is not None: a.plot(v, y + off, mkr, color=col, ms=8, mfc='none' if mkr in 'oD' else col, label=lab if i == 0 or (mkr == 'D' and i == 2) else None)
            elif (mkr == 'o' and km) or (mkr == 's' and kn): a.annotate('> %g' % tend, (tend, y + off), fontsize=8, color=col, va='center')
    a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows][::-1], fontsize=9); a.set_xlim(0, tend + 1.2); a.grid(axis='x', alpha=0.3)
    h, l = a.get_legend_handles_labels(); a.legend(h, l, fontsize=8.5, loc='upper left'); a.set_xlabel('拉伸接管时刻 t')
    a.set_title('(%s) %s：各判据的接管时刻' % ('de'[j], '单次激波（t ≤ 6）' if c == 'single' else '再激波（t ≤ 12）'))
a = ax[1, 2]
for c, col in (('single', 'C0'), ('reshock', 'C3')):
    G = [3e-5, 1e-4, 3e-4]; tt = [MT['%s_G%g' % (c, g)]['a1'] for g in G]
    tend = 6 if c == 'single' else 12
    a.plot([g for g, v in zip(G, tt) if v], [v for v in tt if v], 'o-', color=col, label='网络反事实扩散（%s）' % ('单次' if c == 'single' else '再激波'))
    for g, v in zip(G, tt):
        if v is None: a.plot(g, tend, '^', color=col, ms=9); a.annotate('t=%g 前未接管' % tend, (g, tend), fontsize=8, xytext=(-60, 6), textcoords='offset points')
    a.axhline(doc[c]['mm1'], color=col, ls=':', lw=1.2); a.text(1.1e-5, doc[c]['mm1'] + 0.15, '文档亚分辨率片层极限 %.1f' % doc[c]['mm1'], color=col, fontsize=8.5)
a.set_xscale('log'); a.set_xlim(1e-5, 6e-4); a.set_xlabel('Γ（马尔可夫时间尺度）'); a.set_ylabel('混合质量判据 α=1 的接管时刻')
a.legend(fontsize=9, loc='center right'); a.set_title('(f) 混合质量接管随 Pe 的变化：Γ→0 趋向片层极限')
fig.savefig('/home/claude/w/out/图5_RM接管判据_网络与参照.png', dpi=170); print('ok')
