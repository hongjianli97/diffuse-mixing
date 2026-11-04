import numpy as np, json, sys
sys.path.insert(0, '/home/claude/w'); import cjk
import matplotlib.pyplot as plt
cases = [('dg', '双环流'), ('lo', 'L–O 涡对'), ('sod', 'Sod'), ('blast', '爆炸波'), ('sv', '激波–涡'), ('sb', '激波–气泡'), ('sbv', '粘性激波–气泡'), ('rm', 'RM'), ('rmv', 'RM μ=5e-4')]
R = {c: json.load(open('res_%s.json' % c)) for c, _ in cases}
rep = {'dg': ((1.3, 5.7), (11.4, 65.3), (32.1, 254.3)), 'lo': ((0.41, 13.0), (6.4, 49.2), (26.3, 167.6)), 'sod': ((1e-4, 0.97), (1.9, 7.1), (0.20, 37.8)),
       'blast': ((0.58, 2.1), (6.8, 38.9), (20.2, 107.2)), 'sv': ((0.34, 3.4), (3.4, 32.2), (16.8, 97.7)), 'sb': ((0.67, 29.7), (7.1, 43.6), (25.9, 141.7)), 'rm': ((0.73, 35.3), (4.6, 40.6), (17.1, 72.8))}
def last(c): k = sorted([k for k in R[c] if k.startswith('s') and k[1:].isdigit()], key=lambda k: int(k[1:])); return R[c][k[-1]]
def dens(c, which):
    r = last(c); key = [k for k in r if k.startswith(('density ' + which, 'J ' + which))][0]; return r[key]
fig, ax = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
for j, (idx, lab) in enumerate(((0, '中位'), (1, '95 分位'))):
    a = ax[0, j]; cs = [c for c, _ in cases if c in rep]; x = np.arange(len(cs)); wd = 0.16
    vals = [[100 * dens(c, 'ledger')[idx] for c in cs], [100 * dens(c, 'DTFE')[idx] for c in cs], [rep[c][0][idx] for c in cs], [rep[c][1][idx] for c in cs], [rep[c][2][idx] for c in cs]]
    labs = ['LTN 账本（本次重算）', 'DTFE（本次，同一示踪点）', '报告：账本', '报告：SPH', '报告：DTFE']; cols = ['C0', 'C3', '#9ecae1', '#fdae6b', '#fc9272']
    for k in range(5): a.bar(x + (k - 2) * wd, np.maximum(vals[k], 1e-4), wd, label=labs[k], color=cols[k])
    a.set_yscale('log'); a.set_xticks(x); a.set_xticklabels([dict(cases)[c] for c in cs]); a.set_ylabel('密度相对误差 %（爆炸波为 J）'); a.legend(fontsize=8.5)
    a.set_title('(%s) 质量通道：密度误差的%s（报告值取自其表 7；爆炸波账本 95%% 取 v4 的 2.1%%）' % ('ab'[j], lab), fontsize=10)
a = ax[1, 0]; mets = [('vorticity corr', '涡量与参照相关'), ('entropy corr', '熵与参照相关'), ('strain corr (pair vs three-point)', '两种激活的应变相关')]
x = np.arange(len(cases)); wd = 0.25
for k, (key, lab) in enumerate(mets):
    a.bar(x + (k - 1) * wd, [last(c).get(key, np.nan) for c, _ in cases], wd, label=lab)
a.set_ylim(0, 1.05); a.set_xticks(x); a.set_xticklabels([n for _, n in cases], fontsize=9); a.legend(fontsize=9, loc='lower left')
for i, (c, _) in enumerate(cases):
    r = last(c); txt = []
    if 'pressure rms/std (off shock)' in r: txt.append('p %.1f%%' % (100 * r['pressure rms/std (off shock)']))
    txt.append('curl %.2g' % r['curl share of rho*a (Euler: ~0)']); txt.append('μ_E %.1e' % r['mu_E'])
    a.text(i, 0.08, '\n'.join(txt), ha='center', fontsize=7.5, color='k')
a.set_title('(c) 动力学通道：涡量、熵、应变一致性；底部为压力误差、ρa 旋度份额（Euler 一致性）与 μ_E（真值：粘性算例 5e-4）', fontsize=9.5)
a = ax[1, 1]; pairs = ['Spearman afrc~ftle', 'Spearman afrc~dbs', 'Spearman ftle~dbs', 'Spearman dbs~enc']; names = ['AFRC~FTLE', 'AFRC~DBS', 'FTLE~DBS', 'DBS~相遇数']
M = np.array([[R[c]['structures'][p] for p in pairs] for c, _ in cases]); im = a.imshow(M.T, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
for i in range(M.shape[0]):
    for k in range(M.shape[1]): a.text(i, k, '%.2f' % M[i, k], ha='center', va='center', fontsize=8.5)
a.set_xticks(range(len(cases))); a.set_xticklabels([n for _, n in cases], fontsize=9); a.set_yticks(range(len(pairs))); a.set_yticklabels(names)
plt.colorbar(im, ax=a, shrink=0.8); a.set_title('(d) 输运结构方法之间的秩相关；LAVD（网络 vs 精确）：双环流 %.4f / Jaccard %.2f，涡对 %.2f / %.2f（报告 T7 动态 Laplacian 涡核 Jaccard 0.82）' % (
    R['dg']['structures']['LAVD network vs exact: Spearman'], R['dg']['structures']['LAVD top-10% Jaccard'], R['lo']['structures']['LAVD network vs exact: Spearman'], R['lo']['structures']['LAVD top-10% Jaccard']), fontsize=9)
fig.suptitle('图 21  报告九个算例的重算：LTN 通道与报告结果、主流方法的定量对比', fontsize=13)
fig.savefig('/home/claude/w/out/图21_报告算例_定量对比.png', dpi=140); print('ok')
