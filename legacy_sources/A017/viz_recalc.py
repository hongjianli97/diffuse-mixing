import numpy as np, json, sys
sys.path.insert(0, '/home/claude/w'); import cjk
import matplotlib.pyplot as plt
cases = [('dg', '双环流'), ('lo', 'Lamb–Oseen 涡对'), ('sod', 'Sod'), ('blast', '柱面爆炸波'), ('sv', '激波–等熵涡'), ('sb', '激波–重气泡'), ('sbv', '粘性激波–重气泡'), ('rm', 'RM（无粘）'), ('rmv', 'RM（μ=5e-4）')]
F = {c: np.load('fields_%s.npz' % c) for c, _ in cases}; R = {c: json.load(open('res_%s.json' % c)) for c, _ in cases}
from matplotlib.tri import Triangulation
_tri_cache = {}
def sc(ax, P, v, cmap, lim=None, title='', sym=False):
    v = np.asarray(v, float); ok = np.isfinite(v); v = np.where(ok, v, np.nanmedian(v))
    if lim is None:
        lo, hi = np.percentile(v[ok], [1, 99]); lim = (-max(abs(lo), abs(hi)), max(abs(lo), abs(hi))) if sym else (lo, hi)
    key = (P.shape, float(P[0, 0]), float(P[-1, 1]))
    if key not in _tri_cache:
        tr = Triangulation(P[:, 0], P[:, 1]); t = tr.triangles; L = np.max(np.linalg.norm(P[t] - P[t][:, [1, 2, 0]], axis=2), 1)
        tr.set_mask(L > 4 * np.median(L)); _tri_cache[key] = tr
    ax.tripcolor(_tri_cache[key], v, cmap=cmap, vmin=lim[0], vmax=lim[1], shading='gouraud', rasterized=True)
    span = np.ptp(P, 0); ax.set_aspect('equal' if span[1] / span[0] > 0.3 else 'auto')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8.5)
def lastS(c): return max(int(k.split('_')[1]) for k in F[c].files if k.startswith('om_'))
# ---------------- figure 19: channels
fig, ax = plt.subplots(len(cases), 5, figsize=(20, 3.0 * len(cases)), constrained_layout=True)
for i, (c, name) in enumerate(cases):
    f = F[c]; s = lastS(c); P = f['P_%d' % s]; r = R[c]['s%d' % s]
    lo, hi = np.percentile(f['rho_%d' % s], [1, 99])
    sc(ax[i, 0], P, f['rho_%d' % s], 'viridis', (lo, hi), '%s  t=%.3g：密度（账本）' % (name, r['t']))
    sc(ax[i, 1], P, f['dtfe_%d' % s], 'viridis', (lo, hi), '密度（DTFE，同一组示踪点）')
    sc(ax[i, 2], P, f['om_%d' % s], 'RdBu_r', None, '涡量（面元环量）' + ('，与 DNS 相关 %.3f' % r['vorticity corr'] if 'vorticity corr' in r else ''), sym=True)
    if 'p_%d' % s in f.files:
        sc(ax[i, 3], P, f['p_%d' % s], 'magma', None, '压力（ρa 梯度势），误差 %.1f%%' % (100 * r['pressure rms/std (off shock)']))
        sc(ax[i, 4], P, f['s_%d' % s], 'cividis', None, '熵 s（网络），与参照相关 %.2f' % r['entropy corr'])
    else:
        ax[i, 3].text(0.5, 0.5, '无参考密度/压力：\n只评运动学通道\nρa 旋度份额 %.2f（%s）' % (r['curl share of rho*a (Euler: ~0)'], '非 Euler 运动学模型' if r['curl share of rho*a (Euler: ~0)'] > 0.05 else '≈Euler'), ha='center', va='center', fontsize=9, transform=ax[i, 3].transAxes); ax[i, 3].axis('off')
        sc(ax[i, 4], F[c]['P0'], f['lavd_net'], 'inferno', None, 'LAVD（由网络面元涡量，初始位置）')
fig.suptitle('图 19  LHD 报告的九个算例：LTN 各通道场图（只由位置 + 参考密度；与 DTFE 并列）', fontsize=13)
fig.savefig('/home/claude/w/out/图19_报告算例_通道场图.png', dpi=110)
# ---------------- figure 20: transport structures (initial positions)
fig, ax = plt.subplots(len(cases), 6, figsize=(24, 3.0 * len(cases)), constrained_layout=True)
for i, (c, name) in enumerate(cases):
    f = F[c]; P = f['P0']; st = R[c]['structures']
    sc(ax[i, 0], P, f['ftle'], 'magma', None, '%s：FTLE' % name)
    sc(ax[i, 1], P, np.log(np.maximum(f['dbs'], 1e-12)), 'magma', None, 'DBS（ln，最小二乘 F）')
    sc(ax[i, 2], P, f['enc'], 'magma', None, '相遇数（Rypina–Pratt）')
    sc(ax[i, 3], P, f['afrc'], 'inferno', None, 'AFRC 屏障强度（LTN 动力学激活）\n与 FTLE 秩相关 %.2f' % st['Spearman afrc~ftle'])
    sc(ax[i, 4], P, f['lavd_net'], 'inferno', None, 'LAVD（LTN 面元涡量）')
    if 'lavd_ex' in f.files: sc(ax[i, 5], P, f['lavd_ex'], 'inferno', None, 'LAVD 精确；秩相关 %.3f，前 10%% Jaccard %.2f' % (st['LAVD network vs exact: Spearman'], st['LAVD top-10% Jaccard']))
    else: ax[i, 5].axis('off')
fig.suptitle('图 20  输运结构：主流方法（FTLE、DBS、相遇数）与 LTN 通道（AFRC、LAVD），同一组示踪点、同一时间窗', fontsize=13)
fig.savefig('/home/claude/w/out/图20_报告算例_输运结构对比.png', dpi=100)
print('ok')
