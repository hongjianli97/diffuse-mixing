import numpy as np, json, sys
sys.path.insert(0, '/home/claude/w'); import cjk
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
F = {c: np.load('fields_%s.npz' % c) for c in ('lamb', 'tg', 'vortex', 'acoustic', 'lamella')}
def fmap(ax, P, tri, v, cmap, lim, title, xl=None, yl=None):
    tr = Triangulation(P[:, 0], P[:, 1], tri)
    if len(v) == len(tri): ax.tripcolor(tr, facecolors=v, cmap=cmap, vmin=lim[0], vmax=lim[1], rasterized=True)
    else: ax.tripcolor(tr, v, cmap=cmap, vmin=lim[0], vmax=lim[1], shading='gouraud', rasterized=True)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9.5)
    if xl: ax.set_xlim(*xl)
    if yl: ax.set_ylim(*yl)
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(*lim)), ax=ax, shrink=0.75, pad=0.01)
def triple(axs, P, tri, ex, net, cmap, name, xl=None, yl=None, nodes=None):
    lo, hi = np.nanpercentile(ex, 0.5), np.nanpercentile(ex, 99.5); err = net - ex; el = np.nanpercentile(np.abs(err), 99) or 1e-12
    fmap(axs[0], P, tri, ex, cmap, (lo, hi), name + '：精确', xl, yl); fmap(axs[1], P, tri, net, cmap, (lo, hi), name + '：网络（只由位置）', xl, yl)
    fmap(axs[2], P, tri, err, 'RdBu_r', (-el, el), name + '：误差（网络 − 精确）', xl, yl)
# ---------------- figure 16
fig, ax = plt.subplots(4, 6, figsize=(26, 16), constrained_layout=True)
L = F['lamb']; z = (-0.16, 0.16)
triple(ax[0, :3], L['P'], L['tri'], L['om_ex'], L['om_net'], 'viridis', 'Lamb–Oseen 涡量', z, z)
pin = np.where(L['inN'], 1, 0); triple(ax[0, 3:], L['P'], L['tri'], L['p_ex'], L['p_net'], 'magma', 'Lamb–Oseen 压力', z, z)
T = F['tg']; Pc = T['Pc']; from scipy.spatial import Delaunay
triple(ax[1, :3], T['P'], T['tri'], T['om_ex'], T['om_net'], 'RdBu_r', 'Taylor–Green 涡量', (0, 2 * np.pi), (0, 2 * np.pi))
tc = Delaunay(Pc).simplices; triple(ax[1, 3:], Pc, tc, T['p_ex'], T['p_net'], 'magma', 'Taylor–Green 压力', (0, 2 * np.pi), (0, 2 * np.pi))
V = F['vortex']; cxy = V['P'][V['inN']].mean(0); zz = lambda c: (c - 3, c + 3)
triple(ax[2, :3], V['P'], V['tri'], V['om_ex'], V['om_net'], 'RdBu_r', '非均熵涡 涡量', zz(cxy[0]), zz(cxy[1]))
m = V['inN']; Pv = V['P'][m]; tv = Delaunay(Pv).simplices
triple(ax[2, 3:], Pv, tv, V['p_ex'][m], V['p_net'][m], 'magma', '非均熵涡 压力（变密度）', zz(cxy[0]), zz(cxy[1]))
triple(ax[3, :3], Pv, tv, V['s_ex'][m] - V['s_ex'][m].mean(), V['s_net'][m] - V['s_net'][m].mean(), 'cividis', '非均熵涡 熵 s', zz(cxy[0]), zz(cxy[1]))
triple(ax[3, 3:], T['P'], T['tri'], T['phi_ex'], T['phi_net'], 'inferno', 'Taylor–Green 耗散密度 Φ（三点映射）', (0, 2 * np.pi), (0, 2 * np.pi))
fig.suptitle('图 16  精确解场图 I：涡量、压力、熵与耗散（每组：精确 / 网络 / 误差）', fontsize=14)
fig.savefig('/home/claude/w/out/图16_精确解场图_涡压力熵耗散.png', dpi=120)
# ---------------- figure 17
fig, ax = plt.subplots(3, 4, figsize=(24, 15), constrained_layout=True)
A = F['acoustic']; lim = (-1e-3, 1e-3)
fmap(ax[0, 0], A['P'], A['tri'], A['rho_ex'], 'RdBu_r', lim, "声驻波 t=0.1：ρ' 精确"); fmap(ax[0, 1], A['P'], A['tri'], A['rho_led'], 'RdBu_r', lim, "ρ' 账本（只由位置）")
fmap(ax[0, 2], A['P'], A['tri'], A['rho_dirac'], 'RdBu_r', lim, "ρ' Dirac 前推（初值 + c² + t=0 复形）")
e = A['rho_led'] - A['rho_ex']; el = np.percentile(np.abs(e), 99); fmap(ax[0, 3], A['P'], A['tri'], e, 'RdBu_r', (-el, el), '账本误差（注意色标为 10⁻⁵ 量级）')
Lm = F['lamella']; Ps = Lm['strain_P']; ts = Lm['strain_tri']
fmap(ax[1, 0], Ps, ts, Lm['strain_cex'], 'viridis', (0, 1), '纯应变片层 t=1：c 精确', (-0.12, 0.12), (-0.4, 0.4))
fmap(ax[1, 1], Ps, ts, Lm['strain_c'], 'viridis', (0, 1), 'c 网络扩散（质量度量）', (-0.12, 0.12), (-0.4, 0.4))
e = Lm['strain_c'] - Lm['strain_cex']; el = np.percentile(np.abs(e), 99.5); fmap(ax[1, 2], Ps, ts, e, 'RdBu_r', (-el, el), '误差', (-0.12, 0.12), (-0.4, 0.4))
a = ax[1, 3]; o = np.argsort(Lm['comp_X']); X0 = Lm['comp_X'][o]
a.plot(X0, Lm['comp_ce1'][o], 'k-', lw=2.5, alpha=0.5, label='精确 α=1'); a.plot(X0, Lm['comp_c1'][o], '.', ms=1.5, label='网络，质量度量，α=1')
a.plot(X0, Lm['comp_ca'][o], '.', ms=1.5, label='网络，面积度量（错误）'); a.plot(X0, Lm['comp_ce0'][o], 'k--', lw=1.5, label='精确 α=0'); a.plot(X0, Lm['comp_c0'][o], '.', ms=1.5, label='网络，质量度量，α=0')
a.set_xlim(-0.08, 0.08); a.set_xlabel('参考坐标 X'); a.set_ylabel('c'); a.legend(fontsize=8.5); a.set_title('一维压缩下的片层（t=1）：两种本构与两种度量', fontsize=9.5)
I = np.load('suite2_igw.npz'); Pf = np.c_[I['igw__cx'], I['igw__cy']]; tf_ = Delaunay(Pf).simplices
lim = np.percentile(np.abs(I['igw__tq_ex']), 99.5)
for k, (v, tt) in enumerate(((I['igw__tq_ex'], '内重力波：斜压力矩 −N²∂ζ/∂x 精确'), (I['igw__tq'], '网络：加速度余链的旋度（Kelvin）'))):
    fmap(ax[2, k], Pf, tf_, v, 'RdBu_r', (-lim, lim), tt)
a = ax[2, 2]; kk = np.random.default_rng(0).choice(len(I['igw__tq']), 5000, replace=False)
a.plot(I['igw__tq_ex'][kk], I['igw__tq'][kk], '.', ms=1.5, alpha=0.5); xx = np.array([-lim, lim]); a.plot(xx, xx, 'k--'); a.set_xlabel('精确力矩'); a.set_ylabel('网络力矩'); a.set_title('逐面元比较（斜率即 N² 反演，误差 0.08%）', fontsize=9.5)
S2 = np.load('suite2_stokes.npz'); a = ax[2, 3]
for key, mk, lab in (('free wall', 'o', '墙作自由边界'), ('wall extension', 's', '墙外延拓（像轨迹）')):
    y = S2[key + '__y']; o = np.argsort(y); bins = np.linspace(0, 0.2, 41); bi = np.digitize(y, bins)
    a.plot([y[bi == q].mean() for q in range(1, 41) if (bi == q).any()], [S2[key + '__om'][bi == q].mean() for q in range(1, 41) if (bi == q).any()], mk, ms=4, mfc='none', label=lab)
y = S2['wall extension__y']; o = np.argsort(y); a.plot(y[o], S2['wall extension__omex'][o], 'k-', lw=1.5, label='精确')
a.set_xlim(0, 0.2); a.set_xlabel('离墙距离 y'); a.set_ylabel('ω'); a.legend(fontsize=9); a.set_title('Stokes 第一问题：壁面涡量（两种处理都对；μ_E 只有延拓后才对）', fontsize=9.5)
fig.suptitle('图 17  精确解场图 II：声学、掺混、斜压力矩与壁面', fontsize=14)
fig.savefig('/home/claude/w/out/图17_精确解场图_声学掺混重力波壁面.png', dpi=120)
# ---------------- figure 18
C = json.load(open('suite2_conv.json')); Nz = json.load(open('suite2_noise.json'))
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
a = ax[0]; hs = np.array([C[k]['h'] for k in C])
for key, lab, mk in (('omega', '涡量', 'o'), ('pressure', '压力', 's'), ('Phi_tpm', '耗散（三点映射）', '^'), ('Phi_pair', '耗散（成对伸长率）', 'v'), ('mu', 'μ_E', 'D')):
    a.loglog(hs, [C[k][key] for k in C], mk + '-', label=lab)
a.loglog(hs, 0.15 * hs, 'k:', lw=1, label='一阶'); a.loglog(hs, 0.35 * hs ** 2, 'k--', lw=1, label='二阶')
a.set_xlabel('示踪点间距 h'); a.set_ylabel('相对误差'); a.legend(fontsize=9); a.set_title('(a) Taylor–Green：示踪点加密的收敛（n = 40, 80, 120）')
a = ax[1]; sg = np.array([float(k.split('=')[1]) for k in Nz]); sg[sg == 0] = 3e-5
for key, lab, mk in (('omega', '涡量', 'o'), ('pressure', '压力', 's'), ('mu', 'μ_E', 'D'), ('Phi_tpm', '耗散（三点映射）', '^'), ('Phi_pair', '耗散（成对伸长率）', 'v')):
    a.loglog(sg, [Nz[k][key] for k in Nz], mk + '-', label=lab)
a.axhline(0.05, color='grey', lw=0.6, ls=':'); a.set_xlabel('位置噪声 σ/h（最左点为无噪声）'); a.set_ylabel('相对误差'); a.legend(fontsize=9)
a2 = a.secondary_xaxis('top', functions=(lambda x: x * 0.0785 / 0.02 ** 2 / 1.0, lambda y: y / (0.0785 / 0.02 ** 2)))
a2.set_xlabel('N_a = σ/(|a|Δt²)'); a.set_title('(b) Taylor–Green：位置噪声敏感性', pad=28)
fig.suptitle('图 18  收敛阶与噪声阈值', fontsize=13)
fig.savefig('/home/claude/w/out/图18_收敛与噪声.png', dpi=140); print('ok')
