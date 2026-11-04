import numpy as np, json, sys
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/ltn'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd'); import cjk
import matplotlib.pyplot as plt, matplotlib.gridspec as gs
import ltn_dyn as LD, cases as CS, lhdx as X
J = lambda f: json.load(open(f))
lin = J('suite_linear.json'); lo = J('suite_lamb.json')['Lamb-Oseen']; lo2 = J('suite_lamb_0.0025.json')
lo2 = lo2[list(lo2)[0]]; tg = J('suite_tg.json')['Taylor-Green']; vx = J('suite_vortex.json')['vortex']; ac = J('suite_acoustic.json')['acoustic']
sod = J('suite_sod.json')['sod']; lam = J('suite_lamella.json')['lamella']; hm = J('/home/claude/w/harm/res_harm_lo.json')['frames']
mx = lambda k: max(lin[c][k] for c in lin if k in lin[c])
hp = max(abs(hm[k]['period_h_u'] / hm[k]['C_theory'] - 1) for k in hm); ha = max(abs(hm[k]['period_h_a'] / hm[k]['dCdt_theory'] - 1) for k in hm)
rows = ['膨胀率 θ（账本）', '涡量（面元环量）', 'Kelvin dΓ/dt（加速度旋度）', '压力（ρa 梯度势）', '粘度 μ_E', '耗散（三点映射）', '耗散（成对伸长率）',
        '声速 c² / Dirac 前推', '熵（标签 / RH 跳变）', '调和周期（u / a）', '掺混（片层剖面 / 混合质量）', '旋度通道"应为零"']
cols = ['线性流', 'Lamb–Oseen', 'Taylor–Green', '非均熵涡', '声驻波', 'Sod', '片层', 'L–O 环形']
V = np.full((len(rows), len(cols)), np.nan); T = [[''] * len(cols) for _ in rows]
def put(i, j, v, txt=None): V[i, j] = np.log10(max(abs(v), 1e-16)); T[i][j] = txt or ('%.0e' % abs(v) if abs(v) < 1e-3 else '%.2g%%' % (100 * abs(v)))
put(0, 0, mx('theta (ledger)')); put(0, 5, sod['25']['theta in rarefaction'], '%.0f%%（稀疏波）' % (100 * sod['25']['theta in rarefaction']))
put(1, 0, mx('omega (face circulation)')); put(1, 1, lo['omega (face)']); put(1, 2, tg['omega (face)']); put(1, 3, vx['omega (face)'])
put(2, 0, mx('Kelvin dGamma/dt')); put(2, 1, lo['Kelvin dGamma/dt (viscous torque)'], '110%（粘性力矩太弱）')
put(3, 0, mx('pressure (rel. or abs.)')); put(3, 1, lo['pressure (cyclostrophic)']); put(3, 2, tg['pressure']); put(3, 3, vx['pressure (variable rho)'])
put(3, 5, sod['25']['pressure / std'], '%.1f%%（含激波）' % (100 * sod['25']['pressure / std']))
put(4, 1, lo['mu_E / nu - 1']); put(4, 2, tg['mu_E / nu - 1'])
put(5, 0, mx('Phi/mu (three-point)')); put(5, 1, lo['Phi three-point']); put(5, 2, tg['Phi three-point'])
put(6, 0, mx('S (pair strain)')); put(6, 1, lo['Phi pair strain']); put(6, 2, tg['Phi pair strain'])
put(7, 4, max(abs(ac['c^2 - 1']), ac['Dirac forward t=0.1']), 'c² %.2f%% / 前推 %.2f%%' % (100 * abs(ac['c^2 - 1']), 100 * ac['Dirac forward t=0.1']))
put(8, 3, vx['entropy s (label)']); put(8, 5, 1e-15, 'η、Δs 精确'); 
put(9, 7, max(hp, ha), '%.1f%% / %.0f%%' % (100 * hp, 100 * ha))
put(10, 6, max(lam['strain: profile'], lam['compression alpha=1, mass metric: profile']), '%.1f%%–%.1f%%' % (100 * lam['compression alpha=1, mass metric: profile'], 100 * lam['strain: profile']))
put(11, 3, vx['curl share of rho a (exact 0)']); put(11, 4, ac['curl share of u (exact 0)'])
fig = plt.figure(figsize=(19, 12.5), constrained_layout=True); g = gs.GridSpec(2, 4, figure=fig, height_ratios=[1.35, 1])
a = fig.add_subplot(g[0, :]); im = a.imshow(V, cmap='RdYlGn_r', vmin=-14, vmax=0, aspect='auto')
for i in range(len(rows)):
    for j in range(len(cols)):
        if T[i][j]: a.text(j, i, T[i][j], ha='center', va='center', fontsize=9)
a.set_xticks(range(len(cols))); a.set_xticklabels(cols, fontsize=11); a.set_yticks(range(len(rows))); a.set_yticklabels(rows, fontsize=10)
fig.colorbar(im, ax=a, shrink=0.8, label='log₁₀ 相对误差（"应为零"一行为份额）'); a.set_title('(a) 精确解验证矩阵：每格为网络结果相对精确解的误差（空格 = 该算例不检验该通道）', fontsize=12)
# Sod profiles
cs = CS.case_sod(h=0.004, Ly=0.12, S=41); G = LD.LTN(cs['frames'], cs['dt'], cs['rho'][0], k=12, mirror_band=0.12); s = 25; N0 = cs['frames'].shape[1]
ch = G.channels(s); hot = G.bad[:, s][:N0]; x = G.fr[s][:N0, 0]; rl = G.rho(s)[:N0]; pex, rex = cs['p'][s], cs['rho'][s]
pr = ch['p'][:N0]; pr = pr - pr[~hot].mean() + pex[~hot].mean(); sr = 2.5 * (np.log(np.maximum(pr, 1e-6)) - 1.4 * np.log(rl)); se = 2.5 * (np.log(pex) - 1.4 * np.log(rex))
o = np.argsort(x); k = o[::3]
for jj, (yn, ye, lab) in enumerate(((rl, rex, '密度 ρ（账本）'), (pr, pex, '压力 p（ρa 梯度势）'), (sr, se, '熵 s = c_v ln(p/ρ^γ)'))):
    a = fig.add_subplot(g[1, jj]); a.plot(x[k], yn[k], '.', ms=2, alpha=0.5, label='网络（只由位置）'); a.plot(x[o], ye[o], 'k-', lw=1.2, label='精确 Riemann 解')
    a.set_xlabel('x'); a.set_title('(%s) Sod，t = %.2f：%s' % ('bcd'[jj], cs['t'][s], lab), fontsize=10.5); a.legend(fontsize=8.5)
a = fig.add_subplot(g[1, 3]); keys = ['omega (face)', 'pressure (cyclostrophic)', 'mu_E / nu - 1', 'Kelvin dGamma/dt (viscous torque)']
names = ['涡量', '压力', 'μ_E', '粘性力矩']; xx = np.arange(4)
a.bar(xx - 0.2, [abs(lo[q]) for q in keys], 0.4, label='h = 0.005'); a.bar(xx + 0.2, [abs(lo2[q]) for q in keys], 0.4, label='h = 0.0025')
a.set_yscale('log'); a.set_xticks(xx); a.set_xticklabels(names); a.legend(fontsize=9); a.set_title('(e) Lamb–Oseen：示踪点加密的收敛', fontsize=10.5)
fig.savefig('/home/claude/w/out/图15_精确解验证矩阵.png', dpi=150); print('ok')
