import numpy as np, json, sys
sys.argv = [sys.argv[0], 'visc']
src = open('visc_hodge.py').read(); exec(src[:src.index('res = {}; maps = {}')])
sys.path.insert(0, '/home/claude/w'); import cjk
import matplotlib.pyplot as plt
G = 1.4; K = MU_TRUE * G / ((G - 1) * 0.72)
z = np.load('entropy_visc.npz'); pr, rl = z['pr'], z['rl']; GC = json.load(open('res_gcv.json')); CR = json.load(open('res_cross_dirichlet.json'))
rows = []
for s in (20, 30, 40, 50, 60):
    P, tri, A, E, w, B1, B2, Ast = ops(s); orgE = (E < N0).all(1)
    T = np.r_[pr[s] / rl[s], pr[s] / rl[s]]; Td = np.r_[Ed['p'][s] / Ed['rho'][s]] ; Td = np.r_[Td, Td]
    L0 = (B1 @ sp.diags(w) @ B1.T).tocsc(); Mm = sp.diags(np.maximum(Ast, 1e-12))
    D = lambda TT: float(K * (w * (np.log(TT[E[:, 1]]) - np.log(TT[E[:, 0]])) ** 2)[orgE].sum())
    sm = lambda l: spsolve((Mm + l ** 2 * L0).tocsc(), Mm @ T)
    lg = GC['frames'][str(s)]['ell_gcv']; d_dns = D(Td)
    rows.append(dict(t=s * dt, raw=D(T) / d_dns, gcv=D(sm(lg)) / d_dns, l2h=D(sm(0.01)) / d_dns, cross=CR[str(s)]['D_cross_all'] / CR[str(s)]['D_dns_all'], rawA=CR[str(s)]['D_raw_all'] / CR[str(s)]['D_dns_all']))
    print(rows[-1], flush=True)
# Lamb-Oseen harmonic results
H = json.load(open('/home/claude/w/harm/res_harm_lo.json'))['frames']; ks = sorted(H, key=lambda k: H[k]['t']); th = np.array([H[k]['t'] for k in ks])
hs = open('/home/claude/w/harm/harm_lo.py').read(); hs = hs[:hs.index('res = dict(N=')]; exec(hs)
fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
a = ax[0, 0]; x = np.arange(len(rows)); wd = 0.18
for i, (k, lab) in enumerate((('raw', '原始（全体示踪点）'), ('gcv', 'GCV 选 ℓ'), ('cross', '分样交叉估计（无平滑）'), ('l2h', '固定 ℓ = 2h'))):
    a.bar(x + (i - 1.5) * wd, [r[k] for r in rows], wd, label=lab)
a.axhline(1, color='k', lw=1); a.set_xticks(x); a.set_xticklabels(['t=%.1f' % r['t'] for r in rows]); a.set_ylabel('导热熵产生 / DNS 温度给出的同一型')
a.legend(fontsize=8.5); a.set_title('(a) 正定型的噪声偏置：四种处理')
a = ax[0, 1]; ells = np.array(GC['ells'])
for s, c in (('30', 'C0'), ('50', 'C1')):
    gg = np.array(GC['frames'][s]['gcv']); a.plot(ells, gg / gg.min(), 'o-', color=c, label='GCV，t=%.1f' % GC['frames'][s]['t'])
    pp = np.array(GC['frames'][s]['preds']); a2 = a.twinx() if s == '30' else a2
    a2.plot(ells, pp / GC['frames'][s]['dns'], 's--', color=c, alpha=0.6)
a2.axhline(1, color='grey', lw=0.6, ls=':'); a2.set_ylabel('重气体熵变率：预测 / DNS（虚线）'); a.set_xscale('log'); a.set_xlabel('低通长度 ℓ'); a.set_ylabel('GCV / min')
a.axvline(0.01, color='k', lw=0.6, ls='--'); a.legend(fontsize=9, loc='upper center'); a.set_title('(b) GCV 选得过小（噪声不白、信号近间断）')
a = ax[0, 2]
Mh = np.load('/home/claude/w/harm/maps_harm.npz'); E = Mh['E']; uh = Mh['uh']; P_ = fr[nF - 2]; mid = 0.5 * (P_[E[:, 0]] + P_[E[:, 1]]); ln = np.linalg.norm(P_[E[:, 1]] - P_[E[:, 0]], axis=1)
tang = (np.c_[-mid[:, 1], mid[:, 0]] / np.hypot(*mid.T)[:, None] * (P_[E[:, 1]] - P_[E[:, 0]])).sum(1) / ln
sc = a.scatter(mid[:, 0], mid[:, 1], c=uh / ln * np.sign(tang + 1e-12) * np.abs(tang), s=0.3, cmap='viridis'); a.set_aspect('equal')
fig.colorbar(sc, ax=a, shrink=0.8); a.set_xticks([]); a.set_yticks([]); a.set_title('(c) 扩散 Lamb–Oseen 环形区（核为洞，β₁ = 1）：速度余链的调和部分')
a = ax[1, 0]
a.plot(th, [H[k]['period_h_u'] for k in ks], 'o-', ms=3, label='网络：调和部分绕洞周期'); a.plot(th, [H[k]['C_theory'] for k in ks], 'k--', label='理论 C = ∫Γ(r)/r dr / ln(r_out/r_in)')
a.plot(th, [H[k]['circ_inner_u'] for k in ks], 's-', ms=3, label='网络：内缘环量'); a.plot(th, [H[k]['Gamma_inner_theory'] for k in ks], 'k:', label='理论 Γ(r_in)')
a.set_xlabel('t'); a.legend(fontsize=8.5); a.set_title('(d) 调和通道 = 拓扑环量')
a = ax[1, 1]
a.plot(th, [H[k]['period_h_a'] for k in ks], 'o-', ms=3, label='网络：加速度调和部分的绕洞周期'); a.plot(th, [H[k]['dCdt_theory'] for k in ks], 'k--', label='理论 dC/dt（粘性把环量扩散出洞）')
a.plot(th, [H[k]['circ_inner_a'] for k in ks], 's-', ms=3, label='网络：内缘 ∮a·dl'); a.plot(th, [-H[k]['dGamma_inner_dt_theory'] for k in ks], 'k:', label='理论 dΓ(r_in)/dt')
a.set_xlabel('t'); a.legend(fontsize=8.5); a.set_title('(e) 非单连通域的 Kelvin 定理：调和通道的变化率')
a = ax[1, 2]
a.stackplot(th, [H[k]['harm_share_u'] for k in ks], [H[k]['curl_share_u'] for k in ks], [1 - H[k]['harm_share_u'] - H[k]['curl_share_u'] for k in ks], labels=['调和', '旋度', '梯度'], colors=['#74c476', '#fc9272', '#9ecae1'])
a.set_ylim(0, 1); a.set_xlabel('t'); a.legend(fontsize=9, loc='lower right'); a.set_title('(f) 能量份额（随度量变化；周期才是不变量）')
fig.savefig('/home/claude/w/out/图11_噪声偏置与调和通道.png', dpi=160); print('ok')
