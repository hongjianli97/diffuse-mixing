import numpy as np, json, sys, pickle
sys.path.insert(0, '.'); from rm_take import cross
from cutlam import proj_label, cut_bins
from marker_ref import marker_metrics, ref_curve
case = sys.argv[1]; dbin = 0.0125
r = json.load(open('res_%s_G0.0001_ser.json' % case)); b = np.load('base_%s.npz' % case); net = pickle.load(open('net_%s.pkl' % case, 'rb'))
N = int(b['N']); sel = b['sel']; fr = b['fr'][sel]; t = b['t'][sel]; lab = b['phi0'] > 0; ylab = proj_label(fr[0])
mk = pickle.load(open('marker_%s.pkl' % case, 'rb'))
r0 = ref_curve(np.arange(0, 0.5 + dbin / 2, dbin)); ds0 = np.linalg.norm(np.diff(r0, axis=0), axis=1); b0 = cut_bins(net, fr, N, lab, ylab, 0, dbin)
cutlam2, mklam2, mklnL, mkmlnl, cutmlnl = [], [], [], [], []
for i in range(len(t)):
    ln_ = cut_bins(net, fr, N, lab, ylab, i, dbin) / b0; mm = marker_metrics(*mk[sel[i]], dbin=dbin)
    cutlam2.append(np.log((ln_ ** 2 * ds0).sum() / ds0.sum())); mklam2.append(np.log((mm['lam'] ** 2 * ds0).sum() / ds0.sum()))
    mklnL.append(mm['lnL']); mkmlnl.append(mm['mean_lnlam']); cutmlnl.append(float((np.log(np.maximum(ln_, 1e-9)) * ds0).sum() / ds0.sum()))
s = r['series']; comp = np.array(s['logmean_comp']); C1, C0 = np.array(s['lnC1']), np.array(s['lnC0'])
res = dict(t=t.tolist(), cut_lnlam2=cutlam2, marker_lnlam2=mklam2, marker_lnL=mklnL, marker_meanlnlam=mkmlnl, cut_meanlnlam=cutmlnl)
tk = {}
tk['几何·界面长度：标记曲线'] = cross(t, mklnL, comp); tk['几何·界面长度：网络割长'] = cross(t, s['lnLnet'], comp)
tk['时钟α=1·界面长度：标记曲线'] = cross(t, 2 * np.array(mklnL), comp); tk['时钟α=1·界面长度：网络割长'] = cross(t, 2 * np.array(s['lnLnet']), comp)
tk['速率α=1·算术<λ²>：标记曲线'] = cross(t, mklam2, C1); tk['速率α=1·算术<λ²>：网络割长分箱'] = cross(t, cutlam2, C1); tk['速率α=1：网络交换势(4h带)'] = cross(t, s['lnS'], C1)
tk['速率α=0·算术<λ²>：标记曲线'] = cross(t, mklam2, C0); tk['速率α=0·算术<λ²>：网络割长分箱'] = cross(t, cutlam2, C0); tk['速率α=0：网络交换势(4h带)'] = cross(t, s['lnS'], C0)
tk['几何·均值<lnλ>：标记曲线'] = cross(t, mkmlnl, comp)
res['takeover'] = tk; json.dump(res, open('res_%s_take.json' % case, 'w'), indent=1)
print(json.dumps(tk, indent=1, ensure_ascii=False))
for tt in (1, 2, 3, 4, 6, 8, 10, 12):
    i = np.argmin(abs(t - tt))
    if abs(t[i] - tt) < 1e-6: print('t=%g  ln<lam2> cut %.2f marker %.2f exch(4h) %.2f | lnL cut %.2f marker %.2f | comp %.2f C1 %.2f C0 %.2f' % (tt, cutlam2[i], mklam2[i], s['lnS'][i], s['lnLnet'][i], mklnL[i], comp[i], C1[i], C0[i]))
