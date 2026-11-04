import numpy as np, json, pickle, sys
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import lhdx as X
G = 1.4; CV = 1 / (G - 1); dt = 0.02; out = {}
def load(case):
    z = np.load('entropy_%s.npz' % case)
    if case == 'visc':
        R = pickle.load(open('sbv_ck.pkl', 'rb'))['R']; fr0 = np.array(R['frames']); pd, rd = np.array(R['p']), np.array(R['rho'])
    else:
        o = pickle.load(open('/home/claude/w/LHD_v3.2/sb_dns_dx0.005.pkl', 'rb')); fr0 = o['frames']; pd, rd = o['E']['p'], o['E']['rho']
    led = np.load('/home/claude/w/hd/ledger_sb.npz' if case == 'inv' else 'ledger_visc.npz')
    return z, fr0, pd, rd, led['lnJ']
res = {}
# ---------------- inviscid: entropy as a material label; conservation; shock jumps
z, fr0, pd, rd, lnJ = load('inv'); S, N0 = pd.shape
s_rec = CV * (np.log(z['pr']) - G * np.log(z['rl'])); s_dns = CV * (np.log(pd) - G * np.log(rd)); hot = z['hot']
heavy = np.hypot(fr0[0][:, 0] - 0.75, fr0[0][:, 1]) < 0.25
for s in (30, 50):
    ok = np.isfinite(s_rec[s]) & ~hot[s]
    res['inv_corr_s_%d' % s] = float(np.corrcoef(s_rec[s][ok], s_dns[s][ok])[0, 1])
    res['inv_heavy_minus_light_rec_%d' % s] = float(np.median(s_rec[s][heavy & ok]) - np.median(s_rec[s][~heavy & ok]))
    res['inv_heavy_minus_light_dns_%d' % s] = float(np.median(s_dns[s][heavy & ok]) - np.median(s_dns[s][~heavy & ok]))
res['contrast_theory'] = float(-CV * G * np.log(3.0))
# conservation over [s1, s2] for tracers never in a shock block in between, away from the interface (|phi0| > 0.02)
phi0 = np.hypot(fr0[0][:, 0] - 0.75, fr0[0][:, 1]) - 0.25
for (s1, s2) in ((20, 40), (40, 60), (50, 69)):
    clean = ~hot[s1:s2 + 1].any(0) & (np.abs(phi0) > 0.02) & np.isfinite(s_rec[s1]) & np.isfinite(s_rec[s2])
    res['inv_rms_ds_rec_%d_%d' % (s1, s2)] = float(np.sqrt(np.mean((s_rec[s2] - s_rec[s1])[clean] ** 2)))
    res['inv_rms_ds_dns_%d_%d' % (s1, s2)] = float(np.sqrt(np.mean((s_dns[s2] - s_dns[s1])[clean] ** 2)))
    res['inv_n_clean_%d_%d' % (s1, s2)] = int(clean.sum())
# shock jumps: first shock block of each tracer
first = np.argmax(hot, axis=0); has = hot.any(0)
sb, sa = first - 3, first + 4; okj = has & (sb >= 1) & (sa <= S - 2)
idx = np.where(okj)[0]
eta = np.exp(lnJ[sb[idx], idx] - lnJ[sa[idx], idx])
p21 = ((G + 1) * eta - (G - 1)) / ((G + 1) - (G - 1) * eta); dsRH = CV * (np.log(p21) - G * np.log(eta))
dsR = s_rec[sa[idx], idx] - s_rec[sb[idx], idx]; dsD = s_dns[sa[idx], idx] - s_dns[sb[idx], idx]
far = np.abs(phi0[idx]) > 0.03; light = ~heavy[idx] & far; hv = heavy[idx] & far
for tag, m in (('light', light), ('heavy', hv)):
    res['jump_%s_n' % tag] = int(m.sum()); res['jump_%s_eta_med' % tag] = float(np.median(eta[m]))
    res['jump_%s_dsRH_med' % tag] = float(np.median(dsRH[m])); res['jump_%s_dsRec_med' % tag] = float(np.median(dsR[m])); res['jump_%s_dsDns_med' % tag] = float(np.median(dsD[m]))
np.savez_compressed('jumps_inv.npz', eta=eta, dsRH=dsRH, dsR=dsR, dsD=dsD, light=light, hv=hv)
# ---------------- viscous: entropy budget
zv, fr0v, pdv, rdv, lnJv = load('visc'); hotv = zv['hot']
s_recv = CV * (np.log(zv['pr']) - G * np.log(zv['rl'])); s_dnsv = CV * (np.log(pdv) - G * np.log(rdv))
tri0 = X.complex_peeled(fr0v[0])['tri']; m = rdv[0] * X._star(fr0v[0], tri0, N0) / 3.0
heavyv = np.hypot(fr0v[0][:, 0] - 0.75, fr0v[0][:, 1]) < 0.25
T = zv['pr'] / zv['rl']; pred = (zv['phiN'] + zv['condN']) / (zv['rl'] * T)
budget = dict(t=[], meas_heavy=[], dns_heavy=[], pred_heavy=[], meas_all=[], prod_net=[], prod_phi=[], prod_cond=[], dns_all=[], cond_heavy=[], phi_heavy=[])
for s in range(3, S - 3):
    clean = ~hotv[s - 1:s + 2].any(0) & np.isfinite(s_recv[s - 1]) & np.isfinite(s_recv[s + 1]) & np.isfinite(pred[s])
    rm = (s_recv[s + 1] - s_recv[s - 1]) / (2 * dt); rd_ = (s_dnsv[s + 1] - s_dnsv[s - 1]) / (2 * dt)
    budget['t'].append(s * dt)
    for tag, sel in (('heavy', heavyv & clean), ('all', clean)):
        budget['meas_' + tag].append(float((m * rm)[sel].sum())); budget['dns_' + tag].append(float((m * rd_)[sel].sum()))
        if tag == 'heavy':
            budget['pred_heavy'].append(float((m * pred[s])[sel].sum()))
            budget['cond_heavy'].append(float((m * zv['condN'][s] / (zv['rl'][s] * T[s]))[sel].sum())); budget['phi_heavy'].append(float((m * zv['phiN'][s] / (zv['rl'][s] * T[s]))[sel].sum()))
    budget['prod_phi'].append(float(zv['prodPhi'][s])); budget['prod_cond'].append(float(zv['prodCond'][s])); budget['prod_net'].append(float(zv['prodPhi'][s] + zv['prodCond'][s]))
res['budget'] = budget
# field correlation of predicted vs measured Ds/Dt at t=1.0 and 1.2 (binned 0.02)
for s in (50, 60):
    clean = ~hotv[s - 1:s + 2].any(0) & np.isfinite(pred[s]) & np.isfinite(s_recv[s + 1]) & np.isfinite(s_recv[s - 1])
    rm = (s_recv[s + 1] - s_recv[s - 1]) / (2 * dt); rd_ = (s_dnsv[s + 1] - s_dnsv[s - 1]) / (2 * dt)
    P = fr0v[s]; key = np.floor(P[clean, 0] / 0.02).astype(int) * 1000 + np.floor(P[clean, 1] / 0.02).astype(int)
    _, inv = np.unique(key, return_inverse=True); W_ = np.bincount(inv, weights=m[clean])
    b = lambda v: np.bincount(inv, weights=(m * v)[clean]) / W_
    res['visc_field_corr_pred_meas_%d' % s] = float(np.corrcoef(b(pred[s]), b(rm))[0, 1]); res['visc_field_corr_pred_dns_%d' % s] = float(np.corrcoef(b(pred[s]), b(rd_))[0, 1])
    res['visc_field_corr_meas_dns_%d' % s] = float(np.corrcoef(b(rm), b(rd_))[0, 1])
json.dump(res, open('res_entropy.json', 'w'), indent=1)
for k, v in res.items():
    if k != 'budget': print(k, '%.4g' % v if isinstance(v, float) else v)
B = budget; tt = np.array(B['t'])
for tq in (0.4, 0.6, 0.8, 1.0, 1.2, 1.3):
    i = np.argmin(abs(tt - tq)); print('t=%.2f heavy: meas %.4g dns %.4g pred %.4g (cond %.4g phi %.4g) | all meas %.4g dns %.4g | prod_net %.4g (phi %.4g cond %.4g)' % (tt[i], B['meas_heavy'][i], B['dns_heavy'][i], B['pred_heavy'][i], B['cond_heavy'][i], B['phi_heavy'][i], B['meas_all'][i], B['dns_all'][i], B['prod_net'][i], B['prod_phi'][i], B['prod_cond'][i]))
