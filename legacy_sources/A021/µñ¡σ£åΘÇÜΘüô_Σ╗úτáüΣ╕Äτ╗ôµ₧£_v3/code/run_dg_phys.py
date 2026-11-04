"""DG: stream function recovered by the Crouzeix-Raviart part (vs exact psi); multi-scale AFRC_H."""
import numpy as np, json, cases, lhd as L, lhdx as X, afrc_h_engine as AH
c = cases.case_dg(); fr, dt = c['frames'], c['dt']; S, N = fr.shape[:2]; tm = S // 2
V = np.gradient(fr, dt, axis=0); P = fr[tm]
cx = X.complex_peeled(P); g = X.galerkin_hodge(P, cx, V[tm])
psi_e = g['psi']; acc = np.zeros(N); cnt = np.zeros(N)
for j in range(2):
    np.add.at(acc, cx['edges'][:, j], psi_e); np.add.at(cnt, cx['edges'][:, j], 1)
psi_n = acc / np.maximum(cnt, 1)
A_, eps, om = 0.1, 0.25, 2 * np.pi / 10; tt = c['t'][tm]; a = eps * np.sin(om * tt); b = 1 - 2 * a
psi_ex = A_ * np.sin(np.pi * (a * P[:, 0] ** 2 + b * P[:, 0])) * np.sin(np.pi * P[:, 1])
r = np.corrcoef(-psi_n, psi_ex)[0, 1]
k = (psi_ex @ -psi_n) / (psi_n @ psi_n)
print('psi corr %.5f  relL2 %.4f' % (r, np.sqrt(np.mean((-psi_n * k - psi_ex) ** 2)) / np.sqrt(np.mean(psi_ex ** 2))), 'scale', k)
traj = np.transpose(fr, (1, 0, 2)); Fs = []
for kk in (12, 16, 24):
    B, _, _ = AH.afrc_h_field(traj, k=kk, dt=c['t'][-1] - c['t'][0]); Fs.append(B)
d = dict(np.load('res_dg.npz')); d.update(psi_lhd=-psi_n, psi_ex=psi_ex, afrcH_k12=Fs[0], afrcH_k16=Fs[1], afrcH_k24=Fs[2])
np.savez_compressed('res_dg.npz', **d)
R = json.load(open('res_dg.json')); R['psi'] = dict(r=float(r)); json.dump(R, open('res_dg.json', 'w'), indent=1, default=float)
