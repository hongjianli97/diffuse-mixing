import numpy as np, json, sys
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import cases as C, netmix as NM
c = C.case_sod(h=0.004, Ly=0.12, S=41); fr, h = c['frames'], c['h']; P0 = fr[0]; x0 = c['x0']
net = NM.Net(fr, c['dt'], np.where(x0 > 0, 0.125, 1.0)); eta = c['info']['rsR'] / 0.125
inner = (P0[:, 0] > 0.13) & (P0[:, 0] < 0.32) & (P0[:, 1] > 0.025) & (P0[:, 1] < 0.095)
out = {}
for ang in (45, 0, 90):
    n = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    csm = 0.5 + 0.5 * np.sin(2 * np.pi * (P0 @ n) / (16 * h))
    def Ecot(s):
        E, w = net.E[s], net.w[s]; k = inner[E[:, 0]] & inner[E[:, 1]]
        return float((w[k] * (csm[E[k, 0]] - csm[E[k, 1]]) ** 2).sum())
    def Ecomb(s):
        E = net.E[s]; k = inner[E[:, 0]] & inner[E[:, 1]]
        return float(((csm[E[k, 0]] - csm[E[k, 1]]) ** 2).sum())
    pred = (eta ** 2 * n[0] ** 2 + n[1] ** 2) / eta
    out[ang] = dict(cot=Ecot(-1) / Ecot(0), comb=Ecomb(-1) / Ecomb(0), pred_GammaConst=pred, pred_rhokappa=pred * eta)
print(json.dumps(out, indent=1)); json.dump(out, open('out/res_t1b_sod_energy.json', 'w'), indent=1)
