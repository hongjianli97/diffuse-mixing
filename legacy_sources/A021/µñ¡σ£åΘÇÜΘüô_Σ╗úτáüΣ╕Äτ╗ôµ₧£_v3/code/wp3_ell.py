import json, pickle, numpy as np
from scipy.stats import spearmanr
from scipy.interpolate import RegularGridInterpolator as RGI
import wp3_core as W
from trajnet import TrajectoryNetwork as TN
out = {}
S = pickle.load(open('/home/claude/bench/sv_dns.pkl', 'rb')); nets = {}
for mode in ('in', 'off'):
    n2 = TN(S['fr_' + mode][10:31], S['dt']); n2.events(); nets[mode] = n2
D = pickle.load(open('/home/claude/bench/sb_dns.pkl', 'rb')); fr, dt, xc, yc, grids = D['frames'], D['dt'], D['xc'], D['yc'], D['grids']; del D
net = TN(fr, dt); net.events(); bts = {}
for tk, s in ((0.6, 30), (1.0, 50)):
    Wg = grids[tk]['W'].astype(float); gyr, gxr = np.gradient(Wg[0], yc, xc); gyp, gxp = np.gradient(Wg[3], yc, xc)
    bts[tk] = RGI((yc, xc), (gxr * gyp - gyr * gxp) / Wg[0] ** 2, bounds_error=False, fill_value=0.0)(np.c_[fr[s][:, 1], fr[s][:, 0]])
for ef in (0.75, 1.5, 3.0):
    a = {m: W.t0_t2(nets[m], 10, ef) for m in ('in', 'off')}; com = np.all([np.isfinite(a[m][k]) for m in a for k in ('T0', 'T2')], 0)
    r = dict(SV=float(spearmanr(a['in']['T2'][com], a['off']['T2'][com])[0]))
    for tk, s in ((0.6, 30), (1.0, 50)):
        T = W.t0_t2(net, s, ef); c = np.isfinite(T['T0']) & np.isfinite(T['T2']); r['SB_%.1f' % tk] = float(np.corrcoef(T['T2'][c], bts[tk][c])[0, 1])
    out['ell=%.2fh' % ef] = r; print('ell=%.2fh' % ef, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
json.dump(out, open('res_wp3_ell.json', 'w'), indent=1)
