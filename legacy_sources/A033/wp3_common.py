"""D-06: WP3 torque denoising evaluated on explicitly defined node sets.
Node sets:
  own     : each T on its own finite nodes (what wp3.py did; T2/T3 cover interpolated censored nodes too)
  clean0  : nodes where T0 is finite (outside the +-1-frame shock stencil)
  common  : nodes finite for T0 AND T1 (T1 needs a clean 5-frame window) -- the 'common clean nodes' of table 7
For SV the masks are intersected over the in-loop and offline tracks.  ell = 1.5 median edge length.
Writes res_wp3_common.json."""
import json, pickle, numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.stats import spearmanr
from trajnet import TrajectoryNetwork as TN
from wp3_core import torques

def corr(a, b, m):
    return float(np.corrcoef(a[m], b[m])[0, 1])
def sp(a, b, m):
    return float(spearmanr(a[m], b[m])[0])

R = {}
S = pickle.load(open('sv_dns.pkl', 'rb')); TQ = {}
for mode in ('in', 'off'):
    n2 = TN(S['fr_' + mode][10:31], S['dt']); n2.events(); TQ[mode] = torques(n2, 10, ell_fac=1.5)
fin = lambda k, mode: np.isfinite(TQ[mode][k])
masks = dict(clean0=fin('T0', 'in') & fin('T0', 'off'),
             common=fin('T0', 'in') & fin('T0', 'off') & fin('T1', 'in') & fin('T1', 'off'))
R['SV_in_vs_off_spearman'] = {}
for k in ('T0', 'T1', 'T2', 'T3'):
    own = fin(k, 'in') & fin(k, 'off')
    R['SV_in_vs_off_spearman'][k] = dict(own=sp(TQ['in'][k], TQ['off'][k], own), own_coverage=float(own.mean()),
                                         clean0=sp(TQ['in'][k], TQ['off'][k], masks['clean0'] & own),
                                         common=sp(TQ['in'][k], TQ['off'][k], masks['common'] & own))
R['SV_mask_coverage'] = {k: float(v.mean()) for k, v in masks.items()}
print('SV', json.dumps(R['SV_in_vs_off_spearman'], indent=1), flush=True)

D = pickle.load(open('sb_dns.pkl', 'rb')); fr, dt, xc, yc, grids = D['frames'], D['dt'], D['xc'], D['yc'], D['grids']; del D
net = TN(fr, dt); net.events(); R['SB_vs_DNS'] = {}
for tk, s in ((0.6, 30), (1.0, 50)):
    T = torques(net, s, ell_fac=1.5); W = grids[tk]['W'].astype(float)
    gyr, gxr = np.gradient(W[0], yc, xc); gyp, gxp = np.gradient(W[3], yc, xc)
    bt = RGI((yc, xc), (gxr * gyp - gyr * gxp) / W[0] ** 2, bounds_error=False, fill_value=0.0)(np.c_[fr[s][:, 1], fr[s][:, 0]])
    m0 = np.isfinite(T['T0']); mc = m0 & np.isfinite(T['T1'])
    R['SB_vs_DNS']['t=%.1f' % tk] = {k: dict(own=corr(v, bt, np.isfinite(v)), own_coverage=float(np.isfinite(v).mean()),
                                            clean0=corr(v, bt, m0 & np.isfinite(v)), common=corr(v, bt, mc & np.isfinite(v)))
                                     for k, v in T.items()}
    R['SB_vs_DNS']['t=%.1f' % tk]['mask_coverage'] = dict(clean0=float(m0.mean()), common=float(mc.mean()))
    print('SB t=%.1f' % tk, json.dumps(R['SB_vs_DNS']['t=%.1f' % tk], indent=1), flush=True)
json.dump(R, open('res_wp3_common.json', 'w'), indent=1)
