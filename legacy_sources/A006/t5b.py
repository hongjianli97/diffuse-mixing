import numpy as np, sys, json
from scipy.stats import spearmanr
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
z = np.load('sb_lite_m.npz'); N0 = int(z['N0']); P0 = z['frames'][0][:N0]
phi0 = np.hypot(P0[:, 0] - 0.75, P0[:, 1]) - 0.25; hv = phi0 < 0
mt3 = np.load('out/maps_t3.npz'); mp = np.load('out/maps_all.npz'); res = {}
for s, win in ((30, '0-0.6'), (70, '0-1.4')):
    mix = mp['mix_%d' % s]
    r = dict(flow=spearmanr(mt3['mfpt_' + win][:N0][hv], mix[hv])[0], frozen=spearmanr(mt3['mfpt_frozen'][:N0][hv], mix[hv])[0],
             dist=spearmanr(-phi0[hv], mix[hv])[0])
    # restrict to the heavy nodes that are not trivially mixed/unmixed: 0.02 < mix < 0.98
    k = hv & (mix > 0.02)
    r.update(flow_active=spearmanr(mt3['mfpt_' + win][:N0][k], mix[k])[0], frozen_active=spearmanr(mt3['mfpt_frozen'][:N0][k], mix[k])[0], n_active=int(k.sum()))
    band = np.abs(phi0) < 0.06
    r.update(farx_vs_mix=spearmanr(mp['farx_%d' % s][band], mix[band])[0], dist_vs_mix_band=spearmanr(-np.abs(phi0[band]), mix[band])[0],
             raw_vs_mix=spearmanr(mp['raw_%d' % s][band], mix[band])[0])
    res[s] = {k: float(v) for k, v in r.items()}
print(json.dumps(res, indent=1)); json.dump(res, open('out/res_t5b.json', 'w'), indent=1)
