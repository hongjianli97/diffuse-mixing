import numpy as np, json, sys
from scipy.stats import spearmanr
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import netmix as NM
net, z = NM.load_sb(); N0 = int(z['N0']); P0 = net.fr[0]; h = float(z['h'])
phi0 = np.hypot(P0[:, 0] - float(z['xb']), P0[:, 1]) - float(z['R']); lab = (phi0 < 0).astype(float)
w0 = 0.02; c0 = 0.5 * (1 - np.tanh(phi0 / (w0 / 2)))
mt3 = np.load('out/maps_t3.npz'); af = np.load('out/maps_afrc.npz')
mt2 = dict(np.load('out/maps_t2_a.npz'))
band = np.abs(phi0[:N0]) < 0.5 * w0
q0 = net.node_dirichlet(0, c0)[:N0]
out = {}; stats = {}
for s, win in ((30, '0-0.6'), (70, '0-1.4')):
    q = net.node_dirichlet(s, c0)[:N0]
    out['lnq_%d' % s] = np.where(band, np.log(np.maximum(q, 1e-30) / np.maximum(q0, 1e-30)), np.nan)
    c = mt2['c_G0.0001_s%d' % s][:N0]; out['mix_%d' % s] = 4 * c * (1 - c)
    fc = net.far_contacts(0, s, 4 * h, lab)
    out['far_%d' % s] = fc['far'][:N0]; out['farx_%d' % s] = fc['far_cross'][:N0]; out['raw_%d' % s] = fc['raw'][:N0]
    hf, hz = mt3['mfpt_' + win][:N0], mt3['mfpt_frozen'][:N0]
    out['mfpt_%d' % s] = np.where(lab[:N0] > 0, hf, np.nan)
    out['mfpt_ratio_%d' % s] = np.where(lab[:N0] > 0, np.log2(hz / np.maximum(hf, 1e-12)), np.nan)
    out['eig1_%d' % s] = np.where(lab[:N0] > 0, mt3['eig1_' + win][:N0], np.nan)
    out['afrc_%d' % s] = af['barrier_s%d' % s]
    # stats on band nodes / heavy nodes
    b = band & np.isfinite(out['lnq_%d' % s]) & np.isfinite(out['afrc_%d' % s])
    hv = (lab[:N0] > 0) & (np.abs(phi0[:N0]) > 0.0)
    st = dict(rho_lnq_afrc=spearmanr(out['lnq_%d' % s][b], out['afrc_%d' % s][b])[0],
              rho_farx_mix_band=spearmanr(out['farx_%d' % s][np.abs(phi0[:N0]) < 0.06], out['mix_%d' % s][np.abs(phi0[:N0]) < 0.06])[0],
              rho_mfpt_unmixed_heavy=spearmanr(out['mfpt_%d' % s][hv], 1 - c[hv])[0] if True else None,
              farx_mean_band=float(out['farx_%d' % s][band].mean()), far_mean_all=float(out['far_%d' % s].mean()),
              raw_new_per_node=float((out['raw_%d' % s] - np.bincount(net.E[0].ravel(), minlength=net.N)[:N0]).mean()))
    stats[s] = {k: float(v) for k, v in st.items()}
print(json.dumps(stats, indent=1))
json.dump(stats, open('out/res_t5_stats.json', 'w'), indent=1)
np.savez_compressed('out/maps_all.npz', **out)
