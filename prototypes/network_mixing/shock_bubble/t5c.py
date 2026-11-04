"""Which network quantity explains WHERE the flow enhanced mixing (flow minus frozen network diffusion, Gamma=1e-4)?"""
import numpy as np, sys, json
from scipy.stats import spearmanr
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import netmix as NM
net, z = NM.load_sb(); N0 = int(z['N0']); P0 = net.fr[0]; h = float(z['h'])
phi0 = np.hypot(P0[:, 0] - 0.75, P0[:, 1]) - 0.25; lab = (phi0 < 0).astype(float)
w0 = 0.02; c0 = 0.5 * (1 - np.tanh(phi0 / (w0 / 2)))
fz = net.diffuse(lab, 1e-4, frozen=True, keep={30, 70})
mt2 = np.load('out/maps_t2_a.npz'); mp = dict(np.load('out/maps_all.npz')); af = np.load('out/maps_afrc.npz')
q0 = net.node_dirichlet(0, c0)
Q = np.zeros(net.N); res = {}
clock = {}
for s in range(1, 71):
    Q += 0.5 * net.dt * (net.node_dirichlet(s - 1, c0) + net.node_dirichlet(s, c0))
    if s in (30, 70): clock[s] = Q / (s * net.dt * np.maximum(q0, 1e-30))
for s in (30, 70):
    cf = mt2['c_G0.0001_s%d' % s][:N0]; cz = fz[s][:N0]
    dmix = 4 * cf * (1 - cf) - 4 * cz * (1 - cz)
    mp['dmix_%d' % s] = dmix; mp['clock_%d' % s] = np.where(np.abs(phi0[:N0]) < 0.5 * w0, np.log(clock[s][:N0]), np.nan)
    band = np.abs(phi0[:N0]) < 0.03
    bb = band & np.isfinite(mp['clock_%d' % s])
    # clock is only defined on the c0 band; spread to the 0.03 band by nearest band node in reference coords
    from scipy.spatial import cKDTree
    src = np.where(bb)[0]; tr = cKDTree(P0[:N0][src]); _, j = tr.query(P0[:N0][band]); clk = mp['clock_%d' % s][src][j]
    r = dict(dist=spearmanr(-np.abs(phi0[:N0][band]), dmix[band])[0],
             net_clock=spearmanr(clk, dmix[band])[0],
             far_cross=spearmanr(mp['farx_%d' % s][band], dmix[band])[0],
             far_all=spearmanr(mp['far_%d' % s][band], dmix[band])[0],
             raw_contacts=spearmanr(mp['raw_%d' % s][band], dmix[band])[0],
             afrc_barrier=spearmanr(af['barrier_s%d' % s][band], dmix[band])[0],
             frac_enhanced=float((dmix[band] > 0).mean()), dmix_band_mean=float(dmix[band].mean()))
    res[s] = {k: float(v) for k, v in r.items()}
print(json.dumps(res, indent=1)); json.dump(res, open('out/res_t5c.json', 'w'), indent=1)
np.savez_compressed('out/maps_all.npz', **mp)
