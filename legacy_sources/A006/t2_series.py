"""T2: shock-bubble, network-native time series: contact (cut length), exchange potential (Dirichlet energy of a
resolved initial profile, Gamma=const and rho*kappa closures), Pe sweep of mixing state (flow vs frozen network)."""
import numpy as np, json, sys, time
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import netmix as NM
net, z = NM.load_sb(); P0 = net.fr[0]; t = z['t']; h = float(z['h'])
N0 = int(z['N0'])
phi0 = np.hypot(P0[:, 0] - float(z['xb']), P0[:, 1]) - float(z['R'])
lab = (phi0 < 0).astype(float)
w0 = 0.02; c0 = 0.5 * (1 - np.tanh(phi0 / (w0 / 2)))
res = dict(t=t.tolist())
L0 = net.cut(0, lab)['Lnet']; E0 = net.dirichlet(0, c0)
res['lnL'] = [float(np.log(net.cut(s, lab)['Lnet'] / L0)) for s in range(net.S)]
res['ln_cutw'] = [float(np.log(net.cut(s, lab)['cutw'] / net.cut(0, lab)['cutw'])) for s in range(net.S)]
res['lnP_Gconst'] = [float(np.log(net.dirichlet(s, c0) / E0)) for s in range(net.S)]
# rho*kappa closure: conductance kappa*rho_T with DNS density at tracers (reference density, illustration only)
rho = z['rho']
def E_rk(s):
    tri = net.tri[s]; P = net.fr[s]; G, A = NM.X._grad_lambda(P, tri); rT = rho[s][tri].mean(1)
    g = (G * c0[tri][:, :, None]).sum(1); return float((rT * A * (g * g).sum(1)).sum())
Er0 = E_rk(0); res['lnP_rhokappa'] = [float(np.log(E_rk(s) / Er0)) for s in range(net.S)]
print('series done', flush=True)
# Pe sweep
keep = {30, 50, 70}
res['sweep'] = {}
maps = {}
GL = [float(g) for g in sys.argv[1].split(',')]
for G in GL:
    t1 = time.time()
    fl = net.diffuse(lab, G); fz = net.diffuse(lab, G, frozen=True)
    res['sweep'][str(G)] = dict(mix_flow=[NM.mixed_fraction(net.m, fl[s], lab) for s in range(net.S)],
                                mix_frozen=[NM.mixed_fraction(net.m, fz[s], lab) for s in range(net.S)],
                                Is_flow=[NM.seg_intensity(net.m, fl[s], lab) for s in range(net.S)],
                                Is_frozen=[NM.seg_intensity(net.m, fz[s], lab) for s in range(net.S)])
    for s in keep: maps['c_G%g_s%d' % (G, s)] = fl[s]
    print('G', G, time.time() - t1, flush=True)
# DNS scalar (numerical mixing only) for orientation
Y = np.clip(z['Y1'], 0, 1)
res['mix_DNS_Y1'] = [NM.mixed_fraction(net.m, Y[s], lab) for s in range(net.S)]
json.dump(res, open('out/res_t2_series_%s.json' % sys.argv[2], 'w'))
np.savez_compressed('out/maps_t2_%s.npz' % sys.argv[2], **maps)
for k in ('lnL', 'ln_cutw', 'lnP_Gconst', 'lnP_rhokappa'):
    print(k, ['%.3f' % res[k][s] for s in (5, 10, 15, 30, 50, 70)])
for G, d in res['sweep'].items():
    print(G, 'flow', ['%.4f' % d['mix_flow'][s] for s in (30, 50, 70)], 'frozen', ['%.4f' % d['mix_frozen'][s] for s in (30, 50, 70)])
print('DNS', ['%.4f' % res['mix_DNS_Y1'][s] for s in (0, 30, 50, 70)])
