import json, pickle, numpy as np, dirac as DR
from trajnet import TrajectoryNetwork as TN
D = pickle.load(open('/home/claude/bench/sb_dns.pkl', 'rb')); fr, dt, GamE = D['frames'], D['dt'], D['Gam']; r0 = np.hypot(fr[0][:, 0] - D['xb'], fr[0][:, 1]); Rb = D['R']; del D
net = TN(fr, dt); net.events(); Acc = DR.acceleration(fr, dt); reg = r0 < Rb + 0.05
out = {}
for span in (30, 10, 5):
    tot = imp = smo = 0.0; kept = []
    for a in range(5, 35, span):
        ks = DR.kelvin_split(net, Acc, reg, a, min(a + span, 35)); tot += ks['total'].sum(); imp += ks['impulsive'].sum(); smo += ks['smooth'].sum(); kept.append(ks['kept'])
    out['retile_every_%d' % span] = dict(total=float(tot), impulsive=float(imp), smooth=float(smo), impulsive_share=float(imp / tot), min_kept=float(min(kept)))
gE = lambda k, t: float(np.interp(t, GamE['t'], GamE[k]))
out['Euler'] = dict(dGamma=gE('Y2', 0.7) - gE('Y2', 0.1), baroclinic=gE('bar', 0.7) - gE('bar', 0.1))
# time series of the smooth (post-shock) and impulsive deposition with 5-frame re-tiling
ts, ci, cs = [0.1], [0.0], [0.0]
for a in range(5, 35, 5):
    ks = DR.kelvin_split(net, Acc, reg, a, a + 5); ts.append((a + 5) * dt); ci.append(ci[-1] + ks['impulsive'].sum()); cs.append(cs[-1] + ks['smooth'].sum())
out['series'] = dict(t=ts, impulsive=ci, smooth=cs, euler_t=list(map(float, GamE['t'])), euler_Y2=list(map(float, GamE['Y2'])), euler_bar=list(map(float, GamE['bar'])))
print(json.dumps({k: v for k, v in out.items() if k != 'series'}, indent=0))
json.dump(out, open('res_kelvin_chain.json', 'w'), indent=1)
