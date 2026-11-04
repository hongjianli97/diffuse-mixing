import json, pickle, numpy as np, lhdx as X, netdiff as ND
from trajnet import TrajectoryNetwork as TN
D = pickle.load(open('/home/claude/bench/sb_diff_dns.pkl', 'rb')); fr, dt, Y, rhoD, kappa = D['frames'], D['dt'], D['Y'], D['rho'], D['kappa']
S, N = fr.shape[:2]; P0 = fr[0]; r0 = np.hypot(P0[:, 0] - D['xb'], P0[:, 1]); Rb = D['R']; h = 0.005
net = TN(fr, dt); net.events(); lnJ = X.chained_lnJ_v2(fr, net.bad)                       # positions only
m = rhoD[0] * X.star_area(P0) / 3.0; rho = rhoD[0][None, :] * np.exp(-lnJ)
keep = (30, 50, 70)
outs = {f: ND.run(fr, dt, Y[0], kappa, f, m=m, rho=rho, keep=keep) for f in ('mass', 'volume')}
inner = (P0[:, 0] > 0.32) & (P0[:, 0] < 1.23) & (P0[:, 1] < 0.40)
R = {'rho_from_positions_vs_DNS_mederr_t1.4': float(np.median(np.abs(rho[70][inner] / rhoD[70][inner] - 1)))}
for s in keep:
    ref = Y[s]; sd = np.std(ref[inner]); e = lambda v: float(np.sqrt(np.mean((v[inner] - ref[inner]) ** 2)) / sd)
    var = lambda v: float(np.sum(m * (v - np.sum(m * v) / m.sum()) ** 2) / m.sum())
    R['t=%.1f' % (s * dt)] = dict(relL2_mass_form=e(outs['mass'][s]), relL2_volume_form=e(outs['volume'][s]), relL2_pure_advection=e(Y[0]),
                                  var_ratio_mass_form=var(outs['mass'][s]) / var(ref), var_ratio_volume_form=var(outs['volume'][s]) / var(ref),
                                  heavy_total_mass_form=float(np.sum(m * outs['mass'][s]) / np.sum(m * Y[0])))
    print('t=%.1f' % (s * dt), json.dumps({k: round(v, 4) for k, v in R['t=%.1f' % (s * dt)].items()}), flush=True)
# ---------------- mixing attribution on the bubble interface: ln|grad c| growth = tangential stretch + compression
ring = np.where(np.abs(r0 - Rb) < 0.6 * h)[0]; ang = np.arctan2(P0[ring, 1], P0[ring, 0] - D['xb']); ring = ring[np.argsort(ang)]
mring = m[ring]; R['mixing'] = {}
for s in (15, 30, 50, 70):
    seg0 = np.linalg.norm(np.diff(P0[ring], axis=0), axis=1); seg = np.linalg.norm(np.diff(fr[s][ring], axis=0), axis=1)
    st_seg = np.log(seg / seg0); st = np.r_[st_seg[0], 0.5 * (st_seg[1:] + st_seg[:-1]), st_seg[-1]]
    comp = -lnJ[s][ring]; w = mring / mring.sum()
    S_, C_ = float(np.sum(w * st)), float(np.sum(w * comp))
    R['mixing']['t=%.1f' % (s * dt)] = dict(stretch=S_, compression=C_, total=S_ + C_, compression_share=C_ / (S_ + C_) if S_ + C_ > 0 else None,
                                            interface_length_ratio=float(seg.sum() / seg0.sum()))
print('mixing', json.dumps({k: {q: (round(v, 3) if isinstance(v, float) else v) for q, v in d.items()} for k, d in R['mixing'].items()}), flush=True)
json.dump(R, open('res_val_sb_diff.json', 'w'), indent=1)
pickle.dump(dict(P=fr[70], Y=Y[70], net=outs['mass'][70], vol=outs['volume'][70], adv=Y[0], ring=ring, P_ring=fr[:, ring]), open('res_val_sb_diff.pkl', 'wb'))
