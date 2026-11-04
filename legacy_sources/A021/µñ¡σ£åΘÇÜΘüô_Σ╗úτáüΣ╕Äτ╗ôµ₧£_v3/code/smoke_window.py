import os, sys, pickle, time, numpy as np
sys.path.insert(0, '/home/claude/pkg')
import lhd2, lhd_window as LW
D = pickle.load(open('/home/claude/bench/sb_dns.pkl', 'rb'))
fr = D['frames']; S, N = fr.shape[:2]; dt = D['dt']; X = np.transpose(fr, (1, 0, 2))
V = np.gradient(X, dt, axis=1)
Vd = np.stack([lhd2.relative_velocity(X[:, t], V[:, t]) for t in range(S)], 1)
dlp = np.diff(np.log(D['E']['p']), axis=0); es, ep = np.where(dlp > 0.15)
rr = D['E']['rho'][np.minimum(es + 3, S - 1), ep] / D['E']['rho'][np.maximum(es - 2, 0), ep]
mock = '/home/claude/pkg/tracking_layers_sb.npz'
np.savez_compressed(mock, **{'fwd__traj': X.astype(np.float32), 'fwd__t_start': np.asarray(0), 'fwd__dt_total': np.asarray(float(D['t'][-1])),
    'fwd__snap_indices': np.arange(S), 'fwd__direction': np.asarray('forward'), 'fwd__oob': np.zeros(N, bool),
    'fwd__objrot__vd': Vd.astype(np.float32), 'fwd__ev_pidx': ep.astype(np.int64), 'fwd__ev_step': es.astype(np.int32),
    'fwd__ev_r_ratio': rr.astype(np.float32), '__layer_names': np.asarray(['fwd']), '__meta__T_CENTER': np.asarray(0), '__meta__DT_OUT': np.asarray(dt)})
g0 = D['grids'][0.3]['W'][0]                                     # density field is only needed at t_c = 0: rebuild it
gx, gy = np.meshgrid(D['xc'], D['yc']); rb = np.hypot(gx - D['xb'], gy)
rho0 = np.where(gx < 0.25, D['post'][0], 1.0 + 2.0 * 0.5 * (1 - np.tanh((rb - D['R']) / (0.75 * D['dx']))))
eul = '/home/claude/pkg/euler_context_sb.npz'; np.savez_compressed(eul, x=gx, y=gy, rho=rho0)
t0 = time.time(); summ = LW.run_window(mock, 'fwd', euler_npz=eul, max_n=None, verbose=False)
z = np.load(summ['path'])
print('sec/snapshot %.2f total %.0fs' % (summ['sec_per_snapshot'], time.time() - t0))
print('RH check exp|dlnJ| / rho ratio:', summ['rh'])
idx = z['idx']; rhoT = z['rho_T']; ref = D['E']['rho'][-1][idx]; ok = np.isfinite(rhoT)
print('rho_T vs DNS rho(t_end): median rel err %.4f' % np.nanmedian(np.abs(rhoT[ok] / ref[ok] - 1)))
print('Efrac(t=1.0):', np.round(z['Efrac'][50], 4))
