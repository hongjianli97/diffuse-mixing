"""rebuild the shock-bubble network with mirror ghost tracers across the symmetry line y=0 (removes hull slivers)."""
import numpy as np, pickle, sys, time
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import netmix as NM
z = np.load('sb_lite.npz'); fr = z['frames']; N = fr.shape[1]
frm = np.concatenate([fr, fr * np.array([1, -1])], axis=1)
rho0 = np.r_[z['rho'][0], z['rho'][0]]
t0 = time.time(); net = NM.Net(frm, float(z['dt']), rho0); print('built', time.time() - t0)
pickle.dump(dict(tri=net.tri, E=net.E, w=net.w, m=net.m), open('sb_net_m.pkl', 'wb'))
np.savez_compressed('sb_lite_m.npz', frames=frm, rho=np.concatenate([z['rho'], z['rho']], 1), Y1=np.concatenate([z['Y1'], z['Y1']], 1),
                    t=z['t'], dt=z['dt'], h=z['h'], xb=z['xb'], R=z['R'], N0=N)
