import numpy as np, pickle, sys, time
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import netmix as NM
o = pickle.load(open('LHD_v3.2/sb_dns_dx0.005.pkl', 'rb'))
fr = o['frames']; t0 = time.time()
net = NM.Net(fr, o['dt'], o['E']['rho'][0])
print('built', time.time() - t0)
pickle.dump(dict(tri=net.tri, E=net.E, w=net.w, m=net.m), open('sb_net.pkl', 'wb'))
np.savez_compressed('sb_lite.npz', frames=fr.astype(np.float64), rho=o['E']['rho'], Y1=o['E']['Y1'], om=o['E']['om'], t=o['t'], dt=o['dt'], h=o['h'], xb=o['xb'], R=o['R'], p=o['E']['p'])
print('saved', time.time() - t0)
