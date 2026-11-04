import numpy as np, sys, scipy.sparse as sp
from scipy.special import j0, jn_zeros
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import netmix as NM
net, z = NM.load_sb(); P0 = net.fr[0]
r = np.hypot(P0[:, 0] - 0.75, P0[:, 1]); H = r < 0.25
f = np.where(H, j0(jn_zeros(0,1)[0] * r / 0.25), 0.0)
m3 = 3 * NM.X._star(P0, net.tri[0], net.N) / 3
print('RQ half', f @ (net.L[0] @ f) / (m3 * f * f).sum())
# contributions of edges by y
E, w = net.E[0], net.w[0]; d = w * (f[E[:, 0]] - f[E[:, 1]]) ** 2
ym = 0.5 * (P0[E[:, 0], 1] + P0[E[:, 1], 1])
for lo, hi in ((0, 0.005), (0.005, 0.02), (0.02, 0.3)):
    k = (ym >= lo) & (ym < hi); print(lo, hi, d[k].sum(), 'neg w', (w[k] == 0).sum(), k.sum())
# reference: exact energy pi*int |grad f|^2 r dr *rho... compute numerically: grad energy over half disk / mass integral
rr = np.linspace(0, 0.25, 20001); jz = jn_zeros(0,1)[0]
from scipy.special import j1
num = np.trapezoid((jz / 0.25 * j1(jz * rr / 0.25)) ** 2 * rr, rr) * np.pi
den = 3 * np.trapezoid(j0(jz * rr / 0.25) ** 2 * rr, rr) * np.pi
print('exact RQ', num / den)
# check node mass vs voronoi area
tri = net.tri[0]; print('total star/3', NM.X._star(P0, tri, net.N).sum() / 3, 'hull area approx', (P0[:,0].max()-P0[:,0].min())*(P0[:,1].max()-P0[:,1].min()))
# how many triangles; any triangles duplicated?
print('tri', len(tri), 'unique', len(np.unique(np.sort(tri, 1), axis=0)))
