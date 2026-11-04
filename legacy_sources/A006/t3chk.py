import numpy as np, sys, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import netmix as NM
net, z = NM.load_sb(); P0 = net.fr[0]
phi0 = np.hypot(P0[:, 0] - 0.75, P0[:, 1]) - 0.25; H = phi0 < 0; Hi = np.where(H)[0]
print('mass H', net.m[H].sum(), 'expected', 3 * np.pi * 0.25**2 / 2, 'rho range on H', z['rho'][0][H].min(), z['rho'][0][H].max())
print('y min', P0[:,1].min(), 'nodes near y=0', (P0[:,1]<0.003).sum())
for mm in ('net', 'uniform3'):
    m = net.m if mm == 'net' else 3 * NM.X._star(P0, net.tri[0], net.N) / 3
    A = net.L[0][Hi][:, Hi].tocsc(); B = sp.diags(m[Hi]).tocsc()
    v, _ = eigsh(A, k=2, M=B, sigma=0, which='LM'); print(mm, np.sort(v))
# check boundary weights along y=0: dual area of boundary nodes etc; also try full-disk mirror test
Pm = np.r_[P0, P0 * [1, -1]]
tri = NM.X.complex_peeled(Pm)['tri']; E, w = NM.edge_weights(Pm, tri); L = NM.laplacian(len(Pm), E, w)
m = 3 * NM.X._star(Pm, tri, len(Pm)) / 3
ph = np.hypot(Pm[:, 0] - 0.75, Pm[:, 1]) - 0.25; Hi2 = np.where(ph < 0)[0]
v, _ = eigsh(L[Hi2][:, Hi2].tocsc(), k=2, M=sp.diags(m[Hi2]).tocsc(), sigma=0, which='LM'); print('mirrored full disk', np.sort(v))
