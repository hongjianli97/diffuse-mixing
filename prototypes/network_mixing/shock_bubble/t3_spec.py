"""T3: time scales on the trajectory network: killed random walk (heavy gas absorbed on light gas).
lambda_1 = principal Dirichlet eigenvalue of (Lbar[H,H], M_H): slowest decay rate of never-mixed heavy material;
MFPT h_i: mean first-passage time from heavy tracer i to light gas.  Both in units Gamma = 1."""
import numpy as np, json, sys, time
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import netmix as NM
net, z = NM.load_sb(); P0 = net.fr[0]
phi0 = np.hypot(P0[:, 0] - float(z['xb']), P0[:, 1]) - float(z['R']); H = phi0 < 0; Hi = np.where(H)[0]
def dirichlet_eig(Lb, k=4):
    A = Lb[Hi][:, Hi].tocsc(); B = sp.diags(net.m[Hi]).tocsc()
    vals, vecs = eigsh(A, k=k, M=B, sigma=0, which='LM'); o = np.argsort(vals)
    return vals[o], vecs[:, o]
res = {}; maps = {}
wins = {'0-0.6': (0, 30), '0.6-1.0': (30, 50), '1.0-1.4': (50, 70), '0-1.4': (0, 70), 'frozen': (0, 0)}
for name, (a, b) in wins.items():
    Lb = net.L[0] if name == 'frozen' else net.Lbar(a, b)
    lam, vec = dirichlet_eig(Lb)
    hm = net.mfpt(Lb, ~H)
    res[name] = dict(lam=lam.tolist(), mfpt_mean=float((net.m[H] * hm[H]).sum() / net.m[H].sum()),
                     mfpt_p90=float(np.percentile(hm[H], 90)), mfpt_max=float(hm[H].max()))
    v = np.zeros(net.N); v[Hi] = np.abs(vec[:, 0]); maps['eig1_' + name] = v; maps['mfpt_' + name] = hm
    print(name, res[name], flush=True)
from scipy.special import jn_zeros
res['analytic_frozen_lam1'] = float(jn_zeros(0, 1)[0] ** 2 / float(z['R']) ** 2 / 3.0)
print('analytic disk lambda1 (rho=3, Gamma=1):', res['analytic_frozen_lam1'])
json.dump(res, open('out/res_t3_spec.json', 'w'), indent=1); np.savez_compressed('out/maps_t3.npz', **maps)
