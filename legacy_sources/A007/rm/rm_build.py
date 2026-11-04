"""stage 1: frames, ledger lnJ (position-only), DNS density J, mirrored trajectory networks (ghosts across y=0 and y=0.5)."""
import numpy as np, pickle, sys, time
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import netmix as NM, lhdx as X
case = sys.argv[1]; stride = int(sys.argv[2])
st = pickle.load(open('rm_%s_ck.pkl' % case, 'rb')); rec = st['rec']
fr = np.array(rec['frames']); t = np.array(rec['t']); rho = np.array(rec['rho']); S, N, _ = fr.shape
xi = lambda y: 0.5 + 0.1 * np.cos(2 * np.pi * y); dxi = lambda y: -0.2 * np.pi * np.sin(2 * np.pi * y)
P0 = fr[0]; phi0 = (P0[:, 0] - xi(P0[:, 1])) / np.sqrt(1 + dxi(P0[:, 1]) ** 2); rho0 = np.where(phi0 > 0, 3.0, 1.0)
t0 = time.time()
# ledger: one-step star-area shock detection (C channel) + chained per-tracer anchors
tris = [X.complex_peeled(fr[s])['tri'] for s in range(S)]
dl = np.zeros((S - 1, N))
for s in range(S - 1):
    A0, A1 = X._star(fr[s], tris[s], N), X._star(fr[s + 1], tris[s], N)
    with np.errstate(all='ignore'): dl[s] = np.where(A0 > 0, np.log(A1 / A0), 0.0)
det = dl < -0.2; det[:-1] |= (dl[:-1] + dl[1:]) < -0.2
cs = det.T.copy(); cs[:, 1:] |= det.T[:, :-1]; cs[:, :-1] |= det.T[:, 1:]
bad = np.zeros((N, S), bool); bad[:, :-1] |= cs; bad[:, 1:] |= cs
lnJ = X.chained_lnJ_v2(fr, bad)
print('ledger', time.time() - t0, flush=True)
lnJ_dns = np.log(rho0[None] / rho)      # reference only (smeared at shocks and at the contact)
sel = np.arange(0, S, stride)
frs = fr[sel]
frm = np.concatenate([frs, frs * np.array([1, -1]), np.stack([frs[..., 0], 1 - frs[..., 1]], -1)], axis=1)
net = NM.Net(frm, float(t[stride] - t[0]), np.r_[rho0, rho0, rho0])
print('net', time.time() - t0, flush=True)
pickle.dump(dict(tri=net.tri, E=net.E, w=net.w, m=net.m), open('net_%s.pkl' % case, 'wb'))
np.savez_compressed('base_%s.npz' % case, fr=fr, t=t, rho=rho, Y1=np.array(rec['Y1']), lnJ=lnJ, lnJ_dns=lnJ_dns, hot=cs, phi0=phi0, rho0=rho0, sel=sel, N=N)
pickle.dump(rec['marker'], open('marker_%s.pkl' % case, 'wb'))
print('saved', time.time() - t0)
