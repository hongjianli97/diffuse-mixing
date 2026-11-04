"""Material-connectivity network (triangulation of t=0 kept, current positions) vs current-Delaunay network on SBI paths."""
import sys, pickle, time, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.special import erfc
import sbcore as C, lhdx as LX
T0 = time.time()

class PrepL:
    def __init__(self, D, idx):
        fr = D['frames'][:, idx].astype(float); self.fr = fr; self.S, self.N = fr.shape[:2]; self.dt = D['dt']
        self.tri0 = LX.complex_peeled(fr[0])['tri']; self.m = D['rho'][0][idx].astype(float) * LX._star(fr[0], self.tri0, self.N) / 3
        self.K1, self.inv = [], []
        for s in range(self.S):
            Gr, A = LX._grad_lambda(fr[s], self.tri0); bad = A <= 1e-14
            self.inv.append(bad.mean())
            A = np.where(bad, np.inf, A)                                     # drop inverted/degenerate triangles (none expected)
            Gr = np.where(bad[:, None, None], 0.0, Gr)
            r, c, v = [], [], []
            for i in range(3):
                for j in range(3):
                    r.append(self.tri0[:, i]); c.append(self.tri0[:, j]); v.append(np.where(bad, 0, A) * (Gr[:, i] * Gr[:, j]).sum(1))
            self.K1.append(sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(self.N, self.N)))
    def diffuse(self, c0, coef, record=()):
        c = c0.copy(); out = {}; M = sp.diags(self.m)
        for s in range(self.S - 1):
            A = (M + self.dt * coef * 0.5 * (self.K1[s] + self.K1[s + 1])).tocsr()
            c, _ = cg(A, self.m * c, x0=c, rtol=1e-11, maxiter=4000, M=sp.diags(1 / A.diagonal()))
            if s + 1 in record: out[s + 1] = c.copy()
        return c, out

part = sys.argv[1]
if part == 'pe':
    D = C.load('inv005.pkl'); P0 = D['frames'][0]; phi = np.hypot(P0[:, 0] - D['xb'], P0[:, 1]) - D['R']; w0 = 0.01
    res = {}
    for sub in (1, 2, 4):
        idx = C.subset(D, sub); pL = PrepL(D, idx)
        print('sub %d N=%d max inverted fraction %.2e  prep %.0fs' % (sub, len(idx), max(pL.inv), time.time() - T0), flush=True)
        c0 = 0.5 * erfc(phi[idx] / w0); mean = (pL.m * c0).sum() / pL.m.sum(); V0 = (pL.m * (c0 - mean) ** 2).sum() / pL.m.sum()
        for kap in (1e-6, 1e-5, 1e-4, 1e-3):
            _, rec = pL.diffuse(c0, kap, record=(30, 70))
            res[(sub, kap)] = {s: V0 - (pL.m * (rec[s] - mean) ** 2).sum() / pL.m.sum() for s in rec}
            print('  material sub=%d kappa=%.0e dV(0.6)=%.4e dV(1.4)=%.4e' % (sub, kap, res[(sub, kap)][30], res[(sub, kap)][70]), flush=True)
        del pL
    pickle.dump(res, open('res_opt1_pe.pkl', 'wb'))
if part == 'visc':
    D = C.load('vis005.pkl'); idx = C.subset(D, 1); pL = PrepL(D, idx); Y = D['Y1'][:, idx].astype(float)
    print('visc max inverted fraction %.2e' % max(pL.inv), flush=True)
    _, rec = pL.diffuse(Y[0], 5e-4, record=(30, 70))
    for s in (30, 70):
        mixed = lambda c: (pL.m * c * (1 - c)).sum()
        print('visc material t=%.1f mixed net %.4e DNS %.4e (%+.3f) L2 %.3f' % (s * .02, mixed(rec[s]), mixed(Y[s]), mixed(rec[s]) / mixed(Y[s]) - 1,
              np.sqrt((pL.m * (rec[s] - Y[s]) ** 2).sum() / (pL.m * Y[s] ** 2).sum())), flush=True)
print('done %.0fs' % (time.time() - T0))
