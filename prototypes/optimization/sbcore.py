"""Shared utilities for the shock-bubble diffusion analysis (positions + one-time density only)."""
import sys, pickle, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.special import erfc
sys.path.insert(0, 'lhd'); sys.path.insert(0, 'st/code')
import lhdx as LX
import stretch_est as SE

BOX = (0.3, 1.25, 0.0, 0.42)

def load(path):
    return pickle.load(open(path, 'rb'))

def lattice_shape(h):
    nx = len(np.arange(BOX[0] + h / 2, BOX[1], h)); ny = len(np.arange(BOX[2] + h / 2, BOX[3], h)); return ny, nx

def subset(D, sub):
    ny, nx = lattice_shape(D['h']); iy, ix = np.divmod(np.arange(ny * nx), nx)
    return np.where((iy % sub == 0) & (ix % sub == 0))[0]

class Prep:
    """triangulations, hot steps, ledger, unit stiffness per frame for a tracer subset."""
    def __init__(self, D, idx, verbose=False):
        self.D = D; self.idx = idx; fr = D['frames'][:, idx].astype(float); self.fr = fr
        S, N = fr.shape[:2]; self.S, self.N = S, N; self.t = D['t']; self.dt = D['dt']
        self.h = D['h'] * round(np.sqrt(len(D['frames'][0]) / N))
        self.rho0 = D['rho'][0][idx].astype(float)
        self.cx = [LX.complex_peeled(fr[s]) for s in range(S)]
        self.tri = [c['tri'] for c in self.cx]
        # hot steps (C channel: one-step or two-step star-area drop below -0.2), censoring as in trajnet.events
        dl = np.zeros((S - 1, N))
        for s in range(S - 1):
            A0, A1 = LX._star(fr[s], self.tri[s], N), LX._star(fr[s + 1], self.tri[s], N)
            with np.errstate(all='ignore'): dl[s] = np.where(A0 > 0, np.log(A1 / A0), 0.0)
        det = dl < -0.2; det[:-1] |= (dl[:-1] + dl[1:]) < -0.2
        cs = det.T.copy(); cs[:, 1:] |= det.T[:, :-1]; cs[:, :-1] |= det.T[:, 1:]
        bad = np.zeros((N, S), bool); bad[:, :-1] |= cs; bad[:, 1:] |= cs
        self.hot = det.T; self.bad = bad
        lut = {fr[s].__array_interface__['data'][0]: self.cx[s] for s in range(S)}
        self.lnJ = LX.chained_lnJ_v2(fr, bad, cplx_fn=lambda P: lut.get(P.__array_interface__['data'][0]) or LX.complex_peeled(P))
        self.rho_led = self.rho0[None] * np.exp(-self.lnJ)
        A0 = LX._star(fr[0], self.tri[0], N); self.m = self.rho0 * A0 / 3
        self.K1 = [self.stiff(s, np.ones(len(self.tri[s]))) for s in range(S)]
        if verbose: print('prep done N=%d' % N, flush=True)

    def stiff(self, s, cond):
        P, tri = self.fr[s], self.tri[s]; G, A = LX._grad_lambda(P, tri); r, c, v = [], [], []
        for i in range(3):
            for j in range(3):
                r.append(tri[:, i]); c.append(tri[:, j]); v.append(cond * A * (G[:, i] * G[:, j]).sum(1))
        return sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(self.N, self.N))

    def K(self, s, closure, coef):
        if closure == 'gamma':
            return coef * self.K1[s]
        rT = self.rho_led[s][self.tri[s]].mean(1)          # rho*kappa with ledger density
        return self.stiff(s, coef * rT)

    def diffuse(self, c0, s0, s1, closure, coef, scheme='avg', record=()):
        c = c0.astype(float).copy(); out = {}; M = sp.diags(self.m)
        for s in range(s0, s1):
            if scheme == 'avg':
                Kb = 0.5 * (self.K(s, closure, coef) + self.K(s + 1, closure, coef))
            else:
                Kb = self.K(s, closure, coef)
            A = (M + self.dt * Kb).tocsr(); d = A.diagonal()
            c, info = cg(A, self.m * c, x0=c, rtol=1e-11, maxiter=2000, M=sp.diags(1 / d))
            if s + 1 in record: out[s + 1] = c.copy()
        return c, out

def chained_lnlam(pr, idx_local, T0, keep):
    """chained-tensor tangential stretch (user's estimator) for tracers idx_local of a Prep."""
    F = SE.chained_F(pr.fr, pr.h, bad=pr.bad, idx=idx_local, keep=keep)
    out = {}
    for s in keep:
        lnl, A = SE.vector_lnlam(F[s][0], T0); out[s] = (lnl, A)
    return out
