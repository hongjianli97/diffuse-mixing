"""Shock-bubble DNS with an explicitly diffusing passive scalar (rho*Y in slot 4): d(rho Y)/dt + div(rho Y u) = div(rho kappa grad Y)."""
import pickle, numpy as np, euler2d as E
KAPPA = 1e-3
orig_init, orig_rhs = E.Solver.__init__, E.Solver.rhs
def init2(self, *a, **k):
    orig_init(self, *a, **k)
    Xg, Yg = np.meshgrid(self.xc, self.yc); rb = np.hypot(Xg - self.xb, Yg)
    self.U[4, 2:-2, 2:-2] = self.U[0, 2:-2, 2:-2] * 0.5 * (1 - np.tanh((rb - self.R) / 0.02))
def rhs2(self, U):
    out = orig_rhs(self, U); W = E.prim(self.ghosts(U)); rho, Y, dx = W[0], W[4], self.dx
    fx = 0.5 * (rho[2:-2, 2:-1] + rho[2:-2, 1:-2]) * KAPPA * (Y[2:-2, 2:-1] - Y[2:-2, 1:-2]) / dx
    fy = 0.5 * (rho[2:-1, 2:-2] + rho[1:-2, 2:-2]) * KAPPA * (Y[2:-1, 2:-2] - Y[1:-2, 2:-2]) / dx
    out[4] += (fx[:, 1:] - fx[:, :-1]) / dx + (fy[1:] - fy[:-1]) / dx
    return out
E.Solver.__init__, E.Solver.rhs = init2, rhs2
o = E.run(keep_grid=(0.6, 1.0, 1.4), verbose=False)
pickle.dump(dict(frames=o['frames'], t=o['t'], dt=o['dt'], Y=o['E']['Y1'], rho=o['E']['rho'], kappa=KAPPA, xb=o['xb'], R=o['R']), open('sb_diff_dns.pkl', 'wb'))
print('done', o['nstep'])
