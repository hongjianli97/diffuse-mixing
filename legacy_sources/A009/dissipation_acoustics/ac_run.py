"""Linear acoustic pulse at rest (walls on all sides), rho = 1 + eps exp(-r^2/sig^2), isentropic p = rho^g / g (c0 = 1).
Tracers h = 0.005 in [0.15,0.85]^2; frames every 0.005 to t = 0.3.  DNS rho, p, u, v at tracers kept as references."""
import numpy as np, pickle, sys, time
sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import euler2d as E
dx, L, eps, sig, h, dtf, T = 0.005, 1.0, 0.01, 0.06, 0.005, 0.005, 0.3
class AC(E.Solver):
    def __init__(self):
        E.Solver.__init__(self, dx=dx, Lx=L, Ly=L, Ms=1.5, xs=-1.0, xb=-10.0, R=0.01, rho_b=1.0)
        X, Y = np.meshgrid(self.xc, self.yc); r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2
        rho = 1 + eps * np.exp(-r2 / sig ** 2); p = rho ** E.G / E.G
        self.U[:, 2:-2, 2:-2] = E.cons(np.stack([rho, 0 * X, 0 * X, p, 0 * X + 1, 0 * X]))
    def ghosts(self, U):
        U[:, :, :2] = U[:, :, [3, 2]]; U[1, :, :2] *= -1; U[:, :, -2:] = U[:, :, [-3, -4]]; U[1, :, -2:] *= -1
        U[:, :2, :] = U[:, [3, 2], :]; U[2, :2, :] *= -1; U[:, -2:, :] = U[:, [-3, -4], :]; U[2, -2:, :] *= -1
        return U
sv = AC(); rng = np.random.default_rng(3)
Xg, Yg = np.meshgrid(np.arange(0.15 + h / 2, 0.85, h), np.arange(0.15 + h / 2, 0.85, h))
P = np.c_[Xg.ravel(), Yg.ravel()] + rng.uniform(-0.25, 0.25, (Xg.size, 2)) * h
nF = int(round(T / dtf)) + 1; tf = np.arange(nF) * dtf; R = dict(frames=[], rho=[], p=[], u=[], v=[], t=[])
def rec():
    W = sv.W(); vals = sv.interp(np.stack([W[0], W[3], W[1], W[2]]), P)
    R['frames'].append(P.copy()); R['t'].append(sv.t)
    for k, v in zip(('rho', 'p', 'u', 'v'), vals): R[k].append(v)
rec(); t0 = time.time()
for fi in range(1, nF):
    while sv.t < tf[fi] - 1e-12:
        dt = min(sv.dt_cfl(), tf[fi] - sv.t); W = sv.W(); u0 = sv.interp(W[1:3], P).T; sv.step(dt)
        W = sv.W(); u1 = sv.interp(W[1:3], P + dt * u0).T; P = P + 0.5 * dt * (u0 + u1)
    rec()
print('done %.0fs' % (time.time() - t0)); R = {k: np.array(v) for k, v in R.items()}
np.savez_compressed('/home/claude/w/ac/ac.npz', **R, eps=eps, sig=sig, dtf=dtf)
