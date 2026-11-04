"""Single-mode Richtmyer-Meshkov: Ms=1.5 shock (light, rho=1) -> perturbed interface -> heavy gas rho=3 (A=0.5).
Half wavelength channel y in [0,0.5] (symmetry at y=0, wall at y=0.5): interface x = 0.5 + a0 cos(2 pi y)."""
import numpy as np, pickle, sys, euler2d as E
A0, XI, LAM = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04, 0.5, 1.0
dx = float(sys.argv[1])
orig = E.Solver.__init__
def init(self, dx=dx, **k):
    orig(self, dx=dx, Lx=2.2, Ly=0.5, Ms=1.5, xs=0.3, xb=-10.0, R=0.01, rho_b=3.0)
    X, Y = np.meshgrid(self.xc, self.yc); xi = XI + A0*np.cos(2*np.pi*Y/LAM)
    s = 0.5*(1+np.tanh((X-xi)/(0.75*dx)))
    W = np.stack([1.0+2.0*s, 0*X, 0*X, 1/E.G+0*X, s, s]); sh = X < 0.3
    for kk in range(6): W[kk][sh] = self.post[kk]
    self.U[:, 2:-2, 2:-2] = E.cons(W); self.xb, self.R = XI, A0
E.Solver.__init__ = init
o = E.run(dx=dx, h_p=0.005, t_end=2.0, dt_frame=0.04, box=(0.36, 0.76, 0.0, 0.5), keep_grid=(0.4, 0.8, 1.2, 1.6, 2.0), verbose=True)
o["a0"] = A0
pickle.dump(o, open('rm_dns_dx%g.pkl' % dx, 'wb')); print('ok', o['nstep'], flush=True)
