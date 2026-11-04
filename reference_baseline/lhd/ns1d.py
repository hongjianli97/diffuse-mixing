"""Resolved planar viscous shock: one-dimensional compressible Navier-Stokes DNS with tracers, embedded in 2D.

Same physics and scheme as euler2d / euler2d_visc (gamma=1.4, constant mu, Pr=0.72, MUSCL with van Leer limiter on
primitives, HLLC, SSP-RK2, conservative viscous face fluxes; q = k grad(p/rho), k = mu gamma/((gamma-1) Pr)), plus
the explicit viscous time-step limit.  A planar shock is one-dimensional, so the 1D solution is exact for the 2D
problem; tracers are seeded on a jittered 2D lattice and embedded by a rotation (shock normal n = (cos th, sin th))
and a uniform tangential velocity `slip` (Galilean invariance).  Tracers are advanced inside the solver loop with
Heun's method on linear interpolation of u(x), as in euler2d.

Gas ahead at rest: rho1 = 1, p1 = 1/gamma (c1 = 1).  The exact Rankine-Hugoniot jump is imposed at t = 0 at x = xs,
upstream of the tracer region, so the shock relaxes to its viscous profile before it reaches the first tracer.
"""
import time
import numpy as np
import euler2d as E

G, PR = E.G, 0.72


def rh(Ms):
    r2 = (G + 1) * Ms ** 2 / ((G - 1) * Ms ** 2 + 2); p2 = (1 + 2 * G / (G + 1) * (Ms ** 2 - 1)) / G
    return r2, Ms * (1 - 1 / r2), p2


def _prim(U):
    r = U[0]; u = U[1] / r; p = (G - 1) * (U[2] - 0.5 * r * u * u)
    return r, u, p


def _cons(r, u, p):
    return np.stack([r, r * u, p / (G - 1) + 0.5 * r * u * u])


class NS1D:
    def __init__(self, Ms=2.24, mu=2e-3, dx=0.002, x0=-0.8, x1=3.2, xs=-0.5, cfl=0.4):
        self.Ms, self.mu, self.dx, self.cfl = Ms, mu, dx, cfl
        self.nx = int(round((x1 - x0) / dx)); self.xc = x0 + (np.arange(self.nx) + 0.5) * dx
        r2, u2, p2 = rh(Ms); self.post = (r2, u2, p2)
        r = np.where(self.xc < xs, r2, 1.0); u = np.where(self.xc < xs, u2, 0.0); p = np.where(self.xc < xs, p2, 1 / G)
        self.U = _cons(r, u, p); self.t = 0.0; self.Upost = _cons(np.array(r2), np.array(u2), np.array(p2))

    def rhs(self, U):
        Ug = np.concatenate([np.repeat(self.Upost[:, None], 2, 1), U, np.repeat(U[:, -1:], 2, 1)], 1)
        r, u, p = _prim(Ug); z = 0 * r
        W = np.stack([r, u, z, p, z, z])
        d = E.vanleer(W[:, 1:-1] - W[:, :-2], W[:, 2:] - W[:, 1:-1])
        WL = W[:, 1:-2] + 0.5 * d[:, :-1]; WR = W[:, 2:-1] - 0.5 * d[:, 1:]
        F = E.hllc(WL, WR)[[0, 1, 3]]                                     # faces nx+1
        T = p / r; mu = self.mu; k = mu * G / ((G - 1) * PR)
        ux = (u[2:-1] - u[1:-2]) / self.dx; Tx = (T[2:-1] - T[1:-2]) / self.dx; ua = 0.5 * (u[2:-1] + u[1:-2])
        tau = 4.0 / 3.0 * mu * ux
        Fv = np.stack([0 * tau, tau, ua * tau + k * Tx])
        Ft = F - Fv
        return -(Ft[:, 1:] - Ft[:, :-1]) / self.dx

    def dt(self):
        r, u, p = _prim(self.U); c = np.sqrt(G * p / r)
        dtc = self.cfl * self.dx / np.max(np.abs(u) + c)
        dtv = 0.25 * self.dx ** 2 * r.min() / (self.mu * max(4.0 / 3.0, G / PR))
        return min(dtc, dtv)

    def step(self, dt):
        U0 = self.U; U1 = U0 + dt * self.rhs(U0); U2 = U1 + dt * self.rhs(U1)
        self.U = 0.5 * (U0 + U2); self.t += dt

    def interp(self, f, x):
        fi = (x - self.xc[0]) / self.dx; i0 = np.clip(np.floor(fi).astype(int), 0, self.nx - 2); a = np.clip(fi - i0, 0, 1)
        return f[i0] * (1 - a) + f[i0 + 1] * a

    def thickness(self):
        r, _, _ = _prim(self.U); g = np.abs(np.diff(r)) / self.dx
        return (self.post[0] - 1.0) / g.max(), self.xc[np.argmax(g)]


def run(Ms=2.24, delta_h=2.0, h=0.02, Cs=1.0, theta=0.5, slip=0.3, L=1.0, mu=None, seed=0, mu_per_delta=None,
        verbose=False):
    """delta_h: target max-gradient shock thickness / tracer spacing (mu from the calibration mu_per_delta)."""
    from mfm import lattice
    r2, u2, p2 = rh(Ms); Us = Ms
    delta = delta_h * h
    if mu is None:
        mu = delta * (mu_per_delta[Ms] if mu_per_delta else calib(Ms))
    dx = float(np.clip(delta / 10.0, 0.0015, 0.004))
    n = np.array([np.cos(theta), np.sin(theta)]); tg = np.array([-n[1], n[0]])
    X = lattice(h, (0, L, 0, L), seed); xi0 = X @ n
    sol = NS1D(Ms=Ms, mu=mu, dx=dx, x0=-0.8, x1=xi0.max() + 1.2 + u2 * 1.2 * (xi0.max() + 0.8) / Us, xs=-0.5)
    T = (xi0.max() + 0.5 + 0.25 + 8 * delta) / Us                  # shock beyond the last tracer by > 8 delta
    dtf = Cs * h / Us; S = int(np.ceil(T / dtf)) + 1; tf = np.arange(S) * dtf
    xi = xi0.copy(); fr = np.empty((S, len(X), 2)); rho = np.empty((S, len(X))); xs_hist = np.empty(S); dl = np.empty(S)
    def rec(s):
        r, _, _ = _prim(sol.U); fr[s] = X + (xi - xi0)[:, None] * n + slip * sol.t * tg
        rho[s] = sol.interp(r, xi); dl[s], xs_hist[s] = sol.thickness()
    rec(0); t0 = time.time(); nstep = 0
    for s in range(1, S):
        while sol.t < tf[s] - 1e-13:
            dt = min(sol.dt(), tf[s] - sol.t)
            _, u, _ = _prim(sol.U); u0 = sol.interp(u, xi)
            sol.step(dt); _, u, _ = _prim(sol.U); u1 = sol.interp(u, xi + dt * u0)
            xi = xi + 0.5 * dt * (u0 + u1); nstep += 1
        rec(s)
        if verbose and s % 100 == 0:
            print("   frame %d/%d  %.0fs" % (s, S - 1, time.time() - t0), flush=True)
    return dict(frames=fr, rho=rho, t=tf, dt=dtf, h=h, Ms=Ms, mu=mu, dx=dx, eta=r2, delta=dl, xs=xs_hist, n=n,
                xi0=xi0, slip=slip, theta=theta, nstep=nstep, Us=Us)


def calib(Ms, mu=2e-3, dx=0.0015, t_end=0.6):
    """max-gradient thickness per unit viscosity of the steady NS shock (delta is proportional to mu)"""
    s = NS1D(Ms=Ms, mu=mu, dx=dx, x0=-0.6, x1=2.0, xs=-0.4)
    while s.t < t_end:
        s.step(s.dt())
    d, _ = s.thickness()
    return mu / d
