"""euler2d.py -- planar shock / heavy-bubble interaction (single gamma, density contrast 3, Ms = 1.5).
Finite volume, MUSCL (van Leer limiter on primitives) + HLLC, SSP-RK2, CFL 0.4.
Half domain [0,2.4]x[0,0.6]: y=0 is the symmetry line (reflective), top wall reflective,
left inflow at the post-shock state, right outflow.  Two passive scalars: Y1 = bubble (r<R),
Y2 = bubble + 0.05 margin (r<R+0.05), used for Eulerian circulation of the same material regions.
Tracers are advected in the loop with Heun's method on bilinear cell-centre interpolation.
Per frame we keep tracer positions and Eulerian values interpolated to tracers (rho,u,v,p,omega,theta,Y1);
per step we integrate the Eulerian baroclinic torque over Y2>1/2 (Kelvin reference).
"""
import time
import numpy as np

G = 1.4


def prim(U):
    r = U[0]; u = U[1] / r; v = U[2] / r
    p = (G - 1) * (U[3] - 0.5 * r * (u * u + v * v))
    return np.stack([r, u, v, p, U[4] / r, U[5] / r])


def cons(W):
    r, u, v, p = W[0], W[1], W[2], W[3]
    return np.stack([r, r * u, r * v, p / (G - 1) + 0.5 * r * (u * u + v * v), r * W[4], r * W[5]])


def vanleer(a, b):
    return np.where(a * b > 0, 2 * a * b / (a + b + 1e-300), 0.0)


def hllc(WL, WR):
    """x-normal HLLC flux; W = (rho, un, ut, p, Y1, Y2)."""
    rL, uL, vL, pL = WL[0], WL[1], WL[2], WL[3]
    rR, uR, vR, pR = WR[0], WR[1], WR[2], WR[3]
    cL = np.sqrt(G * pL / rL); cR = np.sqrt(G * pR / rR)
    SL = np.minimum(uL - cL, uR - cR); SR = np.maximum(uL + cL, uR + cR)
    EL = pL / (G - 1) + 0.5 * rL * (uL ** 2 + vL ** 2); ER = pR / (G - 1) + 0.5 * rR * (uR ** 2 + vR ** 2)
    Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (rL * (SL - uL) - rR * (SR - uR))

    def F(r, u, v, p, E, Y1, Y2):
        return np.stack([r * u, r * u * u + p, r * u * v, (E + p) * u, r * u * Y1, r * u * Y2])

    def Uv(r, u, v, E, Y1, Y2):
        return np.stack([r, r * u, r * v, E, r * Y1, r * Y2])
    FL = F(rL, uL, vL, pL, EL, WL[4], WL[5]); FR = F(rR, uR, vR, pR, ER, WR[4], WR[5])
    UL = Uv(rL, uL, vL, EL, WL[4], WL[5]); UR = Uv(rR, uR, vR, ER, WR[4], WR[5])

    def Ustar(r, u, v, p, E, Y1, Y2, S):
        k = r * (S - u) / (S - Ss)
        return np.stack([k, k * Ss, k * v, k * (E / r + (Ss - u) * (Ss + p / (r * (S - u)))), k * Y1, k * Y2])
    FsL = FL + SL * (Ustar(rL, uL, vL, pL, EL, WL[4], WL[5], SL) - UL)
    FsR = FR + SR * (Ustar(rR, uR, vR, pR, ER, WR[4], WR[5], SR) - UR)
    return np.where(SL >= 0, FL, np.where(Ss >= 0, FsL, np.where(SR > 0, FsR, FR)))


class Solver:
    def __init__(self, dx=0.005, Lx=2.4, Ly=0.6, Ms=1.5, xs=0.25, xb=0.75, R=0.25, rho_b=3.0, cfl=0.4):
        self.dx = dx; self.nx = int(round(Lx / dx)); self.ny = int(round(Ly / dx)); self.cfl = cfl
        self.xc = (np.arange(self.nx) + 0.5) * dx; self.yc = (np.arange(self.ny) + 0.5) * dx
        X, Y = np.meshgrid(self.xc, self.yc)
        r1, p1 = 1.0, 1.0 / G
        r2 = r1 * (G + 1) * Ms ** 2 / ((G - 1) * Ms ** 2 + 2)
        p2 = p1 * (1 + 2 * G / (G + 1) * (Ms ** 2 - 1))
        u2 = Ms * (1 - r1 / r2)
        self.post = np.array([r2, u2, 0.0, p2, 0.0, 0.0]); self.Ms = Ms; self.u2 = u2
        rb = np.hypot(X - xb, Y)
        s1 = 0.5 * (1 - np.tanh((rb - R) / (0.75 * dx)))
        s2 = 0.5 * (1 - np.tanh((rb - R - 0.05) / (0.75 * dx)))
        W = np.stack([r1 + (rho_b - r1) * s1, 0 * X, 0 * X, p1 + 0 * X, s1, s2])
        sh = X < xs
        for k in range(6):
            W[k][sh] = self.post[k]
        self.U = np.zeros((6, self.ny + 4, self.nx + 4))
        self.U[:, 2:-2, 2:-2] = cons(W)
        self.Upost = cons(self.post[:, None, None])[:, 0, 0]
        self.t = 0.0; self.xb, self.R = xb, R

    def ghosts(self, U):
        U[:, :, :2] = self.Upost[:, None, None]
        U[:, :, -2:] = U[:, :, -3:-2]
        U[:, :2, :] = U[:, [3, 2], :]; U[2, :2, :] *= -1
        U[:, -2:, :] = U[:, [-3, -4], :]; U[2, -2:, :] *= -1
        return U

    def rhs(self, U):
        U = self.ghosts(U)
        W = prim(U)
        # x-direction
        Wi = W[:, 2:-2, :]
        d = vanleer(Wi[:, :, 1:-1] - Wi[:, :, :-2], Wi[:, :, 2:] - Wi[:, :, 1:-1])     # slopes, cells 1..nx+2
        WL = Wi[:, :, 1:-2] + 0.5 * d[:, :, :-1]; WR = Wi[:, :, 2:-1] - 0.5 * d[:, :, 1:]
        Fx = hllc(WL, WR)                                                                # faces nx+1
        # y-direction (swap u,v)
        Wj = W[:, :, 2:-2][[0, 2, 1, 3, 4, 5]]
        d = vanleer(Wj[:, 1:-1] - Wj[:, :-2], Wj[:, 2:] - Wj[:, 1:-1])
        WL = Wj[:, 1:-2] + 0.5 * d[:, :-1]; WR = Wj[:, 2:-1] - 0.5 * d[:, 1:]
        Fy = hllc(WL, WR)[[0, 2, 1, 3, 4, 5]]
        return -(Fx[:, :, 1:] - Fx[:, :, :-1]) / self.dx - (Fy[:, 1:] - Fy[:, :-1]) / self.dx

    def dt_cfl(self):
        W = prim(self.U[:, 2:-2, 2:-2]); c = np.sqrt(G * W[3] / W[0])
        return self.cfl * self.dx / np.max(np.maximum(np.abs(W[1]), np.abs(W[2])) + c)

    def step(self, dt):
        U0 = self.U.copy()
        U1 = U0.copy(); U1[:, 2:-2, 2:-2] += dt * self.rhs(U0)
        U2 = U1.copy(); U2[:, 2:-2, 2:-2] += dt * self.rhs(U1)
        self.U[:, 2:-2, 2:-2] = 0.5 * (U0[:, 2:-2, 2:-2] + U2[:, 2:-2, 2:-2])
        self.t += dt

    # ------------------------------------------------------------------ diagnostics
    def W(self):
        return prim(self.U[:, 2:-2, 2:-2])

    def derived(self, W=None):
        W = self.W() if W is None else W
        Wp = np.pad(W, ((0, 0), (1, 1), (1, 1)), mode="edge")
        Wp[2, 0, :] = -Wp[2, 1, :]                        # v odd at the symmetry line
        ddx = lambda q: (q[1:-1, 2:] - q[1:-1, :-2]) / (2 * self.dx)
        ddy = lambda q: (q[2:, 1:-1] - q[:-2, 1:-1]) / (2 * self.dx)
        om = ddx(Wp[2]) - ddy(Wp[1]); th = ddx(Wp[1]) + ddy(Wp[2])
        bar = (ddx(Wp[0]) * ddy(Wp[3]) - ddy(Wp[0]) * ddx(Wp[3])) / W[0] ** 2
        return om, th, bar

    def interp(self, F, P, odd=()):
        """bilinear interpolation of cell-centred fields F (k,ny,nx) at points P (n,2)."""
        fx = P[:, 0] / self.dx - 0.5; fy = P[:, 1] / self.dx - 0.5
        i0 = np.clip(np.floor(fx).astype(int), 0, self.nx - 2); j0 = np.floor(fy).astype(int)
        ax = np.clip(fx - i0, 0, 1)
        neg = j0 < 0                                       # below the first centre: mirror
        j0 = np.clip(j0, 0, self.ny - 2); ay = np.clip(fy - j0, 0, 1)
        out = (F[:, j0, i0] * (1 - ax) * (1 - ay) + F[:, j0, i0 + 1] * ax * (1 - ay)
               + F[:, j0 + 1, i0] * (1 - ax) * ay + F[:, j0 + 1, i0 + 1] * ax * ay)
        if neg.any():   # linear to the symmetry line for v (odd), constant for even fields
            a = (fy[neg] + 0.5) / 0.5
            out[:, neg] = F[:, 0, i0[neg]] * (1 - ax[neg]) + F[:, 0, i0[neg] + 1] * ax[neg]
            for k in odd:
                out[k, neg] *= np.clip(a, 0, 1)
        return out


def run(dx=0.005, h_p=0.005, t_end=1.4, dt_frame=0.02, box=(0.3, 1.25, 0.0, 0.42), seed=4,
        keep_grid=(0.3, 0.6, 1.0, 1.4), verbose=True):
    sv = Solver(dx=dx)
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = box
    X, Y = np.meshgrid(np.arange(x0 + h_p / 2, x1, h_p), np.arange(y0 + h_p / 2, y1, h_p))
    P = np.c_[X.ravel(), Y.ravel()] + rng.uniform(-0.25, 0.25, (X.size, 2)) * h_p
    P[:, 1] = np.abs(P[:, 1])
    nF = int(round(t_end / dt_frame)) + 1
    tf = np.arange(nF) * dt_frame
    N = len(P)
    frames = np.empty((nF, N, 2)); E = {k: np.empty((nF, N)) for k in ("rho", "u", "v", "p", "om", "th", "Y1")}
    Gam = dict(t=[], Y1=[], Y2=[], tot=[], bar=[])
    grids = {}
    barint = 0.0; dA = dx * dx

    def record(fi):
        W = sv.W(); om, th, bar = sv.derived(W)
        Fv = np.stack([W[0], W[1], W[2], W[3], om, th, W[4]])
        vals = sv.interp(Fv, P, odd=(2, 4))
        frames[fi] = P
        for k, v in zip(("rho", "u", "v", "p", "om", "th", "Y1"), vals):
            E[k][fi] = v
        if any(abs(sv.t - tk) < 1e-9 for tk in keep_grid):
            grids[round(sv.t, 4)] = dict(W=W.astype(np.float32), om=om.astype(np.float32), th=th.astype(np.float32))
    def gamma_sample(W, om, bar):
        Gam["t"].append(sv.t); Gam["Y1"].append(float((om * (W[4] > 0.5)).sum() * dA))
        Gam["Y2"].append(float((om * (W[5] > 0.5)).sum() * dA)); Gam["tot"].append(float(om.sum() * dA))
        Gam["bar"].append(barint)
    record(0)
    W = sv.W(); om, th, bar = sv.derived(W); gamma_sample(W, om, bar)
    bar_prev = float((bar * (W[5] > 0.5)).sum() * dA)
    t0 = time.time(); nstep = 0
    for fi in range(1, nF):
        while sv.t < tf[fi] - 1e-12:
            dt = min(sv.dt_cfl(), tf[fi] - sv.t)
            W = sv.W()
            u0 = sv.interp(W[1:3], P, odd=(1,)).T
            sv.step(dt)
            W = sv.W()
            Ps = P + dt * u0
            u1 = sv.interp(W[1:3], Ps, odd=(1,)).T
            P = P + 0.5 * dt * (u0 + u1)
            P[:, 1] = np.abs(P[:, 1])
            om, th, bar = sv.derived(W)
            b = float((bar * (W[5] > 0.5)).sum() * dA)
            barint += 0.5 * dt * (b + bar_prev); bar_prev = b
            nstep += 1
            if nstep % 10 == 0:
                gamma_sample(W, om, bar)
        record(fi)
        if verbose and fi % 10 == 0:
            print("  frame %d/%d t=%.3f steps=%d  %.0fs" % (fi, nF - 1, sv.t, nstep, time.time() - t0), flush=True)
    W = sv.W(); om, th, bar = sv.derived(W); gamma_sample(W, om, bar)
    return dict(frames=frames, t=tf, dt=dt_frame, E=E, Gam={k: np.array(v) for k, v in Gam.items()},
                grids=grids, xc=sv.xc, yc=sv.yc, dx=dx, h=h_p, post=sv.post, xb=sv.xb, R=sv.R,
                P0=frames[0], nstep=nstep)


if __name__ == "__main__":
    import sys, pickle
    out = run()
    with open("sb_dns.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print("done", out["nstep"], "steps; Gamma Y2 final", out["Gam"]["Y2"][-1], "bar", out["Gam"]["bar"][-1])
