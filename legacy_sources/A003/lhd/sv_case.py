"""Shock / isentropic-vortex interaction (classic: Ellzey et al. 1995, Inoue & Hattori 1999).
Domain [0,2]x[0,1], walls top/bottom, vortex at rest at (0.7,0.5), R_v = 0.075, swirl Mach 0.5,
planar shock Ms = 1.3 entering from the left.  Snapshots every 0.02 (shock moves 5.2 cells per snapshot).
Tracers are integrated twice from the SAME seeds:
  inloop  : Heun at every solver step (bilinear in space)          -> the physics
  offline : RK4 on stored snapshots, linear in time, bilinear space -> standard post-processing
The physics is identical; any difference between the two is a tracking artefact at the shock."""
import pickle, time
import numpy as np
import euler2d as E

G = E.G


class SV(E.Solver):
    def __init__(self, dx=0.005, Lx=2.0, Ly=1.0, Ms=1.3, xs=0.35, xv=0.7, yv=0.5, Rv=0.075, Uv=0.5):
        super().__init__(dx=dx, Lx=Lx, Ly=Ly, Ms=Ms, xs=xs, xb=-10.0, R=0.0, rho_b=1.0)
        X, Y = np.meshgrid(self.xc, self.yc)
        rr = np.hypot(X - xv, Y - yv) / Rv
        uth = Uv * rr * np.exp(0.5 * (1 - rr ** 2))
        T = 1 / G - (G - 1) / (2 * G) * Uv ** 2 * np.exp(1 - rr ** 2)
        rho = (T * G) ** (1 / (G - 1)); p = rho * T
        th = np.arctan2(Y - yv, X - xv)
        W = np.stack([rho, -uth * np.sin(th), uth * np.cos(th), p, (rr < 1.5).astype(float), (rr < 3.0).astype(float)])
        sh = X < xs
        for k in range(6):
            W[k][sh] = self.post[k]
        self.U[:, 2:-2, 2:-2] = E.cons(W)
        self.xv, self.yv, self.Rv = xv, yv, Rv


def interp_uv(sv, F, P):
    return sv.interp(F, P).T          # no symmetry line here: walls at y=0,1 (P stays inside)


def run(t_end=0.9, dt_frame=0.02, box=(0.45, 1.05, 0.2, 0.8), h_p=0.005, seed=5, keep=(0.2, 0.4, 0.6)):
    sv = SV()
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(np.arange(box[0] + h_p / 2, box[1], h_p), np.arange(box[2] + h_p / 2, box[3], h_p))
    P = np.c_[X.ravel(), Y.ravel()] + rng.uniform(-0.25, 0.25, (X.size, 2)) * h_p
    P0 = P.copy(); N = len(P)
    nF = int(round(t_end / dt_frame)) + 1; tf = np.arange(nF) * dt_frame
    fr = np.empty((nF, N, 2)); Uf = np.empty((nF, 2, sv.ny, sv.nx), np.float32)
    Ev = {k: np.empty((nF, N), np.float32) for k in ("om", "th", "p", "rho")}
    grids = {}

    def record(fi):
        W = sv.W(); om, th, _ = sv.derived(W)
        fr[fi] = P; Uf[fi] = W[1:3].astype(np.float32)
        vals = sv.interp(np.stack([om, th, W[3], W[0]]), P)
        for k, v in zip(("om", "th", "p", "rho"), vals):
            Ev[k][fi] = v
        if any(abs(sv.t - tk) < 1e-9 for tk in keep):
            grids[round(sv.t, 4)] = dict(rho=W[0].astype(np.float32), om=om.astype(np.float32))
    record(0); t0 = time.time(); n = 0
    for fi in range(1, nF):
        while sv.t < tf[fi] - 1e-12:
            dt = min(sv.dt_cfl(), tf[fi] - sv.t)
            W = sv.W(); u0 = interp_uv(sv, W[1:3], P)
            sv.step(dt)
            W = sv.W(); u1 = interp_uv(sv, W[1:3], P + dt * u0)
            P = P + 0.5 * dt * (u0 + u1); n += 1
        record(fi)
        if fi % 10 == 0:
            print("  frame %d/%d  steps %d  %.0fs" % (fi, nF - 1, n, time.time() - t0), flush=True)
    # ---- offline tracking from the stored snapshots (linear in time, bilinear in space, RK4)
    def vel(Pq, tq):
        k = min(int(tq / dt_frame), nF - 2); a = (tq - tf[k]) / dt_frame
        return (1 - a) * sv.interp(Uf[k], Pq).T + a * sv.interp(Uf[k + 1], Pq).T
    out = {}
    for nsub in (4,):
        Q = P0.copy(); frq = np.empty_like(fr); frq[0] = Q; hs = dt_frame / nsub
        for fi in range(nF - 1):
            for j in range(nsub):
                tq = tf[fi] + j * hs
                k1 = vel(Q, tq); k2 = vel(Q + .5 * hs * k1, tq + .5 * hs); k3 = vel(Q + .5 * hs * k2, tq + .5 * hs); k4 = vel(Q + hs * k3, tq + hs)
                Q = Q + hs * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            frq[fi + 1] = Q
        out[nsub] = frq
    return dict(fr_in=fr, fr_off=out[4], t=tf, dt=dt_frame, Ev=Ev, grids=grids, xc=sv.xc, yc=sv.yc, dx=sv.dx,
                P0=P0, Ms=sv.Ms, xs0=0.35, xv=sv.xv, yv=sv.yv, Rv=sv.Rv, nstep=n, post=sv.post)


if __name__ == "__main__":
    o = run()
    pickle.dump(o, open("sv_dns.pkl", "wb"))
    d = np.linalg.norm(o["fr_in"][-1] - o["fr_off"][-1], axis=1)
    print("done; steps", o["nstep"], " in-loop vs offline final position diff: median %.2e, p99 %.2e (h=0.005)" % (np.median(d), np.quantile(d, 0.99)))
