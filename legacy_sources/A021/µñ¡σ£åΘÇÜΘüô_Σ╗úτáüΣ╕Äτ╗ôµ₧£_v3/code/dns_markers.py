"""Re-run the two DNS cases with adaptive material marker curves tracked in-solver (same Heun/bilinear scheme as
the tracers).  The marker curves are the reference for the interface stretch.  Usage: python dns_markers.py sb|sv"""
import sys, time, pickle, numpy as np
import euler2d as E
import stretch_est as SE

which = sys.argv[1]
SNAP = {}


def advance_markers(sv, curves, Wold, Wnew, dt, odd, sym):
    for mc in curves:
        u0 = sv.interp(Wold[1:3], mc.X, odd=odd).T
        u1 = sv.interp(Wnew[1:3], mc.X + dt * u0, odd=odd).T
        mc.X = mc.X + 0.5 * dt * (u0 + u1)
        if sym:
            mc.X[:, 1] = np.abs(mc.X[:, 1])


if which == "sb":
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
    sv = E.Solver(dx=0.005)
    h_p, t_end, dt_frame, box, seed = 0.005, 1.4, 0.02, (0.3, 1.25, 0.0, 0.42), 4
    rng = np.random.default_rng(seed)
    Xg, Yg = np.meshgrid(np.arange(box[0] + h_p / 2, box[1], h_p), np.arange(box[2] + h_p / 2, box[3], h_p))
    P = np.c_[Xg.ravel(), Yg.ravel()] + rng.uniform(-0.25, 0.25, (Xg.size, 2)) * h_p; P[:, 1] = np.abs(P[:, 1])
    radii = (sv.R - 0.01, sv.R, sv.R + 0.01)
    curves = [SE.MarkerCurve((lambda rr: (lambda s: np.c_[sv.xb + rr * np.cos(s), rr * np.sin(s)]))(rr), 0.0, np.pi, 400, False, 0.001,
                             amax=0.15, cap=300000) for rr in radii]
    odd, sym, keepg = (1,), True, (0.3, 0.6, 1.0, 1.4)
else:
    import sv_case as SVC
    sv = SVC.SV()
    h_p, t_end, dt_frame, box, seed = 0.005, 0.6, 0.02, (0.45, 1.05, 0.2, 0.8), 5
    rng = np.random.default_rng(seed)
    Xg, Yg = np.meshgrid(np.arange(box[0] + h_p / 2, box[1], h_p), np.arange(box[2] + h_p / 2, box[3], h_p))
    P = np.c_[Xg.ravel(), Yg.ravel()] + rng.uniform(-0.25, 0.25, (Xg.size, 2)) * h_p
    radii = (2 * sv.Rv,)
    curves = [SE.MarkerCurve(lambda s: np.c_[sv.xv + 2 * sv.Rv * np.cos(s), sv.yv + 2 * sv.Rv * np.sin(s)], -np.pi, np.pi, 600, True,
                             0.001, amax=0.15, cap=300000)]
    odd, sym, keepg = (), False, (0.2, 0.4, 0.6)

P0 = P.copy(); N = len(P); nF = int(round(t_end / dt_frame)) + 1; tf = np.arange(nF) * dt_frame
frames = np.empty((nF, N, 2)); rhoT = np.empty((nF, N), np.float32); YT = np.empty((nF, N), np.float32)
Uf = np.empty((nF, 2, sv.ny, sv.nx), np.float32) if which == "sv" else None
grids = {}; mstats = {}


def record(fi):
    W = sv.W(); frames[fi] = P
    vals = sv.interp(np.stack([W[0], W[4]]), P); rhoT[fi] = vals[0]; YT[fi] = vals[1]
    if Uf is not None:
        Uf[fi] = W[1:3].astype(np.float32)
    if any(abs(sv.t - k) < 1e-9 for k in keepg):
        om, th, _ = sv.derived(W)
        grids[fi] = dict(rho=W[0].astype(np.float32), Y=W[4].astype(np.float32), om=om.astype(np.float32))
    if fi % 5 == 0 and fi > 0:
        mstats[fi] = []
        for mc in curves:
            st = mc.stats(); mstats[fi].append(dict(mean_ln=st["mean_ln"], ln_total=st["ln_total"], n=len(mc.s),
                                                    s_mid=st["s_mid"].astype(np.float32), ln=st["ln"].astype(np.float32),
                                                    X=mc.X.astype(np.float32)))


record(0); t0 = time.time(); nstep = 0
for fi in range(1, nF):
    while sv.t < tf[fi] - 1e-12:
        dt = min(sv.dt_cfl(), tf[fi] - sv.t)
        Wold = sv.W(); u0 = sv.interp(Wold[1:3], P, odd=odd).T
        sv.step(dt)
        Wnew = sv.W(); u1 = sv.interp(Wnew[1:3], P + dt * u0, odd=odd).T
        P = P + 0.5 * dt * (u0 + u1)
        if sym:
            P[:, 1] = np.abs(P[:, 1])
        advance_markers(sv, curves, Wold, Wnew, dt, odd, sym)
        nstep += 1
        if nstep % 3 == 0:
            for mc in curves:
                mc.refine()
    for mc in curves:
        mc.refine()
    record(fi)
    if fi % 5 == 0:
        print("frame %d/%d t=%.2f steps=%d markers=%s %.0fs" % (fi, nF - 1, sv.t, nstep, [len(m.s) for m in curves], time.time() - t0), flush=True)

out = dict(frames=frames, t=tf, dt=dt_frame, rho=rhoT, Y=YT, grids=grids, mstats=mstats, radii=radii, xc=sv.xc, yc=sv.yc, dx=sv.dx)
if which == "sb":
    out.update(xb=sv.xb, R=sv.R)
else:
    def vel(Pq, tq):
        k = min(int(tq / dt_frame), nF - 2); a = (tq - tf[k]) / dt_frame
        return (1 - a) * sv.interp(Uf[k], Pq).T + a * sv.interp(Uf[k + 1], Pq).T
    Q = P0.copy(); frq = np.empty_like(frames); frq[0] = Q; hs = dt_frame / 4
    for fi in range(nF - 1):
        for j in range(4):
            tq = tf[fi] + j * hs
            k1 = vel(Q, tq); k2 = vel(Q + .5 * hs * k1, tq + .5 * hs); k3 = vel(Q + .5 * hs * k2, tq + .5 * hs); k4 = vel(Q + hs * k3, tq + hs)
            Q = Q + hs * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        frq[fi + 1] = Q
    out.update(frames_off=frq, xv=sv.xv, yv=sv.yv, Rv=sv.Rv)
pickle.dump(out, open("dns_%s.pkl" % which, "wb"))
print("done", nstep, time.time() - t0)
