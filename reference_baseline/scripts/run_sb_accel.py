"""Like-for-like reference for T4: the DNS Lagrangian acceleration a = -grad p / rho (+ div tau / rho when viscous,
constant mu: (mu/rho)(lap u + grad(div u)/3)) interpolated to the tracers, and its circulation around every tracer
triangle with the trapezoidal edge rule, sum_e (a_a + a_b)/2 . (x_b - x_a), integrated at EVERY solver step
(trapezoid in time) on the Delaunay triangles of the latest snapshot and accumulated per tracer (one third of each
triangle) and snapshot interval: A[s, i].  This is the continuum counterpart of (d1 g)_t / dt^2 of Theorem 2: summed
over a region it is the boundary integral of the acceleration, so the comparison with the position form involves no
gradient of the (discontinuous) density.  usage: python scripts/run_sb_accel.py inv|visc [budget_s]; repeat until done."""
import sys, os, time, pickle, numpy as np
sys.path[:0] = ["lhd"]
import euler2d as E, lhdx as X
from lhdp import tri_area
kind = sys.argv[1]; mat = len(sys.argv) > 3 and sys.argv[3] == "mat"; budget = float(sys.argv[2]) if len(sys.argv) > 2 else 230
if kind == "visc":
    import euler2d_visc as V; E.Solver = V.ViscSolver; MU = V.MU
else:
    MU = 0.0
ST = "runs/sba_%s%s_state.pkl" % (kind, "_mat" if mat else ""); OUT = "runs/sb_accel_%s%s.npz" % (kind, "_mat" if mat else "")
REF = {"inv": "runs/sb_dns_dx0.005.pkl", "visc": "runs/sbv_dns_dx0.005.pkl"}[kind]
t0 = time.time()
if os.path.exists(ST):
    S_ = pickle.load(open(ST, "rb")); sv, P, fi, A, tf, frames = S_["sv"], S_["P"], S_["fi"], S_["A"], S_["tf"], S_["frames"]
else:
    dx, h_p, t_end, dtf, box, seed = 0.005, 0.005, 1.4, 0.02, (0.3, 1.25, 0.0, 0.42), 4
    sv = E.Solver(dx=dx); rng = np.random.default_rng(seed); x0, x1, y0, y1 = box
    Xg, Yg = np.meshgrid(np.arange(x0 + h_p / 2, x1, h_p), np.arange(y0 + h_p / 2, y1, h_p))
    P = np.c_[Xg.ravel(), Yg.ravel()] + rng.uniform(-0.25, 0.25, (Xg.size, 2)) * h_p; P[:, 1] = np.abs(P[:, 1])
    nF = int(round(t_end / dtf)) + 1; tf = np.arange(nF) * dtf; fi = 1
    A = np.zeros((nF - 1, len(P))); frames = np.empty((nF, len(P), 2)); frames[0] = P
if mat:
    # material triangles of t = 0 (same construction as scripts/v10_t4.py) and hat-kernel weights: A0[s] = int c (t_{s+1}-t)/dt,
    # A1[s] = int c (t - t_s)/dt over interval s, so that the counterpart of dGamma^n (window n-1/2..n+1/2) is A1[n-1] + A0[n]
    tri0 = X.complex_peeled(frames[0])["tri"]; a0 = tri_area(frames[0], tri0); tri0 = np.where((a0 < 0)[:, None], tri0[:, [0, 2, 1]], tri0)
    if not os.path.exists(ST): A0m = np.zeros((len(tf) - 1, len(tri0))); A1m = np.zeros_like(A0m)
    else: A0m, A1m = S_["A0m"], S_["A1m"]


def accel_grid():
    W = sv.W(); d = sv.dx
    Wp = np.pad(W[:4], ((0, 0), (1, 1), (1, 1)), mode="edge"); Wp[2, 0, :] = -Wp[2, 1, :]
    ddx = lambda q: (q[1:-1, 2:] - q[1:-1, :-2]) / (2 * d); ddy = lambda q: (q[2:, 1:-1] - q[:-2, 1:-1]) / (2 * d)
    r = W[0]; ax = -ddx(Wp[3]) / r; ay = -ddy(Wp[3]) / r
    if MU > 0:
        lap = lambda q: (q[1:-1, 2:] + q[1:-1, :-2] + q[2:, 1:-1] + q[:-2, 1:-1] - 4 * q[1:-1, 1:-1]) / d ** 2
        th = np.pad(ddx(Wp[1]) + ddy(Wp[2]), 1, mode="edge")
        ax = ax + MU / r * (lap(Wp[1]) + ddx(th) / 3.0); ay = ay + MU / r * (lap(Wp[2]) + ddy(th) / 3.0)
    return np.stack([ax, ay])


def circ(tri):
    a = sv.interp(accel_grid(), P, odd=(1,)).T; c = np.zeros(len(tri))
    for j in range(3):
        i0, i1 = tri[:, j], tri[:, (j + 1) % 3]; c += (0.5 * (a[i0] + a[i1]) * (P[i1] - P[i0])).sum(1)
    return c


nF = len(tf)
while fi < nF:
    if mat: tri = tri0
    else: tri = X.complex_peeled(P)["tri"]; ar = tri_area(P, tri); tri = np.where((ar < 0)[:, None], tri[:, [0, 2, 1]], tri)
    c0 = circ(tri); acc = np.zeros(len(tri)); ta = tf[fi - 1]; Dt = tf[fi] - tf[fi - 1]
    while sv.t < tf[fi] - 1e-12:
        dt = min(sv.dt_cfl(), tf[fi] - sv.t)
        W = sv.W(); u0 = sv.interp(W[1:3], P, odd=(1,)).T
        sv.step(dt); W = sv.W(); u1 = sv.interp(W[1:3], P + dt * u0, odd=(1,)).T
        P = P + 0.5 * dt * (u0 + u1); P[:, 1] = np.abs(P[:, 1])
        c1 = circ(tri); acc += 0.5 * dt * (c0 + c1)
        if mat:          # trapezoid of c(t) * w(t), w linear in t
            tb = sv.t; wa0, wb0 = (tf[fi] - (tb - dt)) / Dt, (tf[fi] - tb) / Dt
            A0m[fi - 1] += 0.5 * dt * (c0 * wa0 + c1 * wb0); A1m[fi - 1] += 0.5 * dt * (c0 * (1 - wa0) + c1 * (1 - wb0))
        c0 = c1
    if not mat:
        for k in range(3):
            np.add.at(A[fi - 1], tri[:, k], acc / 3.0)
    frames[fi] = P; fi += 1
    if time.time() - t0 > budget and fi < nF:
        pickle.dump(dict(sv=sv, P=P, fi=fi, A=A, tf=tf, frames=frames, **(dict(A0m=A0m, A1m=A1m) if mat else {})), open(ST, "wb")); print("chunk saved at frame", fi, "%.0fs" % (time.time() - t0)); sys.exit()
ref = pickle.load(open(REF, "rb"))["frames"]; md = float(np.abs(frames - ref).max())
if mat: np.savez_compressed(OUT, A0=A0m, A1=A1m, tri0=tri0, t=tf, frames_maxdiff=md)
else: np.savez_compressed(OUT, A=A, t=tf, frames_maxdiff=md)
if os.path.exists(ST): os.remove(ST)
print("done %s; max |frames - stored DNS frames| = %.2e  %.0fs" % (kind, md, time.time() - t0))
