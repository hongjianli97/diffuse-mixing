"""Shock-bubble DNS (identical to run_sb_any.py 0.005: euler2d.run defaults) re-run with the baroclinic torque
B = (grad rho x grad p)/rho^2 integrated at EVERY solver step (trapezoid) on the Delaunay triangles of the latest
snapshot, with P1 gradients of the tracer-interpolated fields, and accumulated per tracer (one third of each triangle)
and per snapshot interval: T[s, i].  This resolves the impulsive deposition in time, which snapshot sampling does not.
Chunked (state pickled in runs/sbt_state.pkl) because of the 300 s command limit: call repeatedly until 'done'."""
import sys, os, time, pickle, numpy as np
sys.path[:0] = ["lhd"]
import euler2d as E, lhdx as X
from t4split import p1_cross
from lhdp import tri_area
ST = "runs/sbt_state.pkl"; OUT = "runs/sb_torque.npz"; budget = float(sys.argv[1]) if len(sys.argv) > 1 else 230
t0 = time.time()
if os.path.exists(ST):
    S_ = pickle.load(open(ST, "rb")); sv, P, fi, T, tf = S_["sv"], S_["P"], S_["fi"], S_["T"], S_["tf"]; frames = S_["frames"]
else:
    dx, h_p, t_end, dtf, box, seed = 0.005, 0.005, 1.4, 0.02, (0.3, 1.25, 0.0, 0.42), 4
    sv = E.Solver(dx=dx); rng = np.random.default_rng(seed); x0, x1, y0, y1 = box
    Xg, Yg = np.meshgrid(np.arange(x0 + h_p / 2, x1, h_p), np.arange(y0 + h_p / 2, y1, h_p))
    P = np.c_[Xg.ravel(), Yg.ravel()] + rng.uniform(-0.25, 0.25, (Xg.size, 2)) * h_p; P[:, 1] = np.abs(P[:, 1])
    nF = int(round(t_end / dtf)) + 1; tf = np.arange(nF) * dtf; fi = 1
    T = np.zeros((nF - 1, len(P))); frames = np.empty((nF, len(P), 2)); frames[0] = P
def torque(tri):
    W = sv.W(); r, p = sv.interp(np.stack([W[0], W[3]]), P)
    return p1_cross(P, tri, r, p)
nF = len(tf)
while fi < nF:
    tri = X.complex_peeled(P)["tri"]; A = tri_area(P, tri); tri = np.where((A < 0)[:, None], tri[:, [0, 2, 1]], tri)
    b0 = torque(tri); acc = np.zeros(len(tri))
    while sv.t < tf[fi] - 1e-12:
        dt = min(sv.dt_cfl(), tf[fi] - sv.t)
        W = sv.W(); u0 = sv.interp(W[1:3], P, odd=(1,)).T
        sv.step(dt); W = sv.W(); u1 = sv.interp(W[1:3], P + dt * u0, odd=(1,)).T
        P = P + 0.5 * dt * (u0 + u1); P[:, 1] = np.abs(P[:, 1])
        b1 = torque(tri); acc += 0.5 * dt * (b0 + b1); b0 = b1
    for k in range(3):
        np.add.at(T[fi - 1], tri[:, k], acc / 3.0)
    frames[fi] = P; fi += 1
    if time.time() - t0 > budget and fi < nF:
        pickle.dump(dict(sv=sv, P=P, fi=fi, T=T, tf=tf, frames=frames), open(ST, "wb")); print("chunk saved at frame", fi, "t=%.2f" % sv.t, "%.0fs" % (time.time() - t0)); sys.exit()
ref = pickle.load(open("runs/sb_dns_dx0.005.pkl", "rb"))["frames"]
np.savez_compressed(OUT, T=T, t=tf, frames_maxdiff=np.abs(frames - ref).max())
if os.path.exists(ST): os.remove(ST)
print("done; max |frames - sb_dns frames| = %.2e" % np.abs(frames - ref).max(), "%.0fs" % (time.time() - t0))
