"""Viscous shock-bubble (euler2d_visc: mu=5e-4, Pr=0.72, Sc=1), same seeding as euler2d.run (seed 4, h_p=0.005),
check-pointed.  Per frame at the tracers: rho,u,v,p,om,th,Y1 and DNS references of the viscous force div(tau) and the
dissipation Phi = tau:grad u (reference only)."""
import numpy as np, pickle, sys, os, time
sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import euler2d as E, euler2d_visc as V
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 250
CK = '/home/claude/w/visc/sbv_ck.pkl'; MU = V.MU
dx, h_p, t_end, dtf, box, seed = 0.005, 0.005, 1.4, 0.02, (0.3, 1.25, 0.0, 0.42), 4
nF = int(round(t_end / dtf)) + 1; tf = np.arange(nF) * dtf; keep = (0.3, 0.6, 1.0, 1.4)
if os.path.exists(CK):
    st = pickle.load(open(CK, 'rb')); sv = V.ViscSolver.__new__(V.ViscSolver); sv.__dict__.update(st['sv']); P, R, fi = st['P'], st['R'], st['fi']
else:
    sv = V.ViscSolver(dx=dx); rng = np.random.default_rng(seed); x0, x1, y0, y1 = box
    Xg, Yg = np.meshgrid(np.arange(x0 + h_p / 2, x1, h_p), np.arange(y0 + h_p / 2, y1, h_p))
    P = np.c_[Xg.ravel(), Yg.ravel()] + rng.uniform(-0.25, 0.25, (Xg.size, 2)) * h_p; P[:, 1] = np.abs(P[:, 1])
    R = dict(frames=[], t=[], grids={}, **{k: [] for k in ('rho', 'u', 'v', 'p', 'om', 'th', 'Y1', 'fvx', 'fvy', 'Phi')}); fi = 0


def viscous_fields(W):
    Wp = np.pad(W[:3], ((0, 0), (1, 1), (1, 1)), mode='edge'); Wp[2, 0, :] = -Wp[2, 1, :]
    ddx = lambda q: (q[1:-1, 2:] - q[1:-1, :-2]) / (2 * dx); ddy = lambda q: (q[2:, 1:-1] - q[:-2, 1:-1]) / (2 * dx)
    ux, uy, vx, vy = ddx(Wp[1]), ddy(Wp[1]), ddx(Wp[2]), ddy(Wp[2])
    txx = MU * (4 / 3 * ux - 2 / 3 * vy); tyy = MU * (4 / 3 * vy - 2 / 3 * ux); txy = MU * (uy + vx)
    T = np.stack([txx, txy, tyy]); Tp = np.pad(T, ((0, 0), (1, 1), (1, 1)), mode='edge'); Tp[1, 0, :] = -Tp[1, 1, :]
    fvx = ddx(Tp[0]) + ddy(Tp[1]); fvy = ddx(Tp[1]) + ddy(Tp[2])
    Phi = txx * ux + txy * (uy + vx) + tyy * vy
    return fvx, fvy, Phi


def record():
    W = sv.W(); om, th, bar = sv.derived(W); fvx, fvy, Phi = viscous_fields(W)
    vals = sv.interp(np.stack([W[0], W[1], W[2], W[3], om, th, W[4], fvx, fvy, Phi]), P, odd=(2, 4, 8))
    R['frames'].append(P.copy()); R['t'].append(sv.t)
    for k, v in zip(('rho', 'u', 'v', 'p', 'om', 'th', 'Y1', 'fvx', 'fvy', 'Phi'), vals): R[k].append(v)
    if any(abs(sv.t - tk) < 1e-9 for tk in keep):
        R['grids'][round(sv.t, 4)] = dict(W=W.astype(np.float32), om=om.astype(np.float32), th=th.astype(np.float32), Phi=Phi.astype(np.float32),
                                         fvx=fvx.astype(np.float32), fvy=fvy.astype(np.float32))


t0 = time.time()
if fi == 0: record(); fi = 1
while fi < nF:
    while sv.t < tf[fi] - 1e-12:
        dt = min(sv.dt_cfl(), tf[fi] - sv.t); W = sv.W(); u0 = sv.interp(W[1:3], P, odd=(1,)).T; sv.step(dt)
        W = sv.W(); Ps = P + dt * u0; Ps[:, 1] = np.abs(Ps[:, 1]); u1 = sv.interp(W[1:3], Ps, odd=(1,)).T
        P = P + 0.5 * dt * (u0 + u1); P[:, 1] = np.abs(P[:, 1])
    record(); fi += 1
    if fi % 10 == 0: print('frame %d t=%.2f %.0fs' % (fi - 1, sv.t, time.time() - t0), flush=True)
    if time.time() - t0 > budget: break
pickle.dump(dict(sv=dict(sv.__dict__), P=P, R=R, fi=fi), open(CK, 'wb'))
print('DONE' if fi >= nF else 'PARTIAL', fi, flush=True)
