"""Single-mode RM (Ms=1.5, light rho=1 -> heavy rho=3, At=0.5, a0=0.1, lambda=1, half wavelength y in [0,0.5]).
case 'single' : frame moving with the post-shock contact velocity (1D Riemann), both x-ends transmissive, t_end=6
case 'reshock': lab frame, reflecting wall at x=1.3, t_end=12
Tracers (h=0.005) in the band |phi0|<0.18 around the interface; an adaptive material marker curve (insert at segment
>0.001 or turning >0.15 rad, 4-point Lagrange in the label) is advected with the SAME Heun/bilinear scheme (reference).
Check-pointed: each call runs for <= budget seconds of wall time and resumes from rm_<case>_ck.pkl."""
import numpy as np, pickle, sys, time, os
sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import euler2d as E, cases as C, euler2d_visc as VS
VS.MU = 2e-3
case = sys.argv[1]; budget = float(sys.argv[2]) if len(sys.argv) > 2 else 270.0
ck = '/home/claude/w/rm/rmv2_%s_ck.pkl' % case
A0, XI, LAM, dx, h_p, xs = 0.1, 0.5, 1.0, 0.005, 0.005, 0.2
if case == 'single': Lx, t_end, dtf, shift, wall = 1.2, 6.0, 0.05, True, False
else: Lx, t_end, dtf, shift, wall = 1.3, 12.0, 0.05, False, True
xi = lambda y: XI + A0 * np.cos(2 * np.pi * y / LAM)
dxi = lambda y: -2 * np.pi / LAM * A0 * np.sin(2 * np.pi * y / LAM)


class RM(VS.ViscSolver):
    def __init__(self):
        E.Solver.__init__(self, dx=dx, Lx=Lx, Ly=0.5, Ms=1.5, xs=xs, xb=-10.0, R=0.01, rho_b=3.0)
        X, Y = np.meshgrid(self.xc, self.yc); s = 0.5 * (1 + np.tanh((X - xi(Y)) / (0.75 * dx)))
        W = np.stack([1.0 + 2.0 * s, 0 * X, 0 * X, 1 / E.G + 0 * X, s, s]); sh = X < xs
        for k in range(6): W[k][sh] = self.post[k]
        self.Uf = 0.0
        if shift:
            r2, u2, _, p2 = self.post[:4]; ps, us = C.riemann_star(r2, u2, p2, 3.0, 0.0, 1 / E.G); self.Uf = us; W[1] -= us
        self.U[:, 2:-2, 2:-2] = E.cons(W)

    def ghosts(self, U):
        U[:, :, :2] = U[:, :, 2:3]
        if wall:
            U[:, :, -2:] = U[:, :, [-3, -4]]; U[1, :, -2:] *= -1
        else:
            U[:, :, -2:] = U[:, :, -3:-2]
        U[:, :2, :] = U[:, [3, 2], :]; U[2, :2, :] *= -1
        U[:, -2:, :] = U[:, [-3, -4], :]; U[2, -2:, :] *= -1
        return U


def fold_y(P):
    P[:, 1] = np.abs(P[:, 1]); P[:, 1] = np.where(P[:, 1] > 0.5, 1.0 - P[:, 1], P[:, 1]); return P


def refine(M, lab, smax=0.001, amax=0.15):
    for _ in range(3):
        d = np.diff(M, axis=0); L = np.linalg.norm(d, axis=1)
        ang = np.zeros(len(L)); a = np.arctan2(d[:, 1], d[:, 0]); da = np.abs(np.angle(np.exp(1j * np.diff(a))))
        ang[:-1] = np.maximum(ang[:-1], da); ang[1:] = np.maximum(ang[1:], da)
        need = np.where((L > smax) | ((ang > amax) & (L > 5e-5)))[0]
        if len(need) == 0: break
        n = len(M); i0 = np.clip(need - 1, 0, n - 1); i1 = need; i2 = need + 1; i3 = np.clip(need + 2, 0, n - 1)
        newM = (-M[i0] + 9 * M[i1] + 9 * M[i2] - M[i3]) / 16
        ends = (need == 0) | (need == n - 2); newM[ends] = 0.5 * (M[i1[ends]] + M[i2[ends]])
        newl = 0.5 * (lab[i1] + lab[i2])
        M = np.insert(M, need + 1, newM, axis=0); lab = np.insert(lab, need + 1, newl)
    return M, lab


if os.path.exists(ck):
    st = pickle.load(open(ck, 'rb')); sv = RM.__new__(RM); sv.__dict__.update(st['sv']); P, M, lab = st['P'], st['M'], st['lab']
    rec, fi, nstep = st['rec'], st['fi'], st['nstep']
else:
    sv = RM(); rng = np.random.default_rng(7)
    X0, Y0 = np.meshgrid(np.arange(0.2 + h_p / 2, 0.8, h_p), np.arange(h_p / 2, 0.5, h_p))
    P = np.c_[X0.ravel(), Y0.ravel()] + rng.uniform(-0.25, 0.25, (X0.size, 2)) * h_p; P = fold_y(P)
    phi = (P[:, 0] - xi(P[:, 1])) / np.sqrt(1 + dxi(P[:, 1]) ** 2); P = P[np.abs(phi) < 0.18]
    y0 = np.linspace(0, 0.5, 501); M = np.c_[xi(y0), y0]; lab = y0.copy()
    rec = dict(t=[], frames=[], rho=[], Y1=[], p=[], marker=[]); fi = 0; nstep = 0
nF = int(round(t_end / dtf)) + 1; tf = np.arange(nF) * dtf


def record():
    W = sv.W(); vals = sv.interp(np.stack([W[0], W[3], W[4]]), P)
    rec['t'].append(sv.t); rec['frames'].append(P.copy()); rec['rho'].append(vals[0]); rec['p'].append(vals[1]); rec['Y1'].append(vals[2])
    rec['marker'].append((M.copy(), lab.copy()))


t0 = time.time()
if fi == 0: record(); fi = 1
done = False
while fi < nF:
    while sv.t < tf[fi] - 1e-12:
        dt = min(sv.dt_cfl(), tf[fi] - sv.t); Q = np.r_[P, M]
        W = sv.W(); u0 = sv.interp(W[1:3], Q, odd=(1,)).T; sv.step(dt)
        W = sv.W(); u1 = sv.interp(W[1:3], fold_y(Q + dt * u0), odd=(1,)).T
        Q = fold_y(Q + 0.5 * dt * (u0 + u1)); P, M = Q[:len(P)], Q[len(P):]
        M[0, 1] = 0.0; M[-1, 1] = 0.5; nstep += 1
        if nstep % 4 == 0: M, lab = refine(M, lab)
    M, lab = refine(M, lab); record(); fi += 1
    if fi % 10 == 0: print('frame %d/%d t=%.2f steps=%d markers=%d %.0fs' % (fi - 1, nF - 1, sv.t, nstep, len(M), time.time() - t0), flush=True)
    if time.time() - t0 > budget: break
st = dict(sv={k: v for k, v in sv.__dict__.items()}, P=P, M=M, lab=lab, rec=rec, fi=fi, nstep=nstep)
pickle.dump(st, open(ck, 'wb'))
print('DONE' if fi >= nF else 'PARTIAL', fi, sv.t, flush=True)
