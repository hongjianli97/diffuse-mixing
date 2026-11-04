"""suite2.py <stokes|igw|conv|noise> -- further exact-solution checks, all through ltn_dyn.LTN, fields saved for plotting.
stokes : Stokes' first problem (impulsively moved wall, exact NS).  Wall treated (a) as a free network boundary,
         (b) with the wall extension (image tracers; the extension is itself the exact full-plane solution).
         Tests vorticity at the wall, mu_E (boundary term!), dissipation, 'no pressure'.
igw    : linear internal gravity wave (Boussinesq, exact linear).  Strong baroclinic torque: curl(a) = -N^2 dzeta/dx,
         zeta = vertical displacement from the equilibrium height (positions + one reference).  N^2 inverted from positions.
conv   : Taylor-Green at n = 40, 80, 120 tracers per period: convergence orders.
noise  : Taylor-Green with Gaussian position noise sigma/h = 0 ... 1e-2, versus N_g and N_a."""
import numpy as np, sys, json, time
from scipy.special import erfc, erf
sys.path.insert(0, '/home/claude/w/ltn'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd'); sys.path.insert(0, '/home/claude/w')
import ltn_dyn as LD, lhdx as X
case = sys.argv[1]; t0w = time.time(); out = {}; save = {}


def lattice(x0, x1, y0, y1, h, seed):
    rng = np.random.default_rng(seed); gx = np.arange(x0 + h / 2, x1, h); gy = np.arange(y0 + h / 2, y1, h)
    Xg, Yg = np.meshgrid(gx, gy); return np.c_[Xg.ravel(), Yg.ravel()] + rng.uniform(-0.25, 0.25, (Xg.size, 2)) * h


def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def tg_case(n, noise=0.0, nu=0.01, dtf=0.02, S=7, s=3, seed=5):
    L = 2 * np.pi; h = L / n; P0 = lattice(0, L, 0, L, h, seed); N0 = len(P0)
    vel = lambda P, t: np.exp(-2 * nu * t) * np.c_[np.sin(P[:, 0]) * np.cos(P[:, 1]), -np.cos(P[:, 0]) * np.sin(P[:, 1])]
    P = P0.copy(); frames = [P.copy()]; t = 0.0; sub = 40; hs = dtf / sub
    for f in range(1, S):
        for _ in range(sub):
            k1 = vel(P, t); k2 = vel(P + 0.5 * hs * k1, t + 0.5 * hs); k3 = vel(P + 0.5 * hs * k2, t + 0.5 * hs); k4 = vel(P + hs * k3, t + hs)
            P = P + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += hs
        frames.append(P.copy())
    frames = np.array(frames)
    if noise: frames = frames + np.random.default_rng(seed + 1).normal(0, noise * h, frames.shape)
    sh = [(a * L, b * L) for a in (0, -1, 1) for b in (0, -1, 1)]
    G = LD.LTN(np.concatenate([frames + np.array(v) for v in sh], axis=1), dtf, np.ones(9 * N0), k=12)
    ch = G.channels(s); c = ch['c']; tri, E, w = c['tri'], c['E'], c['w']; tt = s * dtf; F = np.exp(-2 * nu * tt)
    cT = (tri < N0).all(1); cE = (E < N0).all(1); cN = np.arange(G.N) < N0; ip = lambda x, y: float((w * x * y)[cE].sum())
    enst = float((ch['Gam'] ** 2 / c['A'])[cT].sum()); Pc = G.fr[s][:N0]
    pex = (np.cos(2 * Pc[:, 0]) + np.cos(2 * Pc[:, 1])) * F ** 2 / 4; pr = ch['p'][:N0] - ch['p'][:N0].mean() + pex.mean()
    ps = G.pair_strain(s, rmax=2.5 * h); cen = c['P'][tri].mean(1); omex = 2 * np.sin(cen[:, 0]) * np.sin(cen[:, 1]) * F
    Phi_ex = nu * 4 * np.pi ** 2 * F ** 2
    r = dict(h=h, omega=rel(ch['Gam'][cT] / c['A'][cT], omex[cT]), pressure=rel(pr - pr.mean(), pex - pex.mean()),
             mu=abs((-ip(ch['Hf']['c'], ch['Hu']['c']) / enst) / nu - 1), Phi_tpm=abs(float(nu * (ch['phi_face'] * c['A'])[cT].sum()) / Phi_ex - 1),
             Phi_pair=abs(float(nu * (ps['phi'] * c['Ast'])[cN & np.isfinite(ps['phi'])].sum()) / Phi_ex - 1),
             N_g=noise * h / (1.0 * h * dtf), N_a=noise * h / (1.0 * dtf ** 2))
    return r


# =============================================================== Stokes' first problem
if case == 'stokes':
    U, nu, h, Lx, Ly, tA, dt, S, s = 1.0, 1e-3, 0.005, 0.3, 0.3, 1.0, 0.0025, 7, 3
    P0 = lattice(0, Lx, 0, Ly, h, 4); y0 = P0[:, 1]
    tf = tA + np.arange(S) * dt; tfine = np.linspace(tA, tf[-1], 400 * (S - 1) + 1)          # fine grid aligned with the frames
    def xdisp(y):
        uu = U * erfc(y[None] / (2 * np.sqrt(nu * tfine[:, None])))
        cum = np.concatenate([np.zeros((1, len(y))), np.cumsum(0.5 * (uu[1:] + uu[:-1]) * np.diff(tfine)[:, None], 0)])
        return cum[np.searchsorted(tfine, tf - 1e-12)]
    dx = xdisp(y0); base = np.stack([P0[None, :, 0] + dx, np.repeat(y0[None], S, 0)], -1)
    for variant in ('free wall', 'wall extension'):
        fr = base
        if variant == 'wall extension':
            dxi = xdisp(-y0); img = np.stack([P0[None, :, 0] + dxi, np.repeat(-y0[None], S, 0)], -1); fr = np.concatenate([base, img], axis=1)
        N0 = base.shape[1]
        fr = np.concatenate([fr + np.array([k * Lx, 0]) for k in (0, -1, 1, -2, 2, -3, 3, -4, 4)], axis=1)   # long in x: far ends decay
        G = LD.LTN(fr, dt, np.ones(fr.shape[1]), k=12); ch = G.channels(s); c = ch['c']; tri, E, w = c['tri'], c['E'], c['w']; t = tf[s]
        cen = c['P'][tri].mean(1); orgT = (tri < N0).all(1); orgE = (E < N0).all(1)
        omex = U / np.sqrt(np.pi * nu * t) * np.exp(-cen[:, 1] ** 2 / (4 * nu * t))                 # omega = -du/dy (positive here)
        om = -ch['Gam'] / c['A'] * -1
        om = ch['Gam'] / c['A']; om = om * np.sign(np.sum(om[orgT] * omex[orgT]))
        nearwall = orgT & (cen[:, 1] < 2 * h)
        enst = float((ch['Gam'] ** 2 / c['A'])[orgT].sum()); ip = lambda x, y: float((w * x * y)[orgE].sum())
        Phi_ex = nu * float((omex ** 2 * c['A'])[orgT].sum())          # exact field integrated over the same faces
        r = {'omega (all)': rel(om[orgT], omex[orgT]), 'omega (first 2h at wall)': rel(om[nearwall], omex[nearwall]),
             'mu_E / nu - 1': (-ip(ch['Hf']['c'], ch['Hu']['c']) / enst) / nu - 1,
             'Phi three-point': float(nu * (ch['phi_face'] * c['A'])[orgT].sum()) / Phi_ex - 1,
             'gradient share of rho a (global; exact 0 up to the mean shear stress)': float((w * ch['Hf']['g'] ** 2).sum() / (w * ch['fe'] ** 2).sum())}
        out[variant] = r; print(variant, {k: '%.3e' % v for k, v in r.items()}, flush=True)
        save[variant] = dict(y=cen[orgT, 1], om=om[orgT], omex=omex[orgT])

# =============================================================== internal gravity wave (Boussinesq, linear)
if case == 'igw':
    n, Nb, Amp, dtf, S, s = 64, 1.0, 0.01 / (2 * np.pi), 0.02, 9, 4; L = 1.0; h = L / n; kv = 2 * np.pi * np.array([1.0, 1.0]); km = np.linalg.norm(kv)
    om_w = Nb * abs(kv[0]) / km; P0 = lattice(0, L, 0, L, h, 8); N0 = len(P0)
    phase = lambda P, t: P @ kv - om_w * t
    vel = lambda P, t: Amp * np.c_[-kv[1] * np.sin(phase(P, t)), kv[0] * np.sin(phase(P, t))]   # u = dpsi/dy, v = -dpsi/dx, psi = A cos
    zeta = lambda P, t: Amp * kv[0] / om_w * np.cos(phase(P, t))                                   # Dzeta/Dt = v
    P = P0.copy(); frames = [P.copy()]; t = 0.0; sub = 10; hs = dtf / sub
    for f in range(1, S):
        for _ in range(sub):
            k1 = vel(P, t); k2 = vel(P + 0.5 * hs * k1, t + 0.5 * hs); k3 = vel(P + 0.5 * hs * k2, t + 0.5 * hs); k4 = vel(P + hs * k3, t + hs)
            P = P + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += hs
        frames.append(P.copy())
    frames = np.array(frames); y_eq = frames[0][:, 1] - zeta(frames[0], 0.0)        # one reference: equilibrium heights at t = 0
    sh = [(a * L, b * L) for a in (0, -1, 1) for b in (0, -1, 1)]
    G = LD.LTN(np.concatenate([frames + np.array(v) for v in sh], axis=1), dtf, np.ones(9 * N0), k=12)
    ch = G.channels(s); c = ch['c']; tri = c['tri']; cT = (tri < N0).all(1); P = c['P']; tt = s * dtf
    zN = np.tile(frames[s][:, 1] - y_eq, 9)                                            # displacement from positions
    Gg, Ar = X._grad_lambda(P, tri); dz = (Gg * zN[tri][:, :, None]).sum(1)            # P1 gradient of zeta per face
    cen = P[tri].mean(1); dzx_ex = -Amp * kv[0] ** 2 / om_w * np.sin(phase(cen, tt))
    tq = ch['torque']; tq_ex = -Nb ** 2 * dzx_ex
    N2 = -float((tq[cT] * dz[cT, 0]).sum() / (dz[cT, 0] ** 2).sum())
    ome = ch['Gam'] / c['A']; ome_ex = km ** 2 * Amp * np.cos(phase(cen, tt))
    r = {'torque (Kelvin) vs exact -N^2 dzeta/dx': rel(tq[cT], tq_ex[cT]), 'N^2 from positions / N^2 - 1': N2 / Nb ** 2 - 1,
         'zeta gradient vs exact': rel(dz[cT, 0], dzx_ex[cT]), 'omega (face)': rel(ome[cT], ome_ex[cT]),
         'dispersion: omega_w': om_w}
    out['igw'] = r; print({k: '%.3e' % v for k, v in r.items()}, flush=True)
    save['igw'] = dict(cx=cen[cT, 0], cy=cen[cT, 1], tq=tq[cT], tq_ex=tq_ex[cT])

if case == 'conv':
    for n in (40, 80, 120):
        r = tg_case(n); out['n=%d' % n] = r; print(n, {k: '%.3e' % v for k, v in r.items()}, '%.0fs' % (time.time() - t0w), flush=True)

if case == 'noise':
    for nz in (0.0, 1e-4, 1e-3, 3e-3, 1e-2):
        r = tg_case(80, noise=nz); out['noise=%g' % nz] = r; print(nz, {k: '%.3e' % v for k, v in r.items()}, flush=True)

json.dump(out, open('/home/claude/w/verify/suite2_%s.json' % case, 'w'), indent=1, default=float)
np.savez_compressed('/home/claude/w/verify/suite2_%s.npz' % case, **{k + '__' + kk: vv for k, d in save.items() for kk, vv in d.items()})
print('done %.0fs' % (time.time() - t0w))
