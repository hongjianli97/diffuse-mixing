"""suite.py <case> -- exact-solution (R1) verification of the LTN channels.  No DNS anywhere.
cases: linear | lamb | tg | vortex | acoustic | sod | lamella
Every case: exact trajectories -> ltn_dyn.LTN -> channels -> compared with the exact fields."""
import numpy as np, sys, json, time
from scipy.linalg import expm
from scipy.special import erf
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
sys.path.insert(0, '/home/claude/w/ltn'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd'); sys.path.insert(0, '/home/claude/w')
import ltn_dyn as LD, lhdx as X, netmix as NM
G_ = 1.4; CV = 1 / (G_ - 1)


def lattice(x0, x1, y0, y1, h, seed):
    rng = np.random.default_rng(seed); gx = np.arange(x0 + h / 2, x1, h); gy = np.arange(y0 + h / 2, y1, h)
    Xg, Yg = np.meshgrid(gx, gy); return np.c_[Xg.ravel(), Yg.ravel()] + rng.uniform(-0.25, 0.25, (Xg.size, 2)) * h


def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def star_rate(G, s, tri):
    Am, Ap = X._star(G.fr[s - 1], tri, G.N), X._star(G.fr[s + 1], tri, G.N)
    with np.errstate(all='ignore'): return np.where((Am > 0) & (Ap > 0), (np.log(Ap) - np.log(Am)) / (2 * G.dt), np.nan)


def tile(frames, Lx, Ly):
    sh = [(a * Lx, b * Ly) for a in (0, -1, 1) for b in (0, -1, 1)]
    return np.concatenate([frames + np.array(v) for v in sh], axis=1)


out = {}; case = sys.argv[1]; t0 = time.time(); SAVE = {}

# =============================================================== A. linear flows: first-order channels must be exact
if case == 'linear':
    P0 = lattice(-1, 1, -1, 1, 0.04, 1); dt, S, s = 0.01, 7, 3
    mats = {'rotation': [[0, -1], [1, 0]], 'pure strain': [[1, 0], [0, -1]], 'simple shear': [[0, 1], [0, 0]],
            'dilatation': [[0.5, 0], [0, 0.5]], 'general': [[0.3, -0.8], [1.1, -0.1]]}
    for name, A in mats.items():
        A = np.array(A, float); fr = np.array([P0 @ expm(A * k * dt).T for k in range(S)])
        G = LD.LTN(fr, dt, np.ones(len(P0)), k=12); ch = G.channels(s); c = ch['c']; P, tri = c['P'], c['tri']
        inN = np.abs(P).max(1) < 0.7; inT = np.abs(P[tri]).max((1, 2)) < 0.7
        Sx = 0.5 * (A + A.T); A2 = A @ A; th_ex = np.trace(A); om_ex = A[1, 0] - A[0, 1]; tq_ex = A2[1, 0] - A2[0, 1]
        th = star_rate(G, s, tri); Ss = 0.5 * (ch['Lg'] + ch['Lg'].transpose(0, 2, 1)); ps = G.pair_strain(s, rmax=2.5 * 0.04)
        phi_ex = 2 * (Sx ** 2).sum() - 2 / 3 * th_ex ** 2
        r = {'theta (ledger)': float(np.nanmax(np.abs(th[inN] - th_ex))), 'omega (face circulation)': float(np.max(np.abs(ch['Gam'][inT] / c['A'][inT] - om_ex))),
             'grad u (three-point)': float(np.max(np.abs(ch['Lg'][inT] - A[None]))), 'S (pair strain)': float(np.nanmax(np.abs(ps['S'][inN] - np.array([Sx[0, 0], Sx[0, 1], Sx[1, 1]])[None]))),
             'Phi/mu (three-point)': float(np.max(np.abs(ch['phi_face'][inT] - phi_ex))), 'Kelvin dGamma/dt': float(np.max(np.abs(ch['torque'][inT] - tq_ex)))}
        rho = np.exp(-th_ex * s * dt)
        if np.allclose(A2, A2.T):
            pex = -rho * 0.5 * np.einsum('ni,ij,nj->n', P, A2, P); pr = ch['p'] - ch['p'][inN].mean() + pex[inN].mean()
            r['pressure (rel. or abs.)'] = rel(pr[inN], pex[inN]) if np.abs(pex[inN]).max() > 1e-9 else float(np.abs(pr[inN] - pex[inN]).max())
        else:
            fcT = (c['B2'].T @ ch['fe']) / c['A']; r['curl(rho a) vs rho*tq'] = float(np.max(np.abs(fcT[inT] - rho * tq_ex)))
        out[name] = r; print(name, {k: '%.2e' % v for k, v in r.items()}, flush=True)

# =============================================================== B. diffusing Lamb-Oseen vortex (exact Navier-Stokes)
if case == 'lamb':
    G0, nu, Rd, tA, S, s = 1.0, 1e-3, 0.3, 1.0, 7, 3; dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0025; h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
    P0 = lattice(-Rd, Rd, -Rd, Rd, h, 2); P0 = P0[np.hypot(*P0.T) < Rd]; r0 = np.hypot(*P0.T); th0 = np.arctan2(P0[:, 1], P0[:, 0])
    Gam = lambda rr, tt: G0 * (1 - np.exp(-rr ** 2 / (4 * nu * tt)))
    tf = tA + np.arange(S) * dt; tfine = np.linspace(tA, tf[-1], 3001); om = Gam(r0[None], tfine[:, None]) / (2 * np.pi * r0[None] ** 2)
    cum = np.concatenate([np.zeros((1, len(r0))), np.cumsum(0.5 * (om[1:] + om[:-1]) * np.diff(tfine)[:, None], 0)])
    ang = th0[None] + cum[np.searchsorted(tfine, tf - 1e-12)]; fr = np.stack([r0[None] * np.cos(ang), r0[None] * np.sin(ang)], -1)
    G = LD.LTN(fr, dt, np.ones(len(P0)), k=12); ch = G.channels(s); c = ch['c']; P, tri = c['P'], c['tri']; t = tf[s]
    rr = np.linspace(1e-6, 0.5, 20001); ut = Gam(rr, t) / (2 * np.pi * rr); pp = np.concatenate([[0], np.cumsum(0.5 * (ut[1:] ** 2 / rr[1:] + ut[:-1] ** 2 / rr[:-1]) * np.diff(rr))])
    rn = np.hypot(*P.T); inN = rn < 0.25; cen = P[tri].mean(1); rc = np.hypot(*cen.T); inT = rc < 0.25
    pex = np.interp(rn, rr, pp); pr = ch['p'] - ch['p'][inN].mean() + pex[inN].mean()
    omex = G0 / (4 * np.pi * nu * t) * np.exp(-rc ** 2 / (4 * nu * t)); dom = omex * (-1 / t + rc ** 2 / (4 * nu * t ** 2))
    Srt = 0.5 * rr * np.gradient(ut / rr, rr); phi_r = 4 * nu * Srt ** 2; Phi_ex = np.trapezoid(2 * np.pi * rr * phi_r * (rr < 0.25), rr)
    enst_ex = np.trapezoid(2 * np.pi * rr * (G0 / (4 * np.pi * nu * t) * np.exp(-rr ** 2 / (4 * nu * t))) ** 2 * (rr < 0.25), rr)
    ps = G.pair_strain(s, rmax=2.5 * h); inNs = inN & np.isfinite(ps['phi'])
    r = {'omega (face)': rel(ch['Gam'][inT] / c['A'][inT], omex[inT]), 'enstrophy': (float((ch['Gam'] ** 2 / c['A'])[inT].sum()) - enst_ex) / enst_ex,
         'Kelvin dGamma/dt (viscous torque)': rel(ch['torque'][inT], dom[inT]), 'pressure (cyclostrophic)': rel(pr[inN] - pr[inN].mean(), pex[inN] - pex[inN].mean()),
         'mu_E / nu - 1': ch['mu_E'] / nu - 1, 'Phi three-point': (float(nu * (ch['phi_face'] * c['A'])[inT].sum()) - Phi_ex) / Phi_ex,
         'Phi pair strain': (float(nu * (ps['phi'] * c['Ast'])[inNs].sum()) - Phi_ex) / Phi_ex}
    SAVE.update(P=P, tri=tri[inT], om_net=(ch['Gam'] / c['A'])[inT], om_ex=omex[inT], p_net=pr, p_ex=pex, inN=inN,
                phi_net=nu * ch['phi_face'][inT], phi_ex=np.interp(rc[inT], rr, phi_r))
    out['Lamb-Oseen dt=%g h=%g' % (dt, h)] = r; print({k: '%.3e' % v for k, v in r.items()}, flush=True)

# =============================================================== C. decaying Taylor-Green (exact Navier-Stokes, periodic)
if case == 'tg':
    nu, n, dtf, S, s = 0.01, 80, 0.02, 7, 3; L = 2 * np.pi; h = L / n; P0 = lattice(0, L, 0, L, h, 5); N0 = len(P0)
    vel = lambda P, t: np.exp(-2 * nu * t) * np.c_[np.sin(P[:, 0]) * np.cos(P[:, 1]), -np.cos(P[:, 0]) * np.sin(P[:, 1])]
    P = P0.copy(); frames = [P.copy()]; t = 0.0; sub = 40; hs = dtf / sub
    for f in range(1, S):
        for _ in range(sub):
            k1 = vel(P, t); k2 = vel(P + 0.5 * hs * k1, t + 0.5 * hs); k3 = vel(P + 0.5 * hs * k2, t + 0.5 * hs); k4 = vel(P + hs * k3, t + hs)
            P = P + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += hs
        frames.append(P.copy())
    G = LD.LTN(tile(np.array(frames), L, L), dtf, np.ones(9 * N0), k=12); ch = G.channels(s); c = ch['c']; tri, E, w = c['tri'], c['E'], c['w']
    tt = s * dtf; F = np.exp(-2 * nu * tt); cT = (tri < N0).all(1); cE = (E < N0).all(1); cN = np.arange(G.N) < N0
    ip = lambda x, y: float((w * x * y)[cE].sum()); enst = float((ch['Gam'] ** 2 / c['A'])[cT].sum())
    Pc = G.fr[s][:N0]; pex = (np.cos(2 * Pc[:, 0]) + np.cos(2 * Pc[:, 1])) * F ** 2 / 4; pr = ch['p'][:N0] - ch['p'][:N0].mean() + pex.mean()
    ps = G.pair_strain(s, rmax=2.5 * h); cen = c['P'][tri].mean(1)
    omex = 2 * np.sin(cen[:, 0]) * np.sin(cen[:, 1]) * F
    r = {'omega (face)': rel(ch['Gam'][cT] / c['A'][cT], omex[cT]), 'enstrophy': enst / (4 * np.pi ** 2 * F ** 2) - 1,
         'pressure': rel(pr - pr.mean(), pex - pex.mean()), 'mu_E / nu - 1': (-ip(ch['Hf']['c'], ch['Hu']['c']) / enst) / nu - 1,
         'Phi three-point': float(nu * (ch['phi_face'] * c['A'])[cT].sum()) / (nu * 4 * np.pi ** 2 * F ** 2) - 1,
         'Phi pair strain': float(nu * (ps['phi'] * c['Ast'])[cN].sum()) / (nu * 4 * np.pi ** 2 * F ** 2) - 1}
    SAVE.update(P=c['P'], tri=tri[cT], om_net=(ch['Gam'] / c['A'])[cT], om_ex=omex[cT], Pc=Pc, p_net=pr, p_ex=pex,
                phi_net=nu * ch['phi_face'][cT], phi_ex=nu * 4 * F ** 2 * np.cos(cen[cT, 0]) ** 2 * np.cos(cen[cT, 1]) ** 2)
    out['Taylor-Green'] = r; print({k: '%.3e' % v for k, v in r.items()}, flush=True)

# =============================================================== D. translating non-homentropic vortex (exact steady Euler)
if case == 'vortex':
    beta, h, Rd, U, dt, S, s = 5.0, 0.06, 4.0, np.array([0.5, 0.3]), 0.02, 7, 3
    P0 = lattice(-Rd, Rd, -Rd, Rd, h, 3); P0 = P0[np.hypot(*P0.T) < Rd]; r0 = np.hypot(*P0.T)
    uth = lambda r: beta / (2 * np.pi) * r * np.exp((1 - r ** 2) / 2); rho_r = lambda r: 1 + 0.5 * np.exp(-r ** 2)
    rr = np.linspace(1e-6, 10, 40001); integrand = rho_r(rr) * uth(rr) ** 2 / rr
    cum = np.concatenate([[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(rr))]); pp = 2.0 - (cum[-1] - cum)                      # background pressure 2 keeps p > 0 in the core
    Om = uth(r0) / r0; tf = np.arange(S) * dt
    fr = np.array([np.c_[np.cos(Om * t) * P0[:, 0] - np.sin(Om * t) * P0[:, 1], np.sin(Om * t) * P0[:, 0] + np.cos(Om * t) * P0[:, 1]] + U * t for t in tf])
    G = LD.LTN(fr, dt, rho_r(r0), k=12); ch = G.channels(s); c = ch['c']; P, tri = c['P'], c['tri']; t = tf[s]
    rn = np.hypot(*(P - U * t).T); inN = rn < 3.0; cen = P[tri].mean(1) - U * t; rc = np.hypot(*cen.T); inT = rc < 3.0
    pex = np.interp(rn, rr, pp); pr = ch['p'] - ch['p'][inN].mean() + pex[inN].mean()
    rho_led = G.rho(s); sex = CV * (np.log(pex) - G_ * np.log(rho_r(rn))); srec = CV * (np.log(np.maximum(pr, 1e-9)) - G_ * np.log(rho_led))
    omex = beta / (2 * np.pi) * np.exp((1 - rc ** 2) / 2) * (2 - rc ** 2)
    acen = (uth(rc) ** 2 / rc)
    r = {'omega (face)': rel(ch['Gam'][inT] / c['A'][inT], omex[inT]), 'pressure (variable rho)': rel(pr[inN] - pr[inN].mean(), pex[inN] - pex[inN].mean()),
         'entropy s (label)': (lambda m: rel(srec[m] - srec[m].mean(), sex[m] - sex[m].mean()))(inN & np.isfinite(srec)),
         'Kelvin torque / omega^2 scale (exact 0)': float(np.sqrt(np.mean(ch['torque'][inT] ** 2)) / np.max(omex ** 2)),
         'curl share of rho a (exact 0)': float((c['w'] * ch['Hf']['c'] ** 2).sum() / (c['w'] * ch['fe'] ** 2).sum()),
         'ledger rho (exact rho0)': rel(rho_led[inN], rho_r(rn[inN]))}
    SAVE.update(P=P, tri=tri[inT], om_net=(ch['Gam'] / c['A'])[inT], om_ex=omex[inT], p_net=pr, p_ex=pex, s_net=srec, s_ex=sex, inN=inN)
    out['vortex'] = r; print({k: '%.3e' % v for k, v in r.items()}, flush=True)

# =============================================================== E. linear acoustic standing wave (periodic)
if case == 'acoustic':
    n, eps, cs, dtf, S = 64, 1e-3, 1.0, 0.005, 51; L = 1.0; h = L / n; kv = 2 * np.pi * np.array([1.0, 1.0]); km = np.linalg.norm(kv); w0 = cs * km
    P0 = lattice(0, L, 0, L, h, 7); N0 = len(P0); khat = kv / km
    vel = lambda P, t: eps * cs * np.sin(P @ kv)[:, None] * np.sin(w0 * t) * khat[None]
    P = P0.copy(); frames = [P.copy()]; t = 0.0; sub = 10; hs = dtf / sub
    for f in range(1, S):
        for _ in range(sub):
            k1 = vel(P, t); k2 = vel(P + 0.5 * hs * k1, t + 0.5 * hs); k3 = vel(P + 0.5 * hs * k2, t + 0.5 * hs); k4 = vel(P + hs * k3, t + hs)
            P = P + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += hs
        frames.append(P.copy())
    fr = tile(np.array(frames), L, L); rho0 = np.tile(1 + eps * np.cos(P0 @ kv), 9)
    G = LD.LTN(fr, dtf, rho0, k=12); N = G.N; c0 = G.active(0); E, w = c0['E'], c0['w']; cE = (E < N0).all(1)
    rho_ex = lambda s: 1 + eps * np.cos(G.fr[s][:N0] @ kv) * np.cos(w0 * s * dtf)
    led = {s: rel(G.rho(s)[:N0] - 1, rho_ex(s) - 1) for s in (10, 20, 40)}
    num = den = 0.0
    for s in range(3, S - 2):
        cs_ = G.active(s); V, A = G.node_kin(s); ae = G.cochain(cs_, A); g = cs_['B1'].T @ np.log(G.rho(s)); mm = (cs_['E'] < N0).all(1)
        num += (cs_['w'] * ae * ae)[mm].sum(); den += -(cs_['w'] * ae * g)[mm].sum()
    c2 = num / den
    ch = G.channels(20); curl_share = float((ch['c']['w'] * ch['Hu']['c'] ** 2).sum() / (ch['c']['w'] * ch['ue'] ** 2).sum())
    B1, Ast = c0['B1'], c0['Ast']
    def rhs(y): rp, u = y[:N], y[N:]; return np.r_[-(-(B1 @ (w * u)) / Ast), -c2 * (B1.T @ rp)]
    y = np.r_[G.rho(0) - 1, np.zeros(len(E))]; hstep = 2e-4; pred = {}; en = lambda y: float(0.5 * c2 * (y[:N] ** 2 * Ast).sum() + 0.5 * (w * y[N:] ** 2).sum()); e0 = en(y)
    for s in range(1, 41):
        for _ in range(int(round(dtf / hstep))):
            k1 = rhs(y); k2 = rhs(y + 0.5 * hstep * k1); k3 = rhs(y + 0.5 * hstep * k2); k4 = rhs(y + hstep * k3); y = y + hstep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if s in (10, 20, 40): pred[s] = y[:N0].copy()
    r = {'ledger rho\' t=0.05': led[10], 'ledger rho\' t=0.1': led[20], 'ledger rho\' t=0.2': led[40], 'c^2 - 1': c2 - 1, 'curl share of u (exact 0)': curl_share,
         'Dirac forward t=0.05': rel(pred[10], rho_ex(10) - 1), 'Dirac forward t=0.1': rel(pred[20], rho_ex(20) - 1), 'Dirac forward t=0.2': rel(pred[40], rho_ex(40) - 1),
         'Dirac energy drift': en(y) / e0 - 1}
    c20 = G.active(20); t20 = c20['tri'][(c20['tri'] < N0).all(1)]
    SAVE.update(P=G.fr[20][:N0], tri=t20, rho_ex=rho_ex(20) - 1, rho_led=G.rho(20)[:N0] - 1, rho_dirac=pred[20])
    out['acoustic'] = r; print({k: '%.3e' % v for k, v in r.items()}, flush=True)

# =============================================================== F. Sod shock tube (exact Riemann kinematics)
if case == 'sod':
    import cases as CS
    cs = CS.case_sod(h=0.004, Ly=0.12, S=41); fr, Ly = cs['frames'], 0.12; N0 = fr.shape[1]; dt = cs['dt']
    G = LD.LTN(fr, dt, cs['rho'][0], k=12, mirror_band=Ly); res = {}
    for s in (15, 25, 35):
        ch = G.channels(s); c = ch['c']; org = np.arange(G.N) < N0; hot = G.bad[:, s][:N0]
        rex, pex = cs['rho'][s], cs['p'][s]; rl = G.rho(s)[:N0]; th = star_rate(G, s, c['tri'])[:N0]; dv = cs['div_ex'][s]
        pr = ch['p'][:N0]; pr = pr - pr[~hot].mean() + pex[~hot].mean()
        sex = CV * (np.log(pex) - G_ * np.log(rex)); srec = CV * (np.log(np.maximum(pr, 1e-6)) - G_ * np.log(rl))
        fan = np.abs(dv) > 1e-9; inner = (fr[s][:, 1] > 0.01) & (fr[s][:, 1] < Ly - 0.01)
        tq = s * dt
        res[s] = {'ledger rho': rel(rl[inner], rex[inner]), 'ledger rho (off shock)': rel(rl[inner & ~hot], rex[inner & ~hot]),
                  'theta in rarefaction': rel(th[inner & fan & ~hot], dv[inner & fan & ~hot]),
                  'pressure / std': float(np.sqrt(np.mean((pr - pex)[inner] ** 2)) / pex[inner].std()),
                  'pressure / std (off shock)': float(np.sqrt(np.mean((pr - pex)[inner & ~hot] ** 2)) / pex[inner & ~hot].std()),
                  'entropy abs rms (off shock)': float(np.sqrt(np.mean((srec - sex)[inner & ~hot] ** 2))), 'entropy contrast exact': float(sex.max() - sex.min())}
        print(s, {k: '%.3e' % v for k, v in res[s].items()}, flush=True)
    # shock: tracers that crossed the shock inside the window
    info = cs['info'] if 'info' in cs else {}
    x0 = cs['x0']; Ssh = (fr[-1][:, 0][x0 > 0].max() - 0) if False else None
    first = np.argmax(G.bad[:N0], axis=1); has = G.bad[:N0].any(1) & (x0 > 0.08)
    sb, sa = first - 3, first + 4; ok = has & (sb >= 1) & (sa <= G.S - 2); idx = np.where(ok)[0]
    eta = np.exp(G.lnJ[sb[idx], idx] - G.lnJ[sa[idx], idx]); p21 = ((G_ + 1) * eta - (G_ - 1)) / ((G_ + 1) - (G_ - 1) * eta)
    dsRH = CV * (np.log(p21) - G_ * np.log(eta)); eta_ex = cs['rho'][-1][x0 > 0.08].max() / 0.125
    p21x = ((G_ + 1) * eta_ex - (G_ - 1)) / ((G_ + 1) - (G_ - 1) * eta_ex); ds_ex = CV * (np.log(p21x) - G_ * np.log(eta_ex))
    res['shock'] = {'eta ledger / exact - 1': float(np.median(eta) / eta_ex - 1), 'Delta s RH(eta ledger) / exact - 1': float(np.median(dsRH) / ds_ex - 1), 'n tracers': int(len(idx))}
    print(res['shock']); out['sod'] = res

# =============================================================== G. lamella mixing: incompressible strain and compressible compression
if case == 'lamella':
    h, w0, dt, S = 0.005, 0.03, 0.02, 51
    def run(kind):
        if kind == 'strain': P0 = lattice(-0.3, 0.3, -0.15, 0.15, h, 9)
        else: P0 = lattice(-0.3, 0.3, -0.3, 0.3, h, 9)
        tf = np.arange(S) * dt
        if kind == 'strain': fr = np.array([np.c_[P0[:, 0] * np.exp(-t), P0[:, 1] * np.exp(t)] for t in tf])
        else: fr = np.array([np.c_[P0[:, 0] * np.exp(-t), P0[:, 1]] for t in tf])
        G = LD.LTN(fr, dt, np.ones(len(P0)), k=12); Xr = P0[:, 0]; c0 = 0.5 * (1 + erf(Xr / w0)); res = {}
        Ls = []; Ws = []; As = []
        for s in range(S):
            c = G.active(s); Ls.append(NM.laplacian(G.N, c['E'], c['w'])); Ws.append(c); As.append(c['Ast'])
        m = G.m
        def diffuse(cond, metric):
            cc = c0.copy()
            for s in range(S - 1):
                Lb = 0.5 * (cond(s) + cond(s + 1)); Mv = metric(s + 1)
                cc = spsolve((sp.diags(Mv) + dt * Lb).tocsc(), Mv * cc) if metric is not None else None
            return cc
        sel = np.abs(Xr) < 0.1
        if kind == 'strain':
            kap = 1e-4; tau = (np.exp(2 * tf[-1]) - 1) / 2; cex = 0.5 * (1 + erf(Xr / np.sqrt(w0 ** 2 + 4 * kap * tau)))
            cn = diffuse(lambda s: kap * Ls[s], lambda s: m)
            cl = G.active(S - 1); SAVE.update(strain_P=G.fr[S - 1], strain_tri=cl['tri'], strain_c=cn, strain_cex=cex)
            res['strain: profile'] = rel(cn[sel], cex[sel]); res['strain: mixed mass'] = float((m * 4 * cn * (1 - cn)).sum() / (m * 4 * cex * (1 - cex)).sum() - 1)
        else:
            Gam = 1e-4; rho = lambda s: G.rho(s)
            tau1 = Gam * (np.exp(tf[-1]) - 1); ce1 = 0.5 * (1 + erf(Xr / np.sqrt(w0 ** 2 + 4 * tau1)))          # alpha = 1 (rho kappa const)
            tau0 = Gam * (np.exp(2 * tf[-1]) - 1) / 2; ce0 = 0.5 * (1 + erf(Xr / np.sqrt(w0 ** 2 + 4 * tau0)))   # alpha = 0 (kappa const)
            c1 = diffuse(lambda s: Gam * Ls[s], lambda s: m)
            c0n = diffuse(lambda s: NM.laplacian(G.N, Ws[s]['E'], Ws[s]['w'] * Gam * 0.5 * (rho(s)[Ws[s]['E'][:, 0]] + rho(s)[Ws[s]['E'][:, 1]])), lambda s: m)
            ca = diffuse(lambda s: Gam * Ls[s], lambda s: As[s])                                                 # area metric (wrong for compressible)
            SAVE.update(comp_X=Xr, comp_x=G.fr[S - 1][:, 0], comp_c1=c1, comp_ce1=ce1, comp_ca=ca, comp_c0=c0n, comp_ce0=ce0)
            res['compression alpha=1, mass metric: profile'] = rel(c1[sel], ce1[sel]); res['compression alpha=0, mass metric: profile'] = rel(c0n[sel], ce0[sel])
            res['compression alpha=1, AREA metric: profile'] = rel(ca[sel], ce1[sel])
            res['compression alpha=1: mixed mass'] = float((m * 4 * c1 * (1 - c1)).sum() / (m * 4 * ce1 * (1 - ce1)).sum() - 1)
            res['compression alpha=1, AREA metric: mixed mass'] = float((m * 4 * ca * (1 - ca)).sum() / (m * 4 * ce1 * (1 - ce1)).sum() - 1)
        return res
    r = run('strain'); r.update(run('compression')); out['lamella'] = r
    print({k: '%.3e' % v for k, v in r.items()}, flush=True)

json.dump(out, open('/home/claude/w/verify/suite_%s%s.json' % (case, ('_' + sys.argv[2]) if len(sys.argv) > 2 else ''), 'w'), indent=1, default=float)
np.savez_compressed('/home/claude/w/verify/fields_%s.npz' % case, **SAVE)
print('done %.0fs' % (time.time() - t0))
