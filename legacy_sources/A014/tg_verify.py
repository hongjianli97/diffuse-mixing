"""Manufactured-solution check without any DNS: decaying Taylor-Green vortex (exact Navier-Stokes solution, rho = 1)
u = sin x cos y F, v = -cos x sin y F, F = exp(-2 nu t), p = (cos 2x + cos 2y) F^2 / 4.
Trajectories integrated with RK4; periodic box tiled 3x3 with image trajectories (periodic LTN), central tile scored.
Exact references: nu, int omega^2 = 4 pi^2 F^2, int Phi = mu * 4 pi^2 F^2, pressure."""
import numpy as np, sys, json, time
sys.path.insert(0, '/home/claude/w/ltn'); import ltn_dyn as LD
nu = 0.01; n = 100; L = 2 * np.pi; h = L / n; out = {}
rng = np.random.default_rng(5); g = (np.arange(n) + 0.5) * h; Xg, Yg = np.meshgrid(g, g)
P0 = np.c_[Xg.ravel(), Yg.ravel()] + rng.uniform(-0.25, 0.25, (n * n, 2)) * h; N0 = len(P0)
vel = lambda P, t: np.exp(-2 * nu * t) * np.c_[np.sin(P[:, 0]) * np.cos(P[:, 1]), -np.cos(P[:, 0]) * np.sin(P[:, 1])]
dtf, nF, sub = 0.02, 7, 40; P = P0.copy(); frames = [P.copy()]; t = 0.0; hs = dtf / sub
for f in range(1, nF):
    for _ in range(sub):
        k1 = vel(P, t); k2 = vel(P + 0.5 * hs * k1, t + 0.5 * hs); k3 = vel(P + 0.5 * hs * k2, t + 0.5 * hs); k4 = vel(P + hs * k3, t + hs)
        P = P + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += hs
    frames.append(P.copy())
frames = np.array(frames); shifts = [(a * L, b * L) for a in (0, -1, 1) for b in (0, -1, 1)]   # central tile first
tiled = np.concatenate([frames + np.array(sh) for sh in shifts], axis=1)
for noise in (0.0, 0.01):
    fr = tiled + (np.random.default_rng(2).normal(0, noise * h, tiled.shape) if noise else 0)
    G = LD.LTN(fr, dtf, np.ones(fr.shape[1]), k=12); N = G.N
    for s in (3,):
        tt = s * dtf; F = np.exp(-2 * nu * tt); ch = G.channels(s); c = ch['c']; tri = c['tri']
        cenT = (tri < N0).all(1); E = c['E']; cenE = (E < N0).all(1); w = c['w']
        ip = lambda x, y, m: float((w * x * y)[m].sum())
        enst = float((ch['Gam'] ** 2 / c['A'])[cenT].sum()); muE = -ip(ch['Hf']['c'], ch['Hu']['c'], cenE) / enst
        phiT = float(nu * (ch['phi_face'] * c['A'])[cenT].sum()); ps = G.pair_strain(s, rmax=2.5 * h); cn = np.arange(N) < N0
        phiP = float(nu * (ps['phi'] * c['Ast'])[cn].sum()); Pc = G.fr[s][:N0]
        pex = (np.cos(2 * Pc[:, 0]) + np.cos(2 * Pc[:, 1])) * F ** 2 / 4; prc = ch['p'][:N0]; prc = prc - prc.mean() + pex.mean()
        r = dict(t=tt, noise_over_h=noise, mu_E=muE, mu_true=nu, enst_net=enst, enst_exact=4 * np.pi ** 2 * F ** 2,
                 Phi_tpm=phiT, Phi_pair=phiP, Phi_exact=nu * 4 * np.pi ** 2 * F ** 2,
                 p_relrms=float(np.sqrt(np.mean((prc - pex) ** 2)) / pex.std()), f_curl_share=float(ip(ch['Hf']['c'], ch['Hf']['c'], cenE) / ip(ch['fe'], ch['fe'], cenE)))
        out['noise%.2f' % noise] = r; print({k: ('%.5g' % v if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(out, open('res_tg_verify.json', 'w'), indent=1)
