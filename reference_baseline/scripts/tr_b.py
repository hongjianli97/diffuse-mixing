"""Transport report, part D-F: diffusion (all cases), mixing attribution on material interfaces, robustness."""
import json, pickle, numpy as np, scipy.sparse as sp, cases, lhd as L, lhdx as X, netdiff as ND
from scipy.sparse.linalg import spsolve
from scipy.ndimage import map_coordinates
from scipy.fft import dctn, idctn
from trajnet import TrajectoryNetwork as TN
R = {'D': {}, 'E': {}, 'F': {}}
rel = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)) / np.std(b))
def eul2d(vel, box, n, T, dt, c0fun, kappa, keep):
    x0, x1, y0, y1 = box; dxg = (x1 - x0) / n[0]; xs = x0 + (np.arange(n[0]) + .5) * dxg; ys = y0 + (np.arange(n[1]) + .5) * dxg; XX, YY = np.meshgrid(xs, ys)
    c = c0fun(XX, YY); kx = np.pi * np.arange(n[0]) / (x1 - x0); ky = np.pi * np.arange(n[1]) / (y1 - y0)
    damp = 1 / (1 + kappa * dt * (kx[None, :] ** 2 + ky[:, None] ** 2)); P = np.c_[XX.ravel(), YY.ravel()]; t = 0.0; out = {}
    for _ in range(int(round(T / dt))):
        um = vel(P - .5 * dt * vel(P, t + .5 * dt), t + .5 * dt); Pd = P - dt * um
        c = map_coordinates(c, [(Pd[:, 1] - y0) / dxg - .5, (Pd[:, 0] - x0) / dxg - .5], order=3, mode='nearest').reshape(n[1], n[0])
        c = idctn(dctn(c, norm='ortho') * damp, norm='ortho'); t += dt
        for k in keep:
            if abs(t - k) < 1e-9: out[k] = c.copy()
    sample = lambda cc, Q: map_coordinates(cc, [(Q[:, 1] - y0) / dxg - .5, (Q[:, 0] - x0) / dxg - .5], order=1, mode='nearest')
    return out, sample
def lag1d(dmu, cond_face_fn, c0, ts, kappa):
    """implicit FD in mass coordinate: dc/dt = d/dmu(k dc/dmu); cond_face_fn(t) -> face conductance k (n-1)."""
    c = c0.copy(); n = len(c)
    for a, b in zip(ts[:-1], ts[1:]):
        k = cond_face_fn(0.5 * (a + b)) * kappa / (0.5 * (dmu[1:] + dmu[:-1])); dt = b - a
        main = np.ones(n); lo = np.zeros(n - 1); up = np.zeros(n - 1)
        main[:-1] += dt * k / dmu[:-1]; main[1:] += dt * k / dmu[1:]; up -= dt * k / dmu[:-1]; lo -= dt * k / dmu[1:]
        c = spsolve(sp.diags([lo, main, up], [-1, 0, 1]).tocsc(), c)
    return c
def polylen(P, order): Q = P[order]; return float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
# ================= D1 double gyre: reuse validated numbers + position-noise robustness (kappa = 1e-3)
R['D']['DG'] = json.load(open('res_val_dg_diff.json'))
vel, vort, grad = cases.dg_fields(); c = cases.case_dg(); fr, dt = c['frames'], c['dt']; S, N = fr.shape[:2]
c0 = 0.5 * (1 + np.tanh((fr[0][:, 0] - 1) / 0.1)); m = X.star_area(fr[0]) / 3.0
E, smp = eul2d(vel, (0, 2, 0, 1), (512, 256), 10.0, 0.01, lambda x, y: 0.5 * (1 + np.tanh((x - 1) / 0.1)), 1e-3, (10.0,))
inn = (fr[-1][:, 0] > .03) & (fr[-1][:, 0] < 1.97) & (fr[-1][:, 1] > .03) & (fr[-1][:, 1] < .97); ref = smp(E[10.0], fr[-1])
rng = np.random.default_rng(3); R['F']['DG_diffusion_noise'] = {}
for nz in (0.0, 0.05, 0.1):
    frn = fr + nz * c['h'] * rng.normal(size=fr.shape) if nz > 0 else fr
    out = ND.run(frn, dt, c0, 1e-3, 'mass', m=m, rho=np.ones((S, N)))
    R['F']['DG_diffusion_noise']['noise=%.2fh' % nz] = rel(out[S - 1][inn], ref[inn])
print('DG', json.dumps(R['F']['DG_diffusion_noise']), flush=True)
# ================= E1 double gyre: material line x0 = 0.5 (dense reference) ; compression must vanish
line = np.where(np.abs(fr[0][:, 0] - 0.5) < c['h'] / 2)[0]; line = line[np.argsort(fr[0][line, 1])]
Q = np.c_[np.full(4000, 0.5), np.linspace(0.005, 0.995, 4000)]; tt = 0.0; h_ = 0.025; Lref = {}
for n in range(400):
    k1 = vel(Q, tt); k2 = vel(Q + .5 * h_ * k1, tt + .5 * h_); k3 = vel(Q + .5 * h_ * k2, tt + .5 * h_); k4 = vel(Q + h_ * k3, tt + h_)
    Q = Q + h_ * (k1 + 2 * k2 + 2 * k3 + k4) / 6; tt += h_
    if n in (199, 399): Lref[round(tt, 3)] = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
lnJ = X.chained_lnJ_v2(fr, None); L0 = polylen(fr[0], line)
R['E']['DG'] = {('t=%g' % t): dict(stretch_network=float(np.log(polylen(fr[s], line) / L0)), stretch_reference=float(np.log(Lref[t] / 0.99)),
                                     compression=float(-lnJ[s][line].mean())) for t, s in ((5.0, 25), (10.0, 50))}
print('DG mixing', json.dumps(R['E']['DG']), flush=True)
# ================= D2/E2 Lamb-Oseen
Gam, a, d = 2 * np.pi, 1.0, 6.0; Wr = Gam * (1 - np.exp(-d ** 2 / a ** 2)) / (np.pi * d ** 2); T = 6 * np.pi
def lo_vel(Xq, t):
    u = np.zeros_like(Xq)
    for sgn in (1, -1):
        cc = sgn * 0.5 * d * np.array([np.cos(Wr * t), np.sin(Wr * t)]); r = Xq - cc; r2 = (r * r).sum(1) + 1e-300
        u += (Gam / (2 * np.pi * r2) * (1 - np.exp(-r2 / a ** 2)))[:, None] * np.c_[-r[:, 1], r[:, 0]]
    return u
c = cases.case_lo(a=1.0, Rd=6.5, N=8000); fr, dt = c['frames'], c['dt']; S, N = fr.shape[:2]; m = X.star_area(fr[0]) / 3.0
c0f = lambda x, y: 0.5 * (1 + np.tanh(x / 0.5)); kap = 0.02
E, smp = eul2d(lo_vel, (-7.5, 7.5, -7.5, 7.5), (384, 384), T, T / 600, c0f, kap, (T / 600 * 300, T))
out = ND.run(fr, dt, c0f(fr[0][:, 0], fr[0][:, 1]), kap, 'mass', m=m, rho=np.ones((S, N)), keep=(20,))
R['D']['LO'] = {}
for tk, s in ((T / 600 * 300, 20), (T, S - 1)):
    P = fr[s]; inn = np.hypot(*P.T) < 5.5; ref = smp(E[tk], P)
    R['D']['LO']['t=%.2f' % tk] = dict(relL2_network=rel(out[s][inn], ref[inn]), relL2_pure_advection=rel(c0f(fr[0][:, 0], fr[0][:, 1])[inn], ref[inn]))
ring = np.where(np.abs(np.hypot(fr[0][:, 0] + 3, fr[0][:, 1]) - 1.5) < c['h'] / 2)[0]; ring = ring[np.argsort(np.arctan2(fr[0][ring, 1], fr[0][ring, 0] + 3))]
th = np.linspace(-np.pi, np.pi, 3001)[:-1]; Q = np.c_[-3 + 1.5 * np.cos(th), 1.5 * np.sin(th)]; tt = 0.0; nst = 600; h_ = T / nst
for n in range(nst):
    k1 = lo_vel(Q, tt); k2 = lo_vel(Q + .5 * h_ * k1, tt + .5 * h_); k3 = lo_vel(Q + .5 * h_ * k2, tt + .5 * h_); k4 = lo_vel(Q + h_ * k3, tt + h_)
    Q = Q + h_ * (k1 + 2 * k2 + 2 * k3 + k4) / 6; tt += h_
clen = lambda Z: float(np.linalg.norm(np.diff(np.r_[Z, Z[:1]], axis=0), axis=1).sum())
lnJ = X.chained_lnJ_v2(fr, None)
R['E']['LO'] = dict(stretch_network=float(np.log(clen(fr[-1][ring]) / clen(fr[0][ring]))), stretch_reference=float(np.log(clen(Q) / (2 * np.pi * 1.5))),
                    compression=float(-lnJ[-1][ring].mean()))
print('LO', json.dumps(R['D']['LO']), json.dumps(R['E']['LO']), flush=True)
# ================= D3/E3 Sod: 1D mass-coordinate reference (exact rho), network mass vs volume form; vertical material line
cs = cases.case_sod(); fr, dt = cs['frames'], cs['dt']; S, N = fr.shape[:2]; kap = 1e-3
c0f = lambda x0: 0.5 * (1 + np.tanh(x0 / 0.02))
fine = cases.case_sod(h=2e-4, Ly=2e-4, t0=0.04, t1=0.24, S=401); o = np.argsort(fine['x0']); x0f = fine['x0'][o]
dmu = np.where(x0f > 0, 0.125, 1.0) * 2e-4; rhoF = fine['rho'][:, o]; tF = fine['t']
cond = lambda t: (lambda r: 2 * r[1:] ** 2 * r[:-1] ** 2 / (r[1:] ** 2 + r[:-1] ** 2))(rhoF[min(int(round((t - tF[0]) / (tF[1] - tF[0]))), len(tF) - 1)])
cref = lag1d(dmu, cond, c0f(x0f), tF, kap)
bad = L.vertex_bad_mask(cs['ev_pidx'], cs['ev_step'], N, S, halo=1); lnJ = X.chained_lnJ_v2(fr, bad); rho = cs['rho'][0][None, :] * np.exp(-lnJ)
inn = cs['interior'](fr[0]); refn = np.interp(cs['x0'], x0f, cref)
R['D']['SOD'] = {f: rel(ND.run(fr, dt, c0f(cs['x0']), kap, f, m=cs['mass'], rho=rho)[S - 1][inn], refn[inn]) for f in ('mass', 'volume')}
R['D']['SOD']['pure_advection'] = rel(c0f(cs['x0'])[inn], refn[inn])
col = np.where(np.abs(cs['x0'] - 0.1) < cs['h'] / 2)[0]; col = col[np.argsort(fr[0][col, 1])]
R['E']['SOD'] = dict(stretch_network=float(np.log(polylen(fr[-1], col) / polylen(fr[0], col))), stretch_exact=0.0,
                     compression_network=float(-lnJ[-1][col].mean()), compression_exact=float(np.log(cs['info']['rsR'] / 0.125)))
print('SOD', json.dumps(R['D']['SOD']), json.dumps(R['E']['SOD']), flush=True)
# ================= D4/E4 blast: 1D radial mass-coordinate reference
cb = cases.case_blast(); fr, dt = cb['frames'], cb['dt']; S, N = fr.shape[:2]; kap = 2e-4; p, R0, t0 = 1 / 2.4, 0.2, 1.0
c0f = lambda rs: 0.5 * (1 + np.tanh((rs - 0.5) / 0.04))
rsF = np.linspace(1e-4, 1.2, 6000); dr = rsF[1] - rsF[0]; rho0F = np.where(rsF <= R0, 6.0, 1.0); dmu = 2 * np.pi * rsF * rho0F * dr
def kin(rs, t):
    tc = t0 * (rs / R0) ** 2; ins = rs <= R0; post = (~ins) & (t >= tc)
    r = np.where(ins, rs * (t / t0) ** p, np.where(post, rs * (np.maximum(t, tc) / tc) ** p, rs))
    lnJ_ = np.where(ins, 2 * p * np.log(t / t0), np.where(post, -np.log(6.0) + 2 * p * np.log(np.maximum(t, tc) / tc), 0.0))
    return r, np.where(ins, 6.0, 1.0) * np.exp(-lnJ_)
def condb(t):
    r, rh = kin(rsF, t); k = 4 * np.pi ** 2 * r ** 2 * rh ** 2; return 2 * k[1:] * k[:-1] / (k[1:] + k[:-1])
cref = lag1d(dmu, condb, c0f(rsF), np.linspace(1.0, 25.0, 2401), kap)
bad = L.vertex_bad_mask(cb['ev_pidx'], cb['ev_step'], N, S, halo=1); lnJ = X.chained_lnJ_v2(fr, bad)
rho0 = np.where(cb['rs'] <= R0, 6.0, 1.0); rho = rho0[None, :] * np.exp(-lnJ); m = rho0 * X.star_area(fr[0]) / 3.0
inn = cb['interior'](fr[0]); refn = np.interp(cb['rs'], rsF, cref)
R['D']['BLAST'] = {f: rel(ND.run(fr, dt, c0f(cb['rs']), kap, f, m=m, rho=rho)[S - 1][inn], refn[inn]) for f in ('mass', 'volume')}
R['D']['BLAST']['pure_advection'] = rel(c0f(cb['rs'])[inn], refn[inn])
ring = np.where(np.abs(cb['rs'] - 0.5) < cb['h'] / 2)[0]; ring = ring[np.argsort(np.arctan2(fr[0][ring, 1], fr[0][ring, 0]))]
rT, _ = kin(np.array([0.5]), 25.0)
R['E']['BLAST'] = dict(stretch_network=float(np.log(clen(fr[-1][ring]) / clen(fr[0][ring]))), stretch_exact=float(np.log(rT[0] / 0.5)),
                       compression_network=float(-lnJ[-1][ring].mean()), compression_exact=float(-cb['lnJ_ex'][-1][ring].mean()))
print('BLAST', json.dumps(R['D']['BLAST']), json.dumps(R['E']['BLAST']), flush=True)
# ================= D5/E5 shock-bubble: reuse validated numbers ; D6/E6 shock-vortex: in-loop vs offline consistency
R['D']['SB'] = json.load(open('res_val_sb_diff.json')); R['E']['SB'] = R['D']['SB'].pop('mixing')
S_ = pickle.load(open('/home/claude/bench/sv_dns.pkl', 'rb')); P0 = S_['P0']; r0 = np.hypot(P0[:, 0] - S_['xv'], P0[:, 1] - S_['yv'])
ring = np.where(np.abs(r0 - 2 * S_['Rv']) < 0.0025)[0]; ring = ring[np.argsort(np.arctan2(P0[ring, 1] - S_['yv'], P0[ring, 0] - S_['xv']))]
c0 = 0.5 * (1 + np.tanh((P0[:, 0] - 0.75) / 0.03)); outs = {}; R['E']['SV'] = {}
for mode in ('in', 'off'):
    fr = S_['fr_' + mode][:31]; n2 = TN(fr, S_['dt']); n2.events(); lnJ = X.chained_lnJ_v2(fr, n2.bad)
    rho = S_['Ev']['rho'][0][None, :] * np.exp(-lnJ); m = S_['Ev']['rho'][0] * X.star_area(fr[0]) / 3.0
    outs[mode] = ND.run(fr, S_['dt'], c0, 2e-4, 'mass', m=m, rho=rho)[30]
    R['E']['SV'][mode] = dict(stretch=float(np.log(clen(fr[30][ring]) / clen(fr[0][ring]))), compression=float(-lnJ[30][ring].mean()))
R['F']['SV_diffusion_inloop_vs_offline_relL2'] = rel(outs['off'], outs['in'])
print('SV', json.dumps(R['E']['SV']), R['F']['SV_diffusion_inloop_vs_offline_relL2'], flush=True)
json.dump(R, open('res_tr_b.json', 'w'), indent=1)
