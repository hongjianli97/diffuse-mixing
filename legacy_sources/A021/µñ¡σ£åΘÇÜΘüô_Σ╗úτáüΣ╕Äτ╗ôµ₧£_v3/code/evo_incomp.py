import json, numpy as np, cases, lhdx as X, evo_lib as EV
# ---------------- double gyre
vel, _, _ = cases.dg_fields(); c = cases.case_dg(); fr, t = c['frames'], c['t']; P0 = fr[0]
c0f = lambda x, y: 0.5 * (1 + np.tanh((x - 1) / 0.1)); side = lambda P: P[:, 0] > 1.0
cd = cases.case_dg(nx=640, ny=320); a0 = side(cd['frames'][0]); md = X.star_area(cd['frames'][0]) / 3.0
fref = np.array([np.sum(md * (~a0 & side(cd['frames'][s]))) for s in range(len(t))]) * (X.star_area(P0).sum() / X.star_area(cd['frames'][0]).sum()); del cd
keep = [float(k) for k in range(1, 11)]; E, (XX, YY) = EV.eul2d(vel, (0, 2, 0, 1), (512, 256), 10.0, 0.01, c0f, 1e-3, keep)
vref = (np.r_[0.0, keep], [float(np.var(c0f(XX, YY)))] + [float(np.var(E[k])) for k in keep])
Q0 = np.c_[np.full(8001, 0.5), np.linspace(0.005, 0.995, 8001)]; Q = Q0.copy(); tt = 0.0; h = 0.025; sref = ([0.0], [0.0])
for n in range(400):
    k1 = vel(Q, tt); k2 = vel(Q + .5 * h * k1, tt + .5 * h); k3 = vel(Q + .5 * h * k2, tt + .5 * h); k4 = vel(Q + h * k3, tt + h); Q = Q + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6; tt += h
    if (n + 1) % 40 == 0: sref[0].append(tt); sref[1].append(float(np.mean(np.log(np.linalg.norm(np.diff(Q, axis=0), axis=1) / np.linalg.norm(np.diff(Q0, axis=0), axis=1)))))
idx = np.where(np.abs(P0[:, 0] - 0.5) < c['h'] / 2)[0]; tang = np.tile([0.0, 1.0], (len(P0), 1))
r, _ = EV.run_case(dict(name='DG', title='Double gyre (incompressible): material transport, diffusion ($\\kappa=10^{-3}$) and mixing', fr=fr, t=t, tidx=(0, 12, 25, 50),
                        rho0=np.ones(len(P0)), compressible=False, kappa=1e-3, c0=c0f(P0[:, 0], P0[:, 1]), label=P0[:, 0] < 1, label_names=('right-gyre fluid', 'left-gyre fluid'),
                        side=side, side_name='right half (x>1)', iface=(idx, tang), refs=dict(flux=(t, fref), var=vref, stretch=sref, comp=(t, np.zeros(len(t)))), coh_nb=(20, 10)))
print('DG', json.dumps(r['windows']), 'flux', r['flux_final'], r.get('flux_final_reference'), 'var', r['var_final_ratio'], r.get('var_final_ratio_reference'), flush=True)
# ---------------- Lamb-Oseen pair
Gam, a, d = 2 * np.pi, 1.0, 6.0; Wr = Gam * (1 - np.exp(-d ** 2 / a ** 2)) / (np.pi * d ** 2); T = 6 * np.pi
def lo_vel(Xq, t_):
    u = np.zeros_like(Xq)
    for sg in (1, -1):
        cc = sg * 0.5 * d * np.array([np.cos(Wr * t_), np.sin(Wr * t_)]); rr = Xq - cc; r2 = (rr * rr).sum(1) + 1e-300
        u += (Gam / (2 * np.pi * r2) * (1 - np.exp(-r2 / a ** 2)))[:, None] * np.c_[-rr[:, 1], rr[:, 0]]
    return u
c = cases.case_lo(a=1.0, Rd=6.5, N=8000); fr, t = c['frames'], c['t']; P0 = fr[0]; side = lambda P: P[:, 0] > 0.0
cl = cases.case_lo(a=1.0, Rd=6.5, N=128000); a0 = side(cl['frames'][0]); ml = X.star_area(cl['frames'][0]) / 3.0
fref = np.array([np.sum(ml * (~a0 & side(cl['frames'][s]))) for s in range(len(t))]) * (X.star_area(P0).sum() / X.star_area(cl['frames'][0]).sum()); del cl
c0f = lambda x, y: 0.5 * (1 + np.tanh(x / 0.5)); dte = T / 600; keep = [round(dte * k, 9) for k in range(60, 601, 60)]
E, (XX, YY) = EV.eul2d(lo_vel, (-7.5, 7.5, -7.5, 7.5), (384, 384), T, dte, c0f, 0.02, keep); disk = np.hypot(XX, YY) < 6.5
vref = (np.r_[0.0, keep], [float(np.var(c0f(XX, YY)[disk]))] + [float(np.var(E[k][disk])) for k in keep])
th = np.linspace(-np.pi, np.pi, 3001)[:-1]; Q0 = np.c_[-3 + 1.5 * np.cos(th), 1.5 * np.sin(th)]; Q = Q0.copy(); tt = 0.0; nst = 600; h = T / nst; sref = ([0.0], [0.0])
seg = lambda Z: np.linalg.norm(np.diff(np.r_[Z, Z[:1]], axis=0), axis=1)
for n in range(nst):
    k1 = lo_vel(Q, tt); k2 = lo_vel(Q + .5 * h * k1, tt + .5 * h); k3 = lo_vel(Q + .5 * h * k2, tt + .5 * h); k4 = lo_vel(Q + h * k3, tt + h); Q = Q + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6; tt += h
    if (n + 1) % 60 == 0: sref[0].append(tt); sref[1].append(float(np.mean(np.log(seg(Q) / seg(Q0)))))
d0 = P0 - [-3.0, 0.0]; idx = np.where(np.abs(np.hypot(*d0.T) - 1.5) < c['h'] / 2)[0]; ang = np.arctan2(d0[:, 1], d0[:, 0]); tang = np.c_[-np.sin(ang), np.cos(ang)]
r, _ = EV.run_case(dict(name='LO', title='Lamb-Oseen co-rotating pair (incompressible): transport, diffusion ($\\kappa=0.02$) and mixing', fr=fr, t=t, tidx=(0, 10, 20, 40),
                        rho0=np.ones(len(P0)), compressible=False, kappa=0.02, c0=c0f(P0[:, 0], P0[:, 1]), label=P0[:, 0] < 0, label_names=('right fluid', 'left fluid'),
                        side=side, side_name='right half (x>0)', iface=(idx, tang), refs=dict(flux=(t, fref), var=vref, stretch=sref, comp=(t, np.zeros(len(t))))))
print('LO', json.dumps(r['windows']), 'flux', r['flux_final'], r.get('flux_final_reference'), 'var', r['var_final_ratio'], r.get('var_final_ratio_reference'), flush=True)
