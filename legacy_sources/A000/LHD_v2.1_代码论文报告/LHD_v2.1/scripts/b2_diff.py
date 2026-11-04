import numpy as np, json, cases, lhd as L, time
exec(open("tr_b.py").read().split("# ================= D1")[0].split("R = {'D'")[0])
exec("def lag1d" + open("tr_b.py").read().split("def lag1d")[1].split("def polylen")[0])
exec("def eul2d" + open("tr_b.py").read().split("def eul2d")[1].split("def lag1d")[0])
from bcom import *
out = {}
def three(fr, dt, c0, kap, m, rho, ref, inn):
    t0 = time.time(); a = ND.run(fr, dt, c0, kap, 'mass', m=m, rho=rho)[fr.shape[0]-1]; t1 = time.time()
    b = ND.run(fr, dt, c0, kap, 'volume', m=m, rho=rho)[fr.shape[0]-1]; t2 = time.time()
    s = sph_diffusion(fr, dt, c0, kap, m, rho); t3 = time.time()
    o = dict(network_mass=rel(a[inn], ref[inn]), network_volume=rel(b[inn], ref[inn]), sph=rel(s[inn], ref[inn]),
             pure_advection=rel(np.asarray(c0)[inn], ref[inn]), time_network=t1-t0, time_sph=t3-t2,
             mass_cons_network=float(abs((m*a).sum()/(m*c0).sum()-1)), mass_cons_sph=float(abs((m*s).sum()/(m*c0).sum()-1)),
             overshoot_network=float(max(a.max()-1, -a.min(), 0)), overshoot_sph=float(max(s.max()-1, -s.min(), 0)))
    return o
# Sod
cs = cases.case_sod(); fr, dt = cs['frames'], cs['dt']; S, N = fr.shape[:2]; kap = 1e-3
c0f = lambda x0: 0.5*(1+np.tanh(x0/0.02))
fine = cases.case_sod(h=2e-4, Ly=2e-4, t0=0.04, t1=0.24, S=401); o = np.argsort(fine['x0']); x0f = fine['x0'][o]
dmu = np.where(x0f > 0, 0.125, 1.0)*2e-4; rhoF = fine['rho'][:, o]; tF = fine['t']
cond = lambda t: (lambda r: 2*r[1:]**2*r[:-1]**2/(r[1:]**2+r[:-1]**2))(rhoF[min(int(round((t-tF[0])/(tF[1]-tF[0]))), len(tF)-1)])
cref = lag1d(dmu, cond, c0f(x0f), tF, kap)
lnJ, hot, bad, tau = ledger(fr); rho = cs['rho'][0][None, :]*np.exp(-lnJ)
inn = cs['interior'](fr[0]); refn = np.interp(cs['x0'], x0f, cref)
out['sod'] = three(fr, dt, c0f(cs['x0']), kap, cs['mass'], rho, refn, inn); print('sod', out['sod'], flush=True)
# blast
cb = cases.case_blast(); fr, dt = cb['frames'], cb['dt']; S, N = fr.shape[:2]; kap = 2e-4; p, R0, t0 = 1/2.4, 0.2, 1.0
c0f = lambda rs: 0.5*(1+np.tanh((rs-0.5)/0.04))
rsF = np.linspace(1e-4, 1.2, 6000); dr = rsF[1]-rsF[0]; rho0F = np.where(rsF <= R0, 6.0, 1.0); dmu = 2*np.pi*rsF*rho0F*dr
exec("def kin" + open("tr_b.py").read().split("def kin")[1].split("cref = lag1d")[0])
cref = lag1d(dmu, condb, c0f(rsF), np.linspace(1.0, 25.0, 2401), kap)
lnJ, hot, bad, tau = ledger(fr); rho0 = np.where(cb['rs'] <= R0, 6.0, 1.0); rho = rho0[None, :]*np.exp(-lnJ); m = rho0*X.star_area(fr[0])/3.0
inn = cb['interior'](fr[0]); refn = np.interp(cb['rs'], rsF, cref)
out['blast'] = three(fr, dt, c0f(cb['rs']), kap, m, rho, refn, inn); print('blast', out['blast'], flush=True)
# double gyre
vel, vort, grad = cases.dg_fields(); c = cases.case_dg(); fr, dt = c['frames'], c['dt']; S, N = fr.shape[:2]
c0 = 0.5*(1+np.tanh((fr[0][:, 0]-1)/0.1)); m = X.star_area(fr[0])/3.0
E, smp = eul2d(vel, (0, 2, 0, 1), (512, 256), 10.0, 0.01, lambda x, y: 0.5*(1+np.tanh((x-1)/0.1)), 1e-3, (10.0,))
inn = (fr[-1][:, 0] > .03) & (fr[-1][:, 0] < 1.97) & (fr[-1][:, 1] > .03) & (fr[-1][:, 1] < .97); ref = smp(E[10.0], fr[-1])
out['dg'] = three(fr, dt, c0, 1e-3, m, np.ones((S, N)), ref, inn); print('dg', out['dg'], flush=True)
json.dump(out, open("res_b2_diff.json", "w"), indent=1)
