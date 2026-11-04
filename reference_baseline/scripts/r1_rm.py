import numpy as np, json, pickle, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bcom import *
from lhdp import disp_cochain_tri, tri_area
import stretch_est as SE
D = pickle.load(open("rm_dns_dx0.005.pkl", "rb")); fr = D["frames"]; t = D["t"]; Ev = D["E"]; S, N, _ = fr.shape; a0 = D["a0"]; dx = D["dx"]
out = dict(N=N, S=S, a0=a0)
P0 = fr[0]; xi = lambda y: 0.5+a0*np.cos(2*np.pi*y)
phi = (P0[:, 0]-xi(P0[:, 1]))/np.sqrt(1+(2*np.pi*a0*np.sin(2*np.pi*P0[:, 1]))**2)
d, _ = cKDTree(P0).query(P0, k=2); h = float(np.median(d[:, 1])); out["h"] = h
# T1 + T2
lnJ, hot, bad, tau = ledger(fr); rho = Ev["rho"]
jump = (np.abs(np.log(rho[1:]/rho[:-1])) > np.log(1.15)).T
out["T1_detect"] = float(hot.any(1)[jump.any(1)].mean()); tn = jump.copy(); tn[:, 1:] |= jump[:, :-1]; tn[:, :-1] |= jump[:, 1:]
out["T1_false"] = float((hot & ~tn).sum()/max((~tn).sum(), 1))
x = fr[-1]; mask = (x[:, 0] > x[:, 0].min()+.02) & (x[:, 0] < x[:, 0].max()-.02) & (x[:, 1] < 0.48)
for s in [10, 25, 50]:
    e = np.abs(rho[0]*np.exp(-lnJ[s])/rho[s]-1)
    out["T2_t=%.1f" % t[s]] = dict(med=float(np.median(e[mask])), p95=float(np.quantile(e[mask], .95)),
                                   p95_away_interface=float(np.quantile(e[mask & (np.abs(phi) > 0.03)], .95)))
# baselines at the final frame
m = rho[0]*X.star_area(P0)/3.0
out["T2_baselines_final"] = {k: errstats(v, rho[-1], mask) for k, v in dict(ledger=rho[0]*np.exp(-lnJ[-1]), dtfe=X.dtfe_density(fr[-1], m), sph=X.sph_density(fr[-1], m)).items()}
# interface amplitude (symmetry line y=0: crest; wall y=0.5: trough) from material interface tracers
band = np.abs(phi) < 0.6*h; top = band & (P0[:, 1] < 0.02); bot = band & (P0[:, 1] > 0.48)
amp = 0.5*(fr[:, top, 0].mean(1)-fr[:, bot, 0].mean(1)); xm = 0.5*(fr[:, top, 0].mean(1)+fr[:, bot, 0].mean(1))
def dns_amp(W, xc, yc):
    Y = W[4]; xe = (1-Y).sum(1)*dx        # equivalent interface position per row
    return 0.5*(xe[0]-xe[-1]), xe.mean()
G = D["grids"]; amp_dns = {float(k): dns_amp(v["W"], D["xc"], D["yc"]) for k, v in G.items()}
out["amp_track"] = [[float(t[s]), float(amp[s])] for s in range(S)]
out["amp_dns"] = [[k, float(v[0])] for k, v in sorted(amp_dns.items())]
# Richtmyer impulsive model from positions only
vx = np.gradient(xm, t)                                         # interface translation
ish = int(np.argmax(vx > 0.5*vx.max()))                          # shock arrival
s1 = min(ish+3, S-1)
du = float(np.median(vx[s1:s1+5])); a0p = float(amp[s1])
rhoL = float(np.median((rho[0]*np.exp(-lnJ[s1]))[band & (phi < 0) | ((phi < -0.02) & (phi > -0.06))]))
rhoH = float(np.median((rho[0]*np.exp(-lnJ[s1]))[(phi > 0.02) & (phi < 0.06)]))
Ap = (rhoH-rhoL)/(rhoH+rhoL); k = 2*np.pi
fit = slice(s1+1, min(s1+8, S)); rate = float(np.polyfit(t[fit], amp[fit], 1)[0])
out["richtmyer"] = dict(t_shock=float(t[ish]), du=du, a0_plus=a0p, rho_light=rhoL, rho_heavy=rhoH, A_plus=Ap,
                        rate_model=float(k*du*Ap*a0p), rate_model_MB=float(k*du*Ap*0.5*(a0+a0p)), rate_measured=rate)
# T6: heavy-gas mass penetrating behind the mean interface (spikes), vs DNS
heavy = Ev["Y1"][0] > 0.5; tr6 = {}
Xg, Yg = np.meshgrid(D["xc"], D["yc"])
for k_, v in sorted(G.items()):
    s = int(round(float(k_)/D["dt"])); W = v["W"]; Xc = amp_dns[float(k_)][1]
    tr6["t=%g" % float(k_)] = [float(m[heavy & (fr[s][:, 0] < Xc)].sum()), float((W[0]*W[4]*(Xg < Xc)).sum()*dx*dx), float(Xc)]
out["T6_heavy_mass_behind_mean_interface"] = tr6
# T4: circulation of the interface material region, position form vs DNS
reg = np.abs(phi) < 0.08; tri0 = X.complex_peeled(P0)["tri"]; triR = tri0[reg[tri0].all(1)]
Gp, Gd = [], []
for s in range(S-1):
    c, _ = disp_cochain_tri(fr[s], fr[s+1], triR); Gp.append(c.sum()/(t[s+1]-t[s]))
    Pm = 0.5*(fr[s]+fr[s+1]); o = 0.5*(Ev["om"][s]+Ev["om"][s+1]); Gd.append((o[triR].mean(1)*tri_area(Pm, triR)).sum())
Gp, Gd = np.array(Gp), np.array(Gd)
out["T4"] = dict(G_final=[float(Gp[-1]), float(Gd[-1])], relerr_med=float(np.median(np.abs(Gp[ish+2:]-Gd[ish+2:])/np.abs(Gd[ish+2:]))),
                 G_after_shock=[float(Gp[ish+3]), float(Gd[ish+3])], G_before=float(Gp[max(ish-2, 0)]))
# circulation-based growth: sinusoidal vortex sheet gamma0 sin(ky): da/dt = gamma0/2, Gamma_half = 2 gamma0/k -> da/dt = k Gamma/4
Gpost = float(np.median(Gp[ish+3:ish+8])); out["richtmyer"]["rate_from_circulation"] = float(k*abs(Gpost)/4); out["richtmyer"]["Gamma_post"] = Gpost
# T9: interface stretch (chained tensor) and compression; total length check vs DNS contour
idx = np.where(band)[0]; yy = P0[idx, 1]; tvec = np.c_[-2*np.pi*a0*np.sin(2*np.pi*yy), np.ones(len(idx))]
tvec = tvec/np.linalg.norm(tvec, axis=1)[:, None]; w = 1/np.abs(tvec[:, 1])   # ds0/dy0
keepf = [int(round(float(k_)/D["dt"])) for k_ in sorted(G)]
F = SE.chained_F(fr, h, bad=bad, idx=idx, keep=set(keepf))
cs = plt.figure(); T9 = {}
for k_, s in zip(sorted(G), keepf):
    ll, A = SE.vector_lnlam(F[s][0], tvec); comp = -lnJ[s][idx]
    cntr = plt.contour(D["xc"], D["yc"], G[k_]["W"][4], levels=[0.5]); Lc = 0.0
    for pth in cntr.allsegs[0]: Lc += np.linalg.norm(np.diff(pth, axis=0), axis=1).sum()
    L0 = np.trapezoid(np.sqrt(1+(2*np.pi*a0*np.sin(2*np.pi*np.linspace(0, .5, 2001)))**2), np.linspace(0, .5, 2001))
    T9["t=%g" % float(k_)] = dict(stretch_mean=float(np.average(ll, weights=w)), compression=float(np.average(comp, weights=w)),
                                  ln_mean_lambda=float(np.log(np.average(np.exp(ll), weights=w))), ln_L_dns=float(np.log(Lc/L0)),
                                  total_growth=float(np.average(ll+comp, weights=w)), A_med=float(np.median(A)))
out["T9"] = T9; plt.close("all")
json.dump(out, open("res_rm.json", "w"), indent=1); print(json.dumps(out)[:3000])
pickle.dump(dict(P0=P0, PT=fr[-1], t=t, amp=amp, amp_dns=out["amp_dns"], Gp=Gp, Gd=Gd, ish=ish, rich=out["richtmyer"], T9=T9,
                 Ws={k_: G[k_]["W"][[0, 4]] for k_ in G}, xc=D["xc"], yc=D["yc"], heavy=heavy, frames_sel={s: fr[s] for s in keepf},
                 lnJ_final=lnJ[-1], rho_final=rho[-1], rho0=rho[0]), open("rm.pkl", "wb"))
