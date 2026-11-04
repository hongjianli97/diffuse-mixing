"""DNS refinement: interface compression reference and the source of the density p95 (shock-bubble, shock-vortex)."""
import numpy as np, json, pickle, sys
from bcom import *
out = {}
bands = [(-0.06, -0.03), (-0.03, -0.015), (-0.015, -0.005), (-0.003, 0.003), (0.005, 0.015), (0.015, 0.03), (0.03, 0.06)]
def sb(dx, pre="sb"):
    D = pickle.load(open("%s_dns_dx%g.pkl" % (pre, dx), "rb")); fr = D["frames"]; t = D["t"]; rho = D["E"]["rho"]
    lnJ, hot, bad, tau = ledger(fr); P0 = fr[0]; phi = np.hypot(P0[:, 0]-D["xb"], P0[:, 1])-D["R"]
    x = fr[-1]; mask = (x[:, 0] > x[:, 0].min()+.02) & (x[:, 0] < x[:, 0].max()-.02) & (x[:, 1] < x[:, 1].max()-.02)
    o = dict(dx=dx)
    for tt in [0.6, 1.0, 1.4]:
        s = int(round(tt/D["dt"])); led = -lnJ[s]; dns = np.log(rho[s]/rho[0])
        o["t=%g" % tt] = dict(bands={"%g..%g" % b: [float(np.mean(led[(phi > b[0]) & (phi < b[1])])), float(np.mean(dns[(phi > b[0]) & (phi < b[1])]))] for b in bands})
        e = np.abs(rho[0]*np.exp(-lnJ[s])/rho[s]-1); e[~mask] = np.nan
        q95 = np.nanquantile(e, .95); top = e >= q95
        o["t=%g" % tt].update(med=float(np.nanmedian(e)), p95=float(q95), p95_outside_interface=float(np.nanquantile(e[np.abs(phi) > 0.03], .95)),
                               top5_in_interface_band=float((np.abs(phi[top]) < 0.03).mean()),
                               top5_hot_recent=float(hot[top][:, max(s-3, 0):s].any(1).mean()))
    W = D["grids"][1.4]["W"]; Xg, Yg = np.meshgrid(D["xc"], D["yc"]); dA = dx*dx
    m = rho[0]*X.star_area(fr[0])/3.0; heavy = D["E"]["Y1"][0] > 0.5
    o["heavy_mass_x>1"] = [float(m[heavy & (fr[-1][:, 0] > 1.0)].sum()), float((W[0]*W[4]*(Xg > 1.0)).sum()*dA)]
    o["heavy_mass_total"] = [float(m[heavy].sum()), float((W[0]*W[4]).sum()*dA)]
    # T9 attribution on the interface band |phi0| < 0.6 h (chained tensor stretch + ledger compression)
    import stretch_est as SE
    band = np.where(np.abs(phi) < 0.6*0.005)[0]; th = np.arctan2(P0[band, 1], P0[band, 0]-D["xb"]); t0 = np.c_[-np.sin(th), np.cos(th)]
    keep = {int(round(tt/D["dt"])) for tt in [0.6, 1.0, 1.4]}
    F = SE.chained_F(fr, 0.005, bad=bad, idx=band, keep=keep)
    o["T9"] = {}
    for s_ in sorted(keep):
        ll, A = SE.vector_lnlam(F[s_][0], t0); comp = -lnJ[s_][band]
        o["T9"]["t=%g" % round(s_*D["dt"], 2)] = dict(stretch=float(ll.mean()), compression=float(comp.mean()),
                                                     share=float(comp.mean()/(comp.mean()+ll.mean())), A_med=float(np.median(A)))
    return o
def sv(path, dxl):
    D = pickle.load(open(path, "rb")); fr = D["fr_in"]; rho = D["Ev"]["rho"]
    lnJ, hot, bad, tau = ledger(fr); x = fr[-1]
    mask = (x[:, 0] > x[:, 0].min()+.02) & (x[:, 0] < x[:, 0].max()-.02) & (x[:, 1] > x[:, 1].min()+.02) & (x[:, 1] < x[:, 1].max()-.02)
    e = np.abs(rho[0]*np.exp(-lnJ[-1])/rho[-1]-1); e[~mask] = np.nan; q95 = np.nanquantile(e, .95); top = e >= q95
    c0 = np.array([D["xv"], D["yv"]]); r0 = np.hypot(*(fr[0]-c0).T)
    return dict(dx=dxl, med=float(np.nanmedian(e)), p95=float(q95), top5_hot_recent=float(hot[top][:, -3:].any(1).mean()),
                top5_in_core=float((r0[top] < 2*D["Rv"]).mean()), core_med=float(np.nanmedian(e[r0 < 2*D["Rv"]])),
                core_p95=float(np.nanquantile(e[r0 < 2*D["Rv"]], .95)))
import os
which = sys.argv[1]
if which == "sbv":
    for dx in [0.01, 0.005, 0.0035]:
        if os.path.exists("sbv_dns_dx%g.pkl" % dx): out["sbv_dx%g" % dx] = sb(dx, "sbv"); print(json.dumps(out["sbv_dx%g" % dx])[:300], flush=True)
elif which == "sb":
    for dx in [0.01, 0.005, 0.0025]:
        if os.path.exists("sb_dns_dx%g.pkl" % dx): out["sb_dx%g" % dx] = sb(dx); print(json.dumps(out["sb_dx%g" % dx])[:600], flush=True)
else:
    for dx, pth in [(0.01, "sv_dns_dx0.01.pkl"), (0.005, "sv_dns.pkl"), (0.0025, "sv_dns_dx0.0025.pkl")]:
        if os.path.exists(pth): out["sv_dx%g" % dx] = sv(pth, dx); print(out["sv_dx%g" % dx], flush=True)
json.dump(out, open("res_dnsref_%s.json" % which, "w"), indent=1)
