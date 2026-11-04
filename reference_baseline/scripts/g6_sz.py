"""Shock-bubble circulation: position-only Kelvin (material band) vs DNS vs Samtaney-Zabusky (1994) models."""
import numpy as np, json, pickle
from bcom import *
from lhdp import disp_cochain_tri, tri_area
g, M, eta, R, c0 = 1.4, 1.5, 3.0, 0.25, 1.0
xi = (1+1/M+2/M**2)*(M-1)
Gp1 = 2/(g+1)*(1-eta**-0.5)*xi*c0           # SZ (4.12) first-order coefficient, converted to physical units (x c0/sqrt(gamma))
out = dict(SZ_Gamma1prime=Gp1, SZ_phase_ii=Gp1*R, SZ1_phase_iii=(1+np.pi/2)*Gp1*R)
for nm, f in [("inviscid_dx0.005", "sb_dns_dx0.005.pkl"), ("inviscid_dx0.0025", "sb_dns_dx0.0025.pkl"), ("viscous_dx0.0035", "sbv_dns_dx0.0035.pkl")]:
    try: D = pickle.load(open(f, "rb"))
    except Exception: continue
    fr = D["frames"]; t = D["t"]; om = D["E"]["om"]; P0 = fr[0]; S = len(t)
    phi = np.hypot(P0[:, 0]-D["xb"], P0[:, 1])-D["R"]; tri0 = X.complex_peeled(P0)["tri"]; o = {}
    for wb in [0.04, 0.08, 0.15]:
        reg = np.abs(phi) < wb; triR = tri0[reg[tri0].all(1)]; Gp, Gd = [], []
        for s in range(S-1):
            c, _ = disp_cochain_tri(fr[s], fr[s+1], triR); Gp.append(c.sum()/(t[s+1]-t[s]))
            Pm = 0.5*(fr[s]+fr[s+1]); w = 0.5*(om[s]+om[s+1]); Gd.append((w[triR].mean(1)*tri_area(Pm, triR)).sum())
        Gp, Gd = np.array(Gp), np.array(Gd); tm = 0.5*(t[1:]+t[:-1])
        o["band%g" % wb] = dict(t=[float(x) for x in tm[::5]], G_pos=[float(x) for x in Gp[::5]], G_dns=[float(x) for x in Gd[::5]],
                                G_pos_at=[float(np.interp(x, tm, Gp)) for x in [0.6, 1.0, 1.4]], G_dns_at=[float(np.interp(x, tm, Gd)) for x in [0.6, 1.0, 1.4]],
                                relerr_med=float(np.median(np.abs(Gp[tm > 0.4]-Gd[tm > 0.4])/np.abs(Gd[tm > 0.4]))), min_pos=float(Gp.min()))
    # DNS whole-domain circulation (upper half) from kept grids
    dx = D["dx"]; o["dns_domain"] = {str(k): float(v["om"].sum()*dx*dx) for k, v in D["grids"].items() if "om" in v}
    out[nm] = o; print(nm, json.dumps({k: (v["G_pos_at"], v["G_dns_at"], round(v["relerr_med"], 4)) if k.startswith("band") else v for k, v in o.items()}), flush=True)
json.dump(out, open("res_g6_sz.json", "w"), indent=1); print("SZ", Gp1, Gp1*R, (1+np.pi/2)*Gp1*R)
