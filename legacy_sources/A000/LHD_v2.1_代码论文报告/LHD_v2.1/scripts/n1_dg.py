import numpy as np, json, pickle, time
import cases
from common import *
t0 = time.time(); out = {}
vel, vort, grad = cases.dg_fields()
c = cases.case_dg(nx=160, ny=80, T=10.0, S=51); P0 = c["frames"][0]; t = c["t"]
fr, Fs = integrate(vel, P0, t, sub=8)
out["traj_vs_case"] = float(np.abs(fr-c["frames"]).max())
rs = readout_series(fr, t, vort, stride=5)
out["om_corr"] = [corr(r["om_hat"], r["om_ex"]) for r in rs]
out["om_relL2"] = [float(np.sqrt(np.nanmean((r["om_hat"]-r["om_ex"])**2)/np.nanmean(r["om_ex"]**2))) for r in rs]
out["pseudo_div"] = [float(np.nanstd(r["th_hat"])/np.nanstd(r["om_hat"])) for r in rs]
kr = []
for s in range(1, 50, 7):
    tri = X.complex_peeled(fr[s])["tri"]; d, dg, Gp, Gm = kelvin_pos(fr[s-1], fr[s], fr[s+1], tri)
    kr.append(float(np.max(np.abs(d-dg))/np.max(np.abs(Gp))))
out["kelvin_res_max"] = max(kr)
for T_ in [10.0, 5.0]:
    S_ = int(round(T_/0.2))+1
    A = run_afrc(fr[:S_]); bex, lns = R_exact(Fs[:S_])
    out["afrc_T%g" % T_] = summary(A, bex, lns)
    if T_ == 10.0: keep = dict(A=A, bex=bex, lns=lns)
out["time"] = time.time()-t0
json.dump(out, open("res_dg.json", "w"), indent=1); print(json.dumps(out, indent=1))
A = keep["A"]
pickle.dump(dict(P0=fr[0], B=A["net"]["B"], beta=A["ind"]["beta"], K=A["ind"]["K"], lab=A["lab"], cand=A["cand"],
                 bex=keep["bex"], lns=keep["lns"], rs=rs[5]), open("dg.pkl", "wb"))
