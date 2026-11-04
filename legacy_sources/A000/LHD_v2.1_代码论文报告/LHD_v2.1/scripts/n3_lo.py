import numpy as np, json, pickle, time
import cases
from common import *
t0 = time.time(); out = {}
GAM, A_, D_ = 2*np.pi, 1.0, 6.0
WR = GAM*(1-np.exp(-D_**2/A_**2))/(np.pi*D_**2)
cen = lambda t: [0.5*D_*np.array([np.cos(WR*t), np.sin(WR*t)]), -0.5*D_*np.array([np.cos(WR*t), np.sin(WR*t)])]
def vel(Q, t):
    u = np.zeros_like(Q)
    for c in cen(t):
        r = Q-c; r2 = (r*r).sum(1)+1e-300
        u += (GAM/(2*np.pi*r2)*(1-np.exp(-r2/A_**2)))[:, None]*np.c_[-r[:, 1], r[:, 0]]
    return u
def vort(Q, t): return sum(GAM/(np.pi*A_**2)*np.exp(-((Q-c)**2).sum(1)/A_**2) for c in cen(t))
c = cases.case_lo(a=1.0, Rd=6.5, N=8000); P0 = c["frames"][0]; t = c["t"]
fr, Fs = integrate(vel, P0, t, sub=10)
out["N"] = len(P0)
rs = readout_series(fr, t, vort, stride=4)
out["om_corr"] = [corr(r["om_hat"], r["om_ex"]) for r in rs]
out["om_relL2"] = [float(np.sqrt(np.nanmean((r["om_hat"]-r["om_ex"])**2)/np.nanmean(r["om_ex"]**2))) for r in rs]
out["pseudo_div"] = [float(np.nanstd(r["th_hat"])/np.nanstd(r["om_hat"])) for r in rs]
# objectivity of readouts: superpose rotating frame + boost, with Kabsch alignment
Q = lambda b: np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
frT = np.stack([fr[s]@Q(0.3*t[s]).T+np.array([0.2*t[s], -0.1*t[s]]) for s in range(len(t))])
s = 20; tri = X.complex_peeled(fr[s])["tri"]
a1 = step_readouts(fr[s], pullback(fr[s], fr[s+1])[0], tri); a2 = step_readouts(frT[s], pullback(frT[s], frT[s+1])[0], tri)
b1 = step_readouts(fr[s], fr[s+1], tri); b2 = step_readouts(frT[s], frT[s+1], tri)
out["obj_aligned_psi"] = float(np.max(np.abs(a1["psi"]-a2["psi"]))); out["obj_raw_psi_shift"] = float(np.median(b2["psi"]-b1["psi"]))
# Galerkin Hodge on displacement: raw vs boosted vs aligned
def hodge(P0_, P1_):
    Pm = 0.5*(P0_+P1_); cx = X.complex_peeled(Pm); H = X.galerkin_hodge(Pm, cx, P1_-P0_); return [H["Eg"], H["Ec"], H["Eh"]]
out["hodge_raw"] = hodge(fr[s], fr[s+1]); out["hodge_moving_frame"] = hodge(frT[s], frT[s+1])
out["hodge_aligned"] = hodge(fr[s], pullback(fr[s], fr[s+1])[0]); out["hodge_aligned_moving"] = hodge(frT[s], pullback(frT[s], frT[s+1])[0])
kr = []
for s in range(1, 40, 6):
    tri = X.complex_peeled(fr[s])["tri"]; d, dg, Gp, Gm = kelvin_pos(fr[s-1], fr[s], fr[s+1], tri)
    kr.append(float(np.max(np.abs(d-dg))/np.max(np.abs(Gp))))
out["kelvin_res_max"] = max(kr)
A = run_afrc(fr); bex, lns = R_exact(Fs)
out["afrc"] = summary(A, bex, lns)
# geometric truth: initial distance to nearest vortex centre; ring region 1.2a<r<3a -> shear (differential rotation)
r0 = np.minimum(np.hypot(*(P0-cen(0)[0]).T), np.hypot(*(P0-cen(0)[1]).T))
cand = A["cand"]; lab = A["lab"]
for nm, m in [("ring_r1.2-2.5", (r0 > 1.2) & (r0 < 2.5)), ("outer_r>2.5", r0 >= 2.5), ("core_r<1.2", r0 <= 1.2)]:
    mm = cand & m
    out["afrc"]["region_"+nm] = dict(n=int(mm.sum()), **{k: float((lab[mm] == v).mean()) if mm.sum() else 0 for k, v in [("hyp", 1), ("shear", 2), ("inactive", 4)]})
out["time"] = time.time()-t0
json.dump(out, open("res_lo.json", "w"), indent=1); print(json.dumps(out, indent=1))
pickle.dump(dict(P0=fr[0], PT=fr[-1], B=A["net"]["B"], beta=A["ind"]["beta"], lab=lab, cand=cand, bex=bex, rs=rs[5], cen0=cen(0)), open("lo.pkl", "wb"))
