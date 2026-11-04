import numpy as np, json, pickle, time
from common import *
from n4_shock_lib import block_signature
D = pickle.load(open("sv_dns.pkl", "rb")); fr = D["fr_in"]; t = D["t"]; Ev = D["Ev"]; S, N, _ = fr.shape
out = {"N": N, "S": S}
hot, Rlog, tau = C.hot_steps(fr); out["tau"] = tau
rho = Ev["rho"].astype(float)
jump = np.abs(np.log(rho[1:]/rho[:-1])).T > np.log(1.15)
out["detect_rate"] = float(hot.any(1)[jump.any(1)].mean())
tn = jump.copy(); tn[:, 1:] |= jump[:, :-1]; tn[:, :-1] |= jump[:, 1:]
out["false_step_rate"] = float((hot & ~tn).sum()/max((~tn).sum(), 1))
bad = np.zeros((N, S), bool); bad[:, :-1] |= hot; bad[:, 1:] |= hot
lnJ = X.chained_lnJ_v2(fr, bad)
ex = np.log(rho[0]/rho[-1]); err = np.abs(np.exp(-(lnJ[-1]-ex))-1); out["rho_err_med"] = float(np.median(err)); out["rho_err_p95"] = float(np.quantile(err, .95))
# step readouts vs DNS
oc, ocs, tc_ = [], [], []
for s in range(0, S-1, 3):
    P0, P1 = fr[s], fr[s+1]; tri = X.complex_peeled(P0)["tri"]; r = step_readouts(P0, P1, tri); dt = t[s+1]-t[s]
    Am = tri_area(0.5*(P0+P1), tri); om_hat = 2*r["A0"]*r["p"].imag/(Am*dt)
    om_ref = 0.25*(Ev["om"][s]+Ev["om"][s+1])[tri].sum(1)*2/3
    th_hat = np.log(np.abs(r["J"]))/dt; th_ref = 0.5*(Ev["th"][s]+Ev["th"][s+1])[tri].mean(1)
    hk = (hot[:, s][tri]).any(1)
    oc.append(corr(om_hat, om_ref)); ocs.append(corr(om_hat[~hk], om_ref[~hk])); tc_.append(corr(th_hat, th_ref))
    if s == 12: snap = dict(cen=0.5*(P0+P1)[tri].mean(1), om_hat=om_hat, om_ref=om_ref, hk=hk)
out["om_corr_all"] = oc; out["om_corr_nonhot"] = ocs; out["th_corr"] = tc_
# Kelvin in position form on a material region: vortex core r0 < 2Rv, fixed material triangles of frame 0
P0 = fr[0]; c0 = np.array([D["xv"], D["yv"]]); r0 = np.hypot(*(P0-c0).T)
tri0 = X.complex_peeled(P0)["tri"]; inR = (r0[tri0] < 2*D["Rv"]).all(1)
triR = tri0[inR]
Gpos = []; Gdns = []
for s in range(S-1):
    circ, _ = disp_cochain_tri(fr[s], fr[s+1], triR); Gpos.append(circ.sum()/(t[s+1]-t[s]))
    Pm = 0.5*(fr[s]+fr[s+1]); om = 0.5*(Ev["om"][s]+Ev["om"][s+1]); Gdns.append(float((om[triR].mean(1)*tri_area(Pm, triR)).sum()))
Gpos = np.array(Gpos); Gdns = np.array(Gdns)
out["kelvin_region_relerr_med"] = float(np.median(np.abs(Gpos-Gdns)/np.abs(Gdns)))
out["kelvin_region_final"] = [float(Gpos[-1]), float(Gdns[-1])]
# impulse split: changes over steps with any hot vertex in region vs smooth
regnodes = np.unique(triR); hs = hot[regnodes].any(0)
dG = np.diff(Gpos); hstep = hs[1:] | hs[:-1]
out["kelvin_dG_total"] = float(Gpos[-1]-Gpos[0]); out["kelvin_dG_shocksteps"] = float(dG[hstep].sum()); out["kelvin_dG_smooth"] = float(dG[~hstep].sum())
# position-Kelvin identity on DNS data
kr = []
for s in range(1, S-1, 4):
    tri = X.complex_peeled(fr[s])["tri"]; d, dg, Gp, Gm = kelvin_pos(fr[s-1], fr[s], fr[s+1], tri); kr.append(float(np.max(np.abs(d-dg))/np.max(np.abs(Gp))))
out["kelvin_res_max"] = max(kr)
# block signatures
sig, pst, pbl = block_signature(fr)
unif = np.tanh(0.5*np.abs(sig[:, 0])); out["n_blocks"] = len(sig)
dv = np.abs(sig[:, 1]-unif); okb = np.isfinite(dv); out["sig_dev_med"] = float(np.median(dv[okb])); out["sig_dev_p90"] = float(np.quantile(dv[okb], .9)); out["blocks_flipped"] = int((~okb).sum()); out["blockrot_med_p95"] = [float(np.median(np.abs(sig[okb, 2]))), float(np.quantile(np.abs(sig[okb, 2]), .95))]
out["pseudo_rot_step_med_p95"] = [float(np.median(pst)), float(np.quantile(pst, .95))]
out["pseudo_rot_block_med_p95"] = [float(np.median(pbl)), float(np.quantile(pbl, .95))]
# Hodge: raw vs Kabsch-aligned displacement
s = 20; out["hodge_raw"] = None
def hodge(P0_, P1_):
    Pm = 0.5*(P0_+P1_); cx = X.complex_peeled(Pm); H = X.galerkin_hodge(Pm, cx, P1_-P0_); return [H["Eg"], H["Ec"], H["Eh"]]
def hodge_meanfree(P0_, P1_):
    D_ = P1_-P0_; return hodge(P0_, P0_+D_-D_.mean(0))
out["hodge_raw"] = hodge(fr[s], fr[s+1]); out["hodge_meanfree"] = hodge_meanfree(fr[s], fr[s+1]); out["hodge_aligned"] = hodge(fr[s], pullback(fr[s], fr[s+1])[0])
P1b = fr[s+1]+np.array([0.01, 0.0]); out["hodge_raw_boost"] = hodge(fr[s], P1b); out["hodge_aligned_boost"] = hodge(fr[s], pullback(fr[s], P1b)[0])
# classification
A = run_afrc(fr, lnJ=lnJ, hot=hot); A["tau"] = tau; out["afrc"] = summary(A)
T = t[-1]-t[0]; Bf = X.ls_ftle(fr[-1], fr[0], T); F = classify_field(fr, Bf, lnJ, hot, reverse=True); out["ftle_bwd"] = summary(F)
Bf2 = X.ls_ftle(fr[0], fr[-1], T); F2 = classify_field(fr, Bf2, lnJ, hot, reverse=False); out["ftle_fwd"] = summary(F2)
json.dump(out, open("res_sv.json", "w"), indent=1); print(json.dumps(out, indent=1))
pickle.dump(dict(P0=fr[0], PT=fr[-1], B=A["net"]["B"], lab=A["lab"], cand=A["cand"], Bf=Bf, labf=F["lab"], candf=F["cand"],
                 Gpos=Gpos, Gdns=Gdns, t=t, sig=sig, snap=snap, lnJ=lnJ[-1], ex=ex, hstep=hstep), open("sv.pkl", "wb"))
