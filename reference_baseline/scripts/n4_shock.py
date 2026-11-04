import numpy as np, json, pickle, time, sys
import cases
from common import *

def block_signature(fr, tau=0.05, gap=2):
    """fixed material triangles of frame 0; merge hot steps (|ln J_k|>tau) into blocks; return block signatures
    and the per-step pseudo-rotation sum vs the block rotation"""
    tri = X.complex_peeled(fr[0])["tri"]; S = fr.shape[0]
    P, Qs, LJ = [], [], []
    for s in range(S-1):
        F, A0 = tri_F(fr[s], fr[s+1], tri); p, q = pq(F); P.append(p); Qs.append(q); LJ.append(np.log(np.abs(np.linalg.det(F))))
    P, Qs, LJ = np.array(P), np.array(Qs), np.array(LJ)
    hot = np.abs(LJ) > tau
    sig = []; ps_step = []; ps_block = []
    for k in range(tri.shape[0]):
        hs = np.where(hot[:, k])[0]
        if hs.size == 0: continue
        # group with gaps
        groups = [[hs[0]]]
        for s in hs[1:]:
            if s-groups[-1][-1] <= gap+1: groups[-1].append(s)
            else: groups.append([s])
        for g in groups:
            a, b = g[0], g[-1]
            pB, qB = 1+0j, 0j
            for s in range(a, b+1): pB, qB = compose(P[s, k], Qs[s, k], pB, qB)
            J = abs(pB)**2-abs(qB)**2
            sig.append((-np.log(J), abs(qB/pB), np.angle(pB)))
            ps_step.append(np.abs(np.angle(P[a:b+1, k])).sum()); ps_block.append(abs(np.angle(pB)))
    sig = np.array(sig)
    return sig, np.array(ps_step), np.array(ps_block)

which = sys.argv[1]
t0 = time.time(); out = {}
c = cases.case_sod(h=0.004) if which == "sod" else cases.case_blast(h=0.022)
fr = c["frames"]; S, N, _ = fr.shape
hot, Rlog, tau = C.hot_steps(fr); out["tau"] = tau
# detection vs exact crossings
truth = np.zeros((N, S-1), bool); truth[c["ev_pidx"], c["ev_step"]] = True
tn = np.zeros_like(truth); tn[:, 1:] |= truth[:, :-1]; tn[:, :-1] |= truth[:, 1:]; tn |= truth
det = hot.any(1); out["detect_rate"] = float(det[c["ev_pidx"]].mean())
out["false_step_rate"] = float((hot & ~tn).sum()/max((~tn).sum(), 1))
bad = np.zeros((N, S), bool); bad[:, :-1] |= hot; bad[:, 1:] |= hot
lnJ = X.chained_lnJ_v2(fr, bad)
inter = c["interior"](fr[0]) if which == "sod" else c["interior"](fr[0])
if which == "sod":
    ex = c["lnJ_ex"][-1]
else:
    ex = c["lnJ_ex"][-1]-c["lnJ_ex"][0]
err = np.abs(np.exp(-(lnJ[-1]-ex))-1)[inter]
out["rho_err_med"] = float(np.median(err)); out["rho_err_p95"] = float(np.quantile(err, .95))
# block signature
sig, pst, pbl = block_signature(fr)
unif = np.tanh(0.5*np.abs(sig[:, 0])); full = sig[:, 0] > 0.9*sig[:, 0].max()
out["n_blocks"] = int(len(sig)); out["sig_dev_med_all"] = float(np.median(np.abs(sig[:, 1]-unif)))
out["sig_dev_med_full"] = float(np.median(np.abs(sig[full, 1]-unif[full])))
out["pseudo_rot_step_med_p95"] = [float(np.median(pst)), float(np.quantile(pst, .95))]
out["pseudo_rot_block_med_p95"] = [float(np.median(pbl)), float(np.quantile(pbl, .95))]
# per-step readouts pseudo-vorticity at shock step
# Kelvin position form across the shock steps
kr = []; imp = []
for s in range(1, S-1, 3):
    tri = X.complex_peeled(fr[s])["tri"]; d, dg, Gp, Gm = kelvin_pos(fr[s-1], fr[s], fr[s+1], tri)
    kr.append(float(np.max(np.abs(d-dg))/max(np.max(np.abs(Gp)), 1e-30)))
out["kelvin_res_max"] = max(kr)
# AFRC_H gated and ungated; classification
for gate in [True, False]:
    A = run_afrc(fr, gate=gate, lnJ=lnJ, hot=hot); A["tau"] = tau
    o = summary(A)
    cand = A["cand"]; lab = A["lab"]
    crossed = np.zeros(N, bool); crossed[c["ev_pidx"]] = True
    o["cand_crossed_frac"] = float(crossed[cand].mean())
    o["shock_label_precision"] = float(crossed[cand & (lab == 3)].mean()) if (cand & (lab == 3)).any() else None
    o["crossed_cand_labelled_shock"] = float((lab[cand & crossed] == 3).mean()) if (cand & crossed).any() else None
    if which == "sod":
        fan = (c["x0"] < 0) & (c["x0"] > -c["info"]["cL"]*0.24)
        o["fan_cand_labels"] = {k: float((lab[cand & fan] == v).mean()) if (cand & fan).any() else None for k, v in [("hyp", 1), ("shear", 2), ("shock", 3), ("inactive", 4), ("wave", 5)]}
        o["fan_cand_n"] = int((cand & fan).sum())
    out["afrc_gate" if gate else "afrc_raw"] = o
    if not gate: keep = A
T = c["t"][-1]-c["t"][0]
crossed = np.zeros(N, bool); crossed[c["ev_pidx"]] = True
for nm, rev in [("ftle_bwd", True), ("ftle_fwd", False)]:
    Bf = X.ls_ftle(fr[-1], fr[0], T) if rev else X.ls_ftle(fr[0], fr[-1], T)
    A = classify_field(fr, Bf, lnJ, hot, reverse=rev); o = summary(A); cand, lab = A["cand"], A["lab"]
    o["cand_crossed_frac"] = float(crossed[cand].mean())
    o["shock_label_precision"] = float(crossed[cand & (lab == 3)].mean()) if (cand & (lab == 3)).any() else None
    o["crossed_cand_labelled_shock"] = float((lab[cand & crossed] == 3).mean()) if (cand & crossed).any() else None
    out[nm] = o
    if rev: keepF = dict(B=Bf, lab=lab, cand=cand)
out["time"] = time.time()-t0
json.dump(out, open("res_%s.json" % which, "w"), indent=1); print(json.dumps(out, indent=1))
pickle.dump(dict(P0=fr[0], PT=fr[-1], sig=sig, B=keep["net"]["B"], lab=keep["lab"], cand=keep["cand"], lnJ=lnJ[-1], ex=ex, inter=inter,
                 hotcount=hot.sum(1), pst=pst, pbl=pbl, ftle=keepF), open("%s.pkl" % which, "wb"))
