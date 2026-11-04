import numpy as np, json, pickle, cases, stretch_est as SE
from bcom import *
out = {}
c = cases.case_sod(h=0.004); sod = dict(fr=c["frames"], rho0=c["rho"][0], ref=c["rho"][-1], mask=c["interior"](c["frames"][0]), ev=(c["ev_pidx"], c["ev_step"]))
c = cases.case_blast(h=0.022); rb0 = np.where(c["rs"] <= 0.2, 6.0, 1.0)
blast = dict(fr=c["frames"], rho0=rb0, ref=rb0*np.exp(-c["lnJ_ex"][-1]), mask=c["interior"](c["frames"][0]), ev=(c["ev_pidx"], c["ev_step"]), rs=c["rs"], h=0.022)
D = pickle.load(open("sv_dns.pkl", "rb")); x = D["fr_in"][-1]
sv = dict(fr=D["fr_in"], rho0=D["Ev"]["rho"][0], ref=D["Ev"]["rho"][-1], mask=(x[:, 0] > x[:, 0].min()+.02) & (x[:, 0] < x[:, 0].max()-.02) & (x[:, 1] > x[:, 1].min()+.02) & (x[:, 1] < x[:, 1].max()-.02))
def det(cs, hot):
    if "ev" not in cs: return None
    pi, st = cs["ev"]; N, S1 = hot.shape; tr = np.zeros((N, S1), bool); tr[pi, st] = True
    tn = tr.copy(); tn[:, 1:] |= tr[:, :-1]; tn[:, :-1] |= tr[:, 1:]
    return [float(hot.any(1)[pi].mean()), float((hot & ~tn).sum()/max((~tn).sum(), 1))]
grid = [("default", {}), ("tau=0.1", dict(tau_min=0.1)), ("tau=0.3", dict(tau_min=0.3)), ("block=3", dict(block=3)), ("block=8", dict(block=8)),
        ("defer=2", dict(defer=2)), ("defer=8", dict(defer=8)), ("defer=0", dict(defer=0))]
for nm, kw in []:
    out[nm] = {}
    for cn, cs in [("sod", sod), ("blast", blast), ("sv", sv)]:
        lnJ, hot, bad, tau = ledger(cs["fr"], **kw)
        o = errstats(cs["rho0"]*np.exp(-lnJ[-1]), cs["ref"], cs["mask"]); o["det"] = det(cs, hot); out[nm][cn] = o
    print(nm, json.dumps(out[nm]), flush=True)
# T9 chained tensor on blast ring r0 = 0.5 (exact ln lambda = p ln 4)
fr = blast["fr"]; lnJ, hot, bad, tau = ledger(fr); rs = blast["rs"]; P0 = fr[0]
band = np.where(np.abs(rs-0.5) < 0.6*blast["h"])[0]; th = np.arctan2(P0[band, 1], P0[band, 0]); t0 = np.c_[-np.sin(th), np.cos(th)]
exact = (1/2.4)*np.log(4.0); out["T9_blast"] = {}
T9grid = [("default", {}), ("k=8", dict(k=8)), ("k=16", dict(k=16)), ("block=3", dict(block=3)), ("block=8", dict(block=8)), ("defer=2", dict(max_defer=2)), ("defer=8", dict(max_defer=8))]
for nm, kw in T9grid:
    F = SE.chained_F(fr, blast["h"], bad=bad, idx=band, keep={fr.shape[0]-1}, **kw)[fr.shape[0]-1][0]
    ll, A = SE.vector_lnlam(F, t0); out["T9_blast"][nm] = dict(mean=float(ll.mean()), med_abs_err=float(np.median(np.abs(ll-exact))))
print("T9", out["T9_blast"], flush=True)
json.dump(out, open("res_b3_t9.json", "w"), indent=1)
