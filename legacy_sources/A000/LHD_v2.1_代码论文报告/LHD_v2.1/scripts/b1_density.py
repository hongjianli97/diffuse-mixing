import numpy as np, json, pickle, cases, sys
from bcom import *
out = {}
def run(name, fr, rho0, ref, mask, extra=None):
    lnJ, hot, bad, tau = ledger(fr)
    m, D = density_all(fr, rho0, lnJ, mask)
    o = {k: errstats(v, ref, mask) for k, v in D.items()}
    if extra is not None:
        for nm, mk in extra.items(): o["band_"+nm] = {k: errstats(v, ref, mask & mk) for k, v in D.items()}
    out[name] = o; print(name, json.dumps(o), flush=True)
    return D, lnJ
c = cases.case_sod(h=0.004); fr = c["frames"]
x1 = fr[-1][:, 0]; refS = c["rho"][-1]; inter = c["interior"](fr[0])
info = c["info"]; xs = [k for k in info if "x" in k.lower()]
near = np.zeros(len(x1), bool)
# discontinuity band: points whose exact density differs from neighbours' by a jump (shock/contact)
from scipy.spatial import cKDTree
d, nb = cKDTree(fr[-1]).query(fr[-1], k=8); jump = np.abs(np.log(refS[nb]/refS[:, None])).max(1) > 0.05
run("sod", fr, c["rho"][0], refS, inter, dict(disc=jump))
c = cases.case_blast(h=0.022); fr = c["frames"]
rho0 = np.where(c["rs"] <= 0.2, 6.0, 1.0); refB = rho0*np.exp(-c["lnJ_ex"][-1]); inter = c["interior"](fr[0])
d, nb = cKDTree(fr[-1]).query(fr[-1], k=8); jump = np.abs(np.log(refB[nb]/refB[:, None])).max(1) > 0.05
run("blast", fr, rho0, refB, inter, dict(disc=jump))
for nm, fn in [("dg", lambda: cases.case_dg(nx=160, ny=80, T=10.0, S=51)), ("lo", lambda: cases.case_lo(a=1.0, Rd=6.5, N=8000))]:
    c = fn(); fr = c["frames"]; N = fr.shape[1]
    bnd = np.zeros(N, bool); tri = X.complex_peeled(fr[-1])["tri"]
    e = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1); u, cnt = np.unique(e, axis=0, return_counts=True)
    bnd[u[cnt == 1].ravel()] = True
    _, nb = cKDTree(fr[-1]).query(fr[-1], k=8); inter = ~bnd[nb].any(1)
    run(nm, fr, np.ones(N), np.ones(N), inter)
D = pickle.load(open("sv_dns.pkl", "rb")); fr = D["fr_in"]; rho = D["Ev"]["rho"]
_, nb = cKDTree(fr[-1]).query(fr[-1], k=8)
x = fr[-1]; inter = (x[:, 0] > x[:, 0].min()+0.02) & (x[:, 0] < x[:, 0].max()-0.02) & (x[:, 1] > x[:, 1].min()+0.02) & (x[:, 1] < x[:, 1].max()-0.02)
run("sv", fr, rho[0], rho[-1], inter)
json.dump(out, open("res_b1_density.json", "w"), indent=1)
