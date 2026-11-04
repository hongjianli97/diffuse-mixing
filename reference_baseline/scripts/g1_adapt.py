import numpy as np, json, pickle, cases, sys
from bcom import *
out = {}
def load(nm):
    if nm == "sod":
        c = cases.case_sod(h=0.004); fr = c["frames"]; return fr, c["rho"][0], c["rho"][-1], c["interior"](fr[0])
    if nm == "blast":
        c = cases.case_blast(h=0.022); fr = c["frames"]; r0 = np.where(c["rs"] <= 0.2, 6.0, 1.0); return fr, r0, r0*np.exp(-c["lnJ_ex"][-1]), c["interior"](fr[0])
    if nm in ("sv", "sb", "rm"):
        if nm == "sv": D = pickle.load(open("sv_dns.pkl", "rb")); fr = D["fr_in"]; rho = D["Ev"]["rho"]
        else: D = pickle.load(open({"sb": "sb_dns_dx0.005.pkl", "rm": "rm_dns_dx0.005.pkl"}[nm], "rb")); fr = D["frames"]; rho = D["E"]["rho"]
        x = fr[-1]; m = (x[:, 0] > x[:, 0].min()+.02) & (x[:, 0] < x[:, 0].max()-.02) & (x[:, 1] < x[:, 1].max()-.02)
        if nm == "sv": m &= x[:, 1] > x[:, 1].min()+.02
        return fr, rho[0], rho[-1], m
    if nm in ("dg", "lo"):
        c = cases.case_dg(nx=160, ny=80, T=10.0, S=51) if nm == "dg" else cases.case_lo(a=1.0, Rd=6.5, N=8000)
        fr = c["frames"]; N = fr.shape[1]; tri = X.complex_peeled(fr[-1])["tri"]
        e = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1); u, cnt = np.unique(e, axis=0, return_counts=True)
        bnd = np.zeros(N, bool); bnd[u[cnt == 1].ravel()] = True; _, nb = cKDTree(fr[-1]).query(fr[-1], k=8)
        return fr, np.ones(N), np.ones(N), ~bnd[nb].any(1)
for nm in sys.argv[1:]:
    fr, rho0, ref, mask = load(nm); hot, R, tau = C.hot_steps(fr)
    S, N, _ = fr.shape; bad = np.zeros((N, S), bool); bad[:, :-1] |= hot; bad[:, 1:] |= hot
    o = {"fixed5": errstats(rho0*np.exp(-X.chained_lnJ_v2(fr, bad)[-1]), ref, mask)}
    for eps in [0.01, 0.02, 0.04]:
        lnJ, nre = chained_lnJ_adapt(fr, bad, eps=eps, max_len=5, max_defer=15)
        o["adapt_eps%g" % eps] = dict(errstats(rho0*np.exp(-lnJ[-1]), ref, mask), reanchors_per_node=nre)
    out[nm] = o; print(nm, json.dumps({k: (round(v["med"], 5), round(v["p95"], 4)) for k, v in o.items()}), flush=True)
    json.dump(out, open("res_g1_adapt_%s.json" % "_".join(sys.argv[1:]), "w"), indent=1)
