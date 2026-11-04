import numpy as np, json, pickle, cases
from scipy.signal import savgol_filter
from bcom import *
out = {}; rng = np.random.default_rng(5)
c = cases.case_sod(h=0.004); S1 = dict(fr=c["frames"], rho0=c["rho"][0], ref=c["rho"][-1], mask=c["interior"](c["frames"][0]))
D = pickle.load(open("sv_dns.pkl", "rb")); x = D["fr_in"][-1]
S2 = dict(fr=D["fr_in"], rho0=D["Ev"]["rho"][0], ref=D["Ev"]["rho"][-1], mask=(x[:, 0] > x[:, 0].min()+.02) & (x[:, 0] < x[:, 0].max()-.02) & (x[:, 1] > x[:, 1].min()+.02) & (x[:, 1] < x[:, 1].max()-.02))
for nm, cs in [("sod", S1), ("sv", S2)]:
    fr = cs["fr"]; d, _ = cKDTree(fr[0]).query(fr[0], k=2); h = np.median(d[:, 1]); out[nm] = {}
    for nz in [0.02, 0.05]:
        frn = fr+nz*h*rng.normal(size=fr.shape)
        for win in [0, 5, 9]:
            frs = frn if win == 0 else savgol_filter(frn, win, 2, axis=0)
            lnJ = ledger(frs)[0]; out[nm]["noise%g_sg%d" % (nz, win)] = errstats(cs["rho0"]*np.exp(-lnJ[-1]), cs["ref"], cs["mask"])
        print(nm, nz, {k: v["med"] for k, v in out[nm].items()}, flush=True)
    for win in [5, 9]:
        lnJ = ledger(savgol_filter(fr, win, 2, axis=0))[0]; out[nm]["clean_sg%d" % win] = errstats(cs["rho0"]*np.exp(-lnJ[-1]), cs["ref"], cs["mask"])
json.dump(out, open("res_b4c.json", "w"), indent=1)
