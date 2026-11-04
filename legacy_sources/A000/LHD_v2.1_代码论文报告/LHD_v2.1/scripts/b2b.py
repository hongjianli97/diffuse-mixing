import numpy as np, json, pickle, time
from bcom import *
exec(open("b2_diff.py").read().split("def three")[1].join(["def three", ""]).split("# Sod")[0]) if False else None
src = open("b2_diff.py").read(); exec("def three" + src.split("def three")[1].split("# Sod")[0])
b2 = json.load(open("res_b2_diff.json"))
D = pickle.load(open("sb_diff_dns.pkl", "rb")); fr = D["frames"]; S, N, _ = fr.shape; rho0 = D["rho"][0]
lnJ, hot, bad, tau = ledger(fr); rho = rho0[None, :]*np.exp(-lnJ); m = rho0*X.star_area(fr[0])/3.0
x = fr[-1]; inn = (x[:, 0] > x[:, 0].min()+.02) & (x[:, 0] < x[:, 0].max()-.02) & (x[:, 1] < x[:, 1].max()-.02)
b2["sb"] = three(fr, D["dt"], D["Y"][0], D["kappa"], m, rho, D["Y"][-1], inn); print(b2["sb"])
json.dump(b2, open("res_b2_diff.json", "w"), indent=1)
