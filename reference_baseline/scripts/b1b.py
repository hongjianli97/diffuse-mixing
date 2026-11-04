import numpy as np, json, pickle
from bcom import *
b = json.load(open("res_b1_density.json"))
D = pickle.load(open("sb_dns_dx0.005.pkl", "rb")); fr = D["frames"]; rho = D["E"]["rho"]; x = fr[-1]
mask = (x[:, 0] > x[:, 0].min()+.02) & (x[:, 0] < x[:, 0].max()-.02) & (x[:, 1] < x[:, 1].max()-.02)
lnJ, hot, bad, tau = ledger(fr); m, Dd = density_all(fr, rho[0], lnJ, mask)
phi = np.hypot(fr[0][:, 0]-D["xb"], fr[0][:, 1])-D["R"]
b["sb"] = {k: errstats(v, rho[-1], mask) for k, v in Dd.items()}
b["sb"]["band_away_interface"] = {k: errstats(v, rho[-1], mask & (np.abs(phi) > 0.03)) for k, v in Dd.items()}
D = pickle.load(open("rm_dns_dx0.005.pkl", "rb")); fr = D["frames"]; rho = D["E"]["rho"]; x = fr[-1]
mask = (x[:, 0] > x[:, 0].min()+.02) & (x[:, 0] < x[:, 0].max()-.02) & (x[:, 1] < 0.48)
lnJ, hot, bad, tau = ledger(fr); m, Dd = density_all(fr, rho[0], lnJ, mask)
b["rm"] = {k: errstats(v, rho[-1], mask) for k, v in Dd.items()}
json.dump(b, open("res_b1_density.json", "w"), indent=1); print(b["sb"], b["rm"])
