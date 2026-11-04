"""noise propagation law of the ledger: static jittered lattice + iid position noise sigma"""
import numpy as np, json
from bcom import *
rng = np.random.default_rng(3); n = 60
X_, Y_ = np.meshgrid(np.arange(n)+.5, np.arange(n)+.5); P = np.c_[X_.ravel(), Y_.ravel()]+rng.uniform(-.25, .25, (n*n, 2)); h = 1.0
N = len(P); inner = (np.abs(P-n/2) < n/2-4).all(1)
tri = X.complex_peeled(P)["tri"]
# predicted std of d ln A* from the link polygon: dA*/dx_k = 1/2 R(x_{k+1}-x_{k-1})  ->  var = sigma^2 sum_k |x_{k+1}-x_{k-1}|^2 / 4
pred = np.zeros(N); As = X._star(P, tri, N)
for i in np.where(inner)[0][:400]:
    tt = tri[(tri == i).any(1)]; nb = np.unique(tt[tt != i]); ang = np.arctan2(*(P[nb]-P[i]).T[::-1]); nb = nb[np.argsort(ang)]
    q = P[np.roll(nb, -1)]-P[np.roll(nb, 1)]; pred[i] = np.sqrt((q**2).sum()/4)/As[i]
c_pred = float(np.median(pred[pred > 0]))
out = dict(c_pred_std_per_frame=c_pred, c_pred_med_lnJ=float(0.6745*np.sqrt(2)*c_pred))
for S in [11, 41]:
    for sig in [0.02, 0.05]:
        fr = np.repeat(P[None], S, 0)+sig*h*rng.normal(size=(S, N, 2))
        lnJ = X.chained_lnJ_v2(fr, None)
        out["S%d_sig%g" % (S, sig)] = dict(med_over_sig=float(np.median(np.abs(lnJ[-1][inner]))/sig), std_over_sig=float(np.std(lnJ[-1][inner])/sig))
print(json.dumps(out, indent=0)); json.dump(out, open("res_g4_noise.json", "w"), indent=1)
