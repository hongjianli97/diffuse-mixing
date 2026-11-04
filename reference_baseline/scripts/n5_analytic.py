import numpy as np, json
from scipy.stats import rankdata
def Fss(lam, gam, t):
    if lam == 0: return np.array([[1, gam*t], [0, 1.]])
    return np.array([[np.exp(lam*t), gam/lam*np.sinh(lam*t)], [0, np.exp(-lam*t)]])
def indices(lam, gam, T, nstep=40, ndir=24, rot=0.0):
    th = np.linspace(0, np.pi, ndir, endpoint=False)+rot
    r0 = np.c_[np.cos(th), np.sin(th)]
    ts = np.linspace(0, T, nstep+1)
    ell = np.array([np.log(np.linalg.norm(r0@Fss(lam, gam, t).T, axis=1)) for t in ts]).T
    h = nstep//2; a = ell[:, h]-ell[:, 0]; b = ell[:, -1]-ell[:, h]
    e = np.argmax(ell[:, -1]-ell[:, 0])
    R = b[e]/a[e]
    Z = np.corrcoef(rankdata(a), rankdata(b))[0, 1]
    F = Fss(lam, gam, T); sv = np.linalg.svd(F, compute_uv=False)
    return dict(R=float(R), Z=float(Z), ftle=float(np.log(sv[0])/T))
out = {}
# pure saddle and pure shear at equal FTLE
T = 1.0
for lamT in [1, 2, 3, 4, 5]:
    sig = np.exp(lamT); gT = sig - 1/sig       # same sigma_max: g T = sigma - 1/sigma
    s = indices(lamT, 0, 1); h_ = indices(0, gT, 1)
    out["saddle_vs_shear_lamT%d" % lamT] = dict(saddle=s, shear=h_, gT=float(gT), R_shear_formula=float(np.log(2)/np.log(gT/2)))
# superposition: lam T = 3, vary gamma/lam
sc = []
for r in [0, 0.5, 1, 2, 5, 10, 20, 50, 100]:
    d = indices(1.0, r, 3.0); d["g_over_l"] = r; sc.append(d)
out["superposition_lamT3"] = sc
# shear scan
out["shear_scan"] = [dict(gT=g, **indices(0, g, 1.0), formula=float(np.log(2)/np.log(g/2))) for g in [5, 10, 20, 50, 100, 300]]
# rotation-invariance of star indices (random star orientation)
out["rot_check"] = [indices(1.0, 2.0, 3.0, rot=r)["Z"] for r in [0, 0.05, 0.1]]
json.dump(out, open("res_n5.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
