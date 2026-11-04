import numpy as np, json, pickle
from common import *
out = {}; store = {}
rng = np.random.default_rng(4)
def grid(x0, x1, y0, y1, nx, ny):
    hx = (x1-x0)/nx; X_, Y_ = np.meshgrid(x0+(np.arange(nx)+.5)*hx, y0+(np.arange(ny)+.5)*hx)
    return np.c_[X_.ravel(), Y_.ravel()]+rng.uniform(-.25, .25, (nx*ny, 2))*hx, hx
# 1 parallel shear layer (Haller counterexample): u = (tanh(y/d), 0)
dl = 0.1
vel_sh = lambda Q, t: np.c_[np.tanh(Q[:, 1]/dl), 0*Q[:, 1]]
P, h = grid(-1, 1, -0.5, 0.5, 160, 80); t = np.linspace(0, 4, 41)
fr, Fs = integrate(vel_sh, P, t, sub=4)
Rex, lns = R_exact(Fs); A = run_afrc(fr)
out["shear_layer"] = summary(A, Rex, lns); store["shear_layer"] = dict(P0=P, B=A["net"]["B"], lab=A["lab"], cand=A["cand"], R=A["ind"]["beta"], Rex=Rex, lns=lns)
# 2 localized saddle: psi = lam x y exp(-(x^2+y^2)/(2L^2))
lam, Lw = 1.0, 0.6
def vel_sa(Q, t):
    x, y = Q[:, 0], Q[:, 1]; g = np.exp(-(x*x+y*y)/(2*Lw**2))
    return np.c_[lam*x*g*(1-y*y/Lw**2), -lam*y*g*(1-x*x/Lw**2)]
P, h = grid(-1, 1, -1, 1, 120, 120); t = np.linspace(0, 4, 41)
fr, Fs = integrate(vel_sa, P, t, sub=4)
Rex, lns = R_exact(Fs); A = run_afrc(fr)
out["saddle"] = summary(A, Rex, lns); store["saddle"] = dict(P0=P, B=A["net"]["B"], lab=A["lab"], cand=A["cand"], R=A["ind"]["beta"], Rex=Rex, lns=lns)
# 3 saddle + uniform shear: u = (lam x + g y, -lam y) * localized
for gr in [2.0, 10.0]:
    def vel_ss(Q, t, gr=gr):
        x, y = Q[:, 0], Q[:, 1]; g = np.exp(-(x*x+y*y)/(2*Lw**2))
        return np.c_[(lam*x*(1-y*y/Lw**2)+gr*np.tanh(y/0.15)*0.15)*g, -lam*y*g*(1-x*x/Lw**2)]
    fr, Fs = integrate(vel_ss, P, t, sub=4); Rex, lns = R_exact(Fs); A = run_afrc(fr)
    out["saddle_shear_%g" % gr] = summary(A, Rex, lns)
json.dump(out, open("res_bench.json", "w"), indent=1); print(json.dumps(out, indent=1))
pickle.dump(store, open("bench.pkl", "wb"))
