"""divergence as the material area rate of each node's star on fixed triangles (Reynolds transport / ledger form):
theta_i = [ln A*_i(t+dt) - ln A*_i(t-dt)] / 2dt ; compare with the cochain-based Galerkin divergence and DNS."""
import numpy as np, json
from scipy.stats import spearmanr
from matplotlib.tri import Triangulation
src = open('hodge_dyn.py').read(); exec(src[:src.index('res = dict(series')])
grid_s = {round(float(k) / dt): k for k in o['grids'].keys()}; out = {}
prev = json.load(open('res_hd_check2.json'))
for s in (15, 30, 50):
    P, tri, A, E, w, B1, B2, Ast = ops(s)
    Am, Ap = X._star(fr[s - 1], tri, N), X._star(fr[s + 1], tri, N)
    with np.errstate(all='ignore'): thL = np.where((Am > 0) & (Ap > 0), (np.log(Ap) - np.log(Am)) / (2 * dt), 0.0)
    ntri = np.asarray(abs(B2).sum(1)).ravel(); bE = E[ntri == 1]; bnode = np.zeros(N, bool); bnode[bE.ravel()] = True
    org = np.arange(N) < N0; hs = np.r_[hot[s], hot[s]]; inner = org & ~bnode & ~hs & (Ast > 0)
    th_d = np.r_[E_dns['th'][s], E_dns['th'][s]]
    # same for triangles: area rate of each material triangle
    def tarea(Q): return 0.5 * np.abs((Q[tri[:, 1], 0] - Q[tri[:, 0], 0]) * (Q[tri[:, 2], 1] - Q[tri[:, 0], 1]) - (Q[tri[:, 1], 1] - Q[tri[:, 0], 1]) * (Q[tri[:, 2], 0] - Q[tri[:, 0], 0]))
    thT = (np.log(tarea(fr[s + 1])) - np.log(tarea(fr[s - 1]))) / (2 * dt)
    orgT = (tri < N0).all(1); hotT = hs[tri].any(1)
    r = dict(spearman_ledger=float(spearmanr(thL[inner], th_d[inner])[0]), pearson_ledger=float(np.corrcoef(thL[inner], th_d[inner])[0, 1]),
             dil2_ledger_tri=float((thT[orgT] ** 2 * A[orgT]).sum()), dil2_ledger_tri_offshock=float((thT[orgT & ~hotT] ** 2 * A[orgT & ~hotT]).sum()),
             dil2_dns=prev[str(s)]['dil2_dns'], dil2_dns_offshock=prev[str(s)]['dil2_dns_offshock'],
             enstrophy_net=prev[str(s)]['enstrophy_net'], enstrophy_dns=prev[str(s)]['enstrophy_dns'])
    out[s] = r; print(s, {k: round(v, 4) for k, v in r.items()}, flush=True)
json.dump(out, open('res_hd_check3.json', 'w'), indent=1)
