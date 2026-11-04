"""extra checks: (i) node divergence vs DNS dilatation (interior nodes, Spearman); (ii) Hodge Dirichlet forms of the
velocity cochain vs DNS integrals of omega^2 and theta^2 over the same region (dissipation identity
Phi = mu*int(omega^2 + 4/3 theta^2) - 4 mu*int det(grad u),  the last term is a boundary term)."""
import numpy as np, json
from scipy.stats import spearmanr
from matplotlib.tri import Triangulation
src = open('hodge_dyn.py').read(); exec(src[:src.index('res = dict(series')])
out = {}; grid_s = {round(float(k) / dt): k for k in o['grids'].keys()}
for s in (15, 30, 50):
    P, tri, A, E, w, B1, B2, Ast = ops(s)
    V = (fr[s + 1] - fr[s - 1]) / (2 * dt); ue = cochain(P, E, V)
    theta = -(B1 @ (w * ue)) / np.maximum(Ast, 1e-30); omT = (B2.T @ ue) / A
    ntri = np.asarray(abs(B2).sum(1)).ravel(); bE = E[ntri == 1]; bnode = np.zeros(N, bool); bnode[bE.ravel()] = True
    org = np.arange(N) < N0; hs = np.r_[hot[s], hot[s]]; inner = org & ~bnode & ~hs & (Ast > 0)
    th_d = np.r_[E_dns['th'][s], E_dns['th'][s]]
    r = dict(spearman_theta=float(spearmanr(theta[inner], th_d[inner])[0]), pearson_theta=float(np.corrcoef(theta[inner], th_d[inner])[0, 1]),
             n_inner=int(inner.sum()), n_bnode=int((bnode & org).sum()))
    # Dirichlet forms on the original half (triangles fully original; interior nodes)
    orgT = (tri < N0).all(1)
    r['enstrophy_net'] = float((omT[orgT] ** 2 * A[orgT]).sum()); r['dil2_net'] = float((theta[org & ~bnode] ** 2 * Ast[org & ~bnode]).sum())
    r['dil2_net_offshock'] = float((theta[inner] ** 2 * Ast[inner]).sum())
    g = o['grids'][grid_s[s]]; dx_ = float(o['dx'])
    Xc, Yc = np.meshgrid(o['xc'], o['yc']); trg = Triangulation(P[:, 0], P[:, 1], tri[orgT]); fnd = trg.get_trifinder()
    inside = fnd(Xc, Yc) >= 0
    r['enstrophy_dns'] = float((g['om'][inside].astype(float) ** 2).sum() * dx_ ** 2); r['dil2_dns'] = float((g['th'][inside].astype(float) ** 2).sum() * dx_ ** 2)
    # DNS off-shock: exclude cells with |theta| above the 99th percentile band of the shock (approx: |th| > 5)
    thg = g['th'].astype(float); off = inside & (np.abs(thg) < 5.0)
    r['dil2_dns_offshock'] = float((thg[off] ** 2).sum() * dx_ ** 2); r['area_net'] = float(A[orgT].sum()); r['area_dns'] = float(inside.sum() * dx_ ** 2)
    out[s] = r; print(s, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
json.dump(out, open('res_hd_check2.json', 'w'), indent=1)
