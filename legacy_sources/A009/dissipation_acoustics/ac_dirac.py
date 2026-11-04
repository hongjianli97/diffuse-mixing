"""Acoustic (Dirac) channel from positions only.
node signal  rho' = rho0/J - rho_b  (ledger, rho0 = reference density at t=0)
edge signal  u = velocity 1-cochain (central differences of positions)
(1) sound speed:  acceleration cochain = -c^2 d0(ln rho); the ledger gradient is the noisy variable, so c^2 = -<a,a>/<a,g>
(2) Kovasznay:    curl share of u (should vanish for acoustics)
(3) Dirac forward model on the t=0 complex:  d/dt (rho', u) = -( rho_b delta u ,  (c^2/rho_b) d0 rho' ),
    delta u = -(B1 W u)/A*  ;  D^2 = c^2 diag(L0, L1_down)  ->  omega^2 = c^2 k^2.  Compared with the ledger and the DNS."""
import numpy as np, json, sys
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import lhdx as X, netmix as NM
z = np.load('ac.npz'); fr = z['frames']; S, N, _ = fr.shape; dt = float(z['dtf']); t = z['t']
P0 = fr[0]; tri = X.complex_peeled(P0)['tri']
a = 0.5 * ((P0[tri[:, 1], 0] - P0[tri[:, 0], 0]) * (P0[tri[:, 2], 1] - P0[tri[:, 0], 1]) - (P0[tri[:, 1], 1] - P0[tri[:, 0], 1]) * (P0[tri[:, 2], 0] - P0[tri[:, 0], 0]))
tri[a < 0] = tri[a < 0][:, [0, 2, 1]]; E, w = NM.edge_weights(P0, tri); nE = len(E)
B1 = sp.csr_matrix((np.r_[-np.ones(nE), np.ones(nE)], (np.r_[E[:, 0], E[:, 1]], np.r_[np.arange(nE), np.arange(nE)])), shape=(N, nE))
kk = E[:, 0] * N + E[:, 1]; order = np.argsort(kk); ks = kk[order]; rows, cols, vals = [], [], []
for (p_, q_) in ((0, 1), (1, 2), (2, 0)):
    i, j = tri[:, p_], tri[:, q_]; k = np.minimum(i, j) * N + np.maximum(i, j)
    rows.append(order[np.searchsorted(ks, k)]); cols.append(np.arange(len(tri))); vals.append(np.where(i < j, 1.0, -1.0))
B2 = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nE, len(tri)))
Ast0 = X._star(P0, tri, N) / 3.0
ntri = np.asarray(abs(B2).sum(1)).ravel(); bnode = np.zeros(N, bool); bnode[E[ntri == 1].ravel()] = True
rho0 = z['rho'][0]; rho_b = float(np.median(rho0))
lnJ = np.array([np.log(X._star(fr[s], tri, N) / X._star(P0, tri, N)) for s in range(S)])
rho_led = rho0[None] * np.exp(-lnJ); rp_led = rho_led - rho_b; rp_dns = z['rho'] - rho_b
def coch(P, V): return 0.5 * ((V[E[:, 0]] + V[E[:, 1]]) * (P[E[:, 1]] - P[E[:, 0]])).sum(1)
res = dict(frames={})
num = den = 0.0
Wi = sp.diags(1 / np.maximum(w, 0.02 * np.median(w)))
L0 = (B1 @ sp.diags(w) @ B1.T).tocsc() + sp.diags(1e-12 * np.ones(N))
for s in range(2, S - 1):
    V = (fr[s + 1] - fr[s - 1]) / (2 * dt); A_ = (fr[s + 1] - 2 * fr[s] + fr[s - 1]) / dt ** 2
    ae = coch(fr[s], A_); ue = coch(fr[s], V); g = B1.T @ np.log(rho_led[s])
    c2s = -float((w * ae * ae).sum() / (w * ae * g).sum()); num += (w * ae * ae).sum(); den += -(w * ae * g).sum()   # g = -a/c^2 + noise(ledger): regress g on a
    chi = spsolve((B2.T @ Wi @ B2).tocsc(), B2.T @ ue); uc = Wi @ (B2 @ chi)
    r = dict(t=float(t[s]), c2=c2s, curl_share_u=float((w * uc * uc).sum() / (w * ue * ue).sum()),
             led_vs_dns_relrms=float(np.sqrt(((rp_led[s] - rp_dns[s])[~bnode] ** 2).mean()) / np.abs(rp_dns[s][~bnode]).max()))
    res['frames'][s] = r
c2 = num / den; res['c2_all'] = c2; print('c^2 from positions:', c2)
# Dirac forward model on the t = 0 complex
def rhs(y):
    rp, u = y[:N], y[N:]
    th = -(B1 @ (w * u)) / Ast0
    return np.r_[-rho_b * th, -(c2 / rho_b) * (B1.T @ rp)]
y = np.r_[rp_led[0], np.zeros(nE)]; h = 0.0001; nsub = int(round(dt / h)); pred = [y[:N].copy()]; energy = []
def en(y): rp, u = y[:N], y[N:]; return float(0.5 * (c2 / rho_b) * (rp ** 2 * Ast0).sum() + 0.5 * rho_b * (w * u * u).sum())
energy.append(en(y))
for s in range(1, S):
    for _ in range(nsub):
        k1 = rhs(y); k2 = rhs(y + 0.5 * h * k1); k3 = rhs(y + 0.5 * h * k2); k4 = rhs(y + h * k3); y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    pred.append(y[:N].copy()); energy.append(en(y))
pred = np.array(pred)
# acoustic energy measured from positions (ledger rho', velocity cochain)
en_meas = []
for s in range(1, S - 1):
    V = (fr[s + 1] - fr[s - 1]) / (2 * dt); ue = coch(fr[s], V)
    en_meas.append(float(0.5 * (c2 / rho_b) * (rp_led[s] ** 2 * Ast0).sum() + 0.5 * rho_b * (w * ue * ue).sum()))
res['energy_dirac'] = energy; res['energy_meas'] = en_meas
r_ = np.hypot(P0[:, 0] - 0.5, P0[:, 1] - 0.5); bins = np.linspace(0, 0.35, 71); bi = np.digitize(r_, bins) - 1
prof = {}
for s in (20, 40, 60):
    for tag, F in (('dns', rp_dns[s]), ('led', rp_led[s]), ('dirac', pred[s])):
        prof['%s_%d' % (tag, s)] = [float(F[(bi == k) & ~bnode].mean()) if ((bi == k) & ~bnode).any() else np.nan for k in range(70)]
    m = ~bnode & (r_ < 0.33)
    res['frames'].setdefault(s, {})['dirac_vs_dns_relrms'] = float(np.sqrt(((pred[s] - rp_dns[s])[m] ** 2).mean()) / np.abs(rp_dns[s][m]).max())
    res['frames'][s]['led_vs_dns_relrms_r033'] = float(np.sqrt(((rp_led[s] - rp_dns[s])[m] ** 2).mean()) / np.abs(rp_dns[s][m]).max())
    res['frames'][s]['dirac_vs_led_relrms'] = float(np.sqrt(((pred[s] - rp_led[s])[m] ** 2).mean()) / np.abs(rp_led[s][m]).max())
res['prof'] = prof; res['rbins'] = (0.5 * (bins[1:] + bins[:-1])).tolist()
json.dump(res, open('res_ac_dirac.json', 'w'), indent=1, default=float)
np.savez_compressed('maps_ac.npz', P0=P0, tri=tri, rp_dns40=rp_dns[40], rp_led40=rp_led[40], rp_dirac40=pred[40])
for s in (10, 20, 30, 40, 50, 59):
    print(s, {k: '%.4g' % v for k, v in res['frames'][s].items()})
for s in (20, 40, 60): print(s, {k: '%.4g' % v for k, v in res['frames'][s].items() if 'dirac' in k or 'r033' in k})
print('energy dirac', energy[0], energy[-1], ' measured', en_meas[5], en_meas[-1])
