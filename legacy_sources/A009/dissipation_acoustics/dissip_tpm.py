"""Dissipation from three-point maps (LHD): per material triangle F = D(t+dt) D(t-dt)^-1, grad u = log F / 2dt (exact
2x2 matrix log), Phi/mu = 2 S:S - 2/3 theta^2 = omega^2 + 4/3 theta^2 - 4 det(grad u).  Positions only (+ mu)."""
import numpy as np, json, sys
sys.argv = [sys.argv[0], 'visc']
src = open('visc_hodge.py').read(); exec(src[:src.index('res = {}; maps = {}')])
def logm2(F):
    a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
    tr = a + d; det = a * d - b * c; disc = np.sqrt((tr * tr / 4 - det).astype(complex))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    same = np.abs(l1 - l2) < 1e-12 * np.abs(l1)
    beta = np.where(same, 1 / l1, (np.log(l1) - np.log(l2)) / np.where(same, 1, l1 - l2))
    alpha = np.log(l1) - beta * l1
    L = beta[:, None, None] * F + alpha[:, None, None] * np.eye(2)[None]
    return L.real
out = {}
for s in sorted(set(list(range(3, S - 1, 4)) + [50, 69])):
    P, tri, A, E, w, B1, B2, Ast = ops(s)
    def Dm(Q): return np.stack([Q[tri[:, 1]] - Q[tri[:, 0]], Q[tri[:, 2]] - Q[tri[:, 0]]], axis=2)
    F = Dm(fr[s + 1]) @ np.linalg.inv(Dm(fr[s - 1])); L = logm2(F) / (2 * dt)
    th = L[:, 0, 0] + L[:, 1, 1]; om = L[:, 1, 0] - L[:, 0, 1]; S_ = 0.5 * (L + L.transpose(0, 2, 1))
    SS = (S_ ** 2).sum((1, 2)); dt_ = L[:, 0, 0] * L[:, 1, 1] - L[:, 0, 1] * L[:, 1, 0]
    phi = 2 * SS - 2 / 3 * th ** 2
    hs = np.r_[hot[s], hot[s]]; hotT = hs[tri].any(1)
    ntri = np.asarray(abs(B2).sum(1)).ravel(); bT = np.asarray(abs(B2[ntri == 1]).sum(0)).ravel() > 0; okT = ~hotT & ~bT
    Gam = B2.T @ cochain(P, E, (fr[s + 1] - fr[s - 1]) / (2 * dt))
    Phi_dns = np.r_[Ed['Phi'][s], Ed['Phi'][s]]
    bnode = np.zeros(N, bool); bnode[E[ntri == 1].ravel()] = True; smooth = ~hs & ~bnode
    r = dict(t=float(t[s]), Phi_net_smooth=float(MU_TRUE * (phi * A)[okT].sum()), Phi_dns_smooth=float((Phi_dns * Ast)[smooth].sum()),
             enst_tpm=float((om ** 2 * A)[okT].sum()), enst_circ=float((Gam ** 2 / A)[okT].sum()),
             dil_term=float((4 / 3 * th ** 2 * A)[okT].sum()), det_term=float((-4 * dt_ * A)[okT].sum()),
             Phi_net_all=float(MU_TRUE * (phi * A).sum()), Phi_dns_all=float((Phi_dns * Ast).sum()))
    r['ratio_smooth'] = r['Phi_net_smooth'] / r['Phi_dns_smooth']
    out[s] = r; print(s, {k: '%.4g' % v for k, v in r.items()}, flush=True)
json.dump(out, open('res_dissip_tpm.json', 'w'), indent=1)
