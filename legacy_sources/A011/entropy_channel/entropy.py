"""entropy.py <inv|visc> -- the entropy channel of the trajectory complex (positions + density at t=0 only).

p_rec  : pressure = -(gradient potential of the rho*a cochain)  (+ one reference constant per frame: mean of a probe set)
rho    : material ledger rho0/J
T      : p/rho   (R = 1 in the solver units);  s = cv ln(p / rho^gamma),  cv = 1/(gamma-1)
Second law on the network (viscous, constant mu, k):
   rho T Ds/Dt = Phi + div(k grad T)
   Phi      from three-point maps:  mu (2 S:S - 2/3 theta^2),  grad u = log F / 2dt
   div(k grad T)_i = -k (L_cot T)_i / A*_i           (the same L0 as scalar diffusion)
   total production:  sum Phi/T dA  +  k * sum_e w_e (ln T_j - ln T_i)^2   (Dirichlet form of ln T)  >= 0
"""
import numpy as np, json, sys
case = sys.argv[1]; sys.argv = [sys.argv[0], case]
src = open('/home/claude/w/visc/visc_hodge.py').read(); exec(src[:src.index('res = {}; maps = {}')])
G = 1.4; CV = 1 / (G - 1); K = MU_TRUE * G / ((G - 1) * 0.72)


def logm2(F):
    a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
    tr = a + d; det = a * d - b * c; disc = np.sqrt((tr * tr / 4 - det).astype(complex))
    l1, l2 = tr / 2 + disc, tr / 2 - disc; same = np.abs(l1 - l2) < 1e-12 * np.abs(l1)
    beta = np.where(same, 1 / l1, (np.log(l1) - np.log(l2)) / np.where(same, 1, l1 - l2)); alpha = np.log(l1) - beta * l1
    return (beta[:, None, None] * F + alpha[:, None, None] * np.eye(2)[None]).real


pr = np.full((S, N0), np.nan); rl = np.full((S, N0), np.nan); phiN = np.zeros((S, N0)); condN = np.zeros((S, N0))
prodPhi = np.zeros(S); prodCond = np.zeros(S); probe = None
for s in range(1, S - 1):
    P, tri, A, E, w, B1, B2, Ast = ops(s)
    V = (fr[s + 1] - fr[s - 1]) / (2 * dt); Acc = (fr[s + 1] - 2 * fr[s] + fr[s - 1]) / dt ** 2
    rho = rho0 * np.exp(-np.r_[lnJ[s], lnJ[s]]); fe = cochain(P, E, rho[:, None] * Acc)
    L0 = (B1 @ sp.diags(w) @ B1.T).tocsc() + sp.diags(1e-12 * np.ones(N)); phi = spsolve(L0, B1 @ (w * fe)); p = -phi
    if probe is None:   # reference "probe": tracers in the upstream corner (never shocked before t ~ 0.05, then post-shock)
        probe = np.where((fr0[0][:, 0] < 0.36) & (fr0[0][:, 1] > 0.36))[0]
    p = p + (Ed['p'][s][probe].mean() - p[probe].mean())               # one reference value per frame
    pr[s] = p[:N0]; rl[s] = rho[:N0]
    if case == 'visc':
        def Dm(Q): return np.stack([Q[tri[:, 1]] - Q[tri[:, 0]], Q[tri[:, 2]] - Q[tri[:, 0]]], axis=2)
        Lg = logm2(Dm(fr[s + 1]) @ np.linalg.inv(Dm(fr[s - 1]))) / (2 * dt)
        th = Lg[:, 0, 0] + Lg[:, 1, 1]; Ssym = 0.5 * (Lg + Lg.transpose(0, 2, 1))
        phiT = MU_TRUE * (2 * (Ssym ** 2).sum((1, 2)) - 2 / 3 * th ** 2)
        pn = np.zeros(N); cn = np.zeros(N)
        for k in range(3): np.add.at(pn, tri[:, k], phiT * A); np.add.at(cn, tri[:, k], A)
        phiN[s] = (pn / np.maximum(cn, 1e-30))[:N0]
        T = p / rho; cond = -K * (L0 @ T) / np.maximum(Ast, 1e-30); condN[s] = cond[:N0]
        orgT = (tri < N0).all(1); orgE = (E < N0).all(1)
        prodPhi[s] = float((phiT / T[tri].mean(1) * A)[orgT].sum())
        prodCond[s] = float(K * (w * (np.log(T[E[:, 1]]) - np.log(T[E[:, 0]])) ** 2)[orgE].sum())
    if s % 10 == 0: print(s, '%.0fs' % (time.time() - t0), flush=True)
np.savez_compressed('/home/claude/w/visc/entropy_%s.npz' % case, pr=pr, rl=rl, phiN=phiN, condN=condN, prodPhi=prodPhi, prodCond=prodCond, hot=hot)
print('done %.0fs' % (time.time() - t0))
