"""Time integrators on the planar-shock kinematic test: implicit Euler (averaged K) vs exponential/Magnus (averaged K)."""
import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu, expm_multiply
from scipy.special import erfc
sys.argv = [sys.argv[0]]
import diffusion_theory_checks as DT
LX = DT.LX
P0 = DT.setup(); N = len(P0); rho0 = np.ones(N); w0 = 0.04
inner = (P0[:, 1] > 0.05) & (P0[:, 1] < 0.25) & (P0[:, 0] > 0.15) & (P0[:, 0] < 0.85)
c_par = 0.5 * erfc((P0[:, 0] - 0.5) / w0)
tri0 = LX.complex_peeled(P0)['tri']; m = rho0 * LX._star(P0, tri0, N) / 3
def run(S, kap, clo, scheme):
    t = np.linspace(0, 1, S); c = c_par.copy(); Kc = {}
    def K(s):
        if s not in Kc:
            P = DT.pos(P0, t[s]); tri = LX.complex_peeled(P)['tri']; rho = DT.rho_of(P0, t[s], rho0)
            cond = kap * rho[tri].mean(1) if clo == 'rhokappa' else kap * np.ones(len(tri)); Kc[s] = DT.stiff(P, tri, cond)
        return Kc[s]
    for n in range(S - 1):
        Kb = 0.5 * (K(n) + K(n + 1)); dt = t[n + 1] - t[n]
        if scheme == 'IE': c = splu((sp.diags(m) + dt * Kb).tocsc()).solve(m * c)
        else: c = expm_multiply(-dt * sp.diags(1 / m) @ Kb, c)
        Kc.pop(n, None)
    return c
for clo, kap in (('rhokappa', 1.5e-4), ('rhokappa', 1.5e-3)):
    ref = run(641, kap, clo, 'EXP'); nrm = np.linalg.norm((ref - c_par)[inner])
    for S in (11, 21, 41, 81):
        cI = run(S, kap, clo, 'IE'); cE = run(S, kap, clo, 'EXP')
        print('%s kappa=%.1e S=%3d  time error IE %.4f  EXP %.4f   EXP min/max %.2e %.6f' % (clo, kap, S,
              np.linalg.norm((cI - ref)[inner]) / nrm, np.linalg.norm((cE - ref)[inner]) / nrm, cE.min(), cE.max()), flush=True)
