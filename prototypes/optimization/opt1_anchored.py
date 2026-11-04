"""Homogeneous deformation x = F(t) X (simple shear gamma(t) = G t, then plane compression eta at t>=0.3).
In material coordinates the coefficient Gamma*J*C^-1 is spatially uniform, so the exact solution is the initial field
convolved with a Gaussian of covariance Sigma(t) = 2 int Gamma J C^-1 dt / rho0  (computed by FFT on a fine grid).
Compares the network built on (i) the current Delaunay complex of each frame and (ii) an anchored complex
(material connectivity kept between anchors every B frames, current positions)."""
import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.special import erfc
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, 'lhd'); import lhdx as LX

G_RATE, ETA, T, R, XC, W0 = 20.0, 3.0, 1.0, 0.15, np.array([0.5, 0.5]), 0.02
def Fmat(t):
    sh = np.array([[1.0, G_RATE * t], [0.0, 1.0]]); cp = np.diag([1 / ETA, 1.0]) if t >= 0.3 else np.eye(2)
    return cp @ sh
def exact(P0, kap, S):
    ts = np.linspace(0, T, S * 20 + 1); Sig = np.zeros((2, 2))
    for a, b in zip(ts[:-1], ts[1:]):
        F = Fmat(0.5 * (a + b)); Ci = np.linalg.inv(F.T @ F); Sig += 2 * kap * np.linalg.det(F) * Ci * (b - a)
    n = 1024; x = (np.arange(n) + .5) / n; X, Y = np.meshgrid(x, x, indexing='ij')
    c0 = 0.5 * erfc((np.hypot(X - XC[0], Y - XC[1]) - R) / W0)
    k = 2 * np.pi * np.fft.fftfreq(n, 1 / n); KX, KY = np.meshgrid(k, k, indexing='ij')
    g = np.exp(-0.5 * (Sig[0, 0] * KX ** 2 + 2 * Sig[0, 1] * KX * KY + Sig[1, 1] * KY ** 2))
    c = np.real(np.fft.ifft2(np.fft.fft2(c0 - 1) * g)) + 1                     # (c0-1) decays to 0 at the box edge
    return RegularGridInterpolator((x, x), c, bounds_error=False, fill_value=0.0)(P0)

def stiff(P, tri):
    Gr, A = LX._grad_lambda(P, tri); N = len(P); r, c, v = [], [], []
    for i in range(3):
        for j in range(3):
            r.append(tri[:, i]); c.append(tri[:, j]); v.append(A * (Gr[:, i] * Gr[:, j]).sum(1))
    return sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(N, N))

def network(P0, kap, S, mode, B=5):
    t = np.linspace(0, T, S); fr = [P0 @ Fmat(tt).T for tt in t]
    tri0 = LX.complex_peeled(P0)['tri']; m = LX._star(P0, tri0, len(P0)) / 3
    c = 0.5 * erfc((np.hypot(*(P0 - XC).T) - R) / W0); anchor_tri = tri0
    for n in range(S - 1):
        if mode == 'delaunay':
            Ks = [stiff(fr[s], LX.complex_peeled(fr[s])['tri']) for s in (n, n + 1)]
        else:
            if B and n % B == 0 and n > 0: anchor_tri = LX.complex_peeled(fr[n])['tri']
            Ks = [stiff(fr[s], anchor_tri) for s in (n, n + 1)]
        K = 0.5 * (Ks[0] + Ks[1]); dt = t[n + 1] - t[n]
        c = splu((sp.diags(m) + dt * kap * K).tocsc()).solve(m * c)
    return c, m

if __name__ == '__main__':
    S = 101
    for h in (0.01, 0.005):
        g = np.random.default_rng(3); n = int(1 / h)
        X, Y = np.meshgrid((np.arange(n) + .5) * h, (np.arange(n) + .5) * h)
        P0 = np.c_[X.ravel(), Y.ravel()] + g.uniform(-.25, .25, (n * n, 2)) * h
        for kap in (1e-7, 1e-6, 1e-5):
            ce = exact(P0, kap, S); c0 = 0.5 * erfc((np.hypot(*(P0 - XC).T) - R) / W0)
            row = 'h=%.3f kappa=%.0e |' % (h, kap)
            for mode, B in (('delaunay', 0), ('anchored', 5), ('anchored', 0)):
                c, m = network(P0, kap, S, mode, B)
                mean = (m * c0).sum() / m.sum(); V0 = (m * (c0 - mean) ** 2).sum()
                dVn = V0 - (m * (c - mean) ** 2).sum(); dVe = V0 - (m * (ce - mean) ** 2).sum()
                e = np.sqrt((m * (c - ce) ** 2).sum() / (m * (ce - c0) ** 2).sum())
                row += ' %s%s: dV %+.3f field %.3f [%.3f,%.3f] |' % (mode, '' if mode == 'delaunay' else ('(B=%d)' % B if B else '(total)'),
                                                                    dVn / dVe - 1, e, c.min(), c.max())
            print(row, flush=True)
