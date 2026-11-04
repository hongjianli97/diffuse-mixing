"""dirac.py -- node-edge-triangle Dirac operator on the geometric activation K(t) of the trajectory network.

D(x0, x1, x2) = (delta1 x1, d0 x0 + delta2 x2, d1 x1),  d0 = B1^T, d1 = B2^T (combinatorial, the network's incidences),
delta1 = M0^{-1} d0^T M1, delta2 = M1^{-1} d1^T M2 (adjoints in the physical inner products):
M0 = lumped star area / 3, M1 = Whitney 1-form mass matrix, M2 = diag(1/area).   D^2 = L0 + L1 + L2 (block diagonal).
First-order joint signal : nodes theta_C (material-area dilatation), edges f (velocity 1-cochain), triangles d1 f (circulation)
Second-order joint signal: edges g_e = 1/2 (a_u + a_v).r_e (acceleration cochain); d1 g = dGamma/dt of the material triangle
                           (discrete Kelvin theorem, exact for the trapezoid cochain) = baroclinic + viscous torque."""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
import lhd as L, lhdx as X


def blocks(P, cx):
    tri, te, E = cx['tri'], cx['tri_e'], cx['edges']; N, M, F = len(P), len(E), len(tri)
    G, A = X._grad_lambda(P, tri)                                        # (F,3,2), (F,)
    Gd = np.einsum('fid,fjd->fij', G, G)
    sgn = np.stack([np.where(E[te[:, j], 0] == tri[:, j], 1.0, -1.0) for j in range(3)], 1)   # local edge j: j -> j+1
    rows, cols, vals = [], [], []
    for j in range(3):
        a, b = j, (j + 1) % 3
        for k in range(3):
            c, d = k, (k + 1) % 3
            m = (A / 12.0) * ((1 + (a == c)) * Gd[:, b, d] - (1 + (a == d)) * Gd[:, b, c] - (1 + (b == c)) * Gd[:, a, d] + (1 + (b == d)) * Gd[:, a, c])
            rows.append(te[:, j]); cols.append(te[:, k]); vals.append(sgn[:, j] * sgn[:, k] * m)
    M1 = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(M, M))
    star = np.zeros(N); [np.add.at(star, tri[:, j], A) for j in range(3)]
    M0 = star / 3.0; M2 = 1.0 / A
    d0 = sp.csr_matrix((np.r_[-np.ones(M), np.ones(M)], (np.r_[np.arange(M), np.arange(M)], np.r_[E[:, 0], E[:, 1]])), shape=(M, N))
    d1 = sp.csr_matrix((sgn.ravel(), (np.repeat(np.arange(F), 3), te.ravel())), shape=(F, M))
    used = M0 > 0
    M1r = M1 + sp.diags(1e-14 * np.abs(M1.diagonal()).mean() * np.ones(M))
    return dict(d0=d0, d1=d1, M0=M0, M1=M1, M2=M2, A=A, lu=splu(M1r.tocsc()), used=used)


def apply_D(B, x0, x1, x2):
    with np.errstate(all='ignore'):
        y0 = np.where(B['used'], (B['d0'].T @ (B['M1'] @ x1)) / B['M0'], 0.0)
    y1 = B['d0'] @ x0 + B['lu'].solve(B['d1'].T @ (B['M2'] * x2))
    y2 = B['d1'] @ x1
    return y0, y1, y2


def cochain(P, U, E):
    return 0.5 * ((U[E[:, 0]] + U[E[:, 1]]) * (P[E[:, 1]] - P[E[:, 0]])).sum(1)


def tri_circ(P, U, tri):
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]; ua, ub, uc = U[tri[:, 0]], U[tri[:, 1]], U[tri[:, 2]]
    return 0.5 * (((ua + ub) * (b - a)).sum(1) + ((ub + uc) * (c - b)).sum(1) + ((uc + ua) * (a - c)).sum(1))


def node_avg(N, val, w, tri, ok):
    num, den = np.zeros(N), np.zeros(N)
    for j in range(3): np.add.at(num, tri[ok, j], (val * w)[ok]); np.add.at(den, tri[ok, j], w[ok])
    with np.errstate(all='ignore'): return np.where(den > 0, num / den, np.nan)


def residual(net, s):
    """Dirac residual at snapshot s: theta_C (material star, central in time) + delta1 f  (should vanish: continuity)."""
    fr, cx, N, dt = net.fr, net.K[s], net.N, net.dt; tri = cx['tri']
    Am, Ap = X._star(fr[s - 1], tri, N), X._star(fr[s + 1], tri, N)
    with np.errstate(all='ignore'): thC = (np.log(Ap) - np.log(Am)) / (2 * dt)
    B = blocks(fr[s], cx); f = cochain(fr[s], net.V[s], cx['edges'])
    y0, _, _ = apply_D(B, np.zeros(N), f, np.zeros(len(tri)))
    rD = thC + y0; rG = thC - X.galerkin_div(fr[s], cx, net.V[s])
    rD[X.open_star_nodes(cx)] = np.nan; rG[X.open_star_nodes(cx)] = np.nan
    return rD, rG, thC


def acceleration(fr, dt):
    a = np.full_like(fr, np.nan); a[1:-1] = (fr[2:] - 2 * fr[1:-1] + fr[:-2]) / dt ** 2
    return a


def torque_map(net, s, Acc):
    """node torque density (area-weighted mean of d1 g / A over clean triangles of K(s))."""
    cx = net.K[s]; tri = cx['tri']; A = L.tri_signed_area(net.fr[s], tri)
    ok = ~net.bad[tri, s - 1:s + 2].any((1, 2)) if hasattr(net, 'bad') else np.ones(len(tri), bool)
    return node_avg(net.N, tri_circ(net.fr[s], Acc[s], tri) / A, np.abs(A), tri, ok)


def kelvin_split(net, Acc, region, s0, s1):
    """Follow the material triangles of K(s0) inside `region` from s0 to s1.  Per triangle:
    total = Gamma(s1) - Gamma(s0); smooth = sum over clean snapshots of d1 g * dt; impulsive = total - smooth (shock passage)."""
    fr, V, dt = net.fr, net.V, net.dt; tri = net.K[s0]['tri']; tri = tri[region[tri].all(1)]
    A0 = L.tri_signed_area(fr[s0], tri); keep = np.ones(len(tri), bool); sm = np.zeros(len(tri))
    for s in range(s0, s1 + 1):
        keep &= np.sign(L.tri_signed_area(fr[s], tri)) == np.sign(A0)
        if s0 < s < s1:
            ok = ~net.bad[tri, s - 1:s + 2].any((1, 2))
            sm += np.where(ok, tri_circ(fr[s], Acc[s], tri), 0.0) * dt
    tot = tri_circ(fr[s1], V[s1], tri) - tri_circ(fr[s0], V[s0], tri)
    return dict(tri=tri[keep], A0=A0[keep], total=tot[keep], smooth=sm[keep], impulsive=(tot - sm)[keep], kept=float(keep.mean()))
