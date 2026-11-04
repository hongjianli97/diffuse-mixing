"""Manufactured flow maps (MFM): prescribed analytic flow maps Phi(X,t) with exact density, Jacobian and stretch.

They remove the DNS from the loop, so every claim about the ledger, the block signature and the chained tensor can be
tested in isolation, with convergence orders.  All maps are kinematically mass-consistent (rho = rho0 / J exactly).

  planar_shock      : straight shock of density ratio eta, arbitrary normal, uniform pre-shock velocity (incl. slip)
  cylinder_shock    : converging cylindrical shock (radius Rs(t) decreasing), uniform post-shock density eta*rho0,
                      sink flow behind it -> curved front AND non-uniform post-shock motion (curvature h/Rs)
  smooth_map        : smooth compressible + rotational map x = X + a(t) grad psi + b(t) R grad psi
  shear_layer       : x = X + U t tanh(Y/delta) e1  (thin layer, exact tangential stretch of vertical material lines)
"""
import numpy as np


def lattice(h, box, seed=0, jitter=0.25):
    x0, x1, y0, y1 = box; rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(np.arange(x0 + h / 2, x1, h), np.arange(y0 + h / 2, y1, h))
    P = np.c_[X.ravel(), Y.ravel()]
    return P + rng.uniform(-jitter, jitter, P.shape) * h


def disk(h, R, seed=0, jitter=0.25):
    P = lattice(h, (-R, R, -R, R), seed, jitter)
    return P[np.hypot(P[:, 0], P[:, 1]) < R - 0.5 * h]


# ------------------------------------------------------------------------------------------------ planar shock
def planar_shock(h=0.01, eta=3.0, theta=0.3, Us=1.0, u1=(0.0, 0.0), L=1.0, Cs=1.0, T=None, seed=0):
    """Square [0,L]^2.  Shock: n.x = s0 + Us t with n=(cos th, sin th).  Pre-shock velocity u1 (uniform, may have a
    tangential component), post-shock u2 = u1 + du n, du = (Us - u1.n)(1 - 1/eta).  Snapshot interval from the
    snapshot shock Courant number Cs = (Us - u1.n) dt / h."""
    n = np.array([np.cos(theta), np.sin(theta)]); u1 = np.asarray(u1, float); w = Us - u1 @ n
    X = lattice(h, (0, L, 0, L), seed)
    proj = X @ n; s0 = proj.min() - 0.1 * L                    # shock starts outside the domain
    T = (proj.max() - s0 + 0.1 * L) / w if T is None else T    # ... and leaves it
    dt = Cs * h / w; S = int(np.ceil(T / dt)) + 1; t = np.arange(S) * dt
    du = w * (1 - 1 / eta); u2 = u1 + du * n
    tc = (proj - s0) / w                                       # crossing time (affine in X)
    fr = np.empty((S, len(X), 2)); lnJ = np.zeros((S, len(X)))
    for s, ts in enumerate(t):
        crossed = ts >= tc
        fr[s] = X + np.where(crossed[:, None], u1 * tc[:, None] + u2 * (ts - tc)[:, None], u1 * ts)
        lnJ[s] = np.where(crossed, -np.log(eta), 0.0)
    return dict(name="MFM-planar", frames=fr, t=t, dt=dt, h=h, lnJ_ex=lnJ, tc=tc, eta=eta, n=n,
                interior=lambda P: (P[:, 0] > P[:, 0].min() + 3 * h) & (P[:, 0] < P[:, 0].max() - 3 * h)
                & (P[:, 1] > P[:, 1].min() + 3 * h) & (P[:, 1] < P[:, 1].max() - 3 * h))


# ------------------------------------------------------------------------------------------------ cylindrical shock
def cylinder_shock(h=0.02, eta=3.0, R=1.0, Us=1.0, Rend=0.25, Cs=1.0, seed=0):
    """Converging cylindrical shock Rs(t) = R0 - Us t in a disk of radius R.  Ahead (r0 < Rs): at rest.
    Behind (r0 > Rs): uniform density eta rho0, so  r^2 = Rs^2 + (r0^2 - Rs^2)/eta  (area ratio 1/eta exactly),
    i.e. a sink flow u_r = Rs dRs/dt (1 - 1/eta) / r that is non-uniform on the star scale."""
    X = disk(h, R, seed); r0 = np.hypot(X[:, 0], X[:, 1]); R0 = R + 0.05
    T = (R0 - Rend) / Us; dt = Cs * h / Us; S = int(np.ceil(T / dt)) + 1; t = np.arange(S) * dt
    fr = np.empty((S, len(X), 2)); lnJ = np.zeros((S, len(X))); tc = (R0 - r0) / Us
    for s, ts in enumerate(t):
        Rs = R0 - Us * ts; crossed = r0 > Rs
        r = np.where(crossed, np.sqrt(Rs ** 2 + (r0 ** 2 - Rs ** 2) / eta), r0)
        fr[s] = X * (r / np.maximum(r0, 1e-300))[:, None]
        lnJ[s] = np.where(crossed, -np.log(eta), 0.0)
    Rs_cross = R0 - Us * tc                                     # = r0  (shock radius at crossing)
    return dict(name="MFM-cylinder", frames=fr, t=t, dt=dt, h=h, lnJ_ex=lnJ, tc=tc, eta=eta, Rs_cross=Rs_cross,
                interior=lambda P: np.hypot(P[:, 0], P[:, 1]) < np.hypot(P[:, 0], P[:, 1]).max() - 3 * h)


# ------------------------------------------------------------------------------------------------ smooth map
def smooth_map(h=0.02, k=2 * np.pi, a=0.06, b=0.08, T=1.0, S=41, seed=0):
    """x = X + a(t) grad psi + b(t) R90 grad psi,  psi = sin(kX)sin(kY)/k^2,  a(t)=a sin(pi t/T), b(t) = b t/T.
    Compressible (Laplacian psi != 0) and rotational.  Exact F = I + (a I + b R90) Hess psi."""
    X = lattice(h, (0, 1, 0, 1), seed); x, y = X[:, 0], X[:, 1]
    g = np.c_[np.cos(k * x) * np.sin(k * y), np.sin(k * x) * np.cos(k * y)] / k
    H = np.stack([np.stack([-np.sin(k * x) * np.sin(k * y), np.cos(k * x) * np.cos(k * y)], -1),
                  np.stack([np.cos(k * x) * np.cos(k * y), -np.sin(k * x) * np.sin(k * y)], -1)], -2)
    R90 = np.array([[0.0, -1.0], [1.0, 0.0]]); t = np.linspace(0, T, S)
    fr = np.empty((S, len(X), 2)); lnJ = np.empty((S, len(X)))
    for s, ts in enumerate(t):
        at, bt = a * np.sin(np.pi * ts / T), b * ts / T
        fr[s] = X + at * g + bt * g @ R90.T
        F = np.eye(2) + np.einsum("ij,njk->nik", at * np.eye(2) + bt * R90, H)
        lnJ[s] = np.log(np.linalg.det(F))
    return dict(name="MFM-smooth", frames=fr, t=t, dt=t[1] - t[0], h=h, lnJ_ex=lnJ,
                interior=lambda P: (P[:, 0] > P[:, 0].min() + 3 * h) & (P[:, 0] < P[:, 0].max() - 3 * h)
                & (P[:, 1] > P[:, 1].min() + 3 * h) & (P[:, 1] < P[:, 1].max() - 3 * h))


# ------------------------------------------------------------------------------------------------ thin shear layer
def shear_layer(h=0.01, delta=0.02, U=1.0, T=0.5, S=26, seed=0, L=0.5):
    """x = X + U t tanh(Y/delta), y = Y on [-L/2, L/2]^2.  Vertical material lines (tangent e2) have exact stretch
    ln lambda = 0.5 ln(1 + (U t sech^2(Y/delta)/delta)^2);  J = 1."""
    X = lattice(h, (-L / 2, L / 2, -L / 2, L / 2), seed); t = np.linspace(0, T, S)
    fr = np.stack([np.c_[X[:, 0] + U * ts * np.tanh(X[:, 1] / delta), X[:, 1]] for ts in t])
    g = U * t[:, None] / delta / np.cosh(X[None, :, 1] / delta) ** 2
    return dict(name="MFM-shear", frames=fr, t=t, dt=t[1] - t[0], h=h, lnlam_ex=0.5 * np.log1p(g ** 2),
                lnJ_ex=np.zeros((S, len(X))), delta=delta)


# ------------------------------------------------------------------------------------------------ resolved (viscous) shock
def _M(xi, eta, delta):
    """cumulative mass of the tanh profile rho(xi) = 1 + (eta-1)/2 (1 - tanh(xi/delta)), rho -> eta behind (xi<0)"""
    return xi + 0.5 * (eta - 1) * (xi - delta * np.logaddexp(xi / delta, -xi / delta) + delta * np.log(2.0))


def _Minv(m, eta, delta, it=60):
    xi = np.where(m > 0, m, m / eta)                         # initial guess from the far-field slopes
    for _ in range(it):
        r = 1 + 0.5 * (eta - 1) * (1 - np.tanh(xi / delta)); xi = xi - (_M(xi, eta, delta) - m) / r
    return xi


def viscous_shock(h=0.02, eta=3.0, delta=0.04, theta=0.5, slip=0.3, Us=1.0, L=1.0, Cs=1.0, stop=1.0, seed=0):
    """Steady resolved shock of thickness delta (tanh density profile, ratio eta) moving at Us into gas at rest, with
    shock normal n = (cos th, sin th) and a uniform tangential velocity `slip`.  Mass conservation in the shock frame
    rho (Us - u_n) = Us gives the exact flow map through the cumulative mass M(xi):  xi(t) = M^-1(M(xi0) - Us t),
    J = rho(xi0)/rho(xi).  stop in (0,1]: fraction of the crossing time at which the record ends (stop<1 leaves the
    shock inside the domain at the final frame)."""
    n = np.array([np.cos(theta), np.sin(theta)]); tg = np.array([-n[1], n[0]])
    X = lattice(h, (0, L, 0, L), seed); proj = X @ n; s0 = proj.min() - 0.1 * L - 6 * delta
    T = stop * (proj.max() + 0.1 * L + 6 * delta - s0) / Us * 1.0
    dt = Cs * h / Us; S = int(np.ceil(T / dt)) + 1; t = np.arange(S) * dt
    xi0 = proj - s0; m0 = _M(xi0, eta, delta); rho = lambda xi: 1 + 0.5 * (eta - 1) * (1 - np.tanh(xi / delta))
    fr = np.empty((S, len(X), 2)); lnJ = np.empty((S, len(X)))
    for s, ts in enumerate(t):
        xi = _Minv(m0 - Us * ts, eta, delta); xn = xi + s0 + Us * ts
        fr[s] = X + (xn - proj)[:, None] * n + slip * ts * tg
        lnJ[s] = np.log(rho(xi0) / rho(xi))
    return dict(name="MFM-viscous", frames=fr, t=t, dt=dt, h=h, lnJ_ex=lnJ, eta=eta, delta=delta, n=n,
                rho0=rho(xi0), interior=lambda P: (P[:, 0] > P[:, 0].min() + 3 * h) & (P[:, 0] < P[:, 0].max() - 3 * h)
                & (P[:, 1] > P[:, 1].min() + 3 * h) & (P[:, 1] < P[:, 1].max() - 3 * h))
