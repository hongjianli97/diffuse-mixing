"""Classic benchmarks with exact Lagrangian references (positions only are handed to the methods).
DG    periodically forced double gyre (Shadden et al. 2005), incompressible, hyperbolic + elliptic
SOD   Sod shock tube (planar, exact Riemann solution, exact particle paths through fan/contact/shock)
BLAST cylindrical blast wave, Sedov-Taylor-type radial kinematics, density ratio 6, irrotational
"""
import numpy as np

GAM = 1.4

# =========================================================================== double gyre


def dg_fields(A=0.1, eps=0.25, om=2 * np.pi / 10):
    def ff(x, t):
        a = eps * np.sin(om * t); b = 1 - 2 * a
        return a * x ** 2 + b * x, 2 * a * x + b, 2 * a

    def vel(X, t):
        f, fx, _ = ff(X[:, 0], t)
        return np.c_[-np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * X[:, 1]),
                     np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * X[:, 1]) * fx]

    def vort(X, t):
        f, fx, fxx = ff(X[:, 0], t)
        s, c, sy = np.sin(np.pi * f), np.cos(np.pi * f), np.sin(np.pi * X[:, 1])
        vx = np.pi * A * sy * (-np.pi * s * fx ** 2 + c * fxx)
        uy = np.pi ** 2 * A * s * sy
        return vx - uy

    def grad(X, t):
        f, fx, fxx = ff(X[:, 0], t)
        s, c = np.sin(np.pi * f), np.cos(np.pi * f)
        sy, cy = np.sin(np.pi * X[:, 1]), np.cos(np.pi * X[:, 1])
        ux = -np.pi ** 2 * A * c * fx * cy
        uy = np.pi ** 2 * A * s * sy
        vx = np.pi * A * sy * (-np.pi * s * fx ** 2 + c * fxx)
        vy = np.pi ** 2 * A * c * cy * fx
        return np.stack([np.stack([ux, uy], -1), np.stack([vx, vy], -1)], 1)   # [i,j] = du_i/dx_j
    return vel, vort, grad


def case_dg(nx=160, ny=80, T=10.0, S=51, sub=8, seed=1, jitter=0.25):
    vel, vort, grad = dg_fields()
    h = 2.0 / nx
    g = np.random.default_rng(seed)
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * h, (np.arange(ny) + 0.5) * h)
    P = np.c_[X.ravel(), Y.ravel()] + g.uniform(-jitter, jitter, (nx * ny, 2)) * h
    P = np.clip(P, 1e-3, [2 - 1e-3, 1 - 1e-3])
    N = len(P)
    frames = np.empty((S, N, 2)); frames[0] = P
    Fm = np.tile(np.eye(2), (N, 1, 1))
    t = np.linspace(0, T, S); dts = (t[1] - t[0]) / sub
    # domain-mean vorticity (area-weighted) on a fine grid
    gx, gy = np.meshgrid(np.linspace(0, 2, 401)[:-1] + 0.0025, np.linspace(0, 1, 201)[:-1] + 0.0025)
    G = np.c_[gx.ravel(), gy.ravel()]
    wbar = lambda tt: vort(G, tt).mean()
    om_ex = np.empty((S, N)); om_ex[0] = vort(P, 0.0)
    lavd = np.zeros(N); X_ = P.copy(); tt = 0.0
    prev = np.abs(om_ex[0] - wbar(0.0))
    for s in range(S - 1):
        for _ in range(sub):
            k1 = vel(X_, tt); K1 = grad(X_, tt) @ Fm
            k2 = vel(X_ + .5 * dts * k1, tt + .5 * dts); K2 = grad(X_ + .5 * dts * k1, tt + .5 * dts) @ (Fm + .5 * dts * K1)
            k3 = vel(X_ + .5 * dts * k2, tt + .5 * dts); K3 = grad(X_ + .5 * dts * k2, tt + .5 * dts) @ (Fm + .5 * dts * K2)
            k4 = vel(X_ + dts * k3, tt + dts); K4 = grad(X_ + dts * k3, tt + dts) @ (Fm + dts * K3)
            X_ = X_ + dts * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            Fm = Fm + dts * (K1 + 2 * K2 + 2 * K3 + K4) / 6
            tt += dts
            cur = np.abs(vort(X_, tt) - wbar(tt)); lavd += 0.5 * dts * (prev + cur); prev = cur
        frames[s + 1] = X_
        om_ex[s + 1] = vort(X_, t[s + 1])
    C = np.einsum("nki,nkj->nij", Fm, Fm)
    ftle = np.log(np.linalg.eigvalsh(C)[:, -1]) / (2 * T)
    return dict(name="DG", frames=frames, t=t, dt=t[1] - t[0], h=h, om_ex=om_ex, lavd_ex=lavd,
                ftle_ex=ftle, div_ex=np.zeros((S, N)), vel=vel, vort=vort)

# =========================================================================== Sod


def riemann_star(rL, uL, pL, rR, uR, pR, g=GAM):
    cL, cR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)

    def fK(p, r, pK, c):
        if p > pK:
            A, B = 2 / ((g + 1) * r), (g - 1) / (g + 1) * pK
            return (p - pK) * np.sqrt(A / (p + B)), np.sqrt(A / (p + B)) * (1 - (p - pK) / (2 * (B + p)))
        return 2 * c / (g - 1) * ((p / pK) ** ((g - 1) / (2 * g)) - 1), (p / pK) ** (-(g + 1) / (2 * g)) / (r * c)
    p = 0.5 * (pL + pR)
    for _ in range(100):
        f1, d1 = fK(p, rL, pL, cL); f2, d2 = fK(p, rR, pR, cR)
        dp = (f1 + f2 + uR - uL) / (d1 + d2); p = max(p - dp, 1e-12)
        if abs(dp) < 1e-14:
            break
    f1, _ = fK(p, rL, pL, cL); f2, _ = fK(p, rR, pR, cR)
    u = 0.5 * (uL + uR) + 0.5 * (f2 - f1)
    return p, u


def case_sod(h=0.004, Ly=0.12, t0=0.04, t1=0.24, S=41, seed=2, g=GAM):
    rL, pL, rR, pR = 1.0, 1.0, 0.125, 0.1
    ps, us = riemann_star(rL, 0, pL, rR, 0, pR)
    cL, cR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)
    rsL = rL * (ps / pL) ** (1 / g)
    csL = np.sqrt(g * ps / rsL)
    rsR = rR * ((ps / pR + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pR + 1))
    Ssh = cR * np.sqrt((g + 1) / (2 * g) * ps / pR + (g - 1) / (2 * g))
    b = 2 / (g + 1); a = 2 * cL / (g + 1); xi_tail = us - csL
    rng = np.random.default_rng(seed)
    nx, ny = int(round(1.0 / h)), int(round(Ly / h))
    X, Y = np.meshgrid(-0.5 + (np.arange(nx) + 0.5) * h, (np.arange(ny) + 0.5) * h)
    P0 = np.c_[X.ravel(), Y.ravel()] + rng.uniform(-0.25, 0.25, (nx * ny, 2)) * h
    x0 = P0[:, 0]
    t = np.linspace(t0, t1, S)
    # crossing / event times
    ts = np.where(x0 > 0, x0 / Ssh, np.inf)
    th = np.where(x0 < 0, -x0 / cL, np.inf)
    C = np.where(x0 < 0, (x0 - a * th / (1 - b)) / np.where(np.isfinite(th), th, 1.0) ** b, 0.0)
    with np.errstate(all="ignore"):
        te = np.where(x0 < 0, ((xi_tail - a / (1 - b)) / C) ** (1 / (b - 1)), np.inf)
    xe = np.where(x0 < 0, C * te ** b + a * te / (1 - b), 0.0)

    def state(tt):
        x = np.where(x0 > 0, np.where(tt < ts, x0, x0 + us * (tt - ts)),
                     np.where(tt < th, x0, np.where(tt < te, C * tt ** b + a * tt / (1 - b), xe + us * (tt - te))))
        infan = (x0 < 0) & (tt >= th) & (tt < te)
        cfan = b * cL - (g - 1) / (g + 1) * x / tt
        rho = np.where(x0 > 0, np.where(tt < ts, rR, rsR),
                       np.where(tt < th, rL, np.where(infan, rL * (np.maximum(cfan, 1e-9) / cL) ** (2 / (g - 1)), rsL)))
        u = np.where(x0 > 0, np.where(tt < ts, 0.0, us),
                     np.where(tt < th, 0.0, np.where(infan, b * (cL + x / tt), us)))
        div = np.where(infan, b / tt, 0.0)
        p = np.where(x0 > 0, np.where(tt < ts, pR, ps),
                     np.where(tt < th, pL, np.where(infan, pL * (rho / rL) ** g, ps)))
        return x, rho, u, div, p
    frames = np.empty((S, len(P0), 2)); rho = np.empty((S, len(P0))); u = np.empty_like(rho)
    div = np.empty_like(rho); pr = np.empty_like(rho)
    for s, tt in enumerate(t):
        x, rho[s], u[s], div[s], pr[s] = state(tt)
        frames[s] = np.c_[x, P0[:, 1]]
    rho0 = np.where(x0 > 0, rR, rL)
    lnJ = np.log(rho0[None, :] / rho)                      # vs t=0 ;  window value = lnJ - lnJ[0]
    ev_step = np.searchsorted(t, ts) - 1                    # crossing between frames s and s+1
    ev = np.where((x0 > 0) & (ev_step >= 0) & (ev_step < S - 1))[0]
    return dict(name="SOD", frames=frames, t=t, dt=t[1] - t[0], h=h, rho=rho, u=u, div_ex=div, p=pr,
                lnJ_ex=lnJ - lnJ[0], ev_pidx=ev, ev_step=ev_step[ev], x0=x0, mass=rho0 * h * h,
                om_ex=np.zeros_like(rho), info=dict(ps=ps, us=us, Ssh=Ssh, rsL=rsL, rsR=rsR, cL=cL, xi_tail=xi_tail),
                interior=lambda P: (P[:, 1] > 0.15 * Ly) & (P[:, 1] < 0.85 * Ly))

# =========================================================================== blast


def case_blast(h=0.022, Rd=1.2, t0=1.0, t1=25.0, S=61, seed=3, g=GAM):
    p = 1.0 / (g + 1.0); R0 = 0.2; eta = (g + 1) / (g - 1)
    rng = np.random.default_rng(seed)
    gr = np.arange(-Rd, Rd + h, h); X, Y = np.meshgrid(gr, gr)
    P = np.c_[X.ravel(), Y.ravel()] + rng.uniform(-0.25, 0.25, (X.size, 2)) * h
    P = P[np.hypot(P[:, 0], P[:, 1]) <= Rd]
    rs = np.hypot(P[:, 0], P[:, 1]); tc = t0 * (rs / R0) ** 2
    t = np.linspace(t0, t1, S)
    inside = rs <= R0
    frames = np.empty((S, len(P), 2)); lnJ = np.zeros((S, len(P))); div = np.zeros_like(lnJ)
    for s, tt in enumerate(t):
        post = (~inside) & (tt >= tc)
        r = np.where(inside, rs * (tt / t0) ** p, np.where(post, rs * (np.maximum(tt, tc) / tc) ** p, rs))
        frames[s] = P * (r / rs)[:, None]
        lnJ[s] = np.where(inside, 2 * p * np.log(tt / t0), np.where(post, -np.log(eta) + 2 * p * np.log(tt / tc), 0.0))
        div[s] = np.where(inside | post, 2 * p / tt, 0.0)
    ev_step = np.searchsorted(t, tc) - 1
    ev = np.where((~inside) & (ev_step >= 0) & (ev_step < S - 1))[0]
    return dict(name="BLAST", frames=frames, t=t, dt=t[1] - t[0], h=h, lnJ_ex=lnJ, div_ex=div,
                om_ex=np.zeros_like(lnJ), ev_pidx=ev, ev_step=ev_step[ev], rs=rs, tc=tc, R0=R0,
                interior=lambda P0: np.hypot(P0[:, 0], P0[:, 1]) < Rd - 4 * h)


def case_lo(a=1.0, Rd=7.0, d=6.0, Gam=2 * np.pi, T=6 * np.pi, S=41, N=4000, sub=10, seed=6):
    """Equal co-rotating Lamb-Oseen pair (paper Sec. III B): u_theta = Gam/(2 pi r)(1-exp(-r^2/a^2)), centres at
    +-(d/2)(cos W t, sin W t), W = Gam (1-exp(-d^2/a^2)) / (pi d^2).  Disc of radius Rd seeded uniformly."""
    rng = np.random.default_rng(seed)
    h = np.sqrt(np.pi * Rd ** 2 / N)
    g = np.arange(-Rd, Rd + h, h); X, Y = np.meshgrid(g, g)
    P = np.c_[X.ravel(), Y.ravel()] + rng.uniform(-0.25, 0.25, (X.size, 2)) * h
    P = P[np.hypot(P[:, 0], P[:, 1]) <= Rd]
    W = Gam * (1 - np.exp(-d ** 2 / a ** 2)) / (np.pi * d ** 2)
    def cen(t):
        c = 0.5 * d * np.array([np.cos(W * t), np.sin(W * t)]); return [c, -c]
    def vel(Xq, t):
        u = np.zeros_like(Xq)
        for c in cen(t):
            r = Xq - c; r2 = (r * r).sum(1) + 1e-300
            f = Gam / (2 * np.pi * r2) * (1 - np.exp(-r2 / a ** 2))
            u += f[:, None] * np.c_[-r[:, 1], r[:, 0]]
        return u
    def vort(Xq, t):
        return sum(Gam / (np.pi * a ** 2) * np.exp(-((Xq - c) ** 2).sum(1) / a ** 2) for c in cen(t))
    t = np.linspace(0, T, S); dts = (t[1] - t[0]) / sub
    fr = np.empty((S, len(P), 2)); fr[0] = P; Xq = P.copy(); tt = 0.0
    lavd = np.zeros(len(P)); w0 = vort(Xq, 0); prev = np.abs(w0 - w0.mean())
    for s in range(S - 1):
        for _ in range(sub):
            k1 = vel(Xq, tt); k2 = vel(Xq + .5 * dts * k1, tt + .5 * dts); k3 = vel(Xq + .5 * dts * k2, tt + .5 * dts); k4 = vel(Xq + dts * k3, tt + dts)
            Xq = Xq + dts * (k1 + 2 * k2 + 2 * k3 + k4) / 6; tt += dts
            w = vort(Xq, tt); cur = np.abs(w - w.mean()); lavd += 0.5 * dts * (prev + cur); prev = cur
        fr[s + 1] = Xq
    return dict(name="LO", frames=fr, t=t, dt=t[1] - t[0], lavd_ex=lavd, h=h, a=a, Rd=Rd, centres=cen(0))
