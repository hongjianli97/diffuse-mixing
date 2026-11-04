"""Kinematic planar-shock tests of the mass-metric network diffusion (positions + masses only).
Flow: pre-shock gas at rest; planar shock x_s(t) = -0.25 + S t, compression eta; post-shock speed u2 = S(1-1/eta).
Exact tracer paths: x = X (t < ts), x = X + u2 (t - ts) (t >= ts), ts = (X + 0.25)/S.
"""
import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.special import erfc
from scipy.spatial import cKDTree
sys.path.insert(0, 'lhd')
import lhdx as LX

def stiff(P, tri, cond):
    G, A = LX._grad_lambda(P, tri); N = len(P); r, c, v = [], [], []
    for i in range(3):
        for j in range(3):
            r.append(tri[:, i]); c.append(tri[:, j]); v.append(cond * A * (G[:, i] * G[:, j]).sum(1))
    return sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(N, N))

def setup(h=0.01, Lx=1.0, Ly=0.3, seed=1):
    g = np.random.default_rng(seed)
    X, Y = np.meshgrid((np.arange(int(Lx / h)) + .5) * h, (np.arange(int(Ly / h)) + .5) * h)
    P0 = np.c_[X.ravel(), Y.ravel()] + g.uniform(-.25, .25, (X.size, 2)) * h
    return P0

S_, ETA = 2.0, 3.0
U2 = S_ * (1 - 1 / ETA)
def ts_of(X): return (X + 0.25) / S_
def pos(P0, t):
    ts = ts_of(P0[:, 0]); x = np.where(t < ts, P0[:, 0], P0[:, 0] + U2 * (t - ts))
    return np.c_[x, P0[:, 1]]
def rho_of(P0, t, rho0):
    return np.where(t < ts_of(P0[:, 0]), rho0, rho0 * ETA)

def run_net(P0, rho0, c0, kappa, closure, S, scheme, T=1.0, nsub=16, ts_est=None):
    t = np.linspace(0, T, S); dt = t[1] - t[0]
    tri0 = LX.complex_peeled(P0)['tri']; A0 = LX._star(P0, tri0, len(P0))
    m = rho0 * A0 / 3; c = c0.copy(); N = len(P0)
    def K_at(P, rho, tri):
        rT = rho[tri].mean(1)
        cond = kappa * rT if closure == 'rhokappa' else kappa * np.ones(len(tri))
        return stiff(P, tri, cond)
    for n in range(S - 1):
        Pn, Pn1 = pos(P0, t[n]), pos(P0, t[n + 1])
        if scheme == 'start':
            tri = LX.complex_peeled(Pn)['tri']; K = K_at(Pn, rho_of(P0, t[n], rho0), tri)
        elif scheme == 'avg':
            tri = LX.complex_peeled(Pn)['tri']; tri1 = LX.complex_peeled(Pn1)['tri']
            K = 0.5 * (K_at(Pn, rho_of(P0, t[n], rho0), tri) + K_at(Pn1, rho_of(P0, t[n + 1], rho0), tri1))
        elif scheme == 'split':
            # positions reconstructed from snapshots only: piecewise-linear with a kink at the estimated crossing time
            tri = LX.complex_peeled(Pn)['tri']
            Pm1 = pos(P0, t[n - 1]) if n > 0 else Pn
            vpre = (Pn - Pm1) / dt
            tsi = ts_est
            hot = (tsi > t[n]) & (tsi <= t[n + 1])
            K = None
            for j in range(nsub):
                tj = t[n] + (j + .5) / nsub * dt
                Pj = Pn + (Pn1 - Pn) * (j + .5) / nsub                     # default: linear
                th = tsi[hot]
                vpost = (Pn1[hot] - Pn[hot] - vpre[hot] * (th - t[n])[:, None]) / (t[n + 1] - th)[:, None]
                Pj[hot] = np.where((tj < th)[:, None], Pn[hot] + vpre[hot] * (tj - t[n]), Pn1[hot] - vpost * (t[n + 1] - tj))
                rhoj = rho_of(P0, t[n], rho0).copy()
                rhoj[(tsi <= tj)] = (rho0 * ETA)[(tsi <= tj)]
                Kj = K_at(Pj, rhoj, LX.complex_peeled(Pj)['tri'])
                K = Kj if K is None else K + Kj
            K = K / nsub
        c = splu((sp.diags(m) + dt * K).tocsc()).solve(m * c)
    return c, m

def ref_1d_x(rho0_fn, closure, kappa, c0_fn, T=1.0, nX=4000, nt=8000):
    """rho0(X) d_t c = d_X( G(X,t) d_X c ) in material X on [0,1], CN; G = Gamma*J*Cinv_xx."""
    X = (np.arange(nX) + .5) / nX; dX = 1 / nX; c = c0_fn(X); r0 = rho0_fn(X); dt = T / nt
    Xf = np.arange(1, nX) / nX
    for k in range(nt):
        tm = (k + .5) * dt
        post = tm >= ts_of(Xf)
        r0f = rho0_fn(Xf)
        Gam = kappa * r0f * np.where(post, ETA, 1.0) if closure == 'rhokappa' else kappa * np.ones_like(Xf)
        G = Gam * np.where(post, ETA, 1.0)          # J*Cinv_xx = eta after crossing
        w = G / dX ** 2
        main = np.zeros(nX); main[:-1] += w; main[1:] += w
        A = sp.diags([main, -w, -w], [0, 1, -1], shape=(nX, nX)).tocsc()
        M = sp.diags(r0)
        c = splu((M + .5 * dt * A).tocsc()).solve((M - .5 * dt * A) @ c)
    return X, c

def main_time():
    P0 = setup(); N = len(P0); rho0 = np.ones(N); w0 = 0.04; kap = 1.5e-4
    inner = (P0[:, 1] > 0.05) & (P0[:, 1] < 0.25) & (P0[:, 0] > 0.15) & (P0[:, 0] < 0.85)
    c_par = 0.5 * erfc((P0[:, 0] - 0.5) / w0)
    for clo in ('rhokappa', 'gamma'):
        cinf, _ = run_net(P0, rho0, c_par, kap, clo, 641, 'start')
        Xr, cr = ref_1d_x(lambda X: np.ones_like(X), clo, kap, lambda X: 0.5 * erfc((X - 0.5) / w0), nt=4000)
        cref = np.interp(P0[:, 0], Xr, cr); nrm = np.linalg.norm((cref - c_par)[inner])
        print(clo, 'space error (S=641 vs 1D ref): %.4f' % (np.linalg.norm((cinf - cref)[inner]) / nrm))
        for S in (11, 21, 41, 81):
            t = np.linspace(0, 1, S); dt = t[1] - t[0]
            k = np.searchsorted(t, ts_of(P0[:, 0])) - 1          # hot-step index from the C channel
            ts_est = np.full(N, np.inf)
            for n in range(S - 1):                                # front-distance interpolation in material coords
                hot = k == n
                if not hot.any(): continue
                done, todo = k < n, k > n
                da = cKDTree(P0[done]).query(P0[hot])[0] if done.any() else np.full(hot.sum(), np.inf)
                db = cKDTree(P0[todo]).query(P0[hot])[0] if todo.any() else np.full(hot.sum(), np.inf)
                fr = np.where(np.isfinite(da) & np.isfinite(db), da / (da + db), np.where(np.isfinite(da), 1.0, 0.0))
                ts_est[hot] = t[n] + np.clip(fr, 0.02, 0.98) * dt
            row = []
            for sch in ('start', 'avg', 'split'):
                c, m = run_net(P0, rho0, c_par, kap, clo, S, sch, ts_est=ts_est, nsub=8)
                row.append(np.linalg.norm((c - cinf)[inner]) / nrm)
            print(clo, S, 'time error start/avg/split: %.4f %.4f %.4f' % tuple(row), 'ts_err/dt %.3f' % np.median(np.abs(ts_est - ts_of(P0[:, 0])) / dt), flush=True)


def clock_check(kap=1.5e-4, w0=0.04):
    """lamella clock vs 1D material reference, plane shock eta=3, lamella parallel to the shock."""
    for clo, Dp in (('rhokappa', kap * ETA ** 2), ('gamma', kap * ETA)):
        tsl = ts_of(0.5); tau = kap * tsl + Dp * (1 - tsl)
        Xr, cr = ref_1d_x(lambda X: np.ones_like(X), clo, kap, lambda X: 0.5 * erfc((X - 0.5) / w0), nt=4000)
        lam = 0.5 * erfc((Xr - 0.5) / np.sqrt(w0 ** 2 + 4 * tau)); sel = np.abs(Xr - 0.5) < 0.2
        print('clock', clo, 'Pi=%.2f' % (4 * tau / w0 ** 2), 'max|1D ref - clock| = %.4f' % np.abs(cr - lam)[sel].max())

def composite_check():
    """static two-density interface: interface value vs effusivity prediction."""
    P0 = setup(h=0.005, Lx=1.0, Ly=0.1); N = len(P0)
    rho0 = np.where(P0[:, 0] < 0.5, 3.0, 1.0); c0 = (P0[:, 0] < 0.5).astype(float)
    tri = LX.complex_peeled(P0)['tri']; m = rho0 * LX._star(P0, tri, N) / 3
    mid = (P0[:, 1] > 0.02) & (P0[:, 1] < 0.08)
    for clo, kap in (('gamma', 2e-3), ('rhokappa', 1e-3)):
        cond = kap * rho0[tri].mean(1) if clo == 'rhokappa' else kap * np.ones(len(tri))
        K = stiff(P0, tri, cond); S = 200; dt = 1.0 / S; lu = splu((sp.diags(m) + dt * K).tocsc()); c = c0.copy()
        for _ in range(S): c = lu.solve(m * c)
        xs, cs = P0[mid, 0], c[mid]; L = (xs > .47) & (xs < .5); R = (xs > .5) & (xs < .53)
        cl = np.polyval(np.polyfit(xs[L], cs[L], 2), .5); cr = np.polyval(np.polyfit(xs[R], cs[R], 2), .5)
        eh, el = (np.sqrt(3 * kap), np.sqrt(kap)) if clo == 'gamma' else (np.sqrt(9 * kap), np.sqrt(kap))
        print('composite', clo, 'interface value %.3f / %.3f, predicted %.3f' % (cl, cr, eh / (eh + el)))

def linear_response_check(w0=0.04):
    """exchanged mass: first-order (DBS-type) / network, vs (1+sqrt(1+Pi))/2."""
    P0 = setup(h=0.005, Lx=1.0, Ly=0.1); N = len(P0)
    tri = LX.complex_peeled(P0)['tri']; m = LX._star(P0, tri, N) / 3
    c0 = 0.5 * erfc((P0[:, 0] - 0.5) / w0); R = P0[:, 0] > 0.5; Ly = 0.1
    for Pi in (0.1, 0.5, 2, 14, 50):
        T = Pi * w0 ** 2 / 4; S = 400; dt = T / S
        K = stiff(P0, tri, np.ones(len(tri))); lu = splu((sp.diags(m) + dt * K).tocsc()); c = c0.copy()
        for _ in range(S): c = lu.solve(m * c)
        Q = (m * (c - c0))[R].sum(); Qlin = Ly * T / (np.sqrt(np.pi) * w0)
        Qex = Ly * (np.sqrt(w0 ** 2 + 4 * T) - w0) / (2 * np.sqrt(np.pi))
        print('Pi=%5.1f  Qnet/Qexact=%.3f  Qlin/Qnet=%.3f  theory=%.3f' % (Pi, Q / Qex, Qlin / Q, (1 + np.sqrt(1 + Pi)) / 2))

if __name__ == '__main__':
    clock_check(); composite_check(); linear_response_check(); main_time()
