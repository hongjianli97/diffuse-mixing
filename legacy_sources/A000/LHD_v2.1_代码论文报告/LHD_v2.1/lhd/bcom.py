import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import lhdx as X, classify as C, netdiff as ND

def ledger(fr, tau_min=0.2, block=5, defer=4, kfac=4.0):
    hot, R, tau = C.hot_steps(fr, tau_min=tau_min, kfac=kfac)
    S, N, _ = fr.shape; bad = np.zeros((N, S), bool); bad[:, :-1] |= hot; bad[:, 1:] |= hot
    return X.chained_lnJ_v2(fr, bad, block=block, max_defer=defer), hot, bad, tau

def errstats(est, ref, mask):
    e = np.abs(est/ref-1)[mask]; e = e[np.isfinite(e)]
    return dict(med=float(np.median(e)), p95=float(np.quantile(e, .95)), p99=float(np.quantile(e, .99)))

def density_all(fr, rho0, lnJ, mask):
    m = rho0*X.star_area(fr[0])/3.0
    led = rho0*np.exp(-lnJ[-1]); dt = X.dtfe_density(fr[-1], m); sph = X.sph_density(fr[-1], m)
    return m, dict(ledger=led, dtfe=dt, sph=sph)

def cubic_F(r, h):
    q = r/h; sig = 10.0/(7*np.pi*h**2)
    fp = np.where(q < 1, -3*q+2.25*q**2, np.where(q < 2, -0.75*(2-q)**2, 0.0))
    return -sig*fp/(h*np.maximum(r, 1e-300))           # -W'(r)/r >= 0

def sph_diffusion(frames, dt, c0, kappa, m, rho, eta=1.3):
    """Cleary-Monaghan/Brookshaw SPH conduction in conservative mass form, implicit Euler per frame:
    m_i dc_i/dt = sum_j 2 V_i V_j kbar_ij F_ij (c_j-c_i),  kbar = harmonic mean of rho*kappa, V = m/rho."""
    S, N, _ = frames.shape; c = np.asarray(c0, float).copy()
    for s in range(S-1):
        P = frames[s]; V = m/rho[s]; h = eta*np.sqrt(np.abs(V)); k = kappa*rho[s]
        tree = cKDTree(P); lst = tree.query_ball_point(P, r=2.0*h)
        I = np.repeat(np.arange(N), [len(l) for l in lst]); J = np.concatenate([np.asarray(l, int) for l in lst])
        keep = I < J; I, J = I[keep], J[keep]
        r = np.linalg.norm(P[I]-P[J], axis=1); hij = 0.5*(h[I]+h[J]); ok = r < 2*hij; I, J, r, hij = I[ok], J[ok], r[ok], hij[ok]
        w = 2*V[I]*V[J]*(2*k[I]*k[J]/(k[I]+k[J]))*cubic_F(r, hij)
        W = sp.csr_matrix((np.r_[w, w], (np.r_[I, J], np.r_[J, I])), shape=(N, N))
        K = sp.diags(np.asarray(W.sum(1)).ravel())-W
        c = spsolve((sp.diags(m)+dt*K).tocsc(), m*c)
    return c

rel = lambda a, b: float(np.sqrt(np.mean((a-b)**2))/np.std(b))

import lhd as L
def _sstar(P, tri, N):
    A = L.tri_signed_area(P, tri); s = np.zeros(N); np.add.at(s, tri.ravel(), np.repeat(A, 3)); return s

def hot_robust(fr, tau_min=0.2, kfac=4.0):
    """C channel with persistence: mean log star area over (t+1,t+2) minus over (t-1,t), triangulation of frame t."""
    S, N, _ = fr.shape; R = np.zeros((N, S-1))
    for s in range(S-1):
        tri = X.complex_peeled(fr[s])["tri"]; fs = [max(s-1, 0), s, s+1, min(s+2, S-1)]
        A = [np.abs(_sstar(fr[f], tri, N)) for f in fs]
        with np.errstate(all="ignore"):
            R[:, s] = np.nan_to_num(0.5*(np.log(A[2])+np.log(A[3]))-0.5*(np.log(A[0])+np.log(A[1])))
    sig = 1.4826*np.median(np.abs(R-np.median(R))); tau = max(tau_min, kfac*sig)
    hot = np.abs(R) > tau
    # keep only the single step with the largest jump inside each run (a crossing is one event)
    return hot, R, tau

def chained_lnJ_v3(frames, bad=None, block=5, max_defer=4):
    """noise-robust ledger: anchor topology from the adjacent frame, anchor area measured in the current frame,
    signed star areas (orientation of the anchor triangulation)."""
    S, N, _ = frames.shape
    tri0 = X.complex_peeled(frames[1 if S > 1 else 0])["tri"]
    tri0 = np.where((L.tri_signed_area(frames[1], tri0) < 0)[:, None], tri0[:, [0, 2, 1]], tri0)
    tris = {0: tri0}; anchor = np.zeros(N, int); A0 = _sstar(frames[0], tri0, N); base = np.zeros(N); out = np.zeros((S, N))
    for t in range(1, S):
        cur = np.full(N, np.nan)
        for a in np.unique(anchor):
            idx = anchor == a; cur[idx] = _sstar(frames[t], tris[a], N)[idx]
        with np.errstate(all="ignore"):
            inc = np.log(cur/A0)
        out[t] = np.where(np.isfinite(inc), base+inc, out[t-1])
        due = (t-anchor) >= block
        if due.any() and t < S-1:
            trit = X.complex_peeled(frames[t-1])["tri"]                      # topology from the adjacent frame
            trit = np.where((L.tri_signed_area(frames[t-1], trit) < 0)[:, None], trit[:, [0, 2, 1]], trit)
            if bad is not None:
                nb = bad[:, max(t-1, 0):t+2].any(1); sb = np.zeros(N, bool)
                for k in range(3): np.logical_or.at(sb, trit[:, k], nb[trit].any(1))
                ok = due & (~sb | ((t-anchor) >= max_defer*block))
            else: ok = due
            if ok.any():
                tris[t] = trit; At = _sstar(frames[t], trit, N)                # area measured in the current frame
                good = ok & (At > 0); A0[good] = At[good]; base[good] = out[t][good]; anchor[good] = t
        live = set(np.unique(anchor))
        for a in list(tris):
            if a not in live: del tris[a]
    return out

def ledger3(fr, tau_min=0.2, block=5, defer=4):
    hot, R, tau = hot_robust(fr, tau_min)
    S, N, _ = fr.shape; bad = np.zeros((N, S), bool); bad[:, :-1] |= hot; bad[:, 1:] |= hot
    return chained_lnJ_v3(fr, bad, block, defer), hot, bad, tau

def ledger_breaks(fr, rho0, p, gap=2, Lw=5, seed=0, v3=True):
    """track breaks: each track breaks after a frame with prob p, the particle is absent for `gap` frames and
    reappears with a new identity.  Ledger is run per window of Lw steps on tracks complete in the window;
    a new identity gets its density from the 8 nearest tracks of known density (inverse distance, log rho)."""
    rng = np.random.default_rng(seed); S, N, _ = fr.shape
    seg = np.zeros((N, S), int); cur = np.zeros(N, int); absent = np.full(N, -1)
    for t in range(S):
        a = t <= absent; seg[:, t] = np.where(a, -1, cur)
        brk = (~a) & (rng.random(N) < p) & (t < S-1); absent[brk] = t+gap; cur[brk] += 1
    D = rho0.astype(float).copy(); Dseg = seg[:, 0].copy()
    wins = list(range(0, S-1, Lw)); used_last = None
    for a in wins:
        b = min(a+Lw, S-1)
        usable = (seg[:, a:b+1] >= 0).all(1) & (seg[:, a] == seg[:, b])
        known = usable & (Dseg == seg[:, a]) & np.isfinite(D)
        unk = usable & ~known
        if unk.any() and known.any():
            d, nb = cKDTree(fr[a][known]).query(fr[a][unk], k=8); w = 1/np.maximum(d, 1e-12)
            kv = np.log(D[known]); D[unk] = np.exp((w*kv[nb]).sum(1)/w.sum(1))
        sub = fr[a:b+1][:, usable]
        if v3: lnJ, _, _, _ = ledger3(sub) if sub.shape[0] >= 3 else (np.zeros((sub.shape[0], usable.sum())), 0, 0, 0)
        else: lnJ, _, _, _ = ledger(sub)
        newD = np.full(N, np.nan); newD[usable] = D[usable]*np.exp(-lnJ[-1])
        D = newD; Dseg = np.where(usable, seg[:, b], -2); used_last = usable
    return D, used_last, float((seg[:, -1] != seg[:, 0]).mean())

def chained_lnJ_adapt(frames, bad=None, eps=0.03, max_len=20, max_defer=8):
    """Adaptive per-tracer anchoring: keep the anchor star while its vertex motion stays affine; re-anchor when the
    least-squares affine residual of the star (relative to sqrt of star area) exceeds eps, or after max_len frames;
    never re-anchor while the star touches a shock stencil (bad), up to max_defer extra frames beyond max_len."""
    S, N, _ = frames.shape
    def star_nb(tri):
        e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]], tri[:, [1, 0]], tri[:, [2, 1]], tri[:, [0, 2]]])
        e = np.unique(e, axis=0); cnt = np.bincount(e[:, 0], minlength=N); K = cnt.max()
        nb = np.full((N, K), -1); st = np.r_[0, np.cumsum(cnt)[:-1]]; pos = np.arange(len(e))-st[e[:, 0]]
        nb[e[:, 0], pos] = e[:, 1]; return nb
    tri0 = X.complex_peeled(frames[0])["tri"]
    tris = {0: tri0}; nbs = {0: star_nb(tri0)}
    anchor = np.zeros(N, int); A0 = X._star(frames[0], tri0, N); base = np.zeros(N); out = np.zeros((S, N))
    nre = 0
    for t in range(1, S):
        cur = np.full(N, np.nan); res = np.zeros(N)
        for a in np.unique(anchor):
            idx = np.where(anchor == a)[0]; cur[idx] = X._star(frames[t], tris[a], N)[idx]
            nb = nbs[a][idx]; ok = nb >= 0; nbc = np.where(ok, nb, idx[:, None])
            Xa = frames[a][nbc]-frames[a][idx][:, None]; Xt = frames[t][nbc]-frames[t][idx][:, None]
            w = ok[:, :, None].astype(float)
            # LS affine fit with the centre fixed (relative coordinates): Xt ~ Xa F^T
            G = np.einsum("nki,nkj->nij", Xa*w, Xa)+1e-30*np.eye(2); H = np.einsum("nki,nkj->nij", Xa*w, Xt)
            Ft = np.linalg.solve(G, H); r = Xt-np.einsum("nki,nij->nkj", Xa, Ft)
            res[idx] = np.sqrt((w[:, :, 0]*(r**2).sum(2)).sum(1)/np.maximum(w[:, :, 0].sum(1), 1))/np.sqrt(np.maximum(cur[idx], 1e-30))
        with np.errstate(all="ignore"):
            inc = np.log(cur/A0)
        out[t] = np.where(np.isfinite(inc), base+inc, out[t-1])
        if t < S-1:
            age = t-anchor
            due = (res > eps) | (age >= max_len)
            if due.any():
                trit = X.complex_peeled(frames[t])["tri"]
                if bad is not None:
                    nbd = bad[:, max(t-1, 0):t+2].any(1); sb = np.zeros(N, bool)
                    for k in range(3): np.logical_or.at(sb, trit[:, k], nbd[trit].any(1))
                    ok = due & (~sb | (age >= max_len+max_defer))
                else: ok = due
                if ok.any():
                    tris[t] = trit; nbs[t] = star_nb(trit); At = X._star(frames[t], trit, N)
                    good = ok & (At > 0); A0[good] = At[good]; base[good] = out[t][good]; anchor[good] = t; nre += int(good.sum())
        live = set(np.unique(anchor))
        for a in list(tris):
            if a not in live: del tris[a]; del nbs[a]
    return out, nre/N
