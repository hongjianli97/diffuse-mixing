"""tcx.py -- trajectory-complex theory (completed part), positions only.

Complex representation of the 3-point affine map of a material triangle:  F zeta = p zeta + q conj(zeta)
    J = |p|^2 - |q|^2,  polar angle = arg p,  Beltrami mu = q/p,  c_t/A_t = 2 Im p,  |F T| = |p + q e^{-2i th}|.
Composition:  (p2,q2) o (p1,q1) = (p2 p1 + q2 conj(q1), p2 q1 + q2 conj(p1)).

Per-node chain (ledger-like): a node is anchored to its star in the peeled Delaunay complex K(t_a).  Each star triangle
is followed materially; per step its map (p_k, q_k) is computed from its own three vertices.  Steps with |mu| > mu* are
merged into blocks (hold = 2 cold steps); a block contributes its exact polar angle via a frame-invariant remainder
  rem = wrap(arg P_block - sum_k arg p_k).
The node's segment ends when a star triangle fails the shape gate (Lambda < 0.2, or inverts) or after Rmax steps, but
never while a star block is open (deferred up to Dmax).  Outputs per node: per-step angle increments, remainders, area
ledger ln J, segment list; afterwards phi = sum(inc - mean) + sum(rem) (exactly objective), LAVD, signatures.
"""
import numpy as np, scipy.sparse as sp
import lhd as L, lhdx as X

wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- triangle geometry & complex representation
def cplx(P):
    return X.complex_peeled(P)


def _edges(P, tri):
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    return np.stack([P[b] - P[a], P[c] - P[a]], -1)          # (nT, 2, 2), columns e1, e2


def tri_area(P, tri):
    D = _edges(P, tri); return 0.5 * (D[:, 0, 0] * D[:, 1, 1] - D[:, 0, 1] * D[:, 1, 0])


def tri_lambda(P, tri):
    A = np.abs(tri_area(P, tri))
    per = sum(np.linalg.norm(P[tri[:, (j + 1) % 3]] - P[tri[:, j]], axis=1) for j in range(3))
    return 12 * np.sqrt(3) * A / np.maximum(per ** 2, 1e-300)


def map_F(P0, P1, tri):
    D0 = _edges(P0, tri); D1 = _edges(P1, tri)
    det = D0[:, 0, 0] * D0[:, 1, 1] - D0[:, 0, 1] * D0[:, 1, 0]
    det = np.where(np.abs(det) > 1e-300, det, 1e-300)
    inv = np.stack([np.stack([D0[:, 1, 1], -D0[:, 0, 1]], -1), np.stack([-D0[:, 1, 0], D0[:, 0, 0]], -1)], 1) / det[:, None, None]
    return D1 @ inv


def pq_of(F):
    p = 0.5 * ((F[:, 0, 0] + F[:, 1, 1]) + 1j * (F[:, 1, 0] - F[:, 0, 1]))
    q = 0.5 * ((F[:, 0, 0] - F[:, 1, 1]) + 1j * (F[:, 1, 0] + F[:, 0, 1]))
    return p, q


def compose(P, Q, p, q):
    return p * P + q * np.conj(Q), p * Q + q * np.conj(P)


def mu_threshold(fr, kappa=8.0, mu_min=1e-3, nsamp=12, kappa_J=8.0, J_min=1e-3):
    """returns (mu*, J*) packed as a tuple-like float subclass: hot = |mu| > mu* and ln J_step < -J*."""
    S = fr.shape[0]; ks = np.unique(np.linspace(0, S - 2, min(nsamp, S - 1)).astype(int)); mus = []; ljs = []
    for k in ks:
        tri = cplx(fr[k])["tri"]; p, q = pq_of(map_F(fr[k], fr[k + 1], tri)); mus.append(np.abs(q / p))
        J = np.abs(p) ** 2 - np.abs(q) ** 2; ljs.append(np.abs(np.log(np.maximum(J, 1e-300))))
    med = float(np.median(np.concatenate(mus))); medJ = float(np.median(np.concatenate(ljs)))
    return HotRule(max(kappa * med, mu_min), max(kappa_J * medJ, J_min)), med


class HotRule(float):
    """float (= mu*) carrying the compression threshold J*; hot step = |mu| > mu* and ln J_step < -J*."""
    def __new__(cls, mu, J):
        o = float.__new__(cls, mu); o.J = J; return o


def sigma_hat(fr):
    """robust position-noise estimate: min over k=2..4 of MAD(k-th time difference)/sqrt(C(2k,k)).
    Smooth trajectories make higher differences small, white noise makes them grow by the binomial factor."""
    from math import comb
    out = []
    for k in (2, 3, 4):
        d = np.diff(fr, n=k, axis=0).ravel()
        out.append(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(comb(2 * k, k)))
    return float(min(out))


# ---------------------------------------------------------------- block scan of one anchored complex
def step_mean_angle(fr, Ks=None):
    """area*Lambda weighted mean of arg p over all triangles of K(t_s), per step.  Under a rigid frame rotation every
    arg p shifts by the same -alpha_s, so (arg p - mean) is frame-indifferent."""
    S = fr.shape[0]; out = np.zeros(S - 1)
    for s in range(S - 1):
        tri = Ks[s] if Ks is not None else cplx(fr[s])["tri"]
        p, q = pq_of(map_F(fr[s], fr[s + 1], tri)); w = np.abs(tri_area(fr[s], tri)) * np.maximum(tri_lambda(fr[s], tri), 1e-3)
        out[s] = float((w * np.angle(p)).sum() / w.sum())
    return out


def scan_blocks(fr, a, tri, L_, mu_star, hold=2, record=False, Bmax=4, Jstrong=0.0, cancel=1.5, psi_bar=None):
    """steps a..a+L_-1 for the fixed triangles 'tri' (vertices at frame a).  Returns per-step arrays."""
    nT = len(tri)
    ang = np.zeros((L_, nT)); W = np.zeros((L_, nT)); openA = np.zeros((L_, nT), bool)
    rr = np.zeros((L_, nT)); CC = np.zeros((L_, nT)); area = np.zeros((L_ + 1, nT)); lam = np.zeros((L_ + 1, nT))
    hotA = np.zeros((L_, nT), bool); shockA = np.zeros((L_, nT), bool)
    area[0] = tri_area(fr[a], tri); lam[0] = tri_lambda(fr[a], tri)
    P = np.ones(nT, complex); Q = np.zeros(nT, complex); inb = np.zeros(nT, bool); quiet = np.zeros(nT, int)
    sarg = np.zeros(nT); sabs = np.zeros(nT); srel = np.zeros(nT); cc = np.zeros(nT); start = np.zeros(nT, int); sig = []
    for j in range(L_):
        k = a + j
        F = map_F(fr[k], fr[k + 1], tri); p, q = pq_of(F)
        area[j + 1] = tri_area(fr[k + 1], tri); lam[j + 1] = tri_lambda(fr[k + 1], tri)
        mu = np.abs(q / p); Jst = np.abs(p) ** 2 - np.abs(q) ** 2
        Jthr = getattr(mu_star, "J", 0.0)
        hot = (mu > mu_star) & (np.log(np.maximum(Jst, 1e-300)) < -Jthr); hotA[j] = hot
        cont = (hot | (inb & (quiet < hold))) & ~(inb & (k - start >= Bmax))
        closing = inb & ~cont
        if closing.any():
            ci = np.where(closing)[0]
            JB = np.abs(P[ci]) ** 2 - np.abs(Q[ci]) ** 2
            strong = (np.log(np.maximum(JB, 1e-300)) < -Jstrong) & (sabs[ci] > cancel * np.abs(srel[ci]) + 1e-12)
            cc[ci] += np.where(strong, wrap(np.angle(P[ci]) - sarg[ci]), 0.0)
            if record:
                Jb = np.abs(P[ci]) ** 2 - np.abs(Q[ci]) ** 2
                sig.append(np.c_[ci, start[ci], np.full(len(ci), k), np.log(np.maximum(Jb, 1e-300)), np.abs(Q[ci] / P[ci]), np.angle(P[ci])])
            P[ci] = 1; Q[ci] = 0; sarg[ci] = 0; sabs[ci] = 0; srel[ci] = 0; inb[ci] = False
        st = cont & ~inb
        start[st] = k; quiet[st] = 0
        ci = np.where(cont)[0]
        rel = np.angle(p[ci]) - (psi_bar[k] if psi_bar is not None else 0.0)
        P[ci], Q[ci] = compose(P[ci], Q[ci], p[ci], q[ci]); sarg[ci] += np.angle(p[ci]); sabs[ci] += np.abs(rel); srel[ci] += rel
        quiet[ci] = np.where(hot[ci], 0, quiet[ci] + 1); inb[ci] = True
        ang[j] = np.angle(p)
        ok = (area[j] > 0) & (area[j + 1] > 0)
        W[j] = np.where(ok, np.abs(area[j]) * np.maximum(lam[j], 1e-3), 0.0)
        JBo = np.abs(P) ** 2 - np.abs(Q) ** 2
        lJo = np.log(np.maximum(JBo, 1e-300))
        shockA[j] = inb & (lJo < -1e-3) & (np.abs(np.abs(Q / P) - np.tanh(0.5 * np.abs(lJo))) < 0.05)   # open block is uniaxial-like
        app = inb & (lJo < -Jstrong) & (sabs > cancel * np.abs(srel) + 1e-12)
        openA[j] = inb; rr[j] = np.where(app, wrap(np.angle(P) - sarg), 0.0); CC[j] = cc
    if record and inb.any():
        ci = np.where(inb)[0]; Jb = np.abs(P[ci]) ** 2 - np.abs(Q[ci]) ** 2
        sig.append(np.c_[ci, start[ci], np.full(len(ci), a + L_), np.log(np.maximum(Jb, 1e-300)), np.abs(Q[ci] / P[ci]), np.angle(P[ci])])
    return dict(ang=ang, W=W, openA=openA, rr=rr, CC=CC, area=area, lam=lam, hot=hotA, shock=shockA,
                sig=(np.concatenate(sig) if sig else np.zeros((0, 6))))


# ---------------------------------------------------------------- per-node chain
def run_chain(fr, mu_star, Rmin=2, Rmax=20, Dmax=8, hold=2, lam_gate=0.2, record_sig=False, cplx_cache=None, Bmax=4, Jstrong=0.0, cancel=1.5,
              psi_bar=None, Ks=None):
    S, N = fr.shape[:2]
    inc = np.zeros((S - 1, N)); Wn = np.zeros((S - 1, N)); lnJ = np.zeros((S, N)); has = np.zeros((S, N), bool)
    segs = [[] for _ in range(N)]                     # (start, end, rem)
    pending = {0: np.arange(N)}; base = np.zeros(N); cache = {} if cplx_cache is None else cplx_cache
    sigs = []; hot_cnt = 0; tot_cnt = 0; nseg = 0
    if psi_bar is None and np.isfinite(float(mu_star)):
        psi_bar = step_mean_angle(fr, Ks)
    while pending:
        a = min(pending); nodes = pending.pop(a)
        if a >= S - 1:
            continue
        if a not in cache:
            cache[a] = cplx(fr[a])["tri"]
        tri = cache[a]; nT = len(tri)
        L_ = min(Rmax + Dmax, S - 1 - a)
        B = scan_blocks(fr, a, tri, L_, mu_star, hold, record_sig, Bmax, Jstrong, cancel, psi_bar)
        hot_cnt += int(B["hot"].sum()); tot_cnt += B["hot"].size
        if record_sig and len(B["sig"]):
            sg = B["sig"].copy(); sigs.append((tri, sg))
        # incidence of group nodes to triangles
        rows = np.repeat(np.arange(nT), 3); cols = tri.ravel()
        Inc = sp.csr_matrix((np.ones(3 * nT), (cols, rows)), shape=(N, nT))[nodes]
        good0 = B["lam"][0] >= lam_gate
        fail = ((B["lam"][1:] < lam_gate) & good0[None, :] & ~B["shock"]) | (B["area"][1:] <= 0)   # shock-like compression is not degeneration
        anyfail = (Inc @ fail.T.astype(float)).T > 0          # (L_, nnodes): fail at frame a+j+1
        anyopen = (Inc @ B["openA"].T.astype(float)).T > 0    # open after step j
        jj = np.arange(1, L_ + 1)[:, None]                    # candidate segment length e (frames a..a+e)
        want = ((anyfail | (jj >= Rmax)) & (jj >= Rmin)) | (a + jj >= S - 1)
        gate_now = anyfail & (jj >= 1)                       # a star triangle degenerates: end now (open blocks closed with rr)
        okend = (want & ~anyopen) | gate_now
        okend[-1] = True                                      # forced at L_
        e = np.argmax(okend, 0) + 1                           # segment length per node
        # accumulate increments, weights, ledger
        w0 = B["W"][0]                                                         # anchor weights (segments end before any star inversion)
        valid = (B["area"][:-1] > 0) & (B["area"][1:] > 0)                     # (L_, nT): step j keeps orientation
        wv = valid * w0[None, :]
        num = (Inc @ (wv * B["ang"]).T).T; den = (Inc @ wv.T).T                 # (L_, nnodes)
        Astar = (Inc @ B["area"].T).T                                           # (L_+1, nnodes) signed
        for j in range(L_):
            m = e > j
            if not m.any():
                break
            idx = nodes[m]
            with np.errstate(all="ignore"):
                inc[a + j, idx] = np.where(den[j, m] > 0, num[j, m] / den[j, m], 0.0)
            Wn[a + j, idx] = den[j, m]
        for j in range(L_ + 1):
            m = e >= j
            idx = nodes[m]
            with np.errstate(all="ignore"):
                lnJ[a + j, idx] = base[idx] + np.log(np.maximum(Astar[j, m], 1e-300) / np.maximum(Astar[0, m], 1e-300))
            has[a + j, idx] = True
        # remainders at segment end: closed-block corrections + open remainder (weights at anchor)
        remT = B["CC"] + B["rr"]                                                # (L_, nT)
        remN = (Inc @ (w0[:, None] * remT.T)).T / np.maximum((Inc @ w0)[None, :], 1e-300)
        rem_end = remN[e - 1, np.arange(len(nodes))]
        for ii, n in enumerate(nodes):
            segs[n].append((a, a + e[ii], float(rem_end[ii])))
        nseg += len(nodes)
        base[nodes] = lnJ[a + e, nodes]
        nxt = a + e
        for v in np.unique(nxt):
            if v < S - 1:
                pending.setdefault(int(v), np.array([], int))
                pending[int(v)] = np.r_[pending[int(v)], nodes[nxt == v]]
    return dict(inc=inc, W=Wn, lnJ=lnJ, segs=segs, hot_frac=hot_cnt / max(tot_cnt, 1), nseg_mean=nseg / N,
                sigs=sigs, cache=cache)


def rotation_fields(ch, region=None, floor_eps=None):
    """phi (relative rotation, exact objectivity) and LAVD (segment-level |.|, minus noise floor)."""
    inc, W = ch["inc"], ch["W"]; S1, N = inc.shape
    reg = np.ones(N, bool) if region is None else region
    psibar = (W[:, reg] * inc[:, reg]).sum(1) / np.maximum(W[:, reg].sum(1), 1e-300)
    d = inc - psibar[:, None]
    phi = d.sum(0); lavd = np.zeros(N); nseg = np.zeros(N)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(d, 0)])
    for n in range(N):
        for (s0, s1, rem) in ch["segs"][n]:
            phi[n] += rem
            lavd[n] += 2 * abs(cs[s1, n] - cs[s0, n] + rem)
            nseg[n] += 1
    if floor_eps is not None:
        lavd = lavd - 2 * nseg * floor_eps * np.sqrt(2 / np.pi)
    return phi, lavd, nseg, psibar


def calib_node_noise(nrep=30, seed=0):
    """segment-angle noise of a node star for unit sigma/ell (identity motion, jittered lattice)."""
    rng = np.random.default_rng(seed); g = np.arange(0, 12); Xg, Yg = np.meshgrid(g, g)
    P = np.c_[Xg.ravel(), Yg.ravel()].astype(float) + rng.uniform(-0.25, 0.25, (Xg.size, 2))
    tri = cplx(P)["tri"]; N = len(P); sig = 0.01; vals = []
    c = np.hypot(P[:, 0] - 5.5, P[:, 1] - 5.5) < 3
    for _ in range(nrep):
        P0 = P + rng.normal(0, sig, P.shape); P1 = P + rng.normal(0, sig, P.shape)
        p, q = pq_of(map_F(P0, P1, tri)); w = np.abs(tri_area(P, tri)) * tri_lambda(P, tri)
        num = np.zeros(N); den = np.zeros(N)
        for k in range(3):
            np.add.at(num, tri[:, k], w * np.angle(p)); np.add.at(den, tri[:, k], w)
        vals.append((num / den)[c])
    return float(np.std(np.concatenate(vals)) / sig)         # per unit sigma/ell (ell = 1)


def local_spacing(P, tri, N):
    s = np.zeros(N); c = np.zeros(N)
    for j in range(3):
        l = np.linalg.norm(P[tri[:, (j + 1) % 3]] - P[tri[:, j]], axis=1)
        np.add.at(s, tri[:, j], l); np.add.at(c, tri[:, j], 1); np.add.at(s, tri[:, (j + 1) % 3], l); np.add.at(c, tri[:, (j + 1) % 3], 1)
    return s / np.maximum(c, 1)


# ---------------------------------------------------------------- old baseline: circulation per step, re-triangulated
def phi_circ_retri(fr, Ks, region=None):
    S, N = fr.shape[:2]; inc = np.zeros((S - 1, N)); W = np.zeros((S - 1, N))
    for s in range(S - 1):
        tri = Ks[s]; P, D = fr[s], fr[s + 1] - fr[s]
        a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]; da, db, dc = D[tri[:, 0]], D[tri[:, 1]], D[tri[:, 2]]
        circ = 0.5 * (((da + db) * (b - a)).sum(1) + ((db + dc) * (c - b)).sum(1) + ((dc + da) * (a - c)).sum(1))
        A = tri_area(P, tri); wdt = circ / np.where(np.abs(A) > 1e-300, A, 1e-300)
        num = np.zeros(N); den = np.zeros(N)
        for k in range(3):
            np.add.at(num, tri[:, k], wdt * np.abs(A)); np.add.at(den, tri[:, k], np.abs(A))
        inc[s] = 0.5 * np.where(den > 0, num / np.maximum(den, 1e-300), 0); W[s] = den
    reg = np.ones(N, bool) if region is None else region
    mean = (W[:, reg] * inc[:, reg]).sum(1) / W[:, reg].sum(1)
    return (inc - mean[:, None]).sum(0)


def phi_argp_retri(fr, Ks, region=None):
    """arg p per step on re-triangulated complexes (no anchoring, no merging)."""
    S, N = fr.shape[:2]; inc = np.zeros((S - 1, N)); W = np.zeros((S - 1, N))
    for s in range(S - 1):
        tri = Ks[s]; p, q = pq_of(map_F(fr[s], fr[s + 1], tri))
        w = np.abs(tri_area(fr[s], tri)) * np.maximum(tri_lambda(fr[s], tri), 1e-3)
        num = np.zeros(N); den = np.zeros(N)
        for k in range(3):
            np.add.at(num, tri[:, k], w * np.angle(p)); np.add.at(den, tri[:, k], w)
        inc[s] = np.where(den > 0, num / np.maximum(den, 1e-300), 0); W[s] = den
    reg = np.ones(N, bool) if region is None else region
    mean = (W[:, reg] * inc[:, reg]).sum(1) / W[:, reg].sum(1)
    return (inc - mean[:, None]).sum(0)


# ---------------------------------------------------------------- cells: agglomeration, circulation jump, non-affinity
def tri_adjacency(tri):
    e = np.sort(np.stack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], 1).reshape(-1, 2), 1)
    tid = np.repeat(np.arange(len(tri)), 3)
    key = e[:, 0].astype(np.int64) * (tri.max() + 1) + e[:, 1]
    o = np.argsort(key); ks = key[o]; same = np.where(ks[1:] == ks[:-1])[0]
    return np.c_[tid[o][same], tid[o][same + 1]]


def agglomerate(tri, P, m=2, seed=0):
    """greedy cells of ~m triangles: m=1 triangles, m=2 pairs (matching), m=4 pairs of pairs."""
    nT = len(tri)
    if m == 1:
        return np.arange(nT)
    adj = tri_adjacency(tri); rng = np.random.default_rng(seed)
    lab = -np.ones(nT, int); order = rng.permutation(len(adj)); nl = 0
    for k in order:
        i, j = adj[k]
        if lab[i] < 0 and lab[j] < 0:
            lab[i] = lab[j] = nl; nl += 1
    for i in np.where(lab < 0)[0]:
        lab[i] = nl; nl += 1
    if m == 2:
        return lab
    # pairs of pairs: cell adjacency
    ca = np.unique(np.sort(np.c_[lab[adj[:, 0]], lab[adj[:, 1]]], 1), axis=0); ca = ca[ca[:, 0] != ca[:, 1]]
    lab2 = -np.ones(nl, int); nl2 = 0
    for k in rng.permutation(len(ca)):
        i, j = ca[k]
        if lab2[i] < 0 and lab2[j] < 0:
            lab2[i] = lab2[j] = nl2; nl2 += 1
    for i in np.where(lab2 < 0)[0]:
        lab2[i] = nl2; nl2 += 1
    return lab2[lab]


def tri_disp_circ(fr, tri, s, dt):
    P, D = fr[s], fr[s + 1] - fr[s]
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]; da, db, dc = D[tri[:, 0]], D[tri[:, 1]], D[tri[:, 2]]
    return 0.5 * (((da + db) * (b - a)).sum(1) + ((db + dc) * (c - b)).sum(1) + ((dc + da) * (a - c)).sum(1)) / dt


def cell_deposit(fr, om, dt, a, b, m=1, block=None, seed=0):
    """circulation jump per unit initial area on material cells, chained over blocks; node maps (net, ref)."""
    S, N = fr.shape[:2]; blk = (b - a) if block is None else block
    net = np.zeros(N); ref = np.zeros(N); cov = np.zeros(N, bool); S0 = None
    k0 = a
    while k0 < b:
        k1 = min(k0 + blk, b); tri = cplx(fr[k0])["tri"]; lab = agglomerate(tri, fr[k0], m, seed)
        nc = lab.max() + 1
        A0 = tri_area(fr[k0], tri); A1 = tri_area(fr[k1], tri)
        Ac0 = np.bincount(lab, A0, nc); Ac1 = np.bincount(lab, A1, nc)
        def Gam(q):
            est = [np.bincount(lab, tri_disp_circ(fr, tri, s, dt), nc) for s in (q - 1, q, q + 1)]
            return np.median(est, 0)
        G0, G1 = Gam(k0), Gam(k1)
        Gr0 = np.bincount(lab, A0 * om[k0][tri].mean(1), nc); Gr1 = np.bincount(lab, A1 * om[k1][tri].mean(1), nc)
        keep = (np.sign(Ac1) == np.sign(Ac0)) & (np.abs(Ac0) > 0)
        dG = np.where(keep, (G1 - G0) / np.where(keep, Ac0, 1), 0); dGr = np.where(keep, (Gr1 - Gr0) / np.where(keep, Ac0, 1), 0)
        # map to nodes (area weighted), carried to initial area by star-area ratio of block anchor
        num = np.zeros(N); numr = np.zeros(N); den = np.zeros(N)
        wt = np.abs(A0) * keep[lab]
        for j in range(3):
            np.add.at(num, tri[:, j], wt * dG[lab]); np.add.at(numr, tri[:, j], wt * dGr[lab]); np.add.at(den, tri[:, j], wt)
        st = X._star(fr[k0], tri, N)
        if S0 is None:
            S0 = st.copy()
        Jk = st / np.maximum(S0, 1e-300)
        ok = den > 0
        net[ok] += num[ok] / den[ok] * Jk[ok]; ref[ok] += numr[ok] / den[ok] * Jk[ok]; cov |= ok
        k0 = k1
    net[~cov] = np.nan; ref[~cov] = np.nan
    return net, ref


def cell_nonaffinity(fr, a, b, m=2, seed=0):
    """per cell (>=4 vertices) rms residual of the best affine map a->b, divided by the cell diameter at b."""
    tri = cplx(fr[a])["tri"]; lab = agglomerate(tri, fr[a], m, seed); nc = lab.max() + 1
    verts = [[] for _ in range(nc)]
    for t, l in enumerate(lab):
        verts[l].extend(tri[t])
    eps = np.full(nc, np.nan); cen = np.zeros((nc, 2))
    for l in range(nc):
        v = np.unique(verts[l]); cen[l] = fr[a][v].mean(0)
        if len(v) < 4:
            continue
        A0 = np.c_[fr[a][v], np.ones(len(v))]; Y = fr[b][v]
        coef, *_ = np.linalg.lstsq(A0, Y, rcond=None); res = Y - A0 @ coef
        diam = np.max(np.linalg.norm(Y[:, None] - Y[None], axis=-1))
        eps[l] = np.sqrt((res ** 2).sum(1).mean()) / max(diam, 1e-300)
    return dict(tri=tri, lab=lab, eps=eps, cen=cen)


# ---------------------------------------------------------------- Hodge with general SPD metrics (dense, subdomain)
def whitney_mass(P, tri, edges):
    """Whitney 1-form mass matrix on triangles (edges oriented low->high vertex index)."""
    from lhdx import _grad_lambda
    G, A = _grad_lambda(P, tri)
    key = {(int(min(u, v)), int(max(u, v))): k for k, (u, v) in enumerate(edges)}
    nE = len(edges); Mm = np.zeros((nE, nE))
    loc = [(0, 1), (1, 2), (2, 0)]
    for t in range(len(tri)):
        idx = []; sgn = []
        for (i, j) in loc:
            u, v = tri[t, i], tri[t, j]
            idx.append(key[(min(u, v), max(u, v))]); sgn.append(1 if u < v else -1)
        lam = lambda i, j: A[t] * (1 + (i == j)) / 12
        for e1, (i1, j1) in enumerate(loc):
            for e2, (i2, j2) in enumerate(loc):
                # W_e = l_i grad l_j - l_j grad l_i
                g = lambda x, y: float(G[t, x] @ G[t, y])
                val = (g(j1, j2) * lam(i1, i2) - g(j1, i2) * lam(i1, j2) - g(i1, j2) * lam(j1, i2) + g(i1, i2) * lam(j1, j2))
                Mm[idx[e1], idx[e2]] += sgn[e1] * sgn[e2] * val
    return Mm


def boundary_mats(tri, N):
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    edges = np.unique(e, axis=0); key = {(int(u), int(v)): k for k, (u, v) in enumerate(edges)}
    B1 = np.zeros((N, len(edges)))
    B1[edges[:, 0], np.arange(len(edges))] = -1; B1[edges[:, 1], np.arange(len(edges))] = 1
    B2 = np.zeros((len(edges), len(tri)))
    for t in range(len(tri)):
        for (i, j) in [(0, 1), (1, 2), (2, 0)]:
            u, v = tri[t, i], tri[t, j]; B2[key[(min(u, v), max(u, v))], t] = 1 if u < v else -1
    return edges, B1, B2


def hodge_dense(f, B1, B2, M0, M1, M2):
    """f = d0 a + delta2 b + h under <,>_{M1}; d0 = B1^T, d1 = B2^T, delta2 = M1^{-1} d1^T M2."""
    d0, d1 = B1.T, B2.T
    K0 = d0.T @ M1 @ d0; a = np.linalg.lstsq(K0, d0.T @ M1 @ f, rcond=None)[0]; g = d0 @ a
    M1i = np.linalg.inv(M1); K2 = d1 @ M1i @ d1.T @ M2
    b = np.linalg.lstsq(K2, d1 @ f, rcond=None)[0]; c = M1i @ d1.T @ M2 @ b
    h = f - g - c
    ip = lambda x, y: float(x @ M1 @ y)
    E = ip(f, f)
    return dict(Eg=ip(g, g) / E, Ec=ip(c, c) / E, Eh=ip(h, h) / E,
                orth=max(abs(ip(g, c)), abs(ip(g, h)), abs(ip(c, h))) / E,
                dh=float(np.linalg.norm(d1 @ h)), cdh=float(np.linalg.norm(d0.T @ M1 @ h)))


def harmonic_dim(B1, B2, M1, tol=1e-9):
    """dim ker L1 = dim {f: d1 f = 0, d0^T M1 f = 0}."""
    A = np.vstack([B2.T, B1 @ M1]); s = np.linalg.svd(A, compute_uv=False)
    return int(A.shape[1] - (s > tol * s.max()).sum())


# ---------------------------------------------------------------- persistence (0-dim, superlevel) and convex boundaries
def graph_edges(tri):
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    return np.unique(e, axis=0)


def persistence0(f, edges, mask=None):
    N = len(f); mask = np.ones(N, bool) if mask is None else mask
    nb = [[] for _ in range(N)]
    for u, v in edges:
        if mask[u] and mask[v]:
            nb[u].append(v); nb[v].append(u)
    order = np.argsort(-f); parent = -np.ones(N, int); birth = np.zeros(N); pairs = []
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in order:
        if not mask[i]:
            continue
        parent[i] = i; birth[i] = f[i]
        roots = {find(j) for j in nb[i] if parent[j] >= 0}
        if not roots:
            continue
        roots = sorted(roots, key=lambda r: -birth[r])
        top = roots[0]; parent[i] = top
        for r in roots[1:]:
            pairs.append((birth[r], f[i], r)); parent[r] = top
    roots = {find(i) for i in range(N) if mask[i] and parent[i] >= 0}
    for r in roots:
        pairs.append((birth[r], -np.inf, r))
    return pairs                                   # (birth, death, root node)


def roughness(f, edges, mask):
    N = len(f); s = np.zeros(N); c = np.zeros(N)
    np.add.at(s, edges[:, 0], f[edges[:, 1]]); np.add.at(c, edges[:, 0], 1)
    np.add.at(s, edges[:, 1], f[edges[:, 0]]); np.add.at(c, edges[:, 1], 1)
    r = f - s / np.maximum(c, 1)
    r = r[mask & (c > 0)]
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def cores(f_abs, edges, mask, thr):
    pr = persistence0(f_abs, edges, mask)
    return [(b, d, r) for (b, d, r) in pr if (b - (d if np.isfinite(d) else 0.0)) > thr and b > thr]


def convex_boundary(P0, tri, fs, core, dstar=1e-2, nlev=60):
    """outermost closed level curve of the signed field fs around node 'core' with convexity deficiency <= dstar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.tri as mtri
    from matplotlib.path import Path
    from scipy.spatial import ConvexHull
    T = mtri.Triangulation(P0[:, 0], P0[:, 1], tri)
    fmax = fs[core]; levels = np.linspace(fmax * 0.98, fmax * 0.02, nlev)
    from matplotlib import _tri
    gen = _tri.TriContourGenerator(T.get_cpp_triangulation(), fs)
    best = None; pc = P0[core]; started = False
    for lev in levels:
        segs, codes = gen.create_contour(lev)
        found = None
        for sgm, cd in zip(segs, codes):
            closed = (len(cd) and cd[-1] == 79) or np.linalg.norm(sgm[0] - sgm[-1]) <= 1e-9 * (1 + np.abs(sgm).max())
            if len(sgm) < 6 or not closed:
                continue
            path = Path(sgm)
            if path.contains_point(pc):
                found = sgm; break
        if found is None:
            if started:
                break
            continue
        started = True
        x, y = found[:, 0], found[:, 1]; A = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        try:
            Ah = ConvexHull(found).volume
        except Exception:
            break
        if A <= 0:
            break
        if (Ah - A) / A <= dstar:
            best = (lev, found, A)
    if best is None:
        return None
    inside = Path(best[1]).contains_points(P0)
    return dict(level=best[0], area=best[2], inside=inside)


# ---------------------------------------------------------------- 3-point chained tangential stretch at chosen nodes
def chain_stretch(fr, ch, nodes, T0):
    """ln lambda along node segments with the star-averaged map; tangent transported by F."""
    out = np.zeros(len(nodes)); T = T0[nodes].astype(float).copy(); cache = ch["cache"]
    pos = {n: i for i, n in enumerate(nodes)}
    for i, n in enumerate(nodes):
        for (s0, s1, _) in ch["segs"][n]:
            tri = cache[s0]; st = np.where((tri == n).any(1))[0]
            if len(st) == 0:
                continue
            F = map_F(fr[s0], fr[s1], tri[st]); w = np.abs(tri_area(fr[s0], tri[st])) * np.maximum(tri_lambda(fr[s0], tri[st]), 1e-3)
            Fm = (w[:, None, None] * F).sum(0) / w.sum()
            v = Fm @ T[i]; nv = np.linalg.norm(v); out[i] += np.log(nv); T[i] = v / nv
    return out
