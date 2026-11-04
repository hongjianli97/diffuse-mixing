"""trajnet.py -- one Lagrangian trajectory network, two activations.

G = (V, E): V = tracers; E = union over ALL snapshots of { mutual-kNN(k) pairs } U { Delaunay edges of K(t) }
           (the paper's Eq. (2) with S = all snapshots and the Delaunay rule added), so every instantaneous
           complex K(t) (outside-peeled Delaunay) is a subcomplex of G.  Every edge is a material pair with a full history.
E_afrc   = pairs that are mutual-kNN at the first or last snapshot (the paper's AFRC graph, S = {first, last}).

dynamic activation  (AFRC, on E_afrc, weights from pair histories):
    robust_stretch()  barrier intensity = node mean of |median censored d ln|r_e||/dt over E_afrc   (no gates, no curvature)
    curvature_topology()  augmented Forman curvature on gated robust-slope weights (bottlenecks / saddles), paper engine
geometric activation (LHD, on K(t) subset of G, Galerkin metric):
    lhd()             Stokes vorticity (triangle circulation), Galerkin divergence, Hodge split, pair attribution s_g/s_c
                      on E_afrc pairs, material lnJ (per-tracer anchors, shock-deferred)
shared:
    events()          position-only shock detection (C channel: one-step material star-area ratio)
    rotation()        Stokes relative rotation |phi| = |1/2 int (omega - <omega>) dt| (censored, gaps interpolated)
    vortex_dome()     paper dome (watershed + screened Poisson) on the network graph, driven by |phi|, absolute gate pi/4
    kelvin()          circulation of a material region over time on K(t)
    turnover()        neighbour exchange: changes of K(t)-activity per incident network edge per unit time
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import lhd as L, lhdx as X, afrc_h_engine as AH, paper_ell as PE


class TrajectoryNetwork:
    def __init__(self, frames, dt, k=12):
        self.fr = np.asarray(frames, float); self.dt = float(dt); S, N = self.fr.shape[:2]; self.S, self.N = S, N
        self.V = np.gradient(self.fr, dt, axis=0)
        self.K = [X.complex_peeled(self.fr[s]) for s in range(S)]          # geometric activation K(t)
        keys = [np.minimum(c['edges'][:, 0], c['edges'][:, 1]) * N + np.maximum(c['edges'][:, 0], c['edges'][:, 1]) for c in self.K]
        knn = []
        for s in range(S):
            e = AH._mutual_knn(self.fr[s], k); knn.append(e[:, 0] * N + e[:, 1])
        self.keys = np.unique(np.concatenate(keys + knn))                     # G
        self.E = np.c_[self.keys // N, self.keys % N]
        end = np.unique(np.concatenate([knn[0], knn[-1]]))
        self.afrc_mask = np.isin(self.keys, end); self.E_afrc = self.E[self.afrc_mask]
        self.kidx = [np.searchsorted(self.keys, kk) for kk in keys]          # local K(t) edge -> global edge index
        self.k = k

    # ------------------------------------------------------------------ shared
    def events(self, tau=0.2):
        S, N = self.S, self.N; dl = np.zeros((S - 1, N))
        for s in range(S - 1):
            tri = self.K[s]['tri']; A0, A1 = X._star(self.fr[s], tri, N), X._star(self.fr[s + 1], tri, N)
            with np.errstate(all='ignore'): dl[s] = np.where(A0 > 0, np.log(A1 / A0), 0.0)
        det = dl < -tau; det[:-1] |= (dl[:-1] + dl[1:]) < -tau
        cs = det.T.copy(); cs[:, 1:] |= det.T[:, :-1]; cs[:, :-1] |= det.T[:, 1:]
        bad = np.zeros((N, S), bool); bad[:, :-1] |= cs; bad[:, 1:] |= cs
        self.censor, self.bad = cs, bad
        return cs

    def turnover(self):
        act = np.zeros((len(self.keys), self.S), bool)
        for s in range(self.S): act[self.kidx[s], s] = True
        ch = np.abs(np.diff(act.astype(np.int8), axis=1)).sum(1).astype(float)
        acc = np.zeros(self.N); cnt = np.zeros(self.N)
        for j in range(2): np.add.at(acc, self.E[:, j], ch); np.add.at(cnt, self.E[:, j], 1)
        return acc / np.maximum(cnt, 1) / (self.dt * (self.S - 1))

    # ------------------------------------------------------------------ geometric activation (LHD)
    def lhd(self, hodge_every=1, attribution=True):
        S, N = self.S, self.N; bad = getattr(self, 'bad', None)
        out = dict(vort=np.full((S, N), np.nan), vort_c=np.full((S, N), np.nan), div=np.full((S, N), np.nan),
                   Efrac=np.full((S, 3), np.nan), wbar=np.zeros(S))
        if attribution:
            out['s_g'] = np.full((S, len(self.E_afrc)), np.nan, np.float32); out['s_c'] = np.full((S, len(self.E_afrc)), np.nan, np.float32)
        for s in range(S):
            P, U, cx = self.fr[s], self.V[s], self.K[s]
            f = L.edge_cochain(P, U, cx['edges']); A = L.tri_signed_area(P, cx['tri'])
            out['wbar'][s] = np.sum(L.circulation(f, cx)) / np.sum(A)
            out['vort'][s] = L.node_vorticity(P, cx, f)
            if bad is not None:
                ok = ~bad[cx['tri'], s].any(1); out['vort_c'][s] = L.node_vorticity(P, cx, f, tri_ok=ok.astype(float))
            dv = X.galerkin_div(P, cx, U); dv[X.open_star_nodes(cx)] = np.nan; out['div'][s] = dv
            if s % hodge_every == 0:
                g = X.galerkin_hodge(P, cx, U); out['Efrac'][s] = [g['Eg'], g['Ec'], g['Eh']]
                if attribution:
                    out['s_g'][s] = L.stretch_rate(P, X.face_to_node(P, cx, g['ug']), self.E_afrc)
                    out['s_c'][s] = L.stretch_rate(P, X.face_to_node(P, cx, g['uc']), self.E_afrc)
        out['lnJ'] = X.chained_lnJ_v2(self.fr, bad)
        self.L_ = out
        return out

    def rotation(self):
        o = self.L_; vc = o['vort_c'] if getattr(self, 'bad', None) is not None else o['vort']
        W = np.array([X._star(self.fr[s], self.K[s]['tri'], self.N) for s in range(self.S)])
        wb = np.array([np.nansum(vc[s] * W[s]) / np.nansum(W[s] * np.isfinite(vc[s])) for s in range(self.S)])
        dev = vc - wb[:, None]; ti = np.arange(self.S)
        for i in np.where(~np.isfinite(dev).all(0))[0]:
            ok = np.isfinite(dev[:, i])
            if ok.sum() >= 2: dev[~ok, i] = np.interp(ti[~ok], ti[ok], dev[ok, i])
        dev = np.nan_to_num(dev)
        self.phi = 0.5 * np.trapezoid(dev, dx=self.dt, axis=0)             # signed relative rotation (rad)
        self.lavd_stokes = np.trapezoid(np.abs(dev), dx=self.dt, axis=0)
        return np.abs(self.phi)

    # ------------------------------------------------------------------ dynamic activation (AFRC)
    def _pair_median_rate(self, E):
        traj = np.ascontiguousarray(np.transpose(self.fr, (1, 0, 2)))
        AH._FORMAN_CENSOR_CURRENT = getattr(self, 'censor', None)
        AH._GRID_CELL = 0.5 * float(np.median(np.linalg.norm(self.fr[0][E[:, 0]] - self.fr[0][E[:, 1]], axis=1)))
        try:
            net, TV, med, J, ss, T, neff = AH._edge_logdist_stats_v2(traj, E)
        finally:
            AH._FORMAN_CENSOR_CURRENT = None
        return np.abs(np.nan_to_num(med)) / self.dt

    def robust_stretch(self):
        w = self._pair_median_rate(self.E_afrc); acc = np.zeros(self.N); cnt = np.zeros(self.N)
        for j in range(2): np.add.at(acc, self.E_afrc[:, j], w); np.add.at(cnt, self.E_afrc[:, j], 1)
        self.stretch = acc / np.maximum(cnt, 1)
        return self.stretch

    def curvature_topology(self):
        traj = np.ascontiguousarray(np.transpose(self.fr, (1, 0, 2)))
        AH._FORMAN_CENSOR_CURRENT = getattr(self, 'censor', None)
        try:
            keep = AH.coherent_edge_keep(traj, self.E_afrc)
        finally:
            AH._FORMAN_CENSOR_CURRENT = None
        Ek = self.E_afrc[keep]; w = np.maximum(self._pair_median_rate(Ek), 1e-12)
        B = AH._forman_node_from_weights(self.N, Ek, w, AH._node_weight(self.N, Ek, w))
        B[np.bincount(Ek.ravel(), minlength=self.N) < 3] = 0.0
        self.bottleneck = B
        return B

    def vortex_dome(self, merge, gate=np.pi / 4, kap_fac=0.2):
        Phi = np.abs(self.phi); N = self.N; P0 = self.fr[0]; supp = Phi >= gate
        self.dome, self.cores = np.zeros(N), np.zeros((0, 2))
        if supp.sum() < 10: return self.dome, self.cores
        grL = dict(u=self.E_afrc[:, 0], v=self.E_afrc[:, 1], N=N, nE=len(self.E_afrc)); adj = PE._adj_list(grL); Pn = Phi / Phi.max()
        mk = PE.objrot_markers(Pn, grL, supp, P0, strength=0.30, merge_dist=merge, interior=None)
        lab = PE._fill_holes(PE.watershed_supp(Pn, adj, mk, supp), adj, iters=8, frac=0.65)
        u, v = grL['u'], grL['v']; A = sp.coo_matrix((np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])), shape=(N, N)).tocsr()
        deg = np.asarray(A.sum(1)).ravel(); Lap = sp.diags(deg) - A; kap2 = kap_fac * np.median(deg[deg > 0]); cores = []
        for c in range(len(mk)):
            iv = np.where(lab == c)[0]
            if iv.size < 10: continue
            sol = np.maximum(spsolve((Lap[iv][:, iv] + kap2 * sp.eye(iv.size)).tocsc(), Pn[iv]), 0)
            if sol.max() > 0: self.dome[iv] = sol / sol.max(); cores.append(P0[iv[np.argmax(sol)]])
        self.cores = np.array(cores).reshape(-1, 2); self.labels = lab
        return self.dome, self.cores

    def kelvin(self, mask):
        out = np.zeros((self.S, 2))
        for s in range(self.S):
            P, cx = self.fr[s], self.K[s]; f = L.edge_cochain(P, self.V[s], cx['edges'])
            circ = L.circulation(f, cx); inreg = mask[cx['tri']].all(1)
            out[s] = [circ[inreg].sum(), float(inreg.sum())]
        return out
