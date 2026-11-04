"""ltn_dyn.py -- the kinematics-dynamics channels, built on ONE Lagrangian trajectory network (trajnet.TrajectoryNetwork).

Base object  G = (V, E_G):  V = trajectories x_i(t)  (+ one mass m_i, fixed at the reference time)
                            E_G = union over all snapshots of {mutual-kNN pairs} U {Delaunay edges of K(t)}  (material pairs)
Activations  geometric:  K(t) = outside-peeled Delaunay complex  (a subcomplex of G: every active edge has a global id)
             dynamic:    E_afrc = pairs that are mutual-kNN at the window ends (AFRC); edge signal = pair stretch rate
Cells        0-cells: trajectories; 1-cells: material pairs (history r_e(t)); 2-cells: material triples active in K(t)
             (three-point maps F_T over [t-dt, t+dt]).  Optional symmetric extension G U sigma(G) (mirror images).
Every operator below is a function of (trajectories, activation, reference density) only; DNS fields are never read here.
"""
import numpy as np, scipy.sparse as sp, sys
from scipy.sparse.linalg import spsolve
sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd'); sys.path.insert(0, '/home/claude/w')
import trajnet as TN, lhdx as X, netmix as NM


def logm2(F):
    a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
    tr = a + d; det = a * d - b * c; disc = np.sqrt((tr * tr / 4 - det).astype(complex))
    l1, l2 = tr / 2 + disc, tr / 2 - disc; same = np.abs(l1 - l2) < 1e-12 * np.abs(l1)
    beta = np.where(same, 1 / l1, (np.log(l1) - np.log(l2)) / np.where(same, 1, l1 - l2)); alpha = np.log(l1) - beta * l1
    return (beta[:, None, None] * F + alpha[:, None, None] * np.eye(2)[None]).real


class LTN(TN.TrajectoryNetwork):
    def __init__(self, frames, dt, rho0, k=12, mirror_y=False, mirror_band=None):
        fr = np.asarray(frames, float); self.N0 = fr.shape[1]; rho0 = np.asarray(rho0, float)
        if mirror_y: fr = np.concatenate([fr, fr * np.array([1, -1])], axis=1); rho0 = np.r_[rho0, rho0]
        if mirror_band is not None:      # symmetric images across y = 0 and y = mirror_band (two walls / symmetry lines)
            fr = np.concatenate([fr, fr * np.array([1, -1]), np.stack([fr[..., 0], 2 * mirror_band - fr[..., 1]], -1)], axis=1); rho0 = np.r_[rho0, rho0, rho0]
        super().__init__(fr, dt, k=k)
        self.events(); self.rho0 = rho0
        self.lnJ = X.chained_lnJ_v2(self.fr, self.bad)                      # material ledger (S, N)
        self.m = rho0 * X._star(self.fr[0], self.K[0]['tri'], self.N) / 3.0  # node masses (0-cell measure)
        self.act = np.zeros((len(self.keys), self.S), bool)
        for s in range(self.S): self.act[self.kidx[s], s] = True            # activation of every G edge

    # ---------------- the active complex K(s) expressed on G
    def active(self, s):
        P = self.fr[s]; tri = self.K[s]['tri'].copy(); N = self.N
        a = 0.5 * ((P[tri[:, 1], 0] - P[tri[:, 0], 0]) * (P[tri[:, 2], 1] - P[tri[:, 0], 1]) - (P[tri[:, 1], 1] - P[tri[:, 0], 1]) * (P[tri[:, 2], 0] - P[tri[:, 0], 0]))
        tri[a < 0] = tri[a < 0][:, [0, 2, 1]]; A = np.abs(a)
        E, w = NM.edge_weights(P, tri); gid = np.searchsorted(self.keys, E[:, 0] * N + E[:, 1])
        assert np.all(self.keys[gid] == E[:, 0] * N + E[:, 1]), 'K(s) is not a subcomplex of G'
        nE = len(E); kk = E[:, 0] * N + E[:, 1]; order = np.argsort(kk); ks = kk[order]; rows, cols, vals = [], [], []
        for (p_, q_) in ((0, 1), (1, 2), (2, 0)):
            i, j = tri[:, p_], tri[:, q_]; k = np.minimum(i, j) * N + np.maximum(i, j)
            rows.append(order[np.searchsorted(ks, k)]); cols.append(np.arange(len(tri))); vals.append(np.where(i < j, 1.0, -1.0))
        B2 = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nE, len(tri)))
        B1 = sp.csr_matrix((np.r_[-np.ones(nE), np.ones(nE)], (np.r_[E[:, 0], E[:, 1]], np.r_[np.arange(nE), np.arange(nE)])), shape=(N, nE))
        return dict(P=P, tri=tri, A=A, E=E, w=w, gid=gid, B1=B1, B2=B2, Ast=X._star(P, tri, N) / 3.0)

    def cochain(self, c, V):
        P, E = c['P'], c['E']; return 0.5 * ((V[E[:, 0]] + V[E[:, 1]]) * (P[E[:, 1]] - P[E[:, 0]])).sum(1)

    def node_kin(self, s):
        dt = self.dt; fr = self.fr
        return (fr[s + 1] - fr[s - 1]) / (2 * dt), (fr[s + 1] - 2 * fr[s] + fr[s - 1]) / dt ** 2

    def rho(self, s): return self.rho0 * np.exp(-self.lnJ[s])

    def hodge(self, c, x, metric=None):
        w = c['w'] if metric is None else metric; B1, B2 = c['B1'], c['B2']
        Wi = sp.diags(1 / np.maximum(w, 0.02 * np.median(w)))
        L0 = (B1 @ sp.diags(w) @ B1.T).tocsc() + sp.diags(1e-12 * np.ones(self.N))
        phi = spsolve(L0, B1 @ (w * x)); xg = B1.T @ phi
        chi = spsolve((B2.T @ Wi @ B2).tocsc(), B2.T @ x); xc = Wi @ (B2 @ chi)
        return dict(phi=phi, chi=chi, g=xg, c=xc, h=x - xg - xc)

    # ---------------- channels
    def channels(self, s, mu=None, probe=None, p_ref=None):
        c = self.active(s); V, Acc = self.node_kin(s); rho = self.rho(s)
        ue, ae, fe = self.cochain(c, V), self.cochain(c, Acc), self.cochain(c, rho[:, None] * Acc)
        Hu, Hf = self.hodge(c, ue), self.hodge(c, fe); ip = lambda x, y: float((c['w'] * x * y).sum())
        Gam = c['B2'].T @ ue; enst = float((Gam ** 2 / c['A']).sum())
        out = dict(c=c, ue=ue, ae=ae, fe=fe, Hu=Hu, Hf=Hf, enst=enst, Gam=Gam,
                   torque=(c['B2'].T @ ae) / c['A'],                            # Kelvin: dGamma/dt per material face / area
                   mu_E=-ip(Hf['c'], Hu['c']) / enst,                           # viscosity from the vortical-channel power
                   P_grad=ip(Hf['g'], Hu['g']), P_curl=ip(Hf['c'], Hu['c']))
        p = -Hf['phi']
        if probe is not None: p = p + (p_ref - p[probe].mean())             # one reference value
        out['p'] = p
        # material velocity gradient on every active face: three-point map over [s-1, s+1]
        tri = c['tri']; Dm = lambda Q: np.stack([Q[tri[:, 1]] - Q[tri[:, 0]], Q[tri[:, 2]] - Q[tri[:, 0]]], axis=2)
        Lg = logm2(Dm(self.fr[s + 1]) @ np.linalg.inv(Dm(self.fr[s - 1]))) / (2 * self.dt)
        th = Lg[:, 0, 0] + Lg[:, 1, 1]; Ss = 0.5 * (Lg + Lg.transpose(0, 2, 1))
        out['Lg'] = Lg; out['phi_face'] = 2 * (Ss ** 2).sum((1, 2)) - 2 / 3 * th ** 2   # Phi / mu
        out['theta_face'] = th
        return out

    def pair_strain(self, s, rmax):
        """dynamic-activation route: strain rate fitted per node from pair stretch rates of all G edges shorter than
        rmax at time s:  d ln|r_e|/dt = r^T S r  (rotation does not change |r|)."""
        P, Pm, Pp = self.fr[s], self.fr[s - 1], self.fr[s + 1]; E = self.E
        r = P[E[:, 1]] - P[E[:, 0]]; L = np.linalg.norm(r, axis=1); use = L < rmax; E, r, L = E[use], r[use], L[use]
        eps = (np.log(np.linalg.norm(Pp[E[:, 1]] - Pp[E[:, 0]], axis=1)) - np.log(np.linalg.norm(Pm[E[:, 1]] - Pm[E[:, 0]], axis=1))) / (2 * self.dt)
        cth, sth = r[:, 0] / L, r[:, 1] / L; ph = np.c_[cth ** 2, 2 * cth * sth, sth ** 2]
        M = np.zeros((self.N, 3, 3)); b = np.zeros((self.N, 3))
        for j in range(2):
            np.add.at(M, E[:, j], ph[:, :, None] * ph[:, None, :]); np.add.at(b, E[:, j], ph * eps[:, None])
        ok = np.linalg.det(M) > 1e-6
        Sv = np.full((self.N, 3), np.nan); Sv[ok] = np.linalg.solve(M[ok], b[ok][:, :, None])[:, :, 0]
        th = Sv[:, 0] + Sv[:, 2]; SS = Sv[:, 0] ** 2 + 2 * Sv[:, 1] ** 2 + Sv[:, 2] ** 2
        return dict(S=Sv, theta=th, phi=2 * SS - 2 / 3 * th ** 2, n_edges=int(use.sum()))

    def edge_ages(self, s):
        """LTN-native activation statistics: for edges active at s, frames since activation; births since s-1."""
        a = self.act; born = a[:, s] & ~a[:, s - 1]; died = ~a[:, s] & a[:, s - 1]
        age = np.zeros(len(self.keys), int); run = np.zeros(len(self.keys), int)
        for q in range(s + 1): run = np.where(a[:, q], run + 1, 0)
        return dict(born=born, died=died, age=run)
