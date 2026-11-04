"""netmix.py -- network-native mixing quantities on a Lagrangian trajectory network.

Graph at frame s: outside-peeled Delaunay complex K(t_s) of the tracers (subcomplex of the LTN G).
Node weight  : m_i = rho_0 * A*_i/3 at the reference time (mass; the invariant measure).
Edge weight  : w_ij(t) = Gamma * sum of half-cotangents (P1 conductance, clipped >= 0 -> M-matrix).
Generator    : A(t) = M^{-1} L(t), L = graph Laplacian of w   (exactly the P1 / material-coordinate discretisation of
               rho Dc/Dt = div(Gamma grad c));  diffusion = continuous-time random walk with stationary law m.
Everything is computed from positions + one-time masses (+ the closure Gamma).
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu
import sys
sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import lhdx as X


def half_cot(P, tri):
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    def cot(u, v): return (u * v).sum(1) / np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    return 0.5 * np.stack([cot(a - c, b - c), cot(b - a, c - a), cot(c - b, a - b)], 1)


def edge_weights(P, tri):
    """unique edges (i<j) and summed half-cot weights clipped at 0 (interior Delaunay edges are >= 0 anyway)."""
    N = len(P); W = half_cot(P, tri); E = ((0, 1), (1, 2), (2, 0))
    I = np.concatenate([tri[:, i] for i, j in E]); J = np.concatenate([tri[:, j] for i, j in E])
    w = np.concatenate([W[:, k] for k in range(3)])
    key = np.minimum(I, J) * N + np.maximum(I, J); u, inv = np.unique(key, return_inverse=True)
    we = np.maximum(np.bincount(inv, weights=w), 0.0)
    return np.c_[u // N, u % N], we


def laplacian(N, E, w):
    W = sp.csr_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(N, N))
    return (sp.diags(np.asarray(W.sum(1)).ravel()) - W).tocsr()


class Net:
    def __init__(self, frames, dt, rho0):
        self.fr = frames; self.dt = dt; self.S, self.N = frames.shape[:2]
        self.tri, self.E, self.w = [], [], []
        for s in range(self.S):
            tri = X.complex_peeled(frames[s])['tri']; E, w = edge_weights(frames[s], tri)
            self.tri.append(tri); self.E.append(E); self.w.append(w)
        self.m = rho0 * X._star(frames[0], self.tri[0], self.N) / 3.0
        iso = self.m <= 0; self.m[iso] = np.median(self.m[~iso])      # isolated tracers (no triangle)
        self.L = [laplacian(self.N, self.E[s], self.w[s]) for s in range(self.S)]

    # -------------------------------------------------------------- diffusion (virtual scalar), averaged-K implicit step
    def diffuse(self, c0, Gamma, frozen=False, keep=None):
        c = np.asarray(c0, float).copy(); out = {0: c.copy()}; M = sp.diags(self.m)
        for s in range(self.S - 1):
            Lb = self.L[0] if frozen else 0.5 * (self.L[s] + self.L[s + 1])
            c = spsolve((M + self.dt * Gamma * Lb).tocsc(), self.m * c)
            if keep is None or (s + 1) in keep: out[s + 1] = c.copy()
        return out

    # -------------------------------------------------------------- contact layer
    def cut(self, s, lab):
        E, w = self.E[s], self.w[s]; P = self.fr[s]
        x = lab[E[:, 0]] != lab[E[:, 1]]
        le = np.linalg.norm(P[E[x, 0]] - P[E[x, 1]], axis=1)
        return dict(Lnet=float((w[x] * le).sum()),          # Voronoi interface length (exact for Delaunay)
                    cutw=float(w[x].sum()),                   # = lab^T L lab  (sharp Dirichlet energy, h-dependent)
                    ncut=int(x.sum()), nE=len(E))

    def dirichlet(self, s, c):
        return float(c @ (self.L[s] @ c))

    def node_dirichlet(self, s, c):
        """per-node exchange-rate density q_i = sum_j w_ij (c_i-c_j)^2 / (2 m_i)   (sum_i m_i q_i = c^T L c)."""
        E, w = self.E[s], self.w[s]; d = w * (c[E[:, 0]] - c[E[:, 1]]) ** 2
        q = np.zeros(self.N); np.add.at(q, E[:, 0], 0.5 * d); np.add.at(q, E[:, 1], 0.5 * d)
        return q / self.m

    # -------------------------------------------------------------- time scales
    def Lbar(self, s0, s1):
        Lb = self.L[s0] * 0.5 + self.L[s1] * 0.5
        for s in range(s0 + 1, s1): Lb = Lb + self.L[s]
        return Lb / (s1 - s0)

    def mfpt(self, Lb, target):
        """mean first-passage time (generator M^{-1} Lb, Gamma=1) from every node to the target set:
        Lb[U,U] h_U = m_U on the complement U, h=0 on target.  Units: (time x Gamma)."""
        U = np.where(~target)[0]; h = np.zeros(self.N)
        h[U] = spsolve(Lb[U][:, U].tocsc(), self.m[U])
        return h

    # -------------------------------------------------------------- stirring potential (Pe-independent)
    def far_contacts(self, s0, s1, dmin, lab=None):
        """distinct K(t)-neighbours met in frames s0..s1 whose REFERENCE distance (frame s0) exceeds dmin.
        Affine compression/strain of a neighbourhood creates only edges between reference-near particles,
        so with dmin of a few spacings this counts only contacts made by folding / filamentation."""
        P0 = self.fr[s0]; keys = set()
        allk = np.unique(np.concatenate([self.E[s][:, 0] * self.N + self.E[s][:, 1] for s in range(s0, s1 + 1)]))
        I, J = allk // self.N, allk % self.N
        far = np.linalg.norm(P0[I] - P0[J], axis=1) > dmin
        cnt = np.zeros(self.N); np.add.at(cnt, I[far], 1); np.add.at(cnt, J[far], 1)
        out = dict(far=cnt)
        if lab is not None:
            x = far & (lab[I] != lab[J]); cx = np.zeros(self.N); np.add.at(cx, I[x], 1); np.add.at(cx, J[x], 1)
            out['far_cross'] = cx
        raw = np.zeros(self.N); np.add.at(raw, I, 1); np.add.at(raw, J, 1)
        out['raw'] = raw
        return out


def mixed_fraction(m, c, lab0):
    return float((m * 4 * c * (1 - c)).sum() / (m * lab0).sum())


def seg_intensity(m, c, c0):
    cb = (m * c).sum() / m.sum(); cb0 = (m * c0).sum() / m.sum()
    return float((m * (c - cb) ** 2).sum() / (m * (c0 - cb0) ** 2).sum())


def load_sb(path_net="/home/claude/w/sb_net_m.pkl", path_lite="/home/claude/w/sb_lite_m.npz"):
    import pickle
    d = pickle.load(open(path_net, 'rb')); z = np.load(path_lite)
    net = Net.__new__(Net); net.fr = z['frames']; net.dt = float(z['dt']); net.S, net.N = net.fr.shape[:2]
    net.tri, net.E, net.w, net.m = d['tri'], d['E'], d['w'], d['m']
    net.L = [laplacian(net.N, net.E[s], net.w[s]) for s in range(net.S)]
    return net, z
