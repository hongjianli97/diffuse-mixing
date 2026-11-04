"""LHD-P: position-only Lagrangian Hodge-Dirac primitives."""
import numpy as np
from scipy.spatial import cKDTree
import lhd as L, lhdx as X


def cross(a, b): return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def tri_F(P0, P1, tri):
    """affine step map of each triangle P0->P1: (T,2,2), and signed area at P0"""
    E0 = np.stack([P0[tri[:, 1]]-P0[tri[:, 0]], P0[tri[:, 2]]-P0[tri[:, 0]]], -1)
    E1 = np.stack([P1[tri[:, 1]]-P1[tri[:, 0]], P1[tri[:, 2]]-P1[tri[:, 0]]], -1)
    F = E1 @ np.linalg.inv(E0)
    return F, 0.5*np.linalg.det(E0)


def pq(F):
    p = 0.5*((F[:, 0, 0]+F[:, 1, 1]) + 1j*(F[:, 1, 0]-F[:, 0, 1]))
    q = 0.5*((F[:, 0, 0]-F[:, 1, 1]) + 1j*(F[:, 1, 0]+F[:, 0, 1]))
    return p, q


def compose(p2, q2, p1, q1):
    return p2*p1+q2*np.conj(q1), p2*q1+q2*np.conj(p1)


def disp_cochain_tri(Pn, Pn1, tri):
    """midpoint-edge displacement cochain summed around each triangle: (circulation, normal flux)"""
    D = Pn1-Pn; c = np.zeros(len(tri)); fl = np.zeros(len(tri))
    for j in range(3):
        a, b = tri[:, j], tri[:, (j+1) % 3]
        ub = 0.5*(D[a]+D[b]); qb = 0.5*((Pn[b]-Pn[a])+(Pn1[b]-Pn1[a]))
        c += (ub*qb).sum(1)
        fl += (ub*np.c_[qb[:, 1], -qb[:, 0]]).sum(1)
    return c, fl


def kelvin_pos(Pm, P0, Pp, tri, edge="q0"):
    """(Gamma^{n+1/2}-Gamma^{n-1/2}, (d1 g)_t, Gamma^+, Gamma^-), g_e = 1/2(D2x_a+D2x_b).q_e"""
    Gp = np.zeros(len(tri)); Gm = np.zeros(len(tri)); dg = np.zeros(len(tri))
    Dp, Dm, D2 = Pp-P0, P0-Pm, Pp-2*P0+Pm
    for j in range(3):
        a, b = tri[:, j], tri[:, (j+1) % 3]
        qm, q0, qp = Pm[b]-Pm[a], P0[b]-P0[a], Pp[b]-Pp[a]
        Gp += (0.5*(Dp[a]+Dp[b])*0.5*(q0+qp)).sum(1)
        Gm += (0.5*(Dm[a]+Dm[b])*0.5*(qm+q0)).sum(1)
        qe = q0 if edge == "q0" else 0.25*(qm+2*q0+qp)
        dg += (0.5*(D2[a]+D2[b])*qe).sum(1)
    return Gp-Gm, dg, Gp, Gm


def kabsch(P0, P1, w=None):
    w = np.ones(len(P0)) if w is None else w
    c0 = (w[:, None]*P0).sum(0)/w.sum(); c1 = (w[:, None]*P1).sum(0)/w.sum()
    H = ((P0-c0)*w[:, None]).T @ (P1-c1)
    U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T@U.T))
    R = Vt.T@np.diag([1, d])@U.T
    return R, c0, c1


def pullback(P0, P1, w=None):
    """P1 expressed in the frame of P0 after removing the best-fit rigid motion (objective step)."""
    R, c0, c1 = kabsch(P0, P1, w)
    return (P1-c1)@R + c0, np.arctan2(R[1, 0], R[0, 0])


def tri_area(P, tri):
    return 0.5*cross(P[tri[:, 1]]-P[tri[:, 0]], P[tri[:, 2]]-P[tri[:, 0]])


def star_area(P, tri, N):
    s = np.zeros(N); np.add.at(s, tri.ravel(), np.repeat(tri_area(P, tri), 3)); return s


def tri_to_node(vals, A, tri, N):
    s = np.zeros(N); w = np.zeros(N); A = np.abs(A)
    for k in range(3):
        np.add.at(s, tri[:, k], vals*A); np.add.at(w, tri[:, k], A)
    with np.errstate(all="ignore"):
        return np.where(w > 0, s/w, np.nan)


def step_readouts(P0, P1, tri):
    F, A0 = tri_F(P0, P1, tri); p, q = pq(F)
    return dict(A0=A0, p=p, q=q, J=np.linalg.det(F), mu=np.abs(q/p), psi=np.angle(p), trF=F[:, 0, 0]+F[:, 1, 1])
