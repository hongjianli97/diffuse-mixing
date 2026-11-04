"""WP3: joint Dirac-Kelvin denoising of the torque 2-cochain.
T0 raw: d1 g (acceleration cochain, second differences)
T1 time (Kelvin): Savitzky-Golay (5 frames) derivative of material-triangle circulation d1 f (first-order signal)
T2 space (Dirac): low-pass with the Dirac-squared triangle block, argmin ||c-c_obs||^2_{W M2} + l^2 ||delta2 c||^2_{M1}
   (censored triangles = missing data, weight 0); T3 = T2 applied to T1."""
import json, pickle, numpy as np, scipy.sparse.linalg as sla, lhd as L, lhdx as X, dirac as DR
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.stats import spearmanr
from trajnet import TrajectoryNetwork as TN
def torques(net, s, ell_fac=1.5):
    P, cx = net.fr[s], net.K[s]; tri = cx['tri']; A = L.tri_signed_area(P, tri); N = net.N
    acc = DR.acceleration(net.fr, net.dt); c0 = DR.tri_circ(P, acc[s], tri)
    ok0 = ~net.bad[tri, s - 1:s + 2].any((1, 2))
    G = np.stack([DR.tri_circ(net.fr[q], net.V[q], tri) for q in range(s - 2, s + 3)]); c1 = (np.array([-2, -1, 0, 1, 2]) @ G) / (10 * net.dt)
    sg_ok = ~net.bad[tri, max(s - 3, 0):s + 4].any((1, 2)) & np.all([np.sign(L.tri_signed_area(net.fr[q], tri)) == np.sign(A) for q in range(s - 2, s + 3)], 0)
    B = DR.blocks(P, cx); h = np.median(np.linalg.norm(P[cx['edges'][:, 1]] - P[cx['edges'][:, 0]], axis=1)); l2 = (ell_fac * h) ** 2
    def lowpass(c, w):
        op = sla.LinearOperator((len(c), len(c)), matvec=lambda x: w * B['M2'] * x + l2 * B['M2'] * (B['d1'] @ B['lu'].solve(B['d1'].T @ (B['M2'] * x))))
        x, _ = sla.cg(op, w * B['M2'] * np.nan_to_num(c), x0=np.where(w > 0, np.nan_to_num(c), 0.0), maxiter=400); return x
    c2 = lowpass(c0, ok0.astype(float)); c3 = lowpass(c1, sg_ok.astype(float))
    allok = np.ones(len(tri), bool); node = lambda c, ok: DR.node_avg(N, c / A, np.abs(A), tri, ok)
    return dict(T0=node(c0, ok0), T1=node(c1, sg_ok), T2=node(c2, allok), T3=node(c3, allok))


def t0_t2(net, s, ell_fac):
    P, cx = net.fr[s], net.K[s]; tri = cx['tri']; A = L.tri_signed_area(P, tri); N = net.N
    acc = DR.acceleration(net.fr, net.dt); c0 = DR.tri_circ(P, acc[s], tri); ok0 = ~net.bad[tri, s - 1:s + 2].any((1, 2))
    B = DR.blocks(P, cx); h = np.median(np.linalg.norm(P[cx['edges'][:, 1]] - P[cx['edges'][:, 0]], axis=1)); l2 = (ell_fac * h) ** 2
    w = ok0.astype(float)
    op = sla.LinearOperator((len(c0), len(c0)), matvec=lambda x: w * B['M2'] * x + l2 * B['M2'] * (B['d1'] @ B['lu'].solve(B['d1'].T @ (B['M2'] * x))))
    c2, _ = sla.cg(op, w * B['M2'] * np.nan_to_num(c0), x0=np.where(w > 0, np.nan_to_num(c0), 0.0), maxiter=400)
    node = lambda c, ok: DR.node_avg(N, c / A, np.abs(A), tri, ok)
    return dict(T0=node(c0, ok0), T2=node(c2, np.ones(len(tri), bool)))
