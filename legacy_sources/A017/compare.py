# -*- coding: utf-8 -*-
"""Reference implementations of mainstream Lagrangian diffusion / mixing diagnostics (positions only, scattered particles)."""
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse import linalg as spla

def _nbrs(X0, nn=8):
    return cKDTree(X0).query(X0, k=nn + 1)[1][:, 1:]

def ls_F(X0, X1, j):
    """least-squares flow-map gradient per particle from its k nearest label neighbours (standard for scattered data)."""
    dA = X0[j] - X0[:, None, :]; dX = X1[j] - X1[:, None, :]
    return np.einsum("nki,nkj->nij", dX, dA) @ np.linalg.inv(np.einsum("nki,nkj->nij", dA, dA))

def ftle(traj, T, nn=8):
    """finite-time Lyapunov exponent (forward), Haller 2001 / Shadden 2005."""
    j = _nbrs(traj[0], nn); F = ls_F(traj[0], traj[-1], j)
    lam = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", F, F))[:, -1]
    return np.log(np.maximum(lam, 1e-300)) / (2 * abs(T))

def dbs_ls(traj, t, nn=8, compressible=True, ref=0):
    """diffusion barrier strength DBS = tr < J C^{-1} >  (Haller, Karrasch & Kogelbauer 2018; = tr<C^{-1}> if J=1),
    flow-map gradients by least squares at every snapshot (mainstream evaluation)."""
    j = _nbrs(traj[ref], nn); w = np.zeros(len(t)); dt = np.abs(np.diff(t)); w[:-1] += dt / 2; w[1:] += dt / 2; w /= w.sum()
    acc = 0.0
    for s in range(len(t)):
        F = ls_F(traj[ref], traj[s], j); J = np.linalg.det(F)
        Ci = np.linalg.inv(np.einsum("nki,nkj->nij", F, F)); acc = acc + w[s] * (J if compressible else 1.0) * (Ci[:, 0, 0] + Ci[:, 1, 1])
    return acc

def dbs_net(G):
    """the same tensor read from the network conductances (Lemma 3): node area-average of tr Kbar_T / k."""
    tr = G.Kbar[:, 0, 0] + G.Kbar[:, 1, 1]; acc = np.zeros(G.N); w = np.zeros(G.N)
    for a in range(3): np.add.at(acc, G.tri[:, a], G.Aref * tr); np.add.at(w, G.tri[:, a], G.Aref)
    return acc / np.maximum(w, 1e-300)

def encounter_number(traj, R, stride=2):
    """number of distinct particles that come within R during the window (Rypina & Pratt 2017; network degree of Padberg-Gehle & Schneide 2017)."""
    N = traj.shape[1]; keys = []
    for s in range(0, len(traj), stride):
        pr = cKDTree(traj[s]).query_pairs(R, output_type="ndarray"); keys.append(pr[:, 0].astype(np.int64) * N + pr[:, 1])
    u = np.unique(np.concatenate(keys)); return np.bincount(u // N, minlength=N) + np.bincount(u % N, minlength=N)

def exchange_degree(G):
    """first-order exchange (your Q10): T * sum_j gbar_ij / m_i."""
    d = np.bincount(G.E[:, 0], weights=G.g, minlength=G.N) + np.bincount(G.E[:, 1], weights=G.g, minlength=G.N)
    return G.T * d / G.m

def fiedler(G, nev=3):
    vals, vecs = spla.eigsh(G.L.tocsc(), k=nev, M=sparse.diags(G.m).tocsc(), sigma=-1e-6, which="LM"); o = np.argsort(vals)
    return vals[o], vecs[:, o]

def grad_norm(X0, f, nn=8):
    j = _nbrs(X0, nn); dA = X0[j] - X0[:, None, :]; df = f[j] - f[:, None]
    g = np.linalg.solve(np.einsum("nki,nkj->nij", dA, dA), np.einsum("nki,nk->ni", dA, df)[:, :, None])[:, :, 0]
    return np.hypot(*g.T)

def fte(K, nodes, A0, A_end, T):
    """finite-time entropy (Froyland & Padberg-Gehle 2012): entropy growth of a localized density diffused along the flow,
    density w.r.t. physical area at the final time; kernels K from MixingGeometry.kernels (first scale)."""
    out = []
    for i in nodes:
        nn, MU = K[int(i)][0], K[int(i)][1][0]; p = np.maximum(MU, 0); p = p / p.sum(); q = p > 0
        out.append((-(p[q] * np.log(p[q] / A_end[nn][q])).sum() - np.log(A0[int(i)])) / T)
    return np.array(out)

def salience(f, band, far):
    med = np.nanmedian(f[far]); z = np.abs(f - med)
    return float(np.nanpercentile(z[band], 95) / max(np.nanpercentile(z[far], 95), 1e-12))
