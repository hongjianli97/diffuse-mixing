"""AFRC_E improvement: the paper's dome construction driven by LHD Stokes relative rotation with an ABSOLUTE gate."""
import json, pickle, time, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import paper_ell as PE, lhdx as X, lhd as L, cases
from improve_test_helpers import events, gkcorr, spacing, box_of
GATE = np.pi / 4                                                   # net relative rotation >= 45 deg over the window
def lhd_rotation(fr, dt, bad):
    o = X.run_lhd(fr, dt, kinds=('galerkin',), op='galerkin', cplx='peel', bad=bad, hodge_every=10 ** 6)
    vc = o['vort_c'] if bad is not None else o['vort']; S = len(fr)
    W = np.array([X.star_area(fr[s]) for s in range(S)])
    wb = np.array([np.nansum(vc[s] * W[s]) / np.nansum(W[s] * np.isfinite(vc[s])) for s in range(S)])
    dev = vc - wb[:, None]; ti = np.arange(S)
    for i in np.where(~np.isfinite(dev).all(0))[0]:
        ok = np.isfinite(dev[:, i])
        if ok.sum() >= 2: dev[~ok, i] = np.interp(ti[~ok], ti[ok], dev[ok, i])
    return np.abs(0.5 * np.trapezoid(np.nan_to_num(dev), dx=dt, axis=0))
def dome(trc, Phi, merge, gate=GATE, kk=12):
    N = trc.shape[0]; P0 = trc[:, 0]; supp = Phi >= gate
    if supp.sum() < 10: return np.zeros(N), np.zeros((0, 2))
    grL = PE.build_graph(trc, kk); adj = PE._adj_list(grL); Pn = Phi / Phi.max()
    mk = PE.objrot_markers(Pn, grL, supp, P0, strength=0.30, merge_dist=merge, interior=None)
    lab = PE._fill_holes(PE.watershed_supp(Pn, adj, mk, supp), adj, iters=8, frac=0.65)
    u, v = grL['u'], grL['v']; A = sp.coo_matrix((np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])), shape=(N, N)).tocsr()
    deg = np.asarray(A.sum(1)).ravel(); Lap = sp.diags(deg) - A; kap2 = 0.2 * np.median(deg[deg > 0])
    E = np.zeros(N); cores = []
    for c in range(len(mk)):
        iv = np.where(lab == c)[0]
        if iv.size < 10: continue
        sol = np.maximum(spsolve((Lap[iv][:, iv] + kap2 * sp.eye(iv.size)).tocsc(), Pn[iv]), 0)
        if sol.max() > 0: E[iv] = sol / sol.max(); cores.append(P0[iv[np.argmax(sol)]])
    return E, np.array(cores).reshape(-1, 2)
