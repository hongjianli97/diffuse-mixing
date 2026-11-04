"""Position-only separation of AFRC_H ridge candidates into hyperbolic / shear / shock.
Indices on each node's bottleneck edge e*(i) = argmin_{e ni i} F#(e):
  K  jump fraction  = sum_hot |dl| / sum |dl|          (hot steps from the area channel)
  R  half-window ratio of the continuous log-length    (1 exponential, ln2/ln(gT/2) shear)
  Z  star relay: Spearman(first-half, second-half stretch) over the node's kept edges (+1 saddle, low shear)
"""
import numpy as np
from scipy.stats import rankdata
import afrc_h_engine as AH
import lhdx as X
from lhdp import star_area


def hot_steps(frames, tau_min=0.2, kfac=4.0, every=1):
    """node x step mask of single-step area jumps (C channel). frames (S,N,2)."""
    S, N, _ = frames.shape; R = np.zeros((N, S-1))
    for s in range(S-1):
        tri = X.complex_peeled(frames[s])["tri"]
        a0 = star_area(frames[s], tri, N); a1 = star_area(frames[s+1], tri, N)
        with np.errstate(all="ignore"):
            R[:, s] = np.where((a0 > 0) & (a1 > 0), np.log(a1/a0), 0.0)
    sig = 1.4826*np.median(np.abs(R-np.median(R)))
    tau = max(tau_min, kfac*sig)
    return np.abs(R) > tau, R, tau


def afrc_h_edges(traj, k=24, gate=True, censor=None):
    """run one-scale AFRC_H; return kept edges, F per edge, node field, bottleneck edge index per node"""
    N, T, _ = traj.shape
    d, _ = AH.cKDTree(traj[:, 0]).query(traj[:, 0], k=2); AH._GRID_CELL = 0.5*float(np.median(d[:, 1]))
    old = (AH.FORMAN_COH_JUMP_MAX, AH.FORMAN_COH_Z, AH.FORMAN_COH_TAU)
    if not gate:
        AH.FORMAN_COH_JUMP_MAX, AH.FORMAN_COH_Z, AH.FORMAN_COH_TAU = 1.0, 0.0, 0.0
    AH._FORMAN_CENSOR_CURRENT = censor
    E = np.unique(np.vstack([AH._mutual_knn(traj[:, 0], k), AH._mutual_knn(traj[:, -1], k)]), axis=0)
    keep = AH.coherent_edge_keep(traj, E); Ek = E[keep]
    w_e = AH._edge_weight(traj, Ek, 1.0); w_v = AH._node_weight(N, Ek, w_e)
    Fe = AH._afrc_Fe(N, Ek, w_e, w_v)
    AH.FORMAN_COH_JUMP_MAX, AH.FORMAN_COH_Z, AH.FORMAN_COH_TAU = old; AH._FORMAN_CENSOR_CURRENT = None
    Fn = np.full(N, np.inf); np.minimum.at(Fn, Ek[:, 0], Fe); np.minimum.at(Fn, Ek[:, 1], Fe)
    B = np.where(np.isfinite(Fn), np.maximum(-Fn, 0), 0.0)
    deg = np.bincount(Ek.ravel(), minlength=N); B[deg < 3] = 0
    # bottleneck edge
    idx = np.concatenate([Ek[:, 0], Ek[:, 1]]); eid = np.concatenate([np.arange(len(Ek))]*2); val = np.concatenate([Fe, Fe])
    o = np.lexsort((val, idx)); first = np.full(N, -1)
    iv = idx[o]; st = np.searchsorted(iv, np.arange(N), "left"); has = st < len(iv)
    has[has] = iv[st[has]] == np.arange(N)[has]
    first[has] = eid[o][st[has]]
    return dict(E=Ek, Fe=Fe, w=w_e, B=B, bott=first, deg=deg)


def indices(traj, net, hot, lnJ=None):
    """K, R, Z per node (NaN where no bottleneck edge)."""
    N, T, _ = traj.shape; E = net["E"]
    ell = np.log(np.maximum(np.linalg.norm(traj[E[:, 0]]-traj[E[:, 1]], axis=2), 1e-12))
    dl = np.diff(ell, axis=1)
    he = hot[E[:, 0]] | hot[E[:, 1]]
    dlc = np.where(he, 0.0, dl)
    h = (T-1)//2
    K_e = np.abs(np.where(he, dl, 0)).sum(1)/np.maximum(np.abs(dl).sum(1), 1e-30)
    a = dlc[:, :h].sum(1); b = dlc[:, h:].sum(1)
    R_e = b/np.where(np.abs(a) > 1e-12, a, np.nan)
    net_e = dl.sum(1)
    # stretching edge per node: largest net separation among incident kept edges
    idx2 = np.concatenate([E[:, 0], E[:, 1]]); eid2 = np.concatenate([np.arange(len(E))]*2); val2 = np.concatenate([net_e, net_e])
    o2 = np.lexsort((-val2, idx2)); iv2 = idx2[o2]; st2 = np.searchsorted(iv2, np.arange(N), "left")
    b_ = np.full(N, -1); has = st2 < len(iv2); has[has] = iv2[st2[has]] == np.arange(N)[has]; b_[has] = eid2[o2][st2[has]]
    ok = b_ >= 0
    beta_e = b/np.log(2.0)
    K = np.full(N, np.nan); R = np.full(N, np.nan); Bt = np.full(N, np.nan)
    K[ok] = K_e[b_[ok]]; R[ok] = R_e[b_[ok]]; Bt[ok] = beta_e[b_[ok]]
    # relay: per node Spearman over incident kept edges
    Z = np.full(N, np.nan)
    idx = np.concatenate([E[:, 0], E[:, 1]]); eid = np.concatenate([np.arange(len(E))]*2)
    o = np.argsort(idx, kind="stable"); iv = idx[o]; ev = eid[o]
    st = np.searchsorted(iv, np.arange(N), "left"); en = np.searchsorted(iv, np.arange(N), "right")
    for i in range(N):
        if en[i]-st[i] < 6: continue
        es = ev[st[i]:en[i]]; x = a[es]; y = b[es]
        if np.std(x) < 1e-14 or np.std(y) < 1e-14: continue
        Z[i] = np.corrcoef(rankdata(x), rankdata(y))[0, 1]
    W = np.full(N, np.nan)
    if lnJ is not None:                       # lnJ (S,N) ledger; continuous part only
        dJ = np.where(hot, 0.0, np.diff(lnJ, axis=0).T).sum(1)
        cont = dlc.sum(1)
        W[ok] = np.abs(dJ[ok])/np.maximum(np.abs(cont[b_[ok]]), 1e-12)
    return dict(K=K, R=R, Z=Z, beta=Bt, W=W, K_e=K_e, R_e=R_e, sedge=b_)


def label(ind, B, q=0.9, Kc=0.5, b_lo=0.5, b_hi=1.5, Wc=0.5):
    """rule on ridge candidates (B above its q-quantile).
    0 none, 1 hyperbolic (beta>b_hi), 2 shear (b_lo<=beta<=b_hi), 3 shock (K>Kc), 4 inactive (beta<b_lo)"""
    lab = np.zeros(len(B), int)
    pos = B[B > 0]
    if pos.size == 0: return lab, np.zeros(len(B), bool)
    cand = B >= np.quantile(pos, q)
    K, bt = ind["K"], ind["beta"]
    sh = cand & (K > Kc)
    wv = cand & ~sh & (np.nan_to_num(ind["W"]) > Wc); sh_wv = sh | wv
    hyp = cand & ~sh_wv & (bt > b_hi); she = cand & ~sh_wv & (bt >= b_lo) & (bt <= b_hi); ina = cand & ~sh_wv & (bt < b_lo)
    lab[hyp] = 1; lab[she] = 2; lab[sh] = 3; lab[ina] = 4; lab[wv] = 5
    return lab, cand
