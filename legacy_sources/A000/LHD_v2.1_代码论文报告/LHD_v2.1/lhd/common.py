import numpy as np, lhdx as X
from lhdp import *
import classify as C

def readout_series(frames, t, vort=None, stride=1, align=False):
    """position-only step vorticity / dilatation on current peeled Delaunay; compare with analytic at midpoints"""
    S, N, _ = frames.shape; res = []
    for s in range(0, S-1, stride):
        P0, P1 = frames[s], frames[s+1]
        if align: P1, _ = pullback(P0, P1)
        tri = X.complex_peeled(P0)["tri"]; r = step_readouts(P0, P1, tri); dt = t[s+1]-t[s]
        Pm = 0.5*(P0+P1); Am = tri_area(Pm, tri)
        om_hat = 2*r["A0"]*r["p"].imag/(Am*dt)        # midpoint-triangle circulation / area / dt
        th_hat = np.log(r["J"])/dt
        cen = Pm[tri].mean(1)
        d = dict(s=s, om_hat=om_hat, th_hat=th_hat, mu=r["mu"], tri=tri, cen=cen)
        if vort is not None: d["om_ex"] = vort(cen, 0.5*(t[s]+t[s+1]))
        res.append(d)
    return res

def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b); return float(np.corrcoef(a[m], b[m])[0, 1])

def integrate(vel, P, t, sub=8, delta=1e-6, withF=True):
    """RK4 trajectories at frames t and finite-difference flow-map gradient F (S,N,2,2) (central, +-delta)."""
    N = len(P); S = len(t)
    off = [np.zeros(2)] + ([np.array([delta, 0]), np.array([-delta, 0]), np.array([0, delta]), np.array([0, -delta])] if withF else [])
    Q = np.concatenate([P+o for o in off]); fr = np.empty((S, N, 2)); fr[0] = P
    Fs = np.empty((S, N, 2, 2)) if withF else None
    if withF: Fs[0] = np.eye(2)
    tt = t[0]
    for s in range(1, S):
        h = (t[s]-t[s-1])/sub
        for _ in range(sub):
            k1 = vel(Q, tt); k2 = vel(Q+.5*h*k1, tt+.5*h); k3 = vel(Q+.5*h*k2, tt+.5*h); k4 = vel(Q+h*k3, tt+h)
            Q = Q+h*(k1+2*k2+2*k3+k4)/6; tt += h
        fr[s] = Q[:N]
        if withF:
            Fs[s, :, :, 0] = (Q[N:2*N]-Q[2*N:3*N])/(2*delta); Fs[s, :, :, 1] = (Q[3*N:4*N]-Q[4*N:])/(2*delta)
    return fr, Fs

def R_exact(Fs):
    """exact beta = [ln s1(T) - ln s1(T/2)]/ln2 and ln s1(T)"""
    S = Fs.shape[0]; h = (S-1)//2
    l = lambda F: np.log(np.linalg.svd(F, compute_uv=False)[:, 0])
    a = l(Fs[h]); b = l(Fs[-1])-a
    return b/np.log(2.0), l(Fs[-1])

def run_afrc(fr, k=24, gate=True, q=0.9, lnJ=None, hot=None):
    traj = np.transpose(fr, (1, 0, 2))
    if hot is None: hot, Rlog, tau = C.hot_steps(fr)
    else: tau = np.nan
    net = C.afrc_h_edges(traj, k=k, gate=gate)
    ind = C.indices(traj, net, hot, lnJ); lab, cand = C.label(ind, net["B"], q=q)
    return dict(hot=hot, tau=tau, net=net, ind=ind, lab=lab, cand=cand)

def summary(A, Rex=None, lnsig=None):
    cand = A["cand"]; lab = A["lab"]; ind = A["ind"]
    o = dict(cand_n=int(cand.sum()), hot_frac=float(A["hot"].mean()), tau=float(A["tau"]),
             lab_frac={k: float((lab[cand] == v).mean()) for k, v in [("hyp", 1), ("shear", 2), ("shock", 3), ("inactive", 4), ("wave", 5)]},
             beta_q=[float(x) for x in np.nanquantile(ind["beta"][cand], [.1, .25, .5, .75, .9])],
             K_q=[float(x) for x in np.nanquantile(ind["K"][cand], [.5, .9])])
    if Rex is not None:
        lab_ex = np.where(Rex > 1.5, 1, np.where(Rex >= 0.5, 2, 4))
        m = cand & np.isfinite(ind["beta"]) & (lab != 3)
        o["agree_with_exact"] = float((lab_ex[m] == lab[m]).mean())
        o["beta_ex_q"] = [float(x) for x in np.nanquantile(Rex[cand], [.1, .25, .5, .75, .9])]
        o["lab_ex_frac"] = {k: float((lab_ex[cand] == v).mean()) for k, v in [("hyp", 1), ("shear", 2), ("inactive", 4)]}
        rr = ind["beta"]; re = Rex; mm = cand & np.isfinite(rr) & np.isfinite(re)
        from scipy.stats import spearmanr
        o["spearman_beta"] = float(spearmanr(rr[mm], re[mm])[0])
    if lnsig is not None:
        o["afrcH_vs_lnsig_spearman"] = float(__import__("scipy.stats").stats.spearmanr(A["net"]["B"], lnsig)[0])
    return o

def classify_field(fr, B, lnJ, hot, reverse=False, q=0.9, k=24):
    """classify the top-(1-q) nodes of an arbitrary ridge field B (e.g. LS-FTLE) with the same indices"""
    if reverse: fr = fr[::-1]; lnJ = lnJ[::-1]; hot = hot[:, ::-1]
    traj = np.transpose(fr, (1, 0, 2))
    net = C.afrc_h_edges(traj, k=k, gate=False)
    ind = C.indices(traj, net, hot, lnJ); lab, cand = C.label(ind, B, q=q)
    return dict(hot=hot, tau=np.nan, net=dict(net, B=B), ind=ind, lab=lab, cand=cand)
