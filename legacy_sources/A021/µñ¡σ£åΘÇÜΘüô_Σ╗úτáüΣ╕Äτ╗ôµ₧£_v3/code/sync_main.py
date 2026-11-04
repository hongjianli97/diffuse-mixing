"""Frequency-cluster hierarchy (merge tree of the triangle rotation phase) as a network strength field.
P_i = height of node i above the level at which its frequency cluster merges into a cluster with a higher peak
(the cluster that first covers half the domain is the background and dies there).  Signed: P = p+(phi) - p+(-phi).
Invariant to phi -> phi + const, i.e. to the ensemble reference rotation."""
import sys, json, numpy as np, tcx, cases, afe_cases as AC, tcx_cases as TC
from scipy.stats import spearmanr
def edge_nodes(tri, mask):
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    u, cnt = np.unique(e, axis=0, return_counts=True); bd = np.zeros(len(mask), bool); bd[u[cnt == 1].ravel()] = True
    for x, y in u:                                       # nodes of the mask touching the outside of the mask
        if mask[x] != mask[y]: bd[x] |= mask[x]; bd[y] |= mask[y]
    return bd & mask
def prominence(f, E, mask, bd):
    """height above the level at which the node's frequency cluster either merges into a cluster with a higher
    peak, or first reaches the edge of the observed material (then it is no longer an enclosed vortex)."""
    N = len(f); nb = [[] for _ in range(N)]
    for a_, b_ in E:
        if mask[a_] and mask[b_]: nb[a_].append(b_); nb[b_].append(a_)
    comp = -np.ones(N, int); peak = {}; mem = {}; saddle = np.full(N, np.nan); dead = set()
    for i in np.argsort(-f):
        if not mask[i]: continue
        cs = {comp[j] for j in nb[i] if comp[j] >= 0}
        if not cs:
            comp[i] = i; peak[i] = f[i]; mem[i] = [i]
        else:
            cs = sorted(cs, key=lambda c: (c not in dead, peak[c]), reverse=True)   # an open (edge) cluster absorbs
            top = cs[0]
            for c in cs[1:]:
                for j in mem[c]:
                    if np.isnan(saddle[j]): saddle[j] = f[i]
                    comp[j] = top
                mem[top].extend(mem.pop(c)); peak.pop(c); dead.discard(c)
            comp[i] = top; mem[top].append(i)
        r = comp[i]
        if bd[i] and r not in dead:
            for j in mem[r]:
                if np.isnan(saddle[j]): saddle[j] = f[i]
            dead.add(r)
        if r in dead and np.isnan(saddle[i]): saddle[i] = f[i]
    out = np.maximum(0.0, f - np.where(np.isnan(saddle), f, saddle)); out[~mask] = 0.0; return out
def signedP(f, E, mask, tri):
    bd = edge_nodes(tri, mask); return prominence(f, E, mask, bd) - prominence(-f, E, mask, bd)
def chain_phi(frx):
    S = len(frx); mus, _ = tcx.mu_threshold(frx); ch = tcx.run_chain(frx, mus, Ks=[tcx.cplx(frx[s])["tri"] for s in range(S-1)])
    return tcx.rotation_fields(ch)[0]
def smooth(fv, E0, N, n=2):
    deg = np.bincount(E0.ravel(), minlength=N).astype(float); g = fv.copy()
    for _ in range(n):
        s_ = np.zeros(N); np.add.at(s_, E0[:, 0], g[E0[:, 1]]); np.add.at(s_, E0[:, 1], g[E0[:, 0]]); g = .5*g + .5*s_/np.maximum(deg, 1)
    return g
name = sys.argv[1]; R = dict(case=name)
c = AC.load(name); fr = c["fr"]; S, N = fr.shape[:2]; h = c["h"]; m = c["inner"]; P0 = fr[0]
phi_ref, lavd_ref = TC.phi_reference(c["om"], fr, c["dt"]); tri0 = tcx.cplx(P0)["tri"]; E0 = tcx.graph_edges(tri0)
phi = chain_phi(fr); ps = smooth(phi, E0, N); prs = smooth(phi_ref, E0, N)
Pn, Pr = signedP(ps, E0, m, tri0), signedP(prs, E0, m, tri0)
rng = np.random.default_rng(5); frn = fr + rng.normal(0, 0.1*h, fr.shape); phin = chain_phi(frn); Pnn = signedP(smooth(phin, E0, N, 8), E0, m, tri0)
sp = lambda a, b: float(spearmanr(a[m], b[m])[0])
vr = m & (np.abs(Pr) > 0.05*np.abs(Pr).max())
spv = lambda a, b: float(spearmanr(a[vr], b[vr])[0]); pe = lambda a, b: float(np.corrcoef(a[m], b[m])[0, 1])
R["in_vortex_region"] = dict(frac=float(vr.sum()/m.sum()), spearman_P_net_ref=spv(Pn, Pr), spearman_P_noise_ref=spv(Pnn, Pr),
                             spearman_phi_net_ref=spv(phi, phi_ref), spearman_phi_noise_ref=spv(phin, phi_ref),
                             pearson_P_net_ref=pe(Pn, Pr), pearson_P_noise_ref=pe(Pnn, Pr), pearson_phi_noise_ref=pe(phin, phi_ref))
R["spearman"] = dict(P_net_vs_P_ref=sp(Pn, Pr), absphi_net_vs_lavdref=sp(np.abs(phi), lavd_ref), P_net_vs_lavdref=sp(np.abs(Pn), lavd_ref),
                     P_noise_vs_P_ref=sp(Pnn, Pr), absphi_noise_vs_lavdref=sp(np.abs(phin), lavd_ref))
core = int(np.argmax(np.abs(Pr)*m)); bg = m & (np.abs(Pr) < 0.05*np.abs(Pr[core]))
R["background"] = dict(frac_nodes=float(bg.sum()/m.sum()), median_absphi_over_core=float(np.median(np.abs(phi[bg]))/abs(phi[core])),
                       median_absP_over_core=float(np.median(np.abs(Pn[bg]))/abs(Pn[core])))
print(name, json.dumps(R), flush=True)
# ensemble-invariance: drop part of the seeded ensemble, recompute everything on the sub-ensemble
if name in ("LO", "DG"):
    keep = (np.hypot(P0[:, 0], P0[:, 1]) < 4.5) if name == "LO" else (P0[:, 0] < 1.85)
    idx = np.where(keep)[0]; frs = fr[:, idx]; phis_ = chain_phi(frs); tris = tcx.cplx(frs[0])["tri"]; Es = tcx.graph_edges(tris)
    ms = m[idx] & ((np.hypot(P0[idx, 0], P0[idx, 1]) < 4.2) if name == "LO" else (P0[idx, 0] < 1.8))
    Ps = signedP(smooth(phis_, Es, len(idx)), Es, ms, tris)
    cmp_ = ms & m[idx]
    # compare at the two strongest cores of the full-ensemble field that lie in the sub-domain
    cores = [i for i in np.argsort(-np.abs(Pn)) if keep[i]][:1]
    k = [int(np.where(idx == cc)[0][0]) for cc in cores]
    dphi = phis_[cmp_] - phi[idx][cmp_]; dP = Ps[cmp_] - Pn[idx][cmp_]
    R["ensemble"] = dict(kept=int(len(idx)), phi_shift_median=float(np.median(dphi)), phi_shift_iqr=float(np.subtract(*np.quantile(dphi, [.75, .25]))),
                         P_change_median=float(np.median(dP)), P_change_iqr=float(np.subtract(*np.quantile(dP, [.75, .25]))),
                         core_phi_full=float(phi[cores[0]]), core_phi_sub=float(phis_[k[0]]), core_P_full=float(Pn[cores[0]]), core_P_sub=float(Ps[k[0]]))
    print(name, "ensemble", json.dumps(R["ensemble"]), flush=True)
json.dump(R, open(f"sync_{name}.json", "w"), indent=1)
np.savez(f"sync_F_{name}.npz", P0=P0, phi=phi, Pn=Pn, Pr=Pr, Pnn=Pnn, mask=m, lavd_ref=lavd_ref)
