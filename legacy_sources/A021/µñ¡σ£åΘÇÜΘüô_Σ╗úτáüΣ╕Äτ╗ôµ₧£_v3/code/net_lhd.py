"""LHD built ON the user's trajectory network (fixed mutual-kNN graph, triangles = common-neighbour cliques).
Level 1: triangle circulation/area (Stokes vorticity), area rate (dilatation), area ratio (material lnJ) on network triangles.
Level 2: Hodge on the 'material Delaunay subcomplex' = current Delaunay triangles whose 3 edges are network edges."""
import json, time, numpy as np, cases, lhd as L, lhdx as X, afrc_h_engine as AH
from scipy.stats import spearmanr
t0 = time.time(); c = cases.case_dg(); fr, dt = c['frames'], c['dt']; S, N = fr.shape[:2]; V = np.gradient(fr, dt, axis=0)
k = 12
E = np.unique(np.vstack([AH._mutual_knn(fr[0], k), AH._mutual_knn(fr[-1], k)]), axis=0)      # the paper's graph (endpoints)
key = set((E[:, 0] * N + E[:, 1]).tolist())
nbr = [[] for _ in range(N)]
for a, b in E: nbr[a].append(b); nbr[b].append(a)
nbr = [np.array(sorted(x)) for x in nbr]
T = []
for a, b in E:                                                      # clique triangles a<b<w
    w = np.intersect1d(nbr[a], nbr[b], assume_unique=True); w = w[w > b]
    for x in w: T.append((a, b, x))
T = np.array(T); print('network: %d edges, %d triangles (%.0fs)' % (len(E), len(T), time.time() - t0), flush=True)
def tri_quantities(P, U, T):
    a, b, cc = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]; ua, ub, uc = U[T[:, 0]], U[T[:, 1]], U[T[:, 2]]
    cr = lambda p, q: p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]
    A = 0.5 * cr(b - a, cc - a)                                     # signed area
    G = 0.5 * (((ua + ub) * (b - a)).sum(1) + ((ub + uc) * (cc - b)).sum(1) + ((uc + ua) * (a - cc)).sum(1))   # circulation
    dA = 0.5 * (cr(ub - ua, cc - a) + cr(b - a, uc - ua))           # area rate
    l2 = ((b - a) ** 2).sum(1) + ((cc - b) ** 2).sum(1) + ((a - cc) ** 2).sum(1)
    q = 4 * np.sqrt(3) * np.abs(A) / np.maximum(l2, 1e-300)
    return A, G, dA, q
s0A = tri_quantities(fr[0], V[0], T)[0]; sgn = np.sign(s0A)          # orientation fixed at t0 (material triangle)
def node_avg(val, w, T, ok):
    num = np.zeros(N); den = np.zeros(N)
    for j in range(3):
        np.add.at(num, T[ok, j], (val * w)[ok]); np.add.at(den, T[ok, j], w[ok])
    with np.errstate(all='ignore'): return np.where(den > 0, num / den, np.nan)
R = {}
for s in (S // 2, S - 2):
    P, U = fr[s], V[s]; A, G, dA, q = tri_quantities(P, U, T)
    ok = (np.sign(A) == sgn) & (q > 0.05)                           # valid material triangle at time s (not inverted / sliver)
    om_net = node_avg(G / A, np.abs(A), T, ok); th_net = node_avg(dA / A, np.abs(A), T, ok)
    cx = X.complex_peeled(P); f = L.edge_cochain(P, U, cx['edges']); om_del = L.node_vorticity(P, cx, f); th_gal = X.galerkin_div(P, cx, U)
    inn = (P[:, 0] > .05) & (P[:, 0] < 1.95) & (P[:, 1] > .05) & (P[:, 1] < .95); ex = c['om_ex'][s]; rms = np.sqrt(np.mean(ex[inn] ** 2))
    m1 = inn & np.isfinite(om_net)
    # material Delaunay subcomplex: Delaunay(t) triangles whose 3 edges are network edges
    tri = cx['tri']; ek = lambda i, j: np.minimum(tri[:, i], tri[:, j]) * N + np.maximum(tri[:, i], tri[:, j])
    inE = np.array([[x in key for x in ek(i, (i + 1) % 3)] for i in range(3)]).all(0)
    sub = L._assemble(tri[inE], N); g = X.galerkin_hodge(P, sub, U)
    # torn neighbourhoods: fraction of a node's Delaunay edges that are NOT material network pairs
    De = cx['edges']; notin = np.array([(min(u, v) * N + max(u, v)) not in key for u, v in De])
    tor = np.zeros(N); cnt = np.zeros(N); np.add.at(tor, De[:, 0], notin); np.add.at(tor, De[:, 1], notin); np.add.at(cnt, De[:, 0], 1); np.add.at(cnt, De[:, 1], 1)
    torn = tor / np.maximum(cnt, 1); i0 = (fr[0][:, 0] > .05) & (fr[0][:, 0] < 1.95) & (fr[0][:, 1] > .05) & (fr[0][:, 1] < .95)
    R['t=%.1f' % c['t'][s]] = dict(
        valid_network_triangles=float(ok.mean()),
        vort_corr_network=float(np.corrcoef(om_net[m1], ex[m1])[0, 1]), vort_relL2_network=float(np.sqrt(np.mean((om_net[m1] - ex[m1]) ** 2)) / rms),
        vort_corr_delaunay=float(np.corrcoef(om_del[inn], ex[inn])[0, 1]),
        div_noise_network=float(np.sqrt(np.nanmean(th_net[m1] ** 2)) / rms), div_noise_galerkin_delaunay=float(np.sqrt(np.nanmean(th_gal[inn] ** 2)) / rms),
        subcomplex_coverage=float(inE.mean()), hodge_on_subcomplex=dict(grad=g['Eg'], coexact=g['Ec'], harmonic=g['Eh']),
        torn_vs_FTLE_spearman=float(spearmanr(torn[i0], c['ftle_ex'][i0])[0]))
    print(json.dumps({k_: R['t=%.1f' % c['t'][s]]}), flush=True) if False else print('t=%.1f' % c['t'][s], json.dumps(R['t=%.1f' % c['t'][s]]), flush=True)
# material lnJ of network triangles over the whole window (incompressible: exact 0)
A0 = tri_quantities(fr[0], V[0], T)[0]; A1 = tri_quantities(fr[-1], V[-1], T)[0]; okm = np.sign(A1) == np.sign(A0)
lj = np.log(np.abs(A1[okm]) / np.abs(A0[okm])); lnJ_node = node_avg(np.log(np.abs(A1) / np.abs(A0)), np.abs(A0), T, okm)
R['window_lnJ'] = dict(triangles_not_inverted=float(okm.mean()), median_abs_lnJ_triangle=float(np.median(np.abs(lj))),
                      abs_lnJ_node_vs_FTLE_spearman=float(spearmanr(np.abs(lnJ_node[np.isfinite(lnJ_node)]), c['ftle_ex'][np.isfinite(lnJ_node)])[0]))
print('window', json.dumps(R['window_lnJ']), '%.0fs' % (time.time() - t0))
json.dump(R, open('res_net_lhd.json', 'w'), indent=1)
