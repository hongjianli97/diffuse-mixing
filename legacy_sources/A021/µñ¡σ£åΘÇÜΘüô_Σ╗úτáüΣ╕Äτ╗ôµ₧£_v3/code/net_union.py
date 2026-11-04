"""Paper Eq. (2) with S = ALL snapshots: is each instantaneous Delaunay complex a subcomplex of the network?"""
import json, numpy as np, cases, lhd as L, lhdx as X, afrc_h_engine as AH
c = cases.case_dg(); fr, dt = c['frames'], c['dt']; S, N = fr.shape[:2]; V = np.gradient(fr, dt, axis=0)
R = {}
for k in (8, 12):
    keys_end = set()
    for s in (0, S - 1):
        e = AH._mutual_knn(fr[s], k); keys_end |= set((e[:, 0] * N + e[:, 1]).tolist())
    keys_all = set(keys_end)
    for s in range(S):
        e = AH._mutual_knn(fr[s], k); keys_all |= set((e[:, 0] * N + e[:, 1]).tolist())
    r = dict(edges_endpoints=len(keys_end), edges_all_snapshots=len(keys_all))
    for s in (S // 2, S - 2):
        P = fr[s]; cx = X.complex_peeled(P); tri = cx['tri']
        ek = lambda i, j: np.minimum(tri[:, i], tri[:, j]) * N + np.maximum(tri[:, i], tri[:, j])
        inE = np.array([[x in keys_all for x in ek(i, (i + 1) % 3)] for i in range(3)]).all(0)
        g = X.galerkin_hodge(P, L._assemble(tri[inE], N), V[s])
        r['t=%.1f' % c['t'][s]] = dict(delaunay_triangles_covered=float(inE.mean()), grad_leak=g['Eg'], harmonic=g['Eh'])
    R['k=%d' % k] = r; print('k=%d' % k, json.dumps(r), flush=True)
json.dump(R, open('res_net_union.json', 'w'), indent=1)
