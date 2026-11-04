"""O-04 ablation (double gyre, mid-window frame): how the complex construction changes the node divergence.
Four variants, two operators:
  median   : Delaunay, drop triangles whose longest edge > 3 x global median edge (old default; opens interior holes)
  median+mask : same, but divergence at open-star nodes set to NaN (what run_lhd does today)
  peel     : Delaunay peeled only from the outside (current default), no interior holes
  raw      : full Delaunay, no filtering at all
Writes res_ablation_complex.json.  Reference: exact divergence = 0 (incompressible)."""
import json, numpy as np, cases, lhd as L, lhdx as X
c = cases.case_dg(); fr, dt, S = c['frames'], c['dt'], len(c['t'])
V = np.gradient(fr, dt, axis=0)
tm = S // 2; P = fr[tm]; Vs = V[tm]
inn = (P[:, 0] > .05) & (P[:, 0] < 1.95) & (P[:, 1] > .05) & (P[:, 1] < .95)
rms = float(np.sqrt(np.mean(c['om_ex'][tm][inn] ** 2)))
def noise(dv, mask):
    return float(np.sqrt(np.nanmean(dv[inn & mask] ** 2)) / rms)
R = dict(frame=int(tm), N=int(fr.shape[1]), rms_vort=rms)
variants = dict(raw=L.complex_from_delaunay(P, max_len_fac=1e9),
                median=L.complex_from_delaunay(P, max_len_fac=3.0),
                peel=X.complex_peeled(P))
for name, cx in variants.items():
    opn = X.open_star_nodes(cx); f = L.edge_cochain(P, Vs, cx['edges'])
    s0, s1, _ = L.hodge_stars(P, cx, kind='cotan')
    dv_cot = L.node_divergence(f, cx, s0, s1); dv_gal = X.galerkin_div(P, cx, Vs)
    R[name] = dict(n_tri=int(len(cx['tri'])), n_open_star_nodes=int(opn.sum()), n_open_star_inner=int((opn & inn).sum()),
                   cotan_div_noise=noise(dv_cot, np.ones_like(opn)), cotan_div_noise_masked=noise(dv_cot, ~opn),
                   galerkin_div_noise=noise(dv_gal, np.ones_like(opn)), galerkin_div_noise_masked=noise(dv_gal, ~opn),
                   cotan_div_p99_open=float(np.nanpercentile(np.abs(dv_cot[opn & inn]), 99) / rms) if (opn & inn).any() else 0.0)
json.dump(R, open('res_ablation_complex.json', 'w'), indent=1)
print(json.dumps(R, indent=1))
