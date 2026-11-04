"""Curvature excess of the verbatim FRC6 kernel = strength x struct + contrast (unsharp mask)."""
import numpy as np, json, cases, afrc_engine as AF
c = cases.case_lo(a=1.0, Rd=6.5, N=8000); P = c["frames"][0]
E = AF._frc_build_topology([P], 8); u, v = E[:, 0], E[:, 1]; nt, te = AF._frc_triangles(E, len(P))
mid = 0.5 * (P[u] + P[v]); out = {}
for amp in (0.05, 1.0):                                  # small (linear) and FRC6-like (W0=0.05 -> 1.05) contrast
    r2 = np.minimum(((mid - [3, 0]) ** 2).sum(1), ((mid - [-3, 0]) ** 2).sum(1))
    we = AF.FRC5_W0 + amp * np.exp(-r2)                 # co-rotation bump of core radius a = 1
    act = np.ones(len(E), bool); an = np.bincount(np.r_[u, v], minlength=len(P)).astype(float)
    sv = np.zeros(len(P)); np.add.at(sv, u, we); np.add.at(sv, v, we); wv = sv / an
    F, nta = AF._frc5_masked_curvature(E, te, we, wv, act, return_ntri_act=True)
    st = 4 - an[u] - an[v] + 3 * nta; what = np.median(we)
    X = F - what * st                                    # FRC6 S3 excess (median baseline)
    # edge-graph Laplacian of w: own minus mean over edges sharing a vertex
    sw = np.zeros(len(P)); np.add.at(sw, u, we); np.add.at(sw, v, we)
    nbm = (sw[u] + sw[v] - 2 * we) / np.maximum(an[u] + an[v] - 2, 1)
    lap = we - nbm                                       # >0 on a concave shoulder, <0 at the foot
    A = np.c_[(we - what) * st, lap * an[u] ** 0 * (an[u] + an[v])]
    coef, *_ = np.linalg.lstsq(A, X, rcond=None); pred = A @ coef
    R2 = 1 - np.sum((X - pred) ** 2) / np.sum((X - X.mean()) ** 2)
    R2a = 1 - np.sum((X - A[:, :1] @ np.linalg.lstsq(A[:, :1], X, rcond=None)[0]) ** 2) / np.sum((X - X.mean()) ** 2)
    rr = np.sqrt(r2); prof = [float(np.mean(-X[(rr >= a) & (rr < a + .25)])) for a in np.arange(0, 2.5, .25)]
    out[amp] = dict(R2_two_terms=float(R2), R2_strength_only=float(R2a), coef=coef.tolist(),
                    struct_median=float(np.median(st)), struct_q10_q90=np.quantile(st, [.1, .9]).tolist(),
                    minusX_radial_profile=np.round(np.array(prof) / max(prof), 2).tolist(),
                    weight_profile=np.round([float(np.mean(we[(rr >= a) & (rr < a + .25)])) for a in np.arange(0, 2.5, .25)], 3).tolist())
print(json.dumps(out, indent=1)); json.dump(out, open("res_curv_decomp.json", "w"), indent=1)
