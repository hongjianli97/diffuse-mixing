"""Resolution test: same Lamb-Oseen pair, k fixed (8,12), N = 1000 / 4000 / 16000."""
import json, pickle, numpy as np, cases, lhdx as X, afrc_engine as AF
out = {}
for N in (1000, 4000, 16000):
    c = cases.case_lo(a=1.0, Rd=6.5, N=N); fr = c['frames']; P0 = fr[0]
    AF._CAP_W = []; F = AF.afrc_fields(np.ascontiguousarray(np.transpose(fr, (1, 0, 2))))
    w = np.nanmean(np.stack(AF._CAP_W), 0); AF._CAP_W = None
    r = np.minimum(np.hypot(*(P0 - c['centres'][0]).T), np.hypot(*(P0 - c['centres'][1]).T))
    prof = lambda v: [float(np.nanmean(v[(r >= a) & (r < a + 0.25)]) / np.nanquantile(v, 0.99)) for a in np.arange(0, 2.5, 0.25)]
    out[N] = dict(h=c['h'], core_radius_in_spacings=1.0 / c['h'], AFRC_E=prof(F['core_med']), weight=prof(w), lavd=prof(c['lavd_ex']),
                  J_AFRC=X.jaccard_top(F['core_med'], c['lavd_ex'], 0.9), J_w=X.jaccard_top(w, c['lavd_ex'], 0.9))
    print(N, json.dumps({k: (np.round(v, 2).tolist() if isinstance(v, list) else round(v, 3)) for k, v in out[N].items()}), flush=True)
json.dump(out, open('res_afrcE_res.json', 'w'), indent=1)
