import numpy as np, json, sys
import cases
from common import *
from lo_flow import vel, vort, cen
res = {}
for N in [2000, 8000, 32000]:
    c = cases.case_lo(a=1.0, Rd=6.5, N=N); P0 = c["frames"][0]; t = c["t"]
    fr, Fs = integrate(vel, P0, t, sub=10)
    A = run_afrc(fr); bex, lns = R_exact(Fs); o = summary(A, bex, lns)
    cand = A["cand"]; bt = A["ind"]["beta"]
    o["beta_bias_med"] = float(np.nanmedian(bt[cand]-bex[cand])); o["beta_absdiff_med"] = float(np.nanmedian(np.abs(bt[cand]-bex[cand])))
    res[N] = o; print(N, o["beta_bias_med"], o["beta_absdiff_med"], o["agree_with_exact"], flush=True)
json.dump(res, open("res_lo_res.json", "w"), indent=1)
