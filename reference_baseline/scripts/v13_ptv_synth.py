"""Synthetic particle-tracking error models on the shock-vortex DNS tracers (real PTV data are not available):
white noise (reference), spatially correlated noise (Gaussian correlation length l), temporally correlated noise
(AR(1)), a static calibration distortion (same smooth displacement field d(x) at every frame, sampled where the
particle is), and inhomogeneous seeding (keep probability 0.3-1, combined with white noise).  sigma = 0.02 h.
usage: python scripts/v13_ptv_synth.py cfg [...]   cfg in white spatial3 spatial8 temporal calib seeding"""
import sys, json, time, numpy as np
sys.path[:0] = ["lhd", "scripts"]
import classify as C
from vcases import load, st
from ledger_ub import chained_lnJ_ind, bad_from_hot
from ledger_v4 import ledger_v4
from scipy.spatial import cKDTree
RES = "results/res_v13_ptv_synth.json"
try: out = json.load(open(RES))
except Exception: out = {}
fr0, rho0, ref, m0 = load("sv"); d, _ = cKDTree(fr0[0]).query(fr0[0], k=2); h = float(np.median(d[:, 1])); sig = 0.02 * h
rng = np.random.default_rng(11)
def rff(P, l, M=64, amp=1.0, r=rng):      # random-Fourier Gaussian field, unit variance per component
    k = r.normal(size=(M, 2)) / l; ph = r.uniform(0, 2 * np.pi, (2, M)); a = r.normal(size=(2, M))
    return amp * np.sqrt(2.0 / M) * np.stack([np.cos(P @ k.T + ph[c]) @ a[c] for c in range(2)], 1)
for cfg in sys.argv[1:]:
    t0 = time.time(); fr, r0, rf, m = fr0.copy(), rho0, ref, m0
    S = len(fr)
    if cfg == "white": fr = fr + sig * rng.normal(size=fr.shape)
    elif cfg.startswith("spatial"):
        l = float(cfg[7:]) * h; fr = fr + sig * np.stack([rff(fr[s], l) for s in range(S)])
    elif cfg == "temporal":
        e = np.empty_like(fr); e[0] = rng.normal(size=fr[0].shape)
        for s in range(1, S): e[s] = 0.9 * e[s - 1] + np.sqrt(1 - 0.81) * rng.normal(size=fr[0].shape)
        fr = fr + sig * e
    elif cfg == "calib":
        r2 = np.random.default_rng(5); k = r2.normal(size=(16, 2)) / (10 * h); ph = r2.uniform(0, 2 * np.pi, (2, 16)); a = r2.normal(size=(2, 16))
        dfun = lambda P: 5 * sig * np.sqrt(2.0 / 16) * np.stack([np.cos(P @ k.T + ph[c]) @ a[c] for c in range(2)], 1)
        fr = np.stack([fr[s] + dfun(fr[s]) for s in range(S)])
    elif cfg == "seeding":
        P = fr[0]; p = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * P[:, 0] / 0.3) * np.cos(2 * np.pi * P[:, 1] / 0.3))
        keep = rng.uniform(size=len(P)) < p; fr = fr[:, keep] + sig * rng.normal(size=fr[:, keep].shape); r0, rf, m = rho0[keep], ref[keep], m0[keep]
    E = lambda lj: r0 * np.exp(-lj) / rf - 1
    l2, _, _ = chained_lnJ_ind(fr, bad_from_hot(C.hot_steps(fr)[0])); r4 = ledger_v4(fr, rho0=r0)
    o = dict(v2=st(E(l2[-1]), m), v4_polygon=st(E(r4["lnJ_poly"]), m), v4=st(E(r4["lnJ_v4"]), m), n=int(m.sum()),
             sigma_hat_over_h=r4["diag"]["noise_sigma_over_h"], tensor_share=r4["diag"]["tensor_share"])
    if cfg == "calib": o["distortion_rms_over_h"] = 5 * 0.02
    out[cfg] = o; json.dump(out, open(RES, "w"), indent=1)
    f = lambda q: "%.2f%%/%.2f%%" % (100 * q["med"], 100 * q["p95"])
    print("%-9s n=%5d sig_hat/h=%.4f | v2 %s | v4poly %s | v4 %s  %.0fs" % (cfg, o["n"], o["sigma_hat_over_h"], f(o["v2"]), f(o["v4_polygon"]), f(o["v4"]), time.time() - t0), flush=True)
