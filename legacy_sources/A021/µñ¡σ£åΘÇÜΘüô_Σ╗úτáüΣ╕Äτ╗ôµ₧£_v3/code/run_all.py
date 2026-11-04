"""python run_all.py GROUP   (GROUP in toy, dg, lo, sod, blast, sb, sv) -> out_GROUP.pkl"""
import sys, pickle, time, numpy as np
import cases_mix as CM, evalcore as EC, stretch_est as SE, lhdx as X

grp = sys.argv[1]
OUT = {}


def run_case(c, keep_per=True, fields=True, w=None):
    o, f, t0 = EC.evaluate(c["frames"], c["h"], c["phi0"], c["gX"], c["band"], c["band_sorted"], c["closed"], c["lnJ"],
                           c["eval_frames"], bad=c["bad"], t0=c["t0"], field_frames=c["field_frames"] if fields else (), w=w)
    res = {}
    for s in c["eval_frames"]:
        r = dict(mean=o[s]["mean"], t=float(c["t"][s]))
        if "exact" in c:
            r["exact_mean"] = float(np.mean(c["exact"][s]))
            r["medabs"] = {k: float(np.nanmedian(np.abs(o[s]["per"][k] - c["exact"][s]))) for k in ("pair", "poly", "vector", "covector", "combined")}
        if "ref" in c and s in c["ref"]:
            r["ref_mean"] = c["ref"][s]["mean_ln"]; r["ref_total"] = c["ref"][s]["ln_total"]
            if "side" in c["ref"][s]:
                r["ref_side"] = c["ref"][s]["side"]
        if "comp_per" in c and s in c["comp_per"]:
            r["comp_ref"] = float(np.mean(c["comp_per"][s]))
        elif "comp_exact" in c:
            r["comp_ref"] = float(c["comp_exact"](s))
        if keep_per and s in c["key_frames"]:
            r["per"] = {k: np.asarray(v, np.float32) for k, v in o[s]["per"].items()}
            if "exact" in c:
                r["per"]["exact"] = c["exact"][s].astype(np.float32)
            if "ref" in c and s in c["ref"] and c["ref"][s].get("X") is not None:
                r["refX"] = np.asarray(c["ref"][s]["X"], np.float32)
                if c["ref"][s].get("s_mid") is not None:
                    r["ref_s"] = np.asarray(c["ref"][s]["s_mid"], np.float32); r["ref_ln"] = np.asarray(c["ref"][s]["ln"], np.float32)
        res[s] = r
    out = dict(res=res, band=c["band"], h=c["h"], name=c["name"], P0band=c["frames"][0][c["band"]])
    if fields:
        out["fields"] = {s: {k: (np.asarray(v, np.float32) if k != "tri" else v) for k, v in d.items()} for s, d in f.items()}
        out["P0"] = c["frames"][0].astype(np.float32); out["phi0"] = c["phi0"].astype(np.float32)
    return out


t0 = time.time()
if grp == "dg":
    OUT["base"] = run_case(CM.setup_dg())
    for nx in (80, 320):
        OUT["nx%d" % nx] = run_case(CM.setup_dg(nx=nx, ref=False), keep_per=False, fields=False)
    for nz in (0.05, 0.1):
        OUT["noise%.2f" % nz] = run_case(CM.setup_dg(noise=nz, ref=False), keep_per=False, fields=False)
elif grp == "lo":
    OUT["base"] = run_case(CM.setup_lo())
    for N in (2000, 32000):
        OUT["N%d" % N] = run_case(CM.setup_lo(N=N, ref=False), keep_per=False, fields=False)
    for nz in (0.05, 0.1):
        OUT["noise%.2f" % nz] = run_case(CM.setup_lo(noise=nz, ref=False), keep_per=False, fields=False)
elif grp == "sod":
    OUT["base"] = run_case(CM.setup_sod())
    for h in (0.008, 0.002):
        OUT["h%g" % h] = run_case(CM.setup_sod(h=h), keep_per=False, fields=False)
    OUT["noise0.10"] = run_case(CM.setup_sod(noise=0.1), keep_per=False, fields=False)
elif grp == "blast":
    OUT["base"] = run_case(CM.setup_blast())
    for h in (0.044, 0.011):
        OUT["h%g" % h] = run_case(CM.setup_blast(h=h), keep_per=False, fields=False)
    OUT["noise0.10"] = run_case(CM.setup_blast(noise=0.1), keep_per=False, fields=False)
elif grp in ("sb", "sv"):
    c = CM.setup_dns(grp)
    OUT["base"] = run_case(c)
    if grp == "sb":
        OUT["base_mass"] = run_case(c, keep_per=False, fields=False, w=c["mass"])
        # compression check away from the smeared interface: 2h < |phi0| < 4h, each side
        P0 = c["frames"][0]; rho = c["D"]["rho"].astype(float); chk = {}
        for nm, sel in (("heavy", (c["phi0"] < -2 * c["h"]) & (c["phi0"] > -4 * c["h"])), ("light", (c["phi0"] > 2 * c["h"]) & (c["phi0"] < 4 * c["h"]))):
            idx = np.where(sel)[0]
            chk[nm] = {int(s): (float(np.mean(-c["lnJ"][s][idx])), float(np.mean(np.log(rho[s][idx] / rho[0][idx])))) for s in c["eval_frames"]}
        OUT["comp_check"] = chk
        OUT["grids"] = {k: v for k, v in c["D"]["grids"].items()}
        OUT["xc"], OUT["yc"] = c["D"]["xc"], c["D"]["yc"]
        OUT["mside"] = {s: [np.asarray(m["X"], np.float32) for m in c["D"]["mstats"][s]] for s in (30, 70)}
        for nz in (0.05, 0.1):
            OUT["noise%.2f" % nz] = run_case(CM.setup_dns(grp, noise=nz), keep_per=False, fields=False)
    else:
        OUT["grids"] = {k: v for k, v in c["D"]["grids"].items()}
        OUT["xc"], OUT["yc"] = c["D"]["xc"], c["D"]["yc"]
        OUT["off"] = run_case(CM.setup_dns(grp, mode="off"), keep_per=True, fields=False)
elif grp == "toy":
    # single Lamb-Oseen vortex: (A) concentric ring R=1.5 (pure shear along the interface, exact ln lambda = 0)
    #                           (B) off-centre ring (0.8,0) r=0.5 wound into a spiral
    Om = lambda r: (1 - np.exp(-r ** 2)) / np.maximum(r, 1e-12) ** 2
    def flow(P, t):
        r = np.hypot(*P.T); th = np.arctan2(P[:, 1], P[:, 0]) + Om(r) * t; return np.c_[r * np.cos(th), r * np.sin(th)]
    def mk(h, centre, Rr, times, box, seed=1, noise=0.0):
        rng = np.random.default_rng(seed); g = np.arange(-box, box, h); Xg, Yg = np.meshgrid(g, g)
        P0 = np.c_[Xg.ravel(), Yg.ravel()] + rng.uniform(-0.3, 0.3, (Xg.size, 2)) * h
        if centre[0] == 0:
            P0 = P0[np.abs(np.hypot(*P0.T) - Rr) < 0.5]
        fr = np.stack([flow(P0, t) for t in times]); fr = CM.add_noise(fr, h, noise)
        d0 = P0 - centre; r0 = np.hypot(*d0.T); phi0 = r0 - Rr
        band = np.where(np.abs(phi0) < 0.6 * h)[0]; bs = band[np.argsort(np.arctan2(d0[band, 1], d0[band, 0]))]
        T0 = np.c_[-d0[:, 1], d0[:, 0]] / r0[:, None]
        ex = {}
        for s in (range(1, len(times)) if len(times) < 100 else range(20, len(times), 20)):
            dl = 1e-5; Q = np.r_[P0[band] + dl * T0[band], P0[band] - dl * T0[band]]; Qt = flow(Q, times[s]); n = len(band)
            ex[s] = np.log(np.linalg.norm(Qt[:n] - Qt[n:], axis=1) / (2 * dl))
        ef = list(range(1, len(times))) if len(times) < 100 else list(range(20, len(times), 20))
        return dict(name="toy", frames=fr, h=h, phi0=phi0, gX=np.ones(len(P0)), band=band, band_sorted=bs, closed=True,
                    lnJ=np.zeros(fr.shape[:2]), bad=None, t=np.asarray(times), eval_frames=ef, key_frames=[ef[-1]], field_frames=[],
                    t0=T0, exact=ex, comp_exact=lambda s: 0.0)
    gam = abs(1.5 * (2 / 1.5) * (np.exp(-1.5 ** 2) - Om(1.5)))
    tA = np.r_[0, np.arange(1, 61) * (30 / gam) / 60]
    for h in (0.05, 0.025):
        for nz in (0.0, 0.05, 0.1):
            if h == 0.025 and nz > 0:
                continue
            OUT["A_h%g_n%g" % (h, nz)] = run_case(mk(h, np.array([0.0, 0.0]), 1.5, tA, 2.2, noise=nz), keep_per=False, fields=False)
            OUT["A_h%g_n%g" % (h, nz)]["gt"] = gam * tA
    tB = np.arange(321) * 0.25
    for h in (0.04, 0.02):
        OUT["B_h%g" % h] = run_case(mk(h, np.array([0.8, 0.0]), 0.5, tB, 1.9), keep_per=False, fields=False)
print(grp, "done %.0fs" % (time.time() - t0), flush=True)
pickle.dump(OUT, open("out_%s.pkl" % grp, "wb"))
