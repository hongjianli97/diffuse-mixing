"""Diffusion verification: (1) oracle decomposition 'equation vs density estimate', (2) conductance variants and
maximum principle, (3) snapshot-interval (time) convergence, (4) rho*D(T) ~ T^0.7 physics with T reconstructed from
positions (Rankine-Hugoniot entropy jumps).  usage: python scripts/v5_diffusion.py sod blast"""
import sys, json, time, numpy as np
sys.path[:0] = ["lhd", "scripts"]
import cases, lhdx as X, netdiff as ND
from bcom import ledger
from ledger_ub import thermo_from_ledger
exec(open("scripts/tr_b.py").read().split("# ================= D1")[0].split("R = {'D'")[0])
exec("def lag1d" + open("scripts/tr_b.py").read().split("def lag1d")[1].split("def polylen")[0])
RES = "results/res_v5_diffusion.json"
try: out = json.load(open(RES))
except Exception: out = {}
rel = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)) / np.std(b))   # paper definition: RMS error / std(reference)


def suite(fr, dt, c0, kap, m, rho_led, rho_ex, refn, inn, extra=()):
    o = {}; S, N, _ = fr.shape
    san = lambda r: np.where(np.isfinite(r) & (r > 0), r, np.nanmedian(np.where(r > 0, r, np.nan)))
    dt_ = np.stack([san(X.dtfe_density(fr[s], m)) for s in range(S)]); sph = np.stack([san(X.sph_density(fr[s], m)) for s in range(S)])
    runs = [("ledger_elem_arith (paper)", dict(rho=rho_led)), ("oracle_exact_rho", dict(rho=rho_ex)), ("dtfe_rho", dict(rho=dt_)),
            ("sph_rho", dict(rho=sph)), ("ledger_elem_harm", dict(rho=rho_led, mode="elem_harm")),
            ("ledger_edge_harm", dict(rho=rho_led, mode="edge_harm")), ("area_metric", dict(rho=rho_led, form="volume")),
            ("ledger_nsub2", dict(rho=rho_led, nsub=2)), ("ledger_nsub4", dict(rho=rho_led, nsub=4)),
            ("ledger_CN", dict(rho=rho_led, theta=0.5)), ("oracle_nsub4", dict(rho=rho_ex, nsub=4))] + list(extra)
    for lab, kw in runs:
        t0 = time.time(); kw = dict(kw); rho = kw.pop("rho"); form = kw.pop("form", "mass")
        c, dg = ND.run2(fr, dt, c0, kap, form=form, m=m, rho=rho, **kw)
        o[lab] = dict(relL2=rel(c[inn], refn[inn]), overshoot=dg["overshoot"], pos_offdiag=dg["pos_offdiag_mean"], secs=round(time.time() - t0, 1))
        print("  %-28s L2 %.4f%%  overshoot %.1e  K_ij>0 per step %.1f" % (lab, 100 * o[lab]["relL2"], dg["overshoot"], dg["pos_offdiag_mean"]), flush=True)
    return o


if "sod" in sys.argv:
    cs = cases.case_sod(); fr, dt = cs["frames"], cs["dt"]; S, N = fr.shape[:2]; kap = 1e-3
    c0f = lambda x0: 0.5 * (1 + np.tanh(x0 / 0.02))
    fine = cases.case_sod(h=2e-4, Ly=2e-4, t0=0.04, t1=0.24, S=401); o_ = np.argsort(fine["x0"]); x0f = fine["x0"][o_]
    dmu = np.where(x0f > 0, 0.125, 1.0) * 2e-4; rhoF = fine["rho"][:, o_]; pF = fine["p"][:, o_]; tF = fine["t"]
    fi = lambda t: min(int(round((t - tF[0]) / (tF[1] - tF[0]))), len(tF) - 1)
    hm = lambda k: 2 * k[1:] * k[:-1] / (k[1:] + k[:-1])
    cref = lag1d(dmu, lambda t: hm(rhoF[fi(t)] ** 2), c0f(x0f), tF, kap)
    lnJ, hot, bad, tau = ledger(fr); rho = cs["rho"][0][None] * np.exp(-lnJ); inn = cs["interior"](fr[0])
    refn = np.interp(cs["x0"], x0f, cref); print("SOD (kappa const)", flush=True)
    o = suite(fr, dt, c0f(cs["x0"]), kap, cs["mass"], rho, cs["rho"], refn, inn)
    # ---- rho D(T) = kappa (T/T_ref)^0.7 ; T_ref = 1 (left state).  T reconstructed from positions.
    pR, TR = thermo_from_ledger(lnJ, hot, cs["rho"][0], cs["p"][0]); Tex = cs["p"] / cs["rho"]
    o["thermo_T_error"] = dict(med=float(np.median(np.abs(TR[-1] / Tex[-1] - 1)[inn])), p95=float(np.quantile(np.abs(TR[-1] / Tex[-1] - 1)[inn], .95)))
    print("  thermo reconstruction |T/T_ex-1| med %.2e p95 %.2e" % (o["thermo_T_error"]["med"], o["thermo_T_error"]["p95"]))
    crefT = lag1d(dmu, lambda t: hm(rhoF[fi(t)] * (pF[fi(t)] / rhoF[fi(t)]) ** 0.7), c0f(x0f), tF, kap); refT = np.interp(cs["x0"], x0f, crefT)
    def Tint(T, s):
        s0 = min(int(np.floor(s)), S - 2); f = s - s0; return (1 - f) * T[s0] + f * T[s0 + 1]
    cm = lambda T: (lambda P, tri, r, s: kap * (Tint(T, s)[tri] ** 0.7).mean(1))
    ca = lambda T: (lambda P, tri, r, s: kap * (Tint(T, s)[tri] ** 0.7 / r[tri]).mean(1))
    oT = {}
    for lab, form, fn in (("mass_T_exact", "mass", cm(Tex)), ("mass_T_from_positions", "mass", cm(TR)), ("area_T_from_positions", "volume", ca(TR))):
        c, dg = ND.run2(fr, dt, c0f(cs["x0"]), kap, form=form, m=cs["mass"], rho=rho, cond_fn=fn)
        oT[lab] = rel(c[inn], refT[inn]); print("  rhoD(T): %-24s L2 %.4f%%" % (lab, 100 * oT[lab]), flush=True)
    oT["pure_advection"] = rel(c0f(cs["x0"])[inn], refT[inn]); o["rhoD_T07"] = oT
    out["sod"] = o; json.dump(out, open(RES, "w"), indent=1)

if "blast" in sys.argv:
    cb = cases.case_blast(); fr, dt = cb["frames"], cb["dt"]; S, N = fr.shape[:2]; kap = 2e-4; p, R0, t0 = 1 / 2.4, 0.2, 1.0
    c0f = lambda rs: 0.5 * (1 + np.tanh((rs - 0.5) / 0.04))
    rsF = np.linspace(1e-4, 1.2, 6000); dr = rsF[1] - rsF[0]; rho0F = np.where(rsF <= R0, 6.0, 1.0); dmu = 2 * np.pi * rsF * rho0F * dr
    def kin(rs, t):
        tc = t0 * (rs / R0) ** 2; ins = rs <= R0; post = (~ins) & (t >= tc)
        r = np.where(ins, rs * (t / t0) ** p, np.where(post, rs * (np.maximum(t, tc) / tc) ** p, rs))
        lnJ_ = np.where(ins, 2 * p * np.log(t / t0), np.where(post, -np.log(6.0) + 2 * p * np.log(np.maximum(t, tc) / tc), 0.0))
        return r, np.where(ins, 6.0, 1.0) * np.exp(-lnJ_)
    def condb(t):
        r, rh = kin(rsF, t); k = 4 * np.pi ** 2 * r ** 2 * rh ** 2; return 2 * k[1:] * k[:-1] / (k[1:] + k[:-1])
    cref = lag1d(dmu, condb, c0f(rsF), np.linspace(1.0, 25.0, 2401), kap)
    lnJ, hot, bad, tau = ledger(fr); rho0 = np.where(cb["rs"] <= R0, 6.0, 1.0); rho = rho0[None] * np.exp(-lnJ)
    rho_ex = rho0[None] * np.exp(-cb["lnJ_ex"]); m = rho0 * X.star_area(fr[0]) / 3.0
    inn = cb["interior"](fr[0]); refn = np.interp(cb["rs"], rsF, cref); print("BLAST", flush=True)
    out["blast"] = suite(fr, dt, c0f(cb["rs"]), kap, m, rho, rho_ex, refn, inn); json.dump(out, open(RES, "w"), indent=1)
