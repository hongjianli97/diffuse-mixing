"""Six cases with identical fields: fr, t, dt, h, om (reference vorticity at tracers), lnJ_ref (window ledger),
inner (interior mask for statistics), circ window, interface (band nodes, T0, reference ln lambda)."""
import pickle, numpy as np, cases, lhdx as X
import sys
sys.path.insert(0, "/home/claude/stretch/run")


def _lo_omega(fr, t, a=1.0, d=6.0, Gam=2 * np.pi):
    W = Gam * (1 - np.exp(-d ** 2 / a ** 2)) / (np.pi * d ** 2); om = np.zeros(fr.shape[:2])
    for s, tt in enumerate(t):
        c = 0.5 * d * np.array([np.cos(W * tt), np.sin(W * tt)])
        for cc in (c, -c):
            om[s] += Gam / (np.pi * a ** 2) * np.exp(-((fr[s] - cc) ** 2).sum(1) / a ** 2)
    return om


def load(name):
    if name == "DG":
        c = cases.case_dg(); fr = c["frames"]; P0 = fr[0]
        inner = (P0[:, 0] > .05) & (P0[:, 0] < 1.95) & (P0[:, 1] > .05) & (P0[:, 1] < .95)
        band = np.where(np.abs(P0[:, 0] - 0.5) < 0.6 * c["h"])[0]; T0 = np.tile([0.0, 1.0], (len(P0), 1))
        out = dict(name=name, fr=fr, t=c["t"], dt=c["dt"], h=c["h"], om=c["om_ex"], lnJ_ref=np.zeros(fr.shape[:2]),
                   inner=inner, circ=(1, 25), band=band, T0=T0, kind="incompressible", ftle=c["ftle_ex"])
        import cases_mix as CM
        vel = cases.dg_fields()[0]; S = len(fr)
        ex = CM.exact_lnlam(vel, P0[band], T0[band], c["t"], [S - 1], 8)
        out["lam_ref"] = dict(point=ex[S - 1], mean=float(ex[S - 1].mean()))
        return out
    if name == "LO":
        c = cases.case_lo(a=1.0, Rd=6.5, N=8000); fr = c["frames"]; P0 = fr[0]
        inner = np.hypot(P0[:, 0], P0[:, 1]) < 6.0
        d0 = P0 - [-3.0, 0.0]; r0 = np.hypot(*d0.T); band = np.where(np.abs(r0 - 1.5) < 0.6 * c["h"])[0]
        T0 = np.c_[-d0[:, 1], d0[:, 0]] / np.maximum(r0, 1e-12)[:, None]
        out = dict(name=name, fr=fr, t=c["t"], dt=c["dt"], h=c["h"], om=_lo_omega(fr, c["t"]), lnJ_ref=np.zeros(fr.shape[:2]),
                   inner=inner, circ=(1, 20), band=band, T0=T0, kind="incompressible")
        import cases_mix as CM
        S = len(fr); ex = CM.exact_lnlam(CM.lo_vel, P0[band], T0[band], c["t"], [S - 1], 10)
        out["lam_ref"] = dict(point=ex[S - 1], mean=float(ex[S - 1].mean()))
        return out
    if name == "SOD":
        c = cases.case_sod(); fr = c["frames"]; P0 = fr[0]
        inner = c["interior"](P0) & (P0[:, 0] > -0.45) & (P0[:, 0] < 0.45)
        band = np.where((np.abs(P0[:, 0] - 0.1) < 0.6 * c["h"]) & c["interior"](P0))[0]
        return dict(name=name, fr=fr, t=c["t"], dt=c["dt"], h=c["h"], om=np.zeros(fr.shape[:2]), lnJ_ref=c["lnJ_ex"],
                    inner=inner, circ=(1, 37), band=band, T0=np.tile([0.0, 1.0], (len(P0), 1)), kind="shock",
                    lam_ref=dict(point=np.zeros(len(band)), mean=0.0))
    if name == "BLAST":
        c = cases.case_blast(); fr = c["frames"]; P0 = fr[0]; S = len(fr)
        inner = c["interior"](P0); rs = c["rs"]
        band = np.where(np.abs(rs - 0.5) < 0.6 * c["h"])[0]
        T0 = np.c_[-P0[:, 1], P0[:, 0]] / rs[:, None]
        ex = np.log(np.hypot(*fr[S - 1][band].T) / rs[band])
        return dict(name=name, fr=fr, t=c["t"], dt=c["dt"], h=c["h"], om=np.zeros(fr.shape[:2]), lnJ_ref=c["lnJ_ex"] - c["lnJ_ex"][0],
                    inner=inner, circ=(1, 58), band=band, T0=T0, kind="shock", lam_ref=dict(point=ex, mean=float(ex.mean())))
    if name == "SB":
        D = pickle.load(open("/home/claude/bench/sb_dns.pkl", "rb")); fr = D["frames"]; E = D["E"]; P0 = fr[0]
        inner = (P0[:, 0] > 0.32) & (P0[:, 0] < 1.23) & (P0[:, 1] < 0.40)
        rho = E["rho"].astype(float)
        out = dict(name=name, fr=fr, t=D["t"], dt=D["dt"], h=D["h"], om=E["om"].astype(float), lnJ_ref=np.log(rho[0][None] / rho),
                   inner=inner, circ=(1, 40), kind="dns")
        M = pickle.load(open("/home/claude/stretch/run/dns_sb.pkl", "rb")); frm = M["frames"]; S = len(frm)
        c0 = np.array([M["xb"], 0.0]); d0 = frm[0] - c0; r0 = np.hypot(*d0.T)
        out["stretch_fr"] = frm; out["stretch_dt"] = M["dt"]
        out["band"] = np.where(np.abs(r0 - M["R"]) < 0.6 * 0.005)[0]
        out["T0"] = np.c_[-d0[:, 1], d0[:, 0]] / np.maximum(r0, 1e-12)[:, None]
        out["lam_ref"] = dict(point=None, mean=float(M["mstats"][S - 1][1]["mean_ln"]),
                              side=[float(m["mean_ln"]) for m in M["mstats"][S - 1]])
        return out
    if name == "SV":
        D = pickle.load(open("/home/claude/bench/sv_dns.pkl", "rb")); k = 31
        fr = D["fr_in"][:k]; Ev = D["Ev"]; P0 = fr[0]; rho = Ev["rho"][:k].astype(float)
        out = dict(name=name, fr=fr, fr_off=D["fr_off"][:k], t=D["t"][:k], dt=D["dt"], h=0.005, om=Ev["om"][:k].astype(float),
                   lnJ_ref=np.log(rho[0][None] / rho), inner=np.ones(len(P0), bool), circ=(1, 28), kind="dns")
        M = pickle.load(open("/home/claude/stretch/run/dns_sv.pkl", "rb")); frm = M["frames"]
        Sm = int(np.argmin(np.abs(M["t"] - 0.6))) + 1; frm = frm[:Sm]
        c0 = np.array([M["xv"], M["yv"]]); d0 = frm[0] - c0; r0 = np.hypot(*d0.T)
        out["stretch_fr"] = frm; out["stretch_dt"] = M["dt"]
        out["band"] = np.where(np.abs(r0 - 2 * M["Rv"]) < 0.6 * 0.005)[0]
        out["T0"] = np.c_[-d0[:, 1], d0[:, 0]] / np.maximum(r0, 1e-12)[:, None]
        out["lam_ref"] = dict(point=None, mean=float(M["mstats"][Sm - 1][0]["mean_ln"]))
        return out
    raise ValueError(name)


def phi_reference(om, fr, dt):
    W = np.array([X.star_area(fr[q]) for q in range(len(fr))]); wb = np.nansum(om * W, 1) / np.nansum(W, 1)
    d = om - wb[:, None]
    return 0.5 * np.trapezoid(d, dx=dt, axis=0), np.trapezoid(np.abs(d), dx=dt, axis=0)
