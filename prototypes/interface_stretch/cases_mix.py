"""Case set-ups for the mixing-attribution study: frames, labels, band, ledger, references."""
import numpy as np
import cases, lhd as L, lhdx as X
import stretch_est as SE

GAM, A_, D_ = 2 * np.pi, 1.0, 6.0
WR = GAM * (1 - np.exp(-D_ ** 2 / A_ ** 2)) / (np.pi * D_ ** 2)


def lo_vel(Xq, t):
    u = np.zeros_like(Xq)
    for sgn in (1, -1):
        cc = sgn * 0.5 * D_ * np.array([np.cos(WR * t), np.sin(WR * t)]); r = Xq - cc; r2 = (r * r).sum(1) + 1e-300
        u += (GAM / (2 * np.pi * r2) * (1 - np.exp(-r2 / A_ ** 2)))[:, None] * np.c_[-r[:, 1], r[:, 0]]
    return u


def exact_lnlam(vel, X0, T0, t, frames_idx, nsub, delta=1e-5):
    """exact tangent stretch of the level set through each point: integrate x +- delta*t0 (RK4)."""
    Q = np.r_[X0 + delta * T0, X0 - delta * T0]; n = len(X0); out = {}; tt = t[0]
    for s in range(1, max(frames_idx) + 1):
        hh = (t[s] - t[s - 1]) / nsub
        for _ in range(nsub):
            k1 = vel(Q, tt); k2 = vel(Q + .5 * hh * k1, tt + .5 * hh); k3 = vel(Q + .5 * hh * k2, tt + .5 * hh); k4 = vel(Q + hh * k3, tt + hh)
            Q = Q + hh * (k1 + 2 * k2 + 2 * k3 + k4) / 6; tt += hh
        if s in frames_idx:
            out[s] = np.log(np.linalg.norm(Q[:n] - Q[n:], axis=1) / (2 * delta))
    return out


def add_noise(fr, h, nz, seed=11):
    if nz <= 0:
        return fr
    rng = np.random.default_rng(seed)
    return fr + nz * h * rng.standard_normal(fr.shape)


def setup_dg(nx=160, noise=0.0, ref=True, dmax=0.002):
    c = cases.case_dg(nx=nx, ny=nx // 2); fr = add_noise(c["frames"], c["h"], noise); h = c["h"]; P0 = c["frames"][0]
    phi0 = P0[:, 0] - 0.5; gX = np.ones(len(P0))
    band = np.where(np.abs(phi0) < 0.6 * h)[0]; band_sorted = band[np.argsort(P0[band, 1])]
    lnJ = X.chained_lnJ_v2(fr, None)
    S = fr.shape[0]; t = c["t"]
    T0 = np.tile([0.0, 1.0], (len(P0), 1))
    out = dict(name="DG", frames=fr, h=h, phi0=phi0, gX=gX, band=band, band_sorted=band_sorted, closed=False, lnJ=lnJ,
               bad=None, t=t, eval_frames=list(range(5, S, 5)), key_frames=[25, 50], field_frames=[50], comp_exact=lambda s: 0.0, t0=T0)
    vel = cases.dg_fields()[0]
    out["exact"] = exact_lnlam(vel, P0[band], T0[band], t, out["eval_frames"], 8)
    if ref:
        mc = SE.MarkerCurve(lambda s: np.c_[0.5 + 0 * s, s], 0.003, 0.997, 400, False, dmax)
        R = {}
        for s in range(1, S):
            SE.rk4_curve(mc, vel, t[s - 1], t[s], 16)
            if s in out["eval_frames"]:
                st = mc.stats(); R[s] = dict(mean_ln=st["mean_ln"], ln_total=st["ln_total"], s_mid=st["s_mid"], ln=st["ln"], X=mc.X.copy())
        out["ref"] = R
    return out


def setup_lo(N=8000, noise=0.0, ref=True, dmax=0.01):
    c = cases.case_lo(a=1.0, Rd=6.5, N=N); fr = add_noise(c["frames"], c["h"], noise); h = c["h"]; P0 = c["frames"][0]
    d0 = P0 - [-3.0, 0.0]; r0 = np.hypot(*d0.T); phi0 = r0 - 1.5; gX = np.ones(len(P0))
    band = np.where(np.abs(phi0) < 0.6 * h)[0]; band_sorted = band[np.argsort(np.arctan2(d0[band, 1], d0[band, 0]))]
    lnJ = X.chained_lnJ_v2(fr, None); S = fr.shape[0]; t = c["t"]
    T0 = np.c_[-d0[:, 1], d0[:, 0]] / np.maximum(r0, 1e-12)[:, None]
    out = dict(name="LO", frames=fr, h=h, phi0=phi0, gX=gX, band=band, band_sorted=band_sorted, closed=True, lnJ=lnJ,
               bad=None, t=t, eval_frames=[5, 10, 15, 20, 25, 30, 35, 40], key_frames=[5, 10, 20, 40], field_frames=[40],
               comp_exact=lambda s: 0.0, t0=T0)
    out["exact"] = exact_lnlam(lo_vel, P0[band], T0[band], t, out["eval_frames"], 10)
    if ref:
        mc = SE.MarkerCurve(lambda s: np.c_[-3 + 1.5 * np.cos(s), 1.5 * np.sin(s)], -np.pi, np.pi, 600, True, dmax)
        R = {}
        for s in range(1, S):
            SE.rk4_curve(mc, lo_vel, t[s - 1], t[s], 20)
            if s in out["eval_frames"]:
                st = mc.stats(); R[s] = dict(mean_ln=st["mean_ln"], ln_total=st["ln_total"], s_mid=st["s_mid"], ln=st["ln"], X=mc.X.copy())
        out["ref"] = R
    return out


def setup_sod(h=0.004, noise=0.0):
    cs = cases.case_sod(h=h); fr = add_noise(cs["frames"], h, noise); P0 = cs["frames"][0]; S, N = fr.shape[:2]
    phi0 = P0[:, 0] - 0.1; gX = np.ones(N)
    Ly = 0.12
    band = np.where((np.abs(phi0) < 0.6 * h) & (P0[:, 1] > 0.1 * Ly) & (P0[:, 1] < 0.9 * Ly))[0]
    band_sorted = band[np.argsort(P0[band, 1])]
    bad = L.vertex_bad_mask(cs["ev_pidx"], cs["ev_step"], N, S, halo=1); lnJ = X.chained_lnJ_v2(fr, bad)
    comp_ex = float(np.log(cs["info"]["rsR"] / 0.125))
    t = cs["t"]; ts_cross = 0.1 / cs["info"]["Ssh"]
    def comp_exact(s):
        return comp_ex if t[s] > ts_cross else 0.0
    R = {s: dict(mean_ln=0.0, ln_total=0.0, s_mid=None, ln=None, X=np.c_[fr[s][band_sorted, 0] * 0 + np.mean(cs['frames'][s][band, 0]), P0[band_sorted, 1]])
         for s in range(1, S)}
    ef = list(range(4, S, 4))
    return dict(name="SOD", frames=fr, h=h, phi0=phi0, gX=gX, band=band, band_sorted=band_sorted, closed=False, lnJ=lnJ, bad=bad,
                t=t, eval_frames=ef, key_frames=[40], field_frames=[40], comp_exact=comp_exact, ref=R, rho=cs["rho"],
                t0=np.tile([0.0, 1.0], (N, 1)), exact={s: np.zeros(len(band)) for s in ef},
                comp_per={s: -cs["lnJ_ex"][s][band] for s in ef})


def setup_blast(h=0.022, noise=0.0):
    cb = cases.case_blast(h=h); fr = add_noise(cb["frames"], h, noise); P0 = cb["frames"][0]; S, N = fr.shape[:2]
    rs = cb["rs"]; phi0 = rs - 0.5; gX = np.ones(N)
    band = np.where(np.abs(phi0) < 0.6 * h)[0]; band_sorted = band[np.argsort(np.arctan2(P0[band, 1], P0[band, 0]))]
    bad = L.vertex_bad_mask(cb["ev_pidx"], cb["ev_step"], N, S, halo=1); lnJ = X.chained_lnJ_v2(fr, bad)
    p, R0, t0 = 1 / 2.4, 0.2, 1.0; tc = t0 * (0.5 / R0) ** 2; t = cb["t"]
    def lam_ex(tt):
        return p * np.log(max(tt, tc) / tc)
    R = {}
    for s in range(1, S):
        th = np.linspace(-np.pi, np.pi, 721); rr = 0.5 * np.exp(lam_ex(t[s]))
        R[s] = dict(mean_ln=lam_ex(t[s]), ln_total=lam_ex(t[s]), s_mid=None, ln=None, X=np.c_[rr * np.cos(th), rr * np.sin(th)])
    comp = lambda s: float(-cb["lnJ_ex"][s][band].mean())
    ef = list(range(5, S, 5))
    T0 = np.c_[-P0[:, 1], P0[:, 0]] / rs[:, None]
    ex = {s: np.log(np.hypot(*cb["frames"][s][band].T) / rs[band]) for s in ef}
    return dict(name="BLAST", frames=fr, h=h, phi0=phi0, gX=gX, band=band, band_sorted=band_sorted, closed=True, lnJ=lnJ, bad=bad,
                t=t, eval_frames=ef, key_frames=[S - 1], field_frames=[S - 1], comp_exact=comp, ref=R, t0=T0, exact=ex,
                comp_per={s: -cb["lnJ_ex"][s][band] for s in ef})


def setup_dns(which, mode="in", noise=0.0):
    import pickle
    from trajnet import TrajectoryNetwork as TN
    D = pickle.load(open("dns_%s.pkl" % which, "rb"))
    fr = D["frames"] if mode == "in" else D["frames_off"]
    h = 0.005; fr = add_noise(fr, h, noise); P0 = D["frames"][0]; S, N = fr.shape[:2]
    if which == "sb":
        c = np.array([D["xb"], 0.0]); R0 = D["R"]; ef = list(range(5, S, 5)); kf = [15, 30, 50, 70]; ff = [30, 70]
    else:
        c = np.array([D["xv"], D["yv"]]); R0 = 2 * D["Rv"]; ef = list(range(5, S, 5)); kf = [10, 20, 30]; ff = [30]
    d0 = P0 - c; r0 = np.hypot(*d0.T); phi0 = r0 - R0; gX = np.ones(N)
    band = np.where(np.abs(phi0) < 0.6 * h)[0]; band_sorted = band[np.argsort(np.arctan2(d0[band, 1], d0[band, 0]))]
    net = TN(fr, D["dt"]); net.events(); bad = net.bad; lnJ = X.chained_lnJ_v2(fr, bad)
    T0 = np.c_[-d0[:, 1], d0[:, 0]] / np.maximum(r0, 1e-12)[:, None]
    rho = D["rho"].astype(float)
    comp_per = {s: np.log(rho[s][band] / rho[0][band]) for s in ef}
    ri = 1 if which == "sb" else 0
    ref = {s: dict(mean_ln=D["mstats"][s][ri]["mean_ln"], ln_total=D["mstats"][s][ri]["ln_total"], X=D["mstats"][s][ri]["X"],
                   s_mid=D["mstats"][s][ri]["s_mid"], ln=D["mstats"][s][ri]["ln"],
                   side=[(m["mean_ln"], m["ln_total"]) for m in D["mstats"][s]]) for s in ef}
    m = rho[0] * X.star_area(P0) / 3.0
    return dict(name=which.upper(), frames=fr, h=h, phi0=phi0, gX=gX, band=band, band_sorted=band_sorted, closed=(which == "sv"),
                lnJ=lnJ, bad=bad, t=D["t"], eval_frames=ef, key_frames=kf, field_frames=ff, ref=ref, t0=T0, comp_per=comp_per,
                mass=m, D=D, centre=c, R0=R0)
