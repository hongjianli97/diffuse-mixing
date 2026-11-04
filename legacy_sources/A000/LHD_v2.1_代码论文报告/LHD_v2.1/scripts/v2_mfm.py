"""Verification on manufactured flow maps (no DNS in the loop).

E1 planar shock      : ledger exact?  (theta, eta, slip, snapshot Courant Cs); deferral on/off; block signature
E2 cylindrical shock : ledger error vs curvature h/Rs (converging shock), observed order; block-signature deviation
E3 smooth map        : ledger error vs h (observed order)
E4 thin shear layer  : chained-tensor stretch error vs delta/h (information limit)
usage: python scripts/v2_mfm.py E1 E2 E3 E4
"""
import sys, json, time, numpy as np
sys.path[:0] = ["lhd", "scripts"]
import mfm, lhdx as X, classify as C, stretch_est as SE
from ledger_ub import chained_lnJ_ind
from lhdp import pq

RES = "results/res_v2_mfm.json"
try:
    out = json.load(open(RES))
except Exception:
    out = {}


def run_ledger(fr, defer=4, block=5):
    hot, R, tau = C.hot_steps(fr); S, N, _ = fr.shape
    bad = np.zeros((N, S), bool); bad[:, :-1] |= hot; bad[:, 1:] |= hot
    lnJ, I, dg = chained_lnJ_ind(fr, bad, block=block, max_defer=defer)
    return lnJ, I, dg, hot


def err_stats(e):
    e = np.abs(e[np.isfinite(e)])
    return dict(med=float(np.median(e)), p95=float(np.quantile(e, .95)), max=float(e.max()), n=int(e.size))


def block_signature(fr, tc, tri, t):
    """direct three-point map between the last frame with all three vertices un-crossed and the first frame with
    all three crossed (= the composition of all step maps of the block, exactly)."""
    tmin, tmax = tc[tri].min(1), tc[tri].max(1)
    sa = np.searchsorted(t, tmin, side="right") - 1; sb = np.searchsorted(t, tmax, side="left")
    ok = (sa >= 0) & (sb < len(t)) & (sb > sa)
    tri, sa, sb = tri[ok], sa[ok], sb[ok]
    E0 = np.stack([fr[sa, tri[:, 1]] - fr[sa, tri[:, 0]], fr[sa, tri[:, 2]] - fr[sa, tri[:, 0]]], -1)
    E1 = np.stack([fr[sb, tri[:, 1]] - fr[sb, tri[:, 0]], fr[sb, tri[:, 2]] - fr[sb, tri[:, 0]]], -1)
    F = E1 @ np.linalg.inv(E0); p, q = pq(F); J = np.linalg.det(F)
    dev_mu = np.abs(np.abs(q / p) - np.tanh(0.5 * np.abs(np.log(J))))
    return np.abs(np.angle(p)), dev_mu, ok


if "E1" in sys.argv:
    o = {}
    for th in (0.0, 0.5):
        for eta in (1.5, 3.0, 6.0):
            for Cs in (0.5, 1.0, 2.0, 4.0):
                c = mfm.planar_shock(h=0.02, eta=eta, theta=th, u1=(0.3 * -np.sin(th), 0.3 * np.cos(th)), Cs=Cs)
                fr = c["frames"]; m = c["interior"](fr[0])
                lnJ, I, dg, hot = run_ledger(fr)
                e = np.exp(-(lnJ[-1] - c["lnJ_ex"][-1])) - 1
                key = "th%.1f_eta%g_Cs%g" % (th, eta, Cs)
                rec = dict(err=err_stats(e[m]), indicator_p95=float(np.quantile(I[-1][m], .95)), fallbacks=dg["fallback_events"],
                           detected=float(hot.any(1)[m].mean()))
                if th == 0.5 and eta == 3.0:
                    lnJ0, _, _, _ = run_ledger(fr, defer=0)
                    rec["no_deferral_err"] = err_stats((np.exp(-(lnJ0[-1] - c["lnJ_ex"][-1])) - 1)[m])
                    tri = X.complex_peeled(fr[0])["tri"]; ang, dev, okb = block_signature(fr, c["tc"], tri, c["t"])
                    rec["block_argp_max"] = float(ang.max()); rec["block_mu_dev_max"] = float(dev.max())
                o[key] = rec
                print(key, json.dumps(rec), flush=True)
    out["E1_planar"] = o; json.dump(out, open(RES, "w"), indent=1)

if "E2" in sys.argv:
    o = {}
    for h in (0.04, 0.02, 0.01):
        t0 = time.time(); c = mfm.cylinder_shock(h=h, eta=3.0, Cs=1.0)
        fr = c["frames"]; m = c["interior"](fr[0]) & (c["Rs_cross"] > 0.3)
        lnJ, I, dg, hot = run_ledger(fr)
        e = np.abs(np.exp(-(lnJ[-1] - c["lnJ_ex"][-1])) - 1); x = h / c["Rs_cross"]
        bins = np.array([0, 0.02, 0.04, 0.08, 0.16])
        tab = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = m & (x >= lo) & (x < hi)
            if mm.sum() > 30:
                tab.append(dict(h_over_Rs=[float(lo), float(hi)], med=float(np.median(e[mm])), p95=float(np.quantile(e[mm], .95)), n=int(mm.sum())))
        tri = X.complex_peeled(fr[0])["tri"]; ang, dev, okb = block_signature(fr, c["tc"], tri, c["t"])
        xr = (h / c["Rs_cross"][tri[okb]].mean(1)); sig = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = (xr >= lo) & (xr < hi)
            if mm.sum() > 30:
                sig.append(dict(h_over_Rs=[float(lo), float(hi)], argp_med=float(np.median(ang[mm])), mudev_med=float(np.median(dev[mm]))))
        o["h%g" % h] = dict(err_all=err_stats(e[m]), by_curvature=tab, signature=sig, fallbacks=dg["fallback_events"], secs=time.time() - t0)
        print("h", h, json.dumps(o["h%g" % h]), flush=True)
    out["E2_cylinder"] = o; json.dump(out, open(RES, "w"), indent=1)

if "E3" in sys.argv:
    o = {}
    for h in (0.04, 0.02, 0.01):
        c = mfm.smooth_map(h=h); fr = c["frames"]; m = c["interior"](fr[0])
        lnJ, I, dg, hot = run_ledger(fr)
        d = lnJ[-1] - c["lnJ_ex"][-1]
        mid = len(fr) // 2; dmid = lnJ[mid] - c["lnJ_ex"][mid]
        o["h%g" % h] = dict(err_final=err_stats(d[m]), err_mid=err_stats(dmid[m]), hot_frac=float(hot[m].mean()),
                            spearman_err_vs_indicator=float(np.corrcoef(np.argsort(np.argsort(np.abs(dmid[m]))), np.argsort(np.argsort(I[mid][m])))[0, 1]))
        print("h", h, json.dumps(o["h%g" % h]), flush=True)
    hs = np.array([0.04, 0.02, 0.01]); med = np.array([o["h%g" % h]["err_mid"]["med"] for h in hs])
    o["observed_order_mid_median"] = float(np.polyfit(np.log(hs), np.log(med), 1)[0])
    print("order", o["observed_order_mid_median"])
    out["E3_smooth"] = o; json.dump(out, open(RES, "w"), indent=1)

if "E4" in sys.argv:
    o = {}; h = 0.01
    for r in (0.5, 1.0, 2.0, 4.0, 8.0):
        c = mfm.shear_layer(h=h, delta=r * h); fr = c["frames"]; P0 = fr[0]
        idx = np.where((np.abs(P0[:, 0]) < 0.06) & (np.abs(P0[:, 1]) < 0.2))[0]
        Fo = SE.chained_F(fr, h, idx=idx, keep=[len(fr) - 1])[len(fr) - 1][0]
        lnlam, A = SE.vector_lnlam(Fo, np.tile([0.0, 1.0], (len(idx), 1)))
        ex = c["lnlam_ex"][-1][idx]; layer = np.abs(P0[idx, 1]) < 2 * c["delta"]
        o["delta/h=%g" % r] = dict(abs_err_med_layer=float(np.median(np.abs(lnlam - ex)[layer])),
                                   rel_err_med_layer=float(np.median(np.abs(lnlam - ex)[layer] / np.maximum(ex[layer], 1e-12))),
                                   exact_med_layer=float(np.median(ex[layer])), A_med_layer=float(np.median(A[layer])))
        print(r, json.dumps(o["delta/h=%g" % r]), flush=True)
    out["E4_shear"] = o; json.dump(out, open(RES, "w"), indent=1)
