"""T4 split on merged blocks vs DNS baroclinic torque (lhd/t4split.py) for the shock-bubble (inviscid, viscous) and RM.
usage: python scripts/v10_t4.py sb sbv rm"""
import sys, os, json, pickle, numpy as np, time
sys.path[:0] = ["lhd", "scripts"]
import lhdx as X, t4split as T4
from lhdp import tri_area
RES = "results/res_v10_t4.json"
try: out = json.load(open(RES))
except Exception: out = {}
FILES = {"sb": "runs/sb_dns_dx0.005.pkl", "sbv": "runs/sbv_dns_dx0.005.pkl", "rm": "runs/rm_dns_dx0.005.pkl"}
for nm in sys.argv[1:]:
    t0 = time.time(); D = pickle.load(open(FILES[nm], "rb")); fr, t, E = D["frames"], D["t"], D["E"]
    hot = np.load("runs/hot_%s_2-3-4-6-8-12-16-24-32_ad.npy" % nm); P0 = fr[0]
    tri0 = X.complex_peeled(P0)["tri"]; A0 = tri_area(P0, tri0); tri0 = np.where((A0 < 0)[:, None], tri0[:, [0, 2, 1]], tri0)
    if nm == "rm":
        a0 = D["a0"]; phi = (P0[:, 0] - 0.5 - a0 * np.cos(2 * np.pi * P0[:, 1])) / np.sqrt(1 + (2 * np.pi * a0 * np.sin(2 * np.pi * P0[:, 1])) ** 2)
        bands = (0.08, 0.03)
    else:
        phi = np.hypot(P0[:, 0] - D["xb"], P0[:, 1]) - D["R"]; bands = (0.15, 0.08)
    o = {}
    for w in bands:
        r = T4.split(fr, t, E["rho"], E["p"], hot, np.abs(phi) < w, tri0)
        tot = r["dG"].sum(); f = lambda x: float(np.sum(x))
        o["band %g" % w] = dict(G0=r["G0"], G_end=r["G_end"], total=f(r["dG"]), impulsive=f(r["imp"]), smooth=f(r["smo"]),
            impulsive_old=f(r["imp_old"]), smooth_old=float(tot - np.sum(r["imp_old"])), closure=r["closure"],
            dns_del_impulsive=f(r["dns_del_imp"]), dns_del_smooth=f(r["dns_del_smo"]), dns_tri_impulsive=f(r["dns_tri_imp"]),
            dns_tri_smooth=f(r["dns_tri_smo"]), hot_tri_frac=r["hot_tri_frac"], blocks=r["blocks"], blocks_dropped=r["blocks_dropped"],
            series={k: [float(x) for x in np.cumsum(r[k])] for k in ("dG", "imp", "smo", "imp_old", "dns_del_imp", "dns_del_smo", "dns_tri_smo")},
            t=[float(x) for x in r["t"]])
        i6 = int(np.argmin(np.abs(r["t"] - (0.6 if nm != "rm" else 0.8)))); S6 = lambda k: float(np.cumsum(r[k])[i6])
        b6 = dict(t=float(r["t"][i6]), total=S6("dG"), impulsive=S6("imp"), smooth=S6("smo"), dns_del_impulsive=S6("dns_del_imp"), dns_del_smooth=S6("dns_del_smo"))
        o["band %g" % w]["at_passage_end"] = b6
        b = o["band %g" % w]
        # solver-step torque (scripts/run_sb_torque.py): per tracer and snapshot interval, split with the SAME
        # filtered merged-block classification (interval s belongs to snapshots s and s+1)
        if nm == "sb" and os.path.exists("runs/sb_torque.npz"):
            Tq = np.load("runs/sb_torque.npz")["T"]; Sm = len(t); th = r["th"]; triR = r["triR"]
            H = np.zeros((Sm - 1, len(P0)), bool)
            for k in range(th.shape[0]):
                n = k + 1; vv = np.unique(triR[th[k]])
                H[n - 1, vv] = True; H[n, vv] = True
            lab = np.abs(phi) < w; ci = np.cumsum((Tq * H)[:, lab].sum(1)); cs = np.cumsum((Tq * ~H)[:, lab].sum(1))
            b.update(dns_step_impulsive=float(ci[-1]), dns_step_smooth=float(cs[-1]))
            i6 = int(np.argmin(np.abs(t[1:] - 0.6))); b["at_passage_end"].update(dns_step_impulsive=float(ci[i6]), dns_step_smooth=float(cs[i6]))
            b["series"]["dns_step_imp"] = [float(x) for x in ci]; b["series"]["dns_step_smo"] = [float(x) for x in cs]; b["t_step"] = [float(x) for x in t[1:]]
            print("    solver-step torque: imp %.4f smo %.4f (t=0.6: %.4f %.4f)" % (ci[-1], cs[-1], ci[i6], cs[i6]), flush=True)
        # acceleration-cochain reference (scripts/run_sb_accel.py): boundary integral of the DNS acceleration, viscous
        # term included, integrated at every solver step, split with the same classification
        af = {"sb": "runs/sb_accel_inv.npz", "sbv": "runs/sb_accel_visc.npz"}.get(nm)
        if af and os.path.exists(af):
            Aq = np.load(af)["A"]; Sm = len(t); th = r["th"]; triR = r["triR"]
            H = np.zeros((Sm - 1, len(P0)), bool)
            for k in range(th.shape[0]):
                n = k + 1; vv = np.unique(triR[th[k]]); H[n - 1, vv] = True; H[n, vv] = True
            lab = np.abs(phi) < w; ci = np.cumsum((Aq * H)[:, lab].sum(1)); cs = np.cumsum((Aq * ~H)[:, lab].sum(1))
            b.update(dns_acc_impulsive=float(ci[-1]), dns_acc_smooth=float(cs[-1]))
            i6 = int(np.argmin(np.abs(t[1:] - 0.6))); b["at_passage_end"].update(dns_acc_impulsive=float(ci[i6]), dns_acc_smooth=float(cs[i6]))
            b["series"]["dns_acc_imp"] = [float(x) for x in ci]; b["series"]["dns_acc_smo"] = [float(x) for x in cs]; b["t_step"] = [float(x) for x in t[1:]]
            print("    acceleration cochain: imp %.4f smo %.4f total %.4f (t=0.6: %.4f %.4f)" % (ci[-1], cs[-1], ci[-1] + cs[-1], ci[i6], cs[i6]), flush=True)
        # like-for-like: acceleration circulation on the SAME material triangles with the SAME hat kernel in time
        # (dGamma^n <-> A1[n-1] + A0[n]) and the SAME merged-block classification
        mf = {"sb": "runs/sb_accel_inv_mat.npz", "sbv": "runs/sb_accel_visc_mat.npz"}.get(nm)
        if mf and os.path.exists(mf):
            Z = np.load(mf); assert np.array_equal(Z["tri0"], tri0)
            sel = (np.abs(phi) < w)[tri0].all(1); A0m, A1m = Z["A0"][:, sel], Z["A1"][:, sel]
            dGd = A1m[:-1] + A0m[1:]; th = r["th"]
            ci = np.cumsum((dGd * th).sum(1)); cs = np.cumsum((dGd * ~th).sum(1))
            i6 = int(np.argmin(np.abs(r["t"] - 0.6)))
            b.update(dns_mat_impulsive=float(ci[-1]), dns_mat_smooth=float(cs[-1]))
            b["at_passage_end"].update(dns_mat_impulsive=float(ci[i6]), dns_mat_smooth=float(cs[i6]))
            b["series"]["dns_mat_imp"] = [float(x) for x in ci]; b["series"]["dns_mat_smo"] = [float(x) for x in cs]
            rel = float(np.median(np.abs(np.cumsum(dGd.sum(1)) - np.cumsum(r["dG"]))[r["t"] > 0.4]) / abs(np.sum(r["dG"])))
            b["dns_mat_total_median_rel_diff_t_gt_0.4"] = rel
            print("    material-triangle acceleration (like-for-like): imp %.4f smo %.4f total %.4f (t=0.6: %.4f %.4f); median |total diff| t>0.4: %.2f%%" % (
                ci[-1], cs[-1], ci[-1] + cs[-1], ci[i6], cs[i6], 100 * rel), flush=True)
        # DNS velocity circulation of the SAME material triangles at the frames (u, v stored at the tracers), with
        # the hat kernel of the position form: dGamma_DNS^n = (Gamma(n+1) - Gamma(n-1)) / 2; same classification
        if "u" in E:
            triR = r["triR"]; Gf = np.zeros((len(t), len(triR)))
            for n in range(len(t)):
                Uv = np.c_[E["u"][n], E["v"][n]]
                for j in range(3):
                    i0, i1 = triR[:, j], triR[:, (j + 1) % 3]; Gf[n] += (0.5 * (Uv[i0] + Uv[i1]) * (fr[n][i1] - fr[n][i0])).sum(1)
            dGv = 0.5 * (Gf[2:] - Gf[:-2]); th = r["th"]
            ci = np.cumsum((dGv * th).sum(1)); cs = np.cumsum((dGv * ~th).sum(1)); i6 = int(np.argmin(np.abs(r["t"] - (0.6 if nm != "rm" else 0.8))))
            b.update(dns_vel_impulsive=float(ci[-1]), dns_vel_smooth=float(cs[-1]))
            b["at_passage_end"].update(dns_vel_impulsive=float(ci[i6]), dns_vel_smooth=float(cs[i6]))
            b["series"]["dns_vel_imp"] = [float(x) for x in ci]; b["series"]["dns_vel_smo"] = [float(x) for x in cs]
            b["dns_vel_total_median_rel_diff_t_gt_0.4"] = float(np.median(np.abs(np.cumsum(dGv.sum(1)) - np.cumsum(r["dG"]))[r["t"] > 0.4]) / abs(np.sum(r["dG"])))
            print("    DNS velocity, same triangles (like-for-like): imp %.4f smo %.4f total %.4f (passage end: %.4f %.4f); median |total diff| t>0.4: %.2f%%" % (
                ci[-1], cs[-1], ci[-1] + cs[-1], ci[i6], cs[i6], 100 * b["dns_vel_total_median_rel_diff_t_gt_0.4"]), flush=True)
        # Samtaney-Zabusky local law: impulsive deposition per unit initial interface length vs angle from the upstream point
        if nm in ("sb", "sbv") and w == 0.15:
            fh = np.where(hot.any(1), hot.argmax(1), 10 ** 6); ok = fh < 10 ** 6
            up = np.sign(-np.corrcoef(fh[ok], P0[ok, 0])[0, 1]); c0 = np.array([D["xb"], 0.0])
            Cn = P0[r["triR"]].mean(1) - c0; th_ = np.arccos(np.clip(up * Cn[:, 0] / np.linalg.norm(Cn, axis=1), -1, 1))
            edges = np.linspace(0, np.pi, 13); ib = np.clip(np.digitize(th_, edges) - 1, 0, 11)
            gi = np.array([-r["imp_tri"][ib == k].sum() for k in range(12)]) / (D["R"] * np.diff(edges))
            gs = np.array([-r["smo_tri"][ib == k].sum() for k in range(12)]) / (D["R"] * np.diff(edges))
            sz = json.load(open("results/res_g6_sz.json")); G1 = sz["SZ_Gamma1prime"]; tc = 0.5 * (edges[1:] + edges[:-1])
            lawv = np.where(tc < np.pi / 2, G1 * np.sin(tc), G1)
            b["SZ"] = dict(theta=[float(x) for x in tc], imp_per_length=[float(x) for x in gi], smooth_per_length=[float(x) for x in gs], SZ1=[float(x) for x in lawv],
                           Gamma1prime=G1, imp_upstream=float(gi[tc < np.pi / 2].sum() * D["R"] * np.diff(edges)[0]), imp_downstream=float(gi[tc > np.pi / 2].sum() * D["R"] * np.diff(edges)[0]),
                           SZ_phase_ii=sz["SZ_phase_ii"], SZ_downstream=float(G1 * D["R"] * np.pi / 2))
            print("    SZ: imp upstream quarter %.4f (SZ phase ii %.4f), downstream %.4f (SZ %.4f)" % (b["SZ"]["imp_upstream"], b["SZ"]["SZ_phase_ii"], b["SZ"]["imp_downstream"], b["SZ"]["SZ_downstream"]), flush=True)
            print("    Gamma'(theta)/Gamma1':", " ".join("%.2f" % (x / G1) for x in gi), "| SZ1:", " ".join("%.2f" % (x / G1) for x in lawv), flush=True)
        print("%s band %g: total %.4f = imp %.4f + smo %.4f (closure %.1e) | old imp %.4f smo %.4f | DNS torque del imp %.4f smo %.4f, tri smo %.4f" % (
            nm, w, b["total"], b["impulsive"], b["smooth"], b["closure"], b["impulsive_old"], b["smooth_old"], b["dns_del_impulsive"], b["dns_del_smooth"], b["dns_tri_smooth"]), flush=True)
        print("    t=%.2f: total %.4f imp %.4f smo %.4f | DNS imp %.4f smo %.4f | blocks %d dropped %d" % (b6["t"], b6["total"], b6["impulsive"], b6["smooth"], b6["dns_del_impulsive"], b6["dns_del_smooth"], b["blocks"], b["blocks_dropped"]), flush=True)
    o["secs"] = time.time() - t0; out[nm] = o; json.dump(out, open(RES, "w"), indent=1)
