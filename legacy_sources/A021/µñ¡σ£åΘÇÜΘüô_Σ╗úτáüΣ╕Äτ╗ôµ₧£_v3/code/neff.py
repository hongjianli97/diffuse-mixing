import sys, json, numpy as np, tcx, afe_cases as AC, tcx_cases as TC
out = {}
for name in sys.argv[1:]:
    c = AC.load(name); fr = c["fr"]; S, N = fr.shape[:2]; m = c["inner"]; h = c["h"]
    phi_ref, _ = TC.phi_reference(c["om"], fr, c["dt"]); tri0 = tcx.cplx(fr[0])["tri"]; E0 = tcx.graph_edges(tri0)
    res = {}
    for lab, frx in [("clean", fr), ("noise", fr + np.random.default_rng(5).normal(0, 0.1*h, fr.shape))]:
        mus, _ = tcx.mu_threshold(frx); ch = tcx.run_chain(frx, mus, Ks=[tcx.cplx(frx[s])["tri"] for s in range(S-1)])
        phi, _, nseg, psib = tcx.rotation_fields(ch); d = ch["inc"] - psib[:, None]
        cc = tcx.cores(np.abs(phi), E0, m, 1e-3)[:40]; rows = []
        for b, dd, r in cc:
            nb = np.r_[r, E0[E0[:, 0] == r, 1], E0[E0[:, 1] == r, 0]]; x = d[:, nb].mean(1)
            rows.append(dict(pers=float(b - (dd if np.isfinite(dd) else 0)), share=float(np.abs(x).max()/max(abs(x.sum()), 1e-300)),
                             neff=float(np.abs(x).sum()**2/max((x**2).sum(), 1e-300)), signed_frac=float(abs(x.sum())/max(np.abs(x).sum(), 1e-300))))
        res[lab] = rows
    out[name] = dict(S=S, res=res)
    for lab in res:
        r = res[lab]; top = [x for x in r if x["pers"] > 0.5]; low = [x for x in r if x["pers"] <= 0.5]
        f = lambda xs, k: (round(min(x[k] for x in xs), 3), round(max(x[k] for x in xs), 3)) if xs else None
        print(name, lab, "S-1 =", S-1, " pers>0.5: n=%d share %s neff %s" % (len(top), f(top, "share"), f(top, "neff")),
              " | pers<=0.5: n=%d share %s neff %s" % (len(low), f(low, "share"), f(low, "neff")), flush=True)
json.dump(out, open("neff_" + "_".join(sys.argv[1:]) + ".json", "w"), indent=1)
