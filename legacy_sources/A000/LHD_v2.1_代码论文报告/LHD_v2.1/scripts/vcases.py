"""case loaders shared by the verification scripts (exact / manufactured / DNS references)"""
import pickle, numpy as np
import mfm, cases


def load(nm):
    if nm == "smooth": c = mfm.smooth_map(h=0.02); fr = c["frames"]; return fr, np.ones(fr.shape[1]), np.exp(-c["lnJ_ex"][-1]), c["interior"](fr[0])
    if nm == "cyl": c = mfm.cylinder_shock(h=0.02); fr = c["frames"]; return fr, np.ones(fr.shape[1]), np.exp(-c["lnJ_ex"][-1]), c["interior"](fr[0])
    if nm == "weak": c = mfm.planar_shock(h=0.02, eta=1.5, theta=0.5, u1=(-0.14, 0.26), Cs=0.5); fr = c["frames"]; return fr, np.ones(fr.shape[1]), np.exp(-c["lnJ_ex"][-1]), c["interior"](fr[0])
    if nm == "slow": c = mfm.planar_shock(h=0.02, eta=3.0, theta=0.5, u1=(-0.14, 0.26), Cs=0.25); fr = c["frames"]; return fr, np.ones(fr.shape[1]), np.exp(-c["lnJ_ex"][-1]), c["interior"](fr[0])
    if nm == "sod": c = cases.case_sod(h=0.004); fr = c["frames"]; return fr, c["rho"][0], c["rho"][-1], c["interior"](fr[0])
    if nm == "blast": c = cases.case_blast(h=0.022); fr = c["frames"]; r0 = np.where(c["rs"] <= 0.2, 6.0, 1.0); return fr, r0, r0 * np.exp(-c["lnJ_ex"][-1]), c["interior"](fr[0])
    if nm == "dg": c = cases.case_dg(); fr = c["frames"]; x = fr[-1]; return fr, np.ones(fr.shape[1]), np.ones(fr.shape[1]), (x[:, 0] > .03) & (x[:, 0] < 1.97) & (x[:, 1] > .03) & (x[:, 1] < .97)
    if nm in ("sv", "rm", "sbv"):
        f = {"sv": "runs/sv_dns.pkl", "rm": "runs/rm_dns_dx0.005.pkl", "sbv": "runs/sbv_dns_dx0.005.pkl"}[nm]
        D = pickle.load(open(f, "rb")); fr = D["fr_in"] if nm == "sv" else D["frames"]; rho = (D["Ev"] if nm == "sv" else D["E"])["rho"]
        if nm == "sv": fr, rho = fr[:31], rho[:31]          # evaluate at t = 0.6 as in the paper (31 snapshots)
        x = fr[-1]
        m = (x[:, 0] > x[:, 0].min() + .02) & (x[:, 0] < x[:, 0].max() - .02) & (x[:, 1] < x[:, 1].max() - .02)
        if nm == "sv": m &= x[:, 1] > x[:, 1].min() + .02
        if nm == "rm": m &= x[:, 1] < 0.48
        return fr, rho[0], rho[-1], m
    if nm == "sb":
        D = pickle.load(open("runs/sb_dns_dx0.005.pkl", "rb")); fr = D["frames"]; rho = D["E"]["rho"]; x = fr[-1]
        m = (x[:, 0] > x[:, 0].min() + .02) & (x[:, 0] < x[:, 0].max() - .02) & (x[:, 1] < x[:, 1].max() - .02)
        return fr, rho[0], rho[-1], m


def st(e, m):
    e = np.abs(e[m]); e = e[np.isfinite(e)]
    return dict(med=float(np.median(e)), p95=float(np.quantile(e, .95)), max=float(e.max()))


