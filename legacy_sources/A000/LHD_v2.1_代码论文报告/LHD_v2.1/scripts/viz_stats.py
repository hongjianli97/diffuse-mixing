"""Quantitative readings of every field figure (numbers quoted in the text)."""
import numpy as np, json, pickle, sys
import viz_fields as V
from viz_fields import cases, X, SE, ND, ledger, omega_hat, stretch_field, crossing_map, run_net
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
q = lambda a, p: float(np.nanquantile(a, p))
def mixfrac(c): c = np.asarray(c); return float(((c > 0.1) & (c < 0.9)).mean())
def varr(c, m): cb = (m*c).sum()/m.sum(); return float((m*(c-cb)**2).sum())
out = {}
def dg():
    c = cases.case_dg(nx=160, ny=80, T=10.0, S=51); fr, t, dt = c["frames"], c["t"], c["dt"]; N = fr.shape[1]; P0 = fr[0]; ss = [0, 17, 33, 50]
    lab = P0[:, 0] < 1.0; d, _ = cKDTree(P0).query(P0, 2); h = np.median(d[:, 1])
    lnJ, hot, bad, tau = ledger(fr); m = X.star_area(P0)/3; c0 = 0.5*(1+np.tanh((P0[:, 0]-1)/0.1)); cn = run_net(fr, dt, c0, 1e-3, m, np.ones((len(t), N)), ss)
    o = dict(left_in_right=[float((lab & (fr[s][:, 0] > 1)).sum()/lab.sum()) for s in ss], mixzone=[mixfrac(cn[s]) for s in ss], var_ratio=[varr(cn[s], m)/varr(cn[0], m) for s in ss])
    om, th = omega_hat(fr, 33, dt); vel, vort, grad = cases.dg_fields(); o["om_range"] = [q(om, .01), q(om, .99)]
    t0 = np.tile([0.0, 1.0], (N, 1)); ll, A = stretch_field(fr, h, bad, t0, 50)
    o.update(ll_q=[q(ll, .05), q(ll, .5), q(ll, .95)], ll_frac_gt2=float((np.abs(ll) > 2).mean()), A_med=q(A, .5), A_q95=q(A, .95), comp_p95=q(np.abs(lnJ[50]), .95),
             corr_ll_logA=float(spearmanr(np.abs(ll), A)[0]), times=[float(t[s]) for s in ss])
    iface = np.abs(P0[:, 0]-0.5) < 0.6*h; o["iface_ll_mean"] = float(ll[iface].mean())
    return o
def lo():
    c = cases.case_lo(a=1.0, Rd=6.5, N=8000); fr, t, dt = c["frames"], c["t"], c["dt"]; N = fr.shape[1]; P0 = fr[0]; ss = [0, 13, 27, 40]; cen = c["centres"]
    lab = P0[:, 0] < 0; lnJ, hot, bad, tau = ledger(fr); m = X.star_area(P0)/3; c0 = 0.5*(1+np.tanh(P0[:, 0]/0.3)); cn = run_net(fr, dt, c0, 0.02, m, np.ones((len(t), N)), ss)
    rr = np.hypot(*(P0-np.array(cen[0])).T); ring = np.abs(rr-1.5) < 0.6*c["h"]
    t0 = np.c_[-(P0[:, 1]-cen[0][1]), P0[:, 0]-cen[0][0]]/np.maximum(rr, 1e-9)[:, None]; ll, A = stretch_field(fr, c["h"], bad, t0, 40)
    cur = c["centres"]
    o = dict(left_mass_right=[float((lab & (fr[s][:, 0] > 0)).sum()/lab.sum()) for s in ss], mixzone=[mixfrac(cn[s]) for s in ss], var_ratio=[varr(cn[s], m)/varr(cn[0], m) for s in ss],
             ring_ll=[q(ll[ring], .05), float(ll[ring].mean()), q(ll[ring], .95)], ring_A_med=q(A[ring], .5), A_med_all=q(A, .5), comp_p95=q(np.abs(lnJ[40]), .95), times=[float(t[s]) for s in ss],
             ring_radius_std=float(np.std(np.hypot(*(fr[40][ring]-fr[40][ring].mean(0)).T))/np.mean(np.hypot(*(fr[40][ring]-fr[40][ring].mean(0)).T))))
    om, th = omega_hat(fr, 20, dt); o["om_max"] = q(om, .995)
    return o
def sod():
    c = cases.case_sod(h=0.004); fr, t, dt = c["frames"], c["t"], c["dt"]; N = fr.shape[1]; x0 = c["x0"]; ss = [0, 13, 27, 40]
    lnJ, hot, bad, tau = ledger(fr); rho = c["rho"][0][None]*np.exp(-lnJ); cm = crossing_map(hot)
    ok = (cm >= 0) & (x0 > 0); sl = np.polyfit(x0[ok], t[cm[ok]], 1)[0]
    c0 = 0.5*(1+np.tanh(x0/0.02)); cn = run_net(fr, dt, c0, 1e-3, c["mass"], rho, ss)
    def width(cc, x):
        o_ = np.argsort(x); xs, cs = x[o_], cc[o_]; i1 = np.searchsorted(np.maximum.accumulate(cs), 0.1); i9 = np.searchsorted(np.maximum.accumulate(cs), 0.9); return float(xs[min(i9, len(xs)-1)]-xs[min(i1, len(xs)-1)])
    err = np.abs(rho[-1]/c["rho"][-1]-1); x = fr[-1][:, 0]
    s = 20; om, th = omega_hat(fr, s, dt)
    rf = rho[-1]; post = (x > 0.5) | False
    info = c["info"]
    o = dict(shock_speed_fit=float(1/sl), plateau_post=float(np.median(rf[(x0 > 0) & (cm >= 0) & (x < x.max()-0.01)])), plateau_left_contact=float(np.median(rf[(x0 < 0) & (x0 > -0.02)])),
             err_frac_gt1pct=float((err > 0.01).mean()), front_width=[width(cn[s_], fr[s_][:, 0]) for s_ in ss], th_min=q(th, .001), om_absmax_shock=q(np.abs(om), .999), th_absmax=q(np.abs(th), .999),
             times=[float(t[s_]) for s_ in ss], hot_frac=float(hot.any(1).mean()), info={k: float(v) for k, v in info.items() if np.isscalar(v)})
    return o
def blast():
    c = cases.case_blast(h=0.022); fr, t, dt = c["frames"], c["t"], c["dt"]; N = fr.shape[1]; P0 = fr[0]; rs = c["rs"]; ss = [0, 20, 40, 60]
    lnJ, hot, bad, tau = ledger(fr); rho0 = np.where(rs <= 0.2, 6.0, 1.0); rho = rho0[None]*np.exp(-lnJ); cm = crossing_map(hot)
    ok = cm >= 0; lt = np.log(t[cm[ok]]); lr = np.log(rs[ok]); expo = np.polyfit(lt, lr, 1)[0]
    r_now = np.hypot(*fr[-1].T); shell = rho[-1] > 3
    th = np.arctan2(P0[:, 1], P0[:, 0]); t0 = np.c_[-np.sin(th), np.cos(th)]; ll, A = stretch_field(fr, 0.022, bad, t0, 60); ring = np.abs(rs-0.5) < 0.6*0.022
    m = rho0*X.star_area(P0)/3; c0 = 0.5*(1+np.tanh((rs-0.5)/0.04)); cn = run_net(fr, dt, c0, 2e-4, m, rho, ss)
    o = dict(crossing_exponent=float(expo), shell_rho_max=q(rho[-1], .995), shell_r=[q(r_now[shell], .05), q(r_now[shell], .95)], center_comp=float(np.median(-lnJ[-1][rs < 0.15])),
             shell_comp=float(np.median(-lnJ[-1][shell])), ring_ll=float(ll[ring].mean()), ring_comp=float((-lnJ[-1][ring]).mean()), mixzone=[mixfrac(cn[s]) for s in ss],
             var_ratio=[varr(cn[s], m)/varr(cn[0], m) for s in ss], times=[float(t[s]) for s in ss], inner_frac_r_lt=float(np.median(np.hypot(*fr[-1][rs <= 0.2].T))))
    return o
def sv():
    D = pickle.load(open("sv_dns.pkl", "rb")); fr, t = D["fr_in"], D["t"]; dt = t[1]-t[0]; Ev = D["Ev"]; P0 = fr[0]; ss = [0, 10, 20, 30]
    cen = np.array([D["xv"], D["yv"]]); core = np.hypot(*(P0-cen).T) < D["Rv"]; ring = np.abs(np.hypot(*(P0-cen).T)-2*D["Rv"]) < 0.6*0.005
    lnJ, hot, bad, tau = ledger(fr); rho = Ev["rho"][0][None]*np.exp(-lnJ)
    o = dict(times=[float(t[s]) for s in ss], core_disp=[float(np.hypot(*(fr[s][core].mean(0)-fr[0][core].mean(0)))) for s in ss],
             ring_comp=[float((-lnJ[s][ring]).mean()) for s in ss], err_med=[float(np.median(np.abs(rho[s]/Ev["rho"][s]-1))) for s in ss],
             err_p95=[q(np.abs(rho[s]/Ev["rho"][s]-1), .95) for s in ss])
    cc = []; corrband = []
    for s in [10, 20, 29]:
        om, th = omega_hat(fr, s, dt); okk = np.isfinite(om); cc.append(float(np.corrcoef(om[okk], 0.5*(Ev["om"][s]+Ev["om"][s+1])[okk])[0, 1]))
    o["om_corr_nodes"] = cc
    s = 20; e = np.abs(rho[s]/Ev["rho"][s]-1); top = e >= np.quantile(e, .95); hotn = hot[:, max(s-2, 0):s+1].any(1)
    o["top5_in_shockband_t"] = float(hotn[top].mean()); o["top5_in_core_final"] = float((np.hypot(*(P0-cen).T) < 2*D["Rv"])[np.abs(rho[-1]/Ev["rho"][-1]-1) >= q(np.abs(rho[-1]/Ev["rho"][-1]-1), .95)].mean())
    return o
def sb(path="sb_dns_dx0.005.pkl", visc=False):
    D = pickle.load(open(path, "rb")); fr, t = D["frames"], D["t"]; dt = D["dt"]; E = D["E"]; P0 = fr[0]; ss = [0, 30, 50, 70]
    phi = np.hypot(P0[:, 0]-D["xb"], P0[:, 1])-D["R"]; heavy = phi < 0; iface = np.abs(phi) < 0.6*0.005
    lnJ, hot, bad, tau = ledger(fr); rho = E["rho"][0][None]*np.exp(-lnJ)
    th = np.arctan2(P0[:, 1], P0[:, 0]-D["xb"]); t0 = np.c_[-np.sin(th), np.cos(th)]; ll, A = stretch_field(fr, 0.005, bad, t0, 70)
    o = dict(times=[float(t[s]) for s in ss], heavy_xext=[[q(fr[s][heavy, 0], .01), q(fr[s][heavy, 0], .99)] for s in ss], heavy_yext=[q(fr[s][heavy, 1], .99) for s in ss],
             err_med=[float(np.median(np.abs(rho[s]/E["rho"][s]-1))) for s in ss], iface_ll=[q(ll[iface], .05), float(ll[iface].mean()), q(ll[iface], .95)], ll_q=[q(ll, .02), q(ll, .98)],
             comp_iface=float((-lnJ[70][iface]).mean()), comp_bulk_heavy=float(np.median(-lnJ[70][phi < -0.05])), comp_bulk_light=float(np.median(-lnJ[70][phi > 0.05])),
             A_iface_med=q(A[iface], .5), A_bulk_med=q(A[np.abs(phi) > 0.05], .5))
    err = np.abs(rho[70]/E["rho"][70]-1); top = err >= np.quantile(err, .95); o["top5_near_iface"] = float((np.abs(phi[top]) < 0.03).mean())
    if not visc:
        Dd = pickle.load(open("sb_diff_dns.pkl", "rb")); lj = ledger(Dd["frames"])[0]
        cn = run_net(Dd["frames"], Dd["dt"], Dd["Y"][0], Dd["kappa"], Dd["rho"][0]*X.star_area(Dd["frames"][0])/3, Dd["rho"][0][None]*np.exp(-lj), ss)
        o["diff_relL2"] = [float(np.sqrt(np.mean((cn[s]-Dd["Y"][s])**2))/max(np.std(Dd["Y"][s]), 1e-12)) for s in ss]
        o["mixzone_net"] = [mixfrac(cn[s]) for s in ss]; o["mixzone_dns"] = [mixfrac(Dd["Y"][s]) for s in ss]
        om, th_ = omega_hat(fr, 69, dt); o["om_q"] = [q(om, .01), q(om, .99)]
    return o
def rm():
    D = pickle.load(open("rm_dns_dx0.005.pkl", "rb")); fr, t = D["frames"], D["t"]; dt = D["dt"]; E = D["E"]; P0 = fr[0]; a0 = D["a0"]; N = fr.shape[1]
    xi = 0.5+a0*np.cos(2*np.pi*P0[:, 1]); phi = (P0[:, 0]-xi)/np.sqrt(1+(2*np.pi*a0*np.sin(2*np.pi*P0[:, 1]))**2); iface = np.abs(phi) < 0.6*0.004
    lnJ, hot, bad, tau = ledger(fr); rho = E["rho"][0][None]*np.exp(-lnJ)
    yy = P0[:, 1]; tv = np.c_[-2*np.pi*a0*np.sin(2*np.pi*yy), np.ones(N)]; tv /= np.linalg.norm(tv, axis=1)[:, None]; ll, A = stretch_field(fr, 0.004, bad, tv, 50)
    om, th_ = omega_hat(fr, 30, dt); band = np.abs(phi) < 0.03
    w = np.abs(np.nan_to_num(om))*X.star_area(fr[30])
    o = dict(om_frac_in_band=float(w[band].sum()/w.sum()), ll_spike=float(ll[iface & (yy < 0.05)].mean()), ll_bubble=float(ll[iface & (yy > 0.45)].mean()), ll_flank=float(ll[iface & (np.abs(yy-0.25) < 0.05)].mean()),
             comp_iface=float((-lnJ[50][iface]).mean()), comp_heavy=float(np.median(-lnJ[50][phi > 0.05])), comp_light=float(np.median(-lnJ[50][phi < -0.05])),
             err_iface_med=float(np.median(np.abs(rho[50]/E["rho"][50]-1)[np.abs(phi) < 0.02])), err_bulk_med=float(np.median(np.abs(rho[50]/E["rho"][50]-1)[np.abs(phi) > 0.05])),
             v_post=float(np.median((fr[11][:, 0]-fr[10][:, 0])/dt)))
    return o
for nm in sys.argv[1:]:
    out[nm] = sb("sbv_dns_dx0.0035.pkl", True) if nm == "sbv" else globals()[nm](); print(nm, json.dumps(out[nm])[:600], flush=True)
    old = json.load(open("res_viz_stats.json")) if __import__("os").path.exists("res_viz_stats.json") else {}; old.update(out); json.dump(old, open("res_viz_stats.json", "w"), indent=1)
