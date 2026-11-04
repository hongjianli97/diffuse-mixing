"""Shock / heavy-bubble interaction (Ms 1.5, density ratio 3): LHD on tracer positions vs Eulerian DNS."""
import pickle, json, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import lhd as L, lhdx as X
import afrc_engine as AF, afrc_h_engine as AH

T0 = time.time()
D = pickle.load(open("sb_dns.pkl", "rb"))
fr, t, dt = D["frames"], D["t"], D["dt"]; S, N = fr.shape[:2]; Ed = D["E"]
P0 = fr[0]; r0 = np.hypot(P0[:, 0] - D["xb"], P0[:, 1]); h = D["h"]
R = dict(N=N, S=S)
# ---- tracker-like RH events from the pressure history; censoring masks
dlp = np.diff(np.log(Ed["p"]), axis=0)                      # (S-1, N)
ev_s, ev_p = np.where(dlp > 0.15)
bad = L.vertex_bad_mask(ev_p, ev_s, N, S, halo=1)
cstep = np.zeros((N, S - 1), bool); cstep[ev_p, ev_s] = True
cs = cstep.copy(); cs[:, 1:] |= cstep[:, :-1]; cs[:, :-1] |= cstep[:, 1:]
R["events_per_particle_median"] = float(np.median(cstep.sum(1)))
regions = dict(bubble=r0 < D["R"], sheet=r0 < D["R"] + 0.05)
inner0 = (P0[:, 0] > 0.3 + 4 * h) & (P0[:, 0] < 1.25 - 4 * h) & (P0[:, 1] < 0.42 - 4 * h)
E = np.unique(np.vstack([AH._mutual_knn(fr[0], 16), AH._mutual_knn(fr[-1], 16)]), axis=0)
R["n_pairs"] = len(E)
# ---- LHD (inertial-frame velocity, positions only)
t1 = time.time()
o = X.run_lhd(fr, dt, kinds=("galerkin",), op="galerkin", cplx="peel", rho_frames=Ed["rho"], rho_kinds=("galerkin",), pairs=E,
              bad=bad, regions=regions, hodge_every=1, verbose=True)
oD = X.run_lhd(fr, dt, kinds=("cotan", "unit", "gauss"), cplx="peel", hodge_every=5)
for k in oD["div"]:
    o["div"][k] = oD["div"][k]; o["Efrac"][k] = oD["Efrac"][k]
R["t_lhd_per_snap"] = o["t_snap"]; print("lhd %.0fs" % (time.time() - t1), flush=True)
om_ls, th_ls = X.ls_vort_div(fr, dt, k=12)
# ---- star areas per frame (for area-weighted ensemble means / DTFE)
W = np.array([X.star_area(fr[s]) for s in range(S)])
lavd_dns = X.lavd_from_series(Ed["om"], dt, A_weights=W)
lavd_ls = X.lavd_from_series(om_ls, dt, A_weights=W)
# ---- per-frame comparisons
R["frames"] = {}
keep = {}
for s in (15, 30, 50, 70):
    m = inner0 & np.isfinite(o["vort"][s])
    mc = m & ~bad[:, s]
    d = dict(t=float(t[s]))
    d["vort_r"] = dict(stokes_raw=X.corr(o["vort"][s], Ed["om"][s], m), stokes_cens=X.corr(o["vort_c"][s], Ed["om"][s], mc),
                       ls=X.corr(om_ls[s], Ed["om"][s], m), ls_clean=X.corr(om_ls[s], Ed["om"][s], mc))
    d["vort_relL2"] = dict(stokes_cens=X.relL2(o["vort_c"][s], Ed["om"][s], mc), ls_clean=X.relL2(om_ls[s], Ed["om"][s], mc))
    d["div_r"] = {k: X.corr(o["div"][k][s], Ed["th"][s], mc) for k in o["div"]}
    d["div_r"]["ls"] = X.corr(th_ls[s], Ed["th"][s], mc)
    d["div_relL2"] = {k: X.relL2(o["div"][k][s], Ed["th"][s], mc, fit_scale=(k not in ("cotan", "galerkin"))) for k in o["div"]}
    d["div_relL2"]["ls"] = X.relL2(th_ls[s], Ed["th"][s], mc)
    rho_lhd = Ed["rho"][0] * np.exp(-o["lnJ"][s])
    m_part = Ed["rho"][0] * W[0] / 3.0
    rho_dt = X.dtfe_density(fr[s], m_part); rho_sph = X.sph_density(fr[s], m_part)
    zones = dict(all=m, inside=m & (r0 < D["R"] - 0.03), interface=m & (np.abs(r0 - D["R"]) < 0.03), outside=m & (r0 > D["R"] + 0.03))
    d["rho_mederr"] = {nm: {z: float(np.nanmedian(np.abs(e[zm] / Ed["rho"][s][zm] - 1))) for z, zm in zones.items()}
                       for nm, e in [("LHD", rho_lhd), ("DTFE", rho_dt), ("SPH", rho_sph)]}
    R["frames"][s] = d
    keep[s] = dict(P=fr[s], om=Ed["om"][s], om_st=o["vort_c"][s], om_raw=o["vort"][s], om_ls=om_ls[s], th=Ed["th"][s],
                   div=o["div"]["galerkin"][s], div_cot=o["div"]["cotan"][s], div_unit=o["div"]["unit"][s], div_ls=th_ls[s], rho=Ed["rho"][s], rho_lhd=rho_lhd,
                   rho_dt=rho_dt, rho_sph=rho_sph, bad=bad[:, s])
# ---- Kelvin budget
R["kelvin"] = {k: dict(L_raw=o["Gam"][k][:, 0].tolist(), L_clean=o["Gam"][k][:, 1].tolist(), per=o["Gam"][k][:, 2].tolist())
               for k in regions}
# ---- energy fractions
R["Efrac"] = {k: v.tolist() for k, v in o["Efrac"].items()}
# ---- pair stretch attribution (clean frames only, and raw)
okp = ~(bad[E[:, 0]] | bad[E[:, 1]]).T                       # (S, M)
sg = np.where(okp, o["s_g"], np.nan); sc = np.where(okp, o["s_c"], np.nan)
Ag, Ac = np.nansum(np.abs(sg), 0), np.nansum(np.abs(sc), 0)
share_g = Ag / (Ag + Ac + 1e-30)
Agr, Acr = np.nansum(np.abs(o["s_g"]), 0), np.nansum(np.abs(o["s_c"]), 0)
share_g_raw = Agr / (Agr + Acr + 1e-30)
def node_med(val):
    idx = np.concatenate([E[:, 0], E[:, 1]]); v = np.concatenate([val, val])
    out = np.full(N, np.nan); o_ = np.lexsort((v, idx)); iv, vv = idx[o_], v[o_]
    st = np.searchsorted(iv, np.arange(N), "left"); en = np.searchsorted(iv, np.arange(N), "right")
    for i in np.where(en > st)[0]:
        out[i] = np.median(vv[st[i]:en[i]])
    return out
node_share = node_med(share_g); node_share_raw = node_med(share_g_raw)
R["share_g"] = dict(far_clean=float(np.nanmedian(node_share[inner0 & (r0 > D["R"] + 0.12)])),
                    far_raw=float(np.nanmedian(node_share_raw[inner0 & (r0 > D["R"] + 0.12)])),
                    interface_clean=float(np.nanmedian(node_share[inner0 & (np.abs(r0 - D["R"]) < 0.04)])),
                    interface_raw=float(np.nanmedian(node_share_raw[inner0 & (np.abs(r0 - D["R"]) < 0.04)])))
# ---- window fields: FTLE, AFRC_H (orig / rebuilt), AFRC_E (orig / Stokes input)
traj = np.transpose(fr, (1, 0, 2))
ftle_ls = X.ls_ftle(fr[0], fr[-1], t[-1] - t[0])
np.savez_compressed("sb_stage1.npz", sc=sc, E=E, cs=cs, vort_c=o["vort_c"], dt=dt, inner0=inner0, lavd_dns=lavd_dns, ftle_ls=ftle_ls)
band = inner0 & (r0 > D["R"] + 0.12) & (lavd_dns < np.median(lavd_dns[inner0]))
core = inner0 & (lavd_dns >= np.quantile(lavd_dns[inner0], 0.95))
sal = lambda F: float(np.nanquantile(F[band], 0.95) / max(np.nanquantile(F[core], 0.95), 1e-30))
win = dict(DNS_LAVD=lavd_dns, Stokes_LAVD=o["lavd_c"], Stokes_LAVD_raw=o["lavd"], LS_LAVD=lavd_ls, FTLE_LS=ftle_ls)
R["window"] = {k: dict(sp_vs_DNSLAVD=X.spearman(v, lavd_dns, inner0), J_vs_DNSLAVD=X.jaccard_top(v, lavd_dns, 0.9, inner0),
                       sp_vs_FTLE=X.spearman(v, ftle_ls, inner0), J_vs_FTLE=X.jaccard_top(v, ftle_ls, 0.9, inner0),
                       salience=sal(v)) for k, v in win.items()}
# ---- position-only shock detection (node channel) vs pressure events
dlnJ = np.diff(o["lnJ"], axis=0); d2 = dlnJ[:-1] + dlnJ[1:]
det = dlnJ < -0.25; det[:-1] |= d2 < -0.25; det[1:] |= d2 < -0.25
strong = dlp > 0.4; weakish = dlp > 0.1
near = weakish.copy(); near[1:] |= weakish[:-1]; near[:-1] |= weakish[1:]
nearS = strong.copy(); nearS[1:] |= strong[:-1]; nearS[:-1] |= strong[1:]
crossed = strong.any(0) & inner0
R["detect"] = dict(rate_strong=float((det & nearS).any(0)[crossed].mean()),
                   false_alarm=float((det & ~near).any(0)[inner0].mean()))
# ---- Hodge vs Eulerian Helmholtz in a box at t = 1.0 (same convention: gradient part carries the
#      boundary normal flux, coexact part tangent)
def euler_helmholtz(Wg, xc, yc, dx, box):
    ix = np.where((xc > box[0]) & (xc < box[1]))[0]; iy = np.where((yc > box[2]) & (yc < box[3]))[0]
    u = Wg[1][np.ix_(iy, ix)].astype(float); v = Wg[2][np.ix_(iy, ix)].astype(float)
    ny, nx = u.shape
    ue = 0.5 * (u[:, 1:] + u[:, :-1]); vn = 0.5 * (v[1:] + v[:-1])          # interior faces
    # boundary-face normal velocities: x faces use neighbouring outside cells, y faces likewise
    uxl = 0.5 * (Wg[1][iy, ix[0] - 1] + Wg[1][iy, ix[0]]); uxr = 0.5 * (Wg[1][iy, ix[-1]] + Wg[1][iy, ix[-1] + 1])
    vyb = np.zeros(nx) if iy[0] == 0 else 0.5 * (Wg[2][iy[0] - 1, ix] + Wg[2][iy[0], ix])
    vyt = 0.5 * (Wg[2][iy[-1], ix] + Wg[2][iy[-1] + 1, ix])
    Fx = np.concatenate([uxl[:, None], ue, uxr[:, None]], 1)              # (ny, nx+1)
    Fy = np.concatenate([vyb[None], vn, vyt[None]], 0)                     # (ny+1, nx)
    div = ((Fx[:, 1:] - Fx[:, :-1]) + (Fy[1:] - Fy[:-1])) / dx
    n = nx * ny; idx = np.arange(n).reshape(ny, nx)
    rows, cols, vals = [], [], []
    for a, b in [(idx[:, :-1], idx[:, 1:]), (idx[:-1], idx[1:])]:
        a, b = a.ravel(), b.ravel()
        rows += [a, b, a, b]; cols += [b, a, a, b]; vals += [np.ones(len(a)), np.ones(len(a)), -np.ones(len(a)), -np.ones(len(a))]
    Lp = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)) / dx ** 2
    rhs = div.ravel() - (np.r_[uxl * 0, 0][0] * 0)
    # Neumann data enter through the boundary fluxes: L phi = div - (boundary flux)/dx
    bflux = np.zeros((ny, nx)); bflux[:, 0] -= uxl; bflux[:, -1] += uxr; bflux[0] -= vyb; bflux[-1] += vyt
    rhs = (div - bflux / dx).ravel()
    Lp = Lp + sp.csr_matrix(([1e-10], ([0], [0])), shape=(n, n))
    phi = spsolve(Lp.tocsc(), rhs - rhs.mean()).reshape(ny, nx)
    gx = np.zeros((ny, nx + 1)); gy = np.zeros((ny + 1, nx))
    gx[:, 1:-1] = (phi[:, 1:] - phi[:, :-1]) / dx; gx[:, 0] = uxl; gx[:, -1] = uxr
    gy[1:-1] = (phi[1:] - phi[:-1]) / dx; gy[0] = vyb; gy[-1] = vyt
    ud = np.stack([0.5 * (gx[:, 1:] + gx[:, :-1]), 0.5 * (gy[1:] + gy[:-1])])
    U = np.stack([u, v])
    return dict(ud=ud, U=U, us=U - ud, xc=xc[ix], yc=yc[iy], Efrac_d=float((ud ** 2).sum() / (U ** 2).sum()),
                div_resid=float(np.abs((Fx[:, 1:] - Fx[:, :-1] + Fy[1:] - Fy[:-1]) / dx).mean()))
tg = 1.0; sg_ = int(round(tg / dt)); Wg = D["grids"][tg]["W"]
Pb = fr[sg_]; inb = Ed["Y1"][sg_] > -1
xcb = np.median(Pb[regions["bubble"]][:, 0])
from scipy.spatial import cKDTree as _KD
_tree = _KD(Pb)
for xl in np.arange(xcb - 0.12, xcb + 0.2, 0.01):
    box = (xl, xl + 0.30, 0.0, 0.24)
    gxx, gyy = np.meshgrid(np.arange(box[0], box[1], 0.005), np.arange(0.0025, box[3], 0.005))
    cov = (_tree.query(np.c_[gxx.ravel(), gyy.ravel()])[0] < 2.0 * h).mean()
    if cov > 0.995:
        break
R["box_coverage"] = float(cov)
EH = euler_helmholtz(Wg, D["xc"], D["yc"], D["dx"], box)
sel = (Pb[:, 0] > box[0]) & (Pb[:, 0] < box[1]) & (Pb[:, 1] < box[3])
Vb = np.gradient(fr, dt, axis=0)[sg_][sel]; Psub = Pb[sel]
cxb = X.complex_peeled(Psub)
gb = X.galerkin_hodge(Psub, cxb, Vb)
ugb = X.face_to_node(Psub, cxb, gb["ug"]); ucb = X.face_to_node(Psub, cxb, gb["uc"])
gr = X.galerkin_hodge(Psub, cxb, Vb, rho_t=Ed["rho"][sg_][sel][cxb["tri"]].mean(1)); ugr = X.face_to_node(Psub, cxb, gr["ug"])
s0b, s1b = X.stars(Psub, cxb, "cotan"); fb = L.edge_cochain(Psub, Vb, cxb["edges"]); dec_ = L.Hodge(cxb, s1b)(fb)
ugd = L.node_vectors(Psub, cxb, dec_["fg"])
dec = dict(Eg=gb["Eg"], E=1.0); decr = dict(Eg=gr["Eg"], E=1.0)
def g_interp(F, P, xc, yc):
    i = np.clip(np.round((P[:, 0] - xc[0]) / D["dx"]).astype(int), 0, len(xc) - 1)
    j = np.clip(np.round((P[:, 1] - yc[0]) / D["dx"]).astype(int), 0, len(yc) - 1)
    return F[:, j, i].T
ud_p = g_interp(EH["ud"], Psub, EH["xc"], EH["yc"]); us_p = g_interp(EH["us"], Psub, EH["xc"], EH["yc"])
inb = (Psub[:, 0] > box[0] + 0.03) & np.isfinite(ugd).all(1) & (Psub[:, 0] < box[1] - 0.03) & (Psub[:, 1] < box[3] - 0.03) & np.isfinite(ugb).all(1)
R["box"] = dict(t=tg, box=box, Efrac_grad_LHD=dec["Eg"] / dec["E"], Efrac_grad_LHD_rho=decr["Eg"] / decr["E"],
                Efrac_dil_Euler=EH["Efrac_d"],
                ug_vs_ud_r=X.corr(ugb[inb].ravel(), ud_p[inb].ravel()), ug_vs_ud_relL2=X.relL2(ugb[inb].ravel(), ud_p[inb].ravel()),
                uc_vs_us_r=X.corr(ucb[inb].ravel(), us_p[inb].ravel()), uc_vs_us_relL2=X.relL2(ucb[inb].ravel(), us_p[inb].ravel()),
                ugrho_vs_ud_r=X.corr(ugr[inb].ravel(), ud_p[inb].ravel()), ugrho_vs_ud_relL2=X.relL2(ugr[inb].ravel(), ud_p[inb].ravel()),
                ugDEC_vs_ud_r=X.corr(ugd[inb].ravel(), ud_p[inb].ravel()), ugDEC_vs_ud_relL2=X.relL2(ugd[inb].ravel(), ud_p[inb].ravel()),
                Efrac_grad_DEC=dec_["Eg"] / dec_["E"])
R["Gam_E"] = {k: np.asarray(v).tolist() for k, v in D["Gam"].items()}
R["total_time"] = time.time() - T0
json.dump(R, open("res_sb.json", "w"), indent=1, default=float)
pickle.dump(dict(keep=keep, win=win, P0=P0, Pend=fr[-1], r0=r0, inner0=inner0, node_share=node_share,
                 node_share_raw=node_share_raw, Gam=o["Gam"], Efrac=o["Efrac"], t=t, box=box, EH=EH, Psub=Psub, ugb=ugb,
                 ucb=ucb, ugr=ugr, inb=inb, grids=D["grids"], xc=D["xc"], yc=D["yc"], GamE=D["Gam"], bad_any=bad.any(1)),
            open("res_sb.pkl", "wb"))
print(json.dumps({k: v for k, v in R.items() if k not in ("kelvin", "Efrac", "Gam_E")}, indent=1, default=float))
