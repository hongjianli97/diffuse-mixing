import numpy as np, json, pickle, cases, sys
from bcom import *
from lhdp import disp_cochain_tri, tri_area
which = sys.argv[1]; out = {}
rng = np.random.default_rng(11)
if which == "sod":
    c = cases.case_sod(h=0.004); fr = c["frames"]; rho0 = c["rho"][0]; ref = c["rho"][-1]; mask = c["interior"](fr[0]); om = None
elif which == "sv":
    D = pickle.load(open("sv_dns.pkl", "rb")); fr = D["fr_in"]; rho0 = D["Ev"]["rho"][0]; ref = D["Ev"]["rho"][-1]; om = D["Ev"]["om"]; t = D["t"]
    x = fr[-1]; mask = (x[:, 0] > x[:, 0].min()+.02) & (x[:, 0] < x[:, 0].max()-.02) & (x[:, 1] > x[:, 1].min()+.02) & (x[:, 1] < x[:, 1].max()-.02)
    c0 = np.array([D["xv"], D["yv"]]); r0 = np.hypot(*(fr[0]-c0).T); region = r0 < 2*D["Rv"]
elif which == "sb":
    D = pickle.load(open("sb_dns_dx0.005.pkl", "rb")); fr = D["frames"]; rho0 = D["E"]["rho"][0]; ref = D["E"]["rho"][-1]; om = None
    x = fr[-1]; mask = (x[:, 0] > x[:, 0].min()+.02) & (x[:, 0] < x[:, 0].max()-.02) & (x[:, 1] < x[:, 1].max()-.02)
S, N, _ = fr.shape
d, _ = cKDTree(fr[0]).query(fr[0], k=2); h = float(np.median(d[:, 1]))
def circ_err(frs, sel):
    if om is None: return None
    P0 = frs[0]; tri = X.complex_peeled(P0)["tri"]; rg = region[sel]; triR = tri[rg[tri].all(1)]
    s = S-2; circ, _ = disp_cochain_tri(frs[s], frs[s+1], triR); G = circ.sum()/(t[s+1]-t[s])
    Pm = 0.5*(frs[s]+frs[s+1]); o = 0.5*(om[s]+om[s+1])[sel]; Gd = (o[triR].mean(1)*tri_area(Pm, triR)).sum()
    return float(abs(G/Gd-1))
def dens(frs, sel, v3=True):
    lnJ = (ledger3(frs) if v3 else ledger(frs))[0]
    return errstats(rho0[sel]*np.exp(-lnJ[-1]), ref[sel], mask[sel])
allidx = np.arange(N)
out["h"] = h
out["clean_v2"] = dens(fr, allidx, False); out["clean_v3"] = dens(fr, allidx, True); out["clean_circ"] = circ_err(fr, allidx)
print("clean", out["clean_v2"], out["clean_v3"], out["clean_circ"], flush=True)
for nz in [0.02, 0.05, 0.1]:
    frn = fr+nz*h*rng.normal(size=fr.shape)
    out["noise_%g" % nz] = dict(v2=dens(frn, allidx, False), v3=dens(frn, allidx, True), circ=circ_err(frn, allidx))
    print("noise", nz, out["noise_%g" % nz], flush=True)
for f in [4, 16]:
    sel = np.sort(rng.choice(N, N//f, replace=False))
    out["sparse_1/%d" % f] = dict(v3=dens(fr[:, sel], sel, True), circ=circ_err(fr[:, sel], sel))
    print("sparse", f, out["sparse_1/%d" % f], flush=True)
for p in [0.0, 0.01, 0.03]:
    Dn, used, fb = ledger_breaks(fr, rho0, p)
    out["breaks_p%g" % p] = dict(errstats(Dn[used], ref[used], mask[used]), usable_frac=float(used.mean()), broken_frac=fb)
    print("breaks", p, out["breaks_p%g" % p], flush=True)
sel = np.sort(rng.choice(N, N//4, replace=False)); frc = fr[:, sel]+0.02*h*2*rng.normal(size=fr[:, sel].shape)
Dn, used, fb = ledger_breaks(frc, rho0[sel], 0.01)
out["combined_1/4_0.02h_p0.01"] = dict(errstats(Dn[used], ref[sel][used], mask[sel][used]), usable_frac=float(used.mean()))
print("combined", out["combined_1/4_0.02h_p0.01"], flush=True)
json.dump(out, open("res_b4_%s.json" % which, "w"), indent=1)
