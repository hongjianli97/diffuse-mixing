import pickle, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.tri as mtri
import tcx, afe_cases as AC
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
def image(P0, tri, PT, Q):
    T = mtri.Triangulation(P0[:, 0], P0[:, 1], tri); it = T.get_trifinder()(Q[:, 0], Q[:, 1]); ok = it >= 0
    out = np.full_like(Q, np.nan); A, B, C = (P0[tri[it[ok], j]] for j in range(3))
    v0, v1, v2 = B - A, C - A, Q[ok] - A
    d00 = (v0*v0).sum(1); d01 = (v0*v1).sum(1); d11 = (v1*v1).sum(1); d20 = (v2*v0).sum(1); d21 = (v2*v1).sum(1)
    den = d00*d11 - d01*d01; l1 = (d11*d20 - d01*d21)/den; l2 = (d00*d21 - d01*d20)/den; l0 = 1 - l1 - l2
    t3 = tri[it[ok]]; out[ok] = l0[:, None]*PT[t3[:, 0]] + l1[:, None]*PT[t3[:, 1]] + l2[:, None]*PT[t3[:, 2]]
    return out
cases = [("DG", 0), ("LO", 0), ("SV", 0), ("SV", 2), ("SB", 1)]
fig, ax = plt.subplots(2, len(cases), figsize=(2.6*len(cases), 5.0))
for j, (nm, k) in enumerate(cases):
    F = pickle.load(open(f"tcx_fields_{nm}.pkl", "rb")); R = json.load(open(f"tcx_res_{nm}.json"))
    c = AC.load(nm); fr = c["fr"]; P0 = fr[0]; PT = fr[-1]; tri0 = tcx.cplx(P0)["tri"]
    rec = R["P11_boundary"][k]; pol = F["P11_polys"][k]; ctr = np.array(rec["pos"])
    cv, co = pol["cdef<=0.02"].get("net"), pol["lnLd<=0.1"].get("net"); rf = pol["lnLd<=0.1"].get("ref")
    span = 3.0*np.sqrt(max([rec[q]["net_area"] for q in ("cdef<=0.02", "lnLd<=0.1") if rec[q].get("net_found")] + [1e-4])/np.pi) + 4*c["h"]
    sel = (np.abs(P0 - ctr) < span).all(1)
    ax[0, j].scatter(P0[sel, 0], P0[sel, 1], c=np.abs(F["phi"][sel]), s=2, cmap="Greys", lw=0)
    for Q, col, lab in [(cv, "tab:orange", "convex (cdef≤0.02)"), (co, "tab:blue", "coherent (lnL≤0.1)"), (rf, "k", "coherent, ref")]:
        if Q is None: continue
        Qc = np.vstack([Q, Q[:1]]); ax[0, j].plot(Qc[:, 0], Qc[:, 1], color=col, lw=1.0 if col != "k" else 0.6, ls="-" if col != "k" else "--", label=lab)
        I = image(P0, tri0, PT, Qc); ax[1, j].plot(I[:, 0], I[:, 1], color=col, lw=0.8, ls="-" if col != "k" else "--")
    for q, col in [("cdef<=0.02", "tab:orange"), ("lnLd<=0.1", "tab:blue")]:
        if rec[q].get("net_found"):
            ax[1, j].text(0.02, 0.97 - 0.07*(col == "tab:blue"), f"lnL={rec[q]['net_lnL']:.2f}", color=col, transform=ax[1, j].transAxes, va="top")
    ax[0, j].set_title(f"{nm} core {k}  t0"); ax[1, j].set_title(f"{nm} core {k}  t_end (material image)")
    for a in ax[:, j]: a.set_aspect("equal"); a.tick_params(labelsize=6)
    ax[0, j].set_xlim(ctr[0]-span, ctr[0]+span); ax[0, j].set_ylim(ctr[1]-span, ctr[1]+span)
ax[0, 0].legend(fontsize=6, loc="lower left")
plt.tight_layout(); plt.savefig("fig_boundaries.pdf"); plt.savefig("fig_boundaries.png", dpi=170)
