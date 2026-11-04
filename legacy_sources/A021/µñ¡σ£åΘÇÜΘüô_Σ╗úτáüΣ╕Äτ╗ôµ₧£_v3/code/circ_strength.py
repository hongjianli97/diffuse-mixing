"""Circulation-based vortex strength on the coherent boundary (loop integrals only; positions -> tracer velocities).
S_G = (1/2) mean_t [ G_C(t)/A_C(t) - G_R(t)/A_R(t) ] * T ,  C = coherent loop, R = ring between C and an outer level curve.
Smooth flow: S_G ~ mean relative rotation of C w.r.t. its surroundings.  Objective: frame rotation adds 2*Omega to both."""
import sys, json, pickle, numpy as np, tcx, afe_cases as AC
import matplotlib.tri as mtri
def image(P0, tri, PT, Q):
    T = mtri.Triangulation(P0[:, 0], P0[:, 1], tri); it = T.get_trifinder()(Q[:, 0], Q[:, 1]); ok = it >= 0
    out = np.full_like(Q, np.nan); A, B, C = (P0[tri[it[ok], j]] for j in range(3))
    v0, v1, v2 = B - A, C - A, Q[ok] - A
    d00 = (v0*v0).sum(1); d01 = (v0*v1).sum(1); d11 = (v1*v1).sum(1); d20 = (v2*v0).sum(1); d21 = (v2*v1).sum(1)
    den = d00*d11 - d01*d01; l1 = (d11*d20 - d01*d21)/den; l2 = (d00*d21 - d01*d20)/den; l0 = 1 - l1 - l2
    t3 = tri[it[ok]]; out[ok] = l0[:, None]*PT[t3[:, 0]] + l1[:, None]*PT[t3[:, 1]] + l2[:, None]*PT[t3[:, 2]]
    return out
def area(Q): x, y = Q[:, 0], Q[:, 1]; return 0.5*(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
def circ(Q, V): Qb, Vb = np.roll(Q, -1, 0), np.roll(V, -1, 0); return float((0.5*(V + Vb)*(Qb - Q)).sum())
def bary(P0, tri, Q):
    T = mtri.Triangulation(P0[:, 0], P0[:, 1], tri); it = T.get_trifinder()(Q[:, 0], Q[:, 1])
    if (it < 0).any(): return None
    t3 = tri[it]; A, B, C = P0[t3[:, 0]], P0[t3[:, 1]], P0[t3[:, 2]]
    v0, v1, v2 = B - A, C - A, Q - A
    d00 = (v0*v0).sum(1); d01 = (v0*v1).sum(1); d11 = (v1*v1).sum(1); d20 = (v2*v0).sum(1); d21 = (v2*v1).sum(1)
    den = d00*d11 - d01*d01; l1 = (d11*d20 - d01*d21)/den; l2 = (d00*d21 - d01*d20)/den
    return t3, np.c_[1 - l1 - l2, l1, l2]
def ccw(Q): return Q if area(Q) > 0 else Q[::-1]
def strength(fr, t, tri0, Cin, Cout):
    Cin, Cout = ccw(Cin), ccw(Cout); b1, b2 = bary(fr[0], tri0, Cin), bary(fr[0], tri0, Cout)
    if b1 is None or b2 is None: return None
    V = np.gradient(fr, t, axis=0); vals = []
    im = lambda X, b: (b[1][:, :, None]*X[b[0]]).sum(1)
    for s in range(len(t)):
        Q1, V1, Q2, V2 = im(fr[s], b1), im(V[s], b1), im(fr[s], b2), im(V[s], b2)
        a1, a2 = area(Q1), area(Q2); g1, g2 = circ(Q1, V1), circ(Q2, V2)
        vals.append(g1/a1 - (g2 - g1)/(a2 - a1))
    return 0.5*float(np.mean(vals))*(t[-1] - t[0])
def rotate(fr, t, ang=0.6, shift=(0.4, -0.3)):
    th = ang*(t - t[0])/(t[-1] - t[0]); c, s_ = np.cos(th), np.sin(th)
    x, y = fr[..., 0], fr[..., 1]
    return np.stack([c[:, None]*x - s_[:, None]*y + shift[0]*th[:, None], s_[:, None]*x + c[:, None]*y + shift[1]*th[:, None]], -1)
out = {}
for name in sys.argv[1:]:
    c = AC.load(name); fr = c["fr"]; t = np.asarray(c["t"], float); S, N = fr.shape[:2]; P0 = fr[0]
    F = pickle.load(open(f"tcx_fields_{name}.pkl", "rb")); R = json.load(open(f"tcx_res_{name}.json"))
    tri0 = tcx.cplx(P0)["tri"]; E0 = tcx.graph_edges(tri0); deg = np.bincount(E0.ravel(), minlength=N).astype(float)
    g = F["phi"].copy()
    for _ in range(2):
        s_ = np.zeros(N); np.add.at(s_, E0[:, 0], g[E0[:, 1]]); np.add.at(s_, E0[:, 1], g[E0[:, 0]]); g = .5*g + .5*s_/np.maximum(deg, 1)
    frames = np.unique(np.linspace(0, S-1, 11).round().astype(int)); rows = []
    for k, b in enumerate(R["P11_boundary"]):
        pol = F["P11_polys"][k]["lnLd<=0.1"].get("net")
        if pol is None: continue
        core = b["core"]; sg = np.sign(g[core]) or 1.0
        rec = [r for r in tcx.scan_levels(fr, tri0, sg*g, core, frames=frames) if r["found"]]
        Ain = abs(area(pol)); outer = [r for r in rec if 1.8*Ain <= r["area"] <= 3.0*Ain and not r["merged"]]
        if not outer: continue
        Cout = outer[0]["poly"]
        sG = strength(fr, t, tri0, pol, Cout); sGr = strength(rotate(fr, t), t, tri0, pol, Cout)
        inside_ = tcx.inside(P0, pol); ring = tcx.inside(P0, Cout) & ~inside_
        sphi = sg*float(np.mean(F["phi"][inside_]) - np.mean(F["phi"][ring]))
        rows.append(dict(pos=b["pos"], area=Ain, S_circ=sG, S_circ_rotframe=sGr, S_phi_inside_minus_ring=sphi, phi_core=float(F["phi"][core])))
        print(name, [round(x, 3) for x in b["pos"]], {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in rows[-1].items() if k_ != "pos"}, flush=True)
    out[name] = rows
json.dump(out, open("circ_strength_" + "_".join(sys.argv[1:]) + ".json", "w"), indent=1)
