"""Well-posedness at a vortex sheet: u = (U tanh(y/ds), 0).  Positions only.
Compare (i) node rotation phi (network 'LAVD'), (ii) region integral of phi, (iii) circulation of a material loop."""
import sys, json, numpy as np, tcx
U, T = 0.5, 1.0
def run(h, ds, nf, seed=1):
    rng = np.random.default_rng(seed)
    xs = np.arange(-0.8, 1.8, h); ys = np.arange(-0.3, 0.3 + 1e-9, h); X, Y = np.meshgrid(xs, ys)
    P0 = np.c_[X.ravel(), Y.ravel()] + rng.uniform(-0.25*h, 0.25*h, (X.size, 2))
    t = np.linspace(0, T, nf); u = U*np.tanh(P0[:, 1]/ds)
    fr = np.stack([P0 + np.c_[u*tt, 0*u] for tt in t]).astype(float); S = nf
    mus, _ = tcx.mu_threshold(fr); ch = tcx.run_chain(fr, mus, Ks=[tcx.cplx(fr[s])["tri"] for s in range(S-1)])
    phi = tcx.rotation_fields(ch)[0]
    inner = (P0[:, 0] > 0.0) & (P0[:, 0] < 1.0) & (np.abs(P0[:, 1]) < 0.25)
    tri = tcx.cplx(P0)["tri"]; Ar = np.abs(tcx.tri_area(P0, tri)); Astar = np.zeros(len(P0)); np.add.at(Astar, tri.ravel(), np.repeat(Ar, 3)/3)
    band = inner & (np.abs(P0[:, 1]) < 0.75*h)
    # exact relative rotation of a material element: 1/2 * omega * T, omega = -U/ds sech^2 (ensemble mean ~ 0 by symmetry)
    # region: material box |x-0.5|<0.1, |y|<0.1 ; circulation from the boundary loop of tracers, trapezoidal rule
    box = (np.abs(P0[:, 0] - 0.5) < 0.1) & (np.abs(P0[:, 1]) < 0.1)
    Q_phi = float((phi*Astar)[box].sum())                                   # region integral of node rotation
    # boundary loop: outermost tracers of the box ordered around its centre at t0
    bx = np.where(box)[0]; c0 = P0[bx].mean(0); ang = np.arctan2(*(P0[bx] - c0).T[::-1])
    rad = np.max(np.abs(P0[bx] - c0)/[0.1, 0.1], axis=1); ring = bx[rad > 1 - 1.5*h/0.1]; ring = ring[np.argsort(ang[rad > 1 - 1.5*h/0.1])]
    vel = np.gradient(fr, t, axis=0)                                         # tracer velocity from positions
    def circ(s):
        a, b = ring, np.roll(ring, -1)
        return float((0.5*(vel[s, a] + vel[s, b])*(fr[s, b] - fr[s, a])).sum())
    G = [circ(s) for s in range(S)]
    return dict(h=h, ds=ds, frames=nf, phi_max=float(np.abs(phi[inner]).max()), phi_band_median=float(np.median(np.abs(phi[band]))),
                Q_phi_box=Q_phi, Gamma_t0=G[0], Gamma_tend=G[-1], Gamma_exact=float(-U*np.tanh(0.1/ds)*0.2*2))
out = []
cfg = [(0.04, 0.1, 41), (0.02, 0.1, 41), (0.01, 0.1, 41), (0.04, 5e-4, 41), (0.02, 5e-4, 41), (0.01, 5e-4, 41), (0.02, 5e-4, 161), (0.02, 0.1, 161)]
for h, ds, nf in cfg[int(sys.argv[1]):int(sys.argv[2])]:
    r = run(h, ds, nf); out.append(r); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
json.dump(out, open(f"sheet_{sys.argv[1]}_{sys.argv[2]}.json", "w"), indent=1)
