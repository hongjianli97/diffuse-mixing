"""verbatim function definitions from routeA_E_validation_v10_lambda3.ipynb cells 99-100 (paper Fig. 5 machinery)."""

import numpy as np, time

from collections import defaultdict

from scipy.spatial import cKDTree

from scipy.sparse import coo_matrix, diags

from scipy.sparse.linalg import spsolve

from scipy.spatial.distance import directed_hausdorff

import numpy as np, time, heapq, warnings, os, json

from collections import defaultdict

from scipy.spatial import cKDTree, ConvexHull

from scipy.spatial.distance import directed_hausdorff

from scipy.sparse import coo_matrix, diags, csr_matrix

from scipy.sparse.linalg import spsolve

from scipy.sparse.csgraph import connected_components

from scipy.special import exp1

from scipy.interpolate import griddata

from scipy.ndimage import binary_fill_holes, gaussian_filter1d

from skimage import measure

from matplotlib.path import Path as MplPath

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.patheffects as pe

_ELL_G = 2*np.pi*1.5

def lamb_oseen(centers, Gammas, a):
    centers = np.array(centers, float); G = np.atleast_1d(Gammas).astype(float)
    if len(G) == 1: G = np.full(len(centers), G[0])
    def vel(P, t=0.0):
        u = np.zeros(len(P)); v = np.zeros(len(P))
        for (cx, cy), Gm in zip(centers, G):
            dx = P[:, 0]-cx; dy = P[:, 1]-cy; r2 = dx*dx+dy*dy+1e-12; r = np.sqrt(r2)
            ut = (Gm/(2*np.pi*r))*(1-np.exp(-r2/a**2)); u += -ut*dy/r; v += ut*dx/r
        return np.column_stack([u, v])
    def psi(P):
        p = np.zeros(len(P))
        for (cx, cy), Gm in zip(centers, G):
            r = np.sqrt((P[:, 0]-cx)**2+(P[:, 1]-cy)**2)+1e-12; s = (r*r)/a**2+1e-30
            p += -(Gm/(4*np.pi))*(np.log(s)+exp1(s))
        return p
    return vel, psi

def seeds_disk(R, N, seed=7):
    rng = np.random.default_rng(seed); rr = R*np.sqrt(rng.uniform(0, 1, N)); th = rng.uniform(0, 2*np.pi, N)
    return np.column_stack([rr*np.cos(th), rr*np.sin(th)])

def rk4_traj(vel, P0, T, dt, unsteady=True, direction="fwd"):
    sgn = 1.0 if direction == "fwd" else -1.0
    nt = int(round(T/dt))+1; tr = np.empty((P0.shape[0], nt, 2)); P = P0.copy(); h = sgn*dt
    for n in range(nt):
        tr[:, n, :] = P
        t0 = (sgn*n*dt) if unsteady else 0.0
        k1 = vel(P, t0); k2 = vel(P+.5*h*k1, t0+.5*h); k3 = vel(P+.5*h*k2, t0+.5*h); k4 = vel(P+h*k3, t0+h)
        P = P+(h/6)*(k1+2*k2+2*k3+k4)
    return tr

def build_graph(P_snaps, k):
    def mknn(Pp, kk):
        _, idx = cKDTree(Pp).query(Pp, k=kk+1); idx = idx[:, 1:]; nb = [set(r) for r in idx]; out = []
        for i in range(len(Pp)):
            for j in idx[i]:
                if j > i and i in nb[j]: out.append((i, j))
        return out
    Ss = P_snaps.shape[1]; Es = set()
    for s in [0, Ss//4, Ss//2, 3*Ss//4, Ss-1]:
        for e in mknn(P_snaps[:, s, :], k): Es.add(e)
    E = np.array(sorted(Es), np.int64); Nn = P_snaps.shape[0]
    adj = defaultdict(set); inc = defaultdict(list)
    for m, (i, j) in enumerate(E): adj[i].add(j); adj[j].add(i); inc[i].append(m); inc[j].append(m)
    deg = np.array([len(adj[i]) for i in range(Nn)]); Emap = {(i, j): m for m, (i, j) in enumerate(E)}
    eid = lambda p, q: (Emap[(p, q)] if p < q else Emap[(q, p)])
    tris = np.array(sorted(set(tuple(sorted((i, j, w))) for (i, j) in E for w in (adj[i] & adj[j]) if w > j)), np.int64)
    tri_of = defaultdict(list)
    for (i, j, kk) in tris:
        e1, e2, e3 = eid(i, j), eid(j, kk), eid(i, kk); tri_of[e1].append((e2, e3)); tri_of[e2].append((e1, e3)); tri_of[e3].append((e1, e2))
    ntri = np.array([len(tri_of.get(m, [])) for m in range(len(E))])
    return dict(E=E, u=E[:, 0], v=E[:, 1], nE=len(E), inc=inc, deg=deg, tris=tris, tri_of=tri_of,
                struct=4.0-deg[E[:, 0]]-deg[E[:, 1]]+3.0*ntri, N=Nn)

def afrc_curv(gr, we, W0=0.1):
    u, v, nE, tri_of, N = gr['u'], gr['v'], gr['nE'], gr['tri_of'], gr['N']
    acc = np.zeros(N); cnt = np.zeros(N); np.add.at(acc, u, we-W0); np.add.at(acc, v, we-W0); np.add.at(cnt, u, 1.); np.add.at(cnt, v, 1.)
    wv = W0 + np.where(cnt > 0, acc/np.maximum(cnt, 1), 0.); invq = 1/np.sqrt(we); sinv = np.zeros(N)
    np.add.at(sinv, u, invq); np.add.at(sinv, v, invq)
    dterm = 2*(wv[u]+wv[v]) - np.sqrt(we)*(wv[u]*sinv[u]+wv[v]*sinv[v]); tterm = np.zeros(nE)
    for m in range(nE):
        sm = 0.
        for (e2, e3) in tri_of.get(m, []): sm += 3*we[m]**2/((we[m]*we[e2]*we[e3])**(1/3)+1e-12)
        tterm[m] = sm
    return dterm + tterm

def node_mean(gr, ev):
    u, v, N = gr['u'], gr['v'], gr['N']; s = np.zeros(N); c = np.zeros(N)
    np.add.at(s, u, ev); np.add.at(s, v, ev); np.add.at(c, u, 1.); np.add.at(c, v, 1.); return s/np.maximum(c, 1)

def vort_fd(vel, P, t, h=1e-4):
    uy = (vel(P+[0, h], t)[:, 0]-vel(P-[0, h], t)[:, 0])/(2*h)
    vx = (vel(P+[h, 0], t)[:, 1]-vel(P-[h, 0], t)[:, 1])/(2*h)
    return vx-uy

def LAVD_from_traj(tr, vel, T, dt, unsteady=True, direction="fwd"):
    sgn = 1.0 if direction == "fwd" else -1.0
    S = tr.shape[1]; W = np.empty((tr.shape[0], S))
    for n in range(S):
        tt = (sgn*n*dt) if unsteady else 0.0; W[:, n] = vort_fd(vel, tr[:, n, :], tt)
    wbar = W.mean(0); return np.trapezoid(np.abs(W-wbar[None, :]), dx=dt, axis=1)/((S-1)*dt)

def _poly_area(P):
    x, y = P[:, 0], P[:, 1]; return 0.5*abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

def lavd_boundary(lv, P0, center, box, ng=170, conv_def=0.08, frac_lo=0.05):
    x0, x1, y0, y1 = box; xs = np.linspace(x0, x1, ng); ys = np.linspace(y0, y1, ng)
    GX, GY = np.meshgrid(xs, ys); G = griddata(P0, lv, (GX, GY), method='linear')
    if np.all(~np.isfinite(G)): return None, None
    vmax = np.nanmax(G); best = None; bestpoly = None
    for lev in np.linspace(0.95*vmax, frac_lo*vmax, 80):
        for c in measure.find_contours(G, lev):
            py = ys[0]+c[:, 0]*(ys[-1]-ys[0])/(ng-1); px = xs[0]+c[:, 1]*(xs[-1]-xs[0])/(ng-1)
            poly = np.column_stack([px, py])
            if np.hypot(*(poly[0]-poly[-1])) > 1e-9 or not MplPath(poly).contains_point(center): continue
            A = _poly_area(poly)
            if A <= 0: continue
            try: H = ConvexHull(poly).volume
            except: continue
            if (H-A)/H <= conv_def: best = np.sqrt(A/np.pi); bestpoly = poly
    return best, bestpoly

def analytic_corot(Gammas, centers, box, n=600):
    _, psi = lamb_oseen(centers, Gammas, _ELL_a)
    G = np.atleast_1d(Gammas).astype(float); Cc = np.array(centers, float)
    if len(G) == 1: G = np.full(len(Cc), G[0])
    xc = (G[:, None]*Cc).sum(0)/G.sum(); d = np.hypot(*(Cc[0]-Cc[1])); Om = G.sum()/(2*np.pi*d*d)
    x0, x1, y0, y1 = box; xs = np.linspace(x0, x1, n); ys = np.linspace(y0, y1, n); X, Y = np.meshgrid(xs, ys)
    Prot = psi(np.column_stack([X.ravel(), Y.ravel()])).reshape(n, n)-0.5*Om*((X-xc[0])**2+(Y-xc[1])**2)
    xln = np.linspace(Cc[:, 0].min(), Cc[:, 0].max(), 4000)
    pl = psi(np.column_stack([xln, np.full_like(xln, xc[1])]))-0.5*Om*((xln-xc[0])**2)
    s = np.where(np.diff(np.sign(np.gradient(pl, xln))))[0]; xsad = xln[s[len(s)//2]] if len(s) else xc[0]
    L = float((psi(np.array([[xsad, xc[1]]]))-0.5*Om*((xsad-xc[0])**2))[0]); loops = []
    for c in measure.find_contours(Prot, L):
        xx = np.interp(c[:, 1], np.arange(n), xs); yy = np.interp(c[:, 0], np.arange(n), ys)
        if len(xx) > 20: loops.append(np.column_stack([xx, yy]))
    return dict(Om=float(Om), xc=xc.tolist(), psi=Prot, xs=xs, ys=ys, ext=[x0, x1, y0, y1], L=L, loops=loops, centers=Cc.tolist())

def _eqr(poly): x, y = poly[:, 0], poly[:, 1]; A = 0.5*abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1))); return float(np.sqrt(A/np.pi))

def _adj_list(grL):
    A = defaultdict(set)
    for a, b in zip(grL['u'], grL['v']): A[a].add(b); A[b].add(a)
    return A

def _area(p): p = np.asarray(p); x, y = p[:, 0], p[:, 1]; return float(0.5*abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1))))

def _perim(p): p = np.asarray(p); return float(np.sum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))))

def _cent(p):
    p = np.asarray(p); x, y = p[:, 0], p[:, 1]; a = x*np.roll(y, 1)-np.roll(x, 1)*y; A = 0.5*np.sum(a)
    if abs(A) < 1e-9: return p.mean(0)
    return np.array([np.sum((x+np.roll(x, 1))*a)/(6*A), np.sum((y+np.roll(y, 1))*a)/(6*A)])

def _haus(p, q): return float(max(directed_hausdorff(np.asarray(p), np.asarray(q))[0], directed_hausdorff(np.asarray(q), np.asarray(p))[0]))

def _near(pt, pts): pts = [np.asarray(a) for a in pts]; return int(np.argmin([np.hypot(*(np.asarray(pt)-a)) for a in pts]))

def elliptic_signed_full(trj, kk=12):
    gr = build_graph(trj, kk); u, v = gr['u'], gr['v']; struct = gr['struct']; inc = gr['inc']
    R = trj[v]-trj[u]; dist = np.linalg.norm(R, axis=2)
    cr = R[:, :-1, 0]*R[:, 1:, 1]-R[:, :-1, 1]*R[:, 1:, 0]; dtp = R[:, :-1, 0]*R[:, 1:, 0]+R[:, :-1, 1]*R[:, 1:, 1]
    dth = np.arctan2(cr, dtp); wrap = lambda x: (x+np.pi) % (2*np.pi)-np.pi
    phi0 = np.angle(np.mean(np.exp(1j*dth), 0)); rsd = wrap(dth-phi0[None, :]); dtheta = wrap(rsd-np.median(rsd, 0)[None, :]); netw = np.sum(dtheta, 1)
    mdt = np.mean(dtheta, 1); sdt = np.std(dtheta, 1); c_e = np.abs(mdt)/(np.abs(mdt)+sdt+1e-6)
    mdist = dist.mean(1); sg = np.median(mdist); g_e = np.exp(-(mdist*mdist)/(2*sg*sg)); q_e = np.tanh(np.abs(netw)/np.pi)
    w_ell = 0.1+c_e*g_e*q_e; Xell = afrc_curv(gr, w_ell)-np.median(w_ell)*struct
    zXr = node_mean(gr, Xell); zX = zXr/(1.4826*np.median(np.abs(zXr-np.median(zXr)))+1e-12)
    Phi = np.array([abs(np.median(netw[inc[i]])) if inc[i] else 0. for i in range(gr['N'])])
    sigstar = np.sign(np.nansum(zX*Phi)) or 1.0
    return sigstar*zX, Phi, gr

def laac_and_objrot(tr, K, kk=12):
    S = tr.shape[1]; bs = max(8, (S-1)//K); accZ = np.zeros(tr.shape[0]); accP = np.zeros(tr.shape[0]); grL = None; nb = 0
    for j in range(K):
        a = j*bs; b = min(S, a+bs+1)
        if b-a < 8: break
        z, P, g = elliptic_signed_full(tr[:, a:b, :], kk); accZ += z; accP += P; nb += 1
        if grL is None: grL = g
    return accZ/max(nb, 1), accP/max(nb, 1), grL

def _graph_smooth(x, grL, keep, iters=3, a=0.6):
    u, v = grL['u'], grL['v']; N = grL['N']; adj = _adj_list(grL)
    interior = np.array([keep[i] and len(adj[i]) > 0 and all(keep[j] for j in adj[i]) for i in range(N)])
    x = x.copy()
    for _ in range(iters):
        sm = np.zeros(N); cn = np.zeros(N)
        np.add.at(sm, u, x[v]); np.add.at(sm, v, x[u]); np.add.at(cn, u, 1.); np.add.at(cn, v, 1.)
        xa = a*x+(1-a)*sm/np.maximum(cn, 1); x = np.where(interior, xa, x)
    return x

def _otsu(vals):
    v = vals[np.isfinite(vals)]
    if v.size < 8: return 0.5*(np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else 1.0)
    hist, edges = np.histogram(v, bins=64); p = hist/hist.sum(); c = np.cumsum(p); m = np.cumsum(p*0.5*(edges[:-1]+edges[1:]))
    mt = m[-1]; sigmab = (mt*c-m)**2/(c*(1-c)+1e-12); k = np.nanargmax(sigmab); return 0.5*(edges[k]+edges[k+1])

def _support_subgraph(grL, supp):
    u, v = grL['u'], grL['v']; N = grL['N']
    m = supp[u] & supp[v]; uu, vv = u[m], v[m]
    A = coo_matrix((np.ones(2*len(uu)), (np.r_[uu, vv], np.r_[vv, uu])), shape=(N, N)).tocsr()
    ncomp, lab = connected_components(A, directed=False)
    adj = defaultdict(set)
    for a, b in zip(uu, vv): adj[a].add(b); adj[b].add(a)
    return adj, lab

def objrot_markers(Phi, grL, supp, P0, strength=0.30, merge_dist=2.0, interior=None, phi_gate=0.25):
    N = grL['N']; sadj, lab = _support_subgraph(grL, supp)
    ok = supp.copy()
    if interior is not None: ok = ok & interior(P0)
    ls = Phi.copy()
    for _ in range(2):
        sm = np.zeros(N); cn = np.zeros(N)
        for i in np.where(supp)[0]:
            nb = list(sadj[i])
            if nb: sm[i] = np.mean(ls[nb]); cn[i] = 1
        ls = np.where(cn > 0, 0.5*ls+0.5*sm, ls)
    gmax = ls[supp].max() if supp.any() else 1.0
    markers = []
    for c in np.unique(lab[supp]):
        comp = np.where((lab == c) & supp)[0]
        if comp.size < 8: continue
        cmax = ls[comp].max()
        if cmax < phi_gate*gmax: continue
        thr = max(strength*cmax, phi_gate*gmax)
        cand = [i for i in comp if ok[i] and sadj[i] and ls[i] >= thr and ls[i] >= max(ls[j] for j in sadj[i])]
        if not cand:
            oki = comp[ok[comp]]; cand = [int(oki[np.argmax(ls[oki])])] if oki.size else []
        for i in sorted(cand, key=lambda i: -ls[i]):
            if all(np.hypot(*(P0[i]-P0[j])) > merge_dist for j in markers): markers.append(i)
    return markers

def watershed_supp(field, adj, markers, supp):
    N = len(field); label = -np.ones(N, int); h = []
    for c, sd in enumerate(markers):
        label[sd] = c
        for j in adj[sd]:
            if label[j] < 0 and supp[j]: heapq.heappush(h, (-field[j], j, c))
    while h:
        _, i, c = heapq.heappop(h)
        if label[i] >= 0 or not supp[i]: continue
        label[i] = c
        for j in adj[i]:
            if label[j] < 0 and supp[j]: heapq.heappush(h, (-field[j], j, c))
    return label

def _fill_holes(label, adj, iters=10, frac=0.6):
    label = label.copy()
    for _ in range(iters):
        changed = 0
        for i in np.where(label < 0)[0]:
            nb = adj[i]
            if not nb: continue
            lb = [label[j] for j in nb if label[j] >= 0]
            if len(lb) >= frac*len(nb):
                vals, cnts = np.unique(lb, return_counts=True); label[i] = int(vals[np.argmax(cnts)]); changed += 1
        if changed == 0: break
    return label

def unified_field(trc, K, cfg, kk=12, lam=2.0, kappa2_fac=0.30, taufrac=0.5, smooth_core=6, strength=0.30, afrc_mix=0.45):
    P0 = trc[:, 0, :]
    laac, Phi, grL = laac_and_objrot(trc, K, kk)
    N = grL['N']; u, v = grL['u'], grL['v']; adj = _adj_list(grL)
    phi_core = np.clip(laac, 0, None)
    if phi_core.max() > 0: phi_core = phi_core/phi_core.max()
    Phi = np.clip(Phi, 0, None)
    if Phi.max() > 0: Phi = Phi/Phi.max()
    tauplus = taufrac*_otsu(phi_core[phi_core > 1e-6])
    supp0 = phi_core > tauplus
    if cfg.get('interior') is not None: supp0 = supp0 & cfg['interior'](P0)
    markers = objrot_markers(Phi, grL, supp0, P0, strength=strength, merge_dist=cfg['merge'], interior=cfg.get('interior'))
    label = watershed_supp(Phi, adj, markers, supp0)
    label = _fill_holes(label, adj, iters=8, frac=0.65)
    filled = label >= 0
    A = coo_matrix((np.ones(2*len(u)), (np.r_[u, v], np.r_[v, u])), shape=(N, N)).tocsr()
    deg = np.asarray(A.sum(1)).ravel(); L = diags(deg)-A
    kap2 = kappa2_fac*np.median(deg[deg > 0])
    Psi = np.zeros(N)
    phi_core_s = _graph_smooth(phi_core, grL, filled, iters=smooth_core)
    basins = []
    for c in range(len(markers)):
        interior = np.where(label == c)[0]
        if interior.size < 10: continue
        M = (L[interior][:, interior].tocsc()+kap2*diags(np.ones(interior.size))).tocsc()
        try: sol = np.maximum(spsolve(M, Phi[interior]), 0.0)
        except Exception: continue
        if sol.max() <= 0: continue
        Psi[interior] = sol/sol.max(); basins.append((c, interior))
    E = np.zeros(N)
    for c, iv in basins:
        psi = Psi[iv]; pl = phi_core_s[iv]; plm = pl.max(); pl = pl/plm if plm > 0 else pl
        e = psi*((1.0-afrc_mix)+afrc_mix*pl); em = e.max(); E[iv] = e/em if em > 0 else e
    cores = [P0[iv[np.argmax(Psi[iv])]] for c, iv in basins]
    return dict(E=E, phi_core=phi_core, phi_core_s=phi_core_s, Phi=Phi, Psi=Psi,
                label=label, basins=basins, cores=cores, grL=grL, tauplus=tauplus, P0=P0)

def _largest_closed_contour(binimg, xs, ys):
    from skimage.measure import find_contours
    best = None
    for c in find_contours(binimg.astype(float), 0.5):
        py = ys[0]+c[:, 0]*(ys[-1]-ys[0])/(binimg.shape[0]-1)
        px = xs[0]+c[:, 1]*(xs[-1]-xs[0])/(binimg.shape[1]-1)
        poly = np.column_stack([px, py])
        if np.hypot(*(poly[0]-poly[-1])) > 1e-6: continue
        A = 0.5*abs(np.dot(poly[:, 0], np.roll(poly[:, 1], 1))-np.dot(poly[:, 1], np.roll(poly[:, 0], 1)))
        if best is None or A > best[0]: best = (A, poly)
    return best[1] if best else None

def _smooth_poly(poly, frac=0.035):
    if poly is None or len(poly) < 10: return poly
    p = poly.copy()
    if np.hypot(*(p[0]-p[-1])) < 1e-9: p = p[:-1]
    s = max(1.2, frac*len(p))
    x = gaussian_filter1d(p[:, 0], s, mode='wrap'); y = gaussian_filter1d(p[:, 1], s, mode='wrap')
    q = np.column_stack([x, y]); return np.vstack([q, q[0]])

def unified_boundaries(res, cloud, box, ng=(360, 320), trim=0.5):
    label = res['label']; basins = res['basins']
    cloud = np.asarray(cloud); x0, x1, y0, y1 = box
    gx = np.linspace(x0, x1, ng[0]); gy = np.linspace(y0, y1, ng[1]); GX, GY = np.meshgrid(gx, gy)
    labgood = label >= 0
    ZL = griddata(cloud[labgood], label[labgood].astype(float), (GX, GY), method='nearest')
    dd, _ = cKDTree(cloud[labgood]).query(np.column_stack([GX.ravel(), GY.ravel()])); far = (dd > trim).reshape(GX.shape)
    polys = []; radii = []
    for c, iv in basins:
        reg = binary_fill_holes((np.round(ZL) == c) & (~far))
        if reg.sum() < 12: polys.append(None); radii.append(np.nan); continue
        poly = _smooth_poly(_largest_closed_contour(reg, gx, gy))
        polys.append(poly); radii.append(_eqr(poly) if poly is not None else np.nan)
    return polys, radii, None, [x0, x1, y0, y1]

def gk_grid(cloud, vals, box, n=(340, 300), sigph=0.40, trim=0.5, k=80):
    """scatter->grid via k-NN Gaussian-kernel smoothing (facet-free; no Delaunay notches)."""
    cloud = np.asarray(cloud); vals = np.asarray(vals, float); good = np.isfinite(vals)
    cl = cloud[good]; vl = vals[good]; x0, x1, y0, y1 = box
    gx = np.linspace(x0, x1, n[0]); gy = np.linspace(y0, y1, n[1]); GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()]); tree = cKDTree(cl); kk = min(k, len(cl))
    dist, idx = tree.query(pts, k=kk, workers=-1)
    if kk == 1: dist = dist[:, None]; idx = idx[:, None]
    w = np.exp(-(dist*dist)/(2*sigph*sigph)); Z = (w*vl[idx]).sum(1)/np.maximum(w.sum(1), 1e-12)
    Z[dist[:, 0] > trim] = np.nan
    return Z.reshape(GX.shape), [x0, x1, y0, y1]
