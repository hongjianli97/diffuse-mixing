"""Validate 'co-rotation weight -> core, curvature (AFRC_E) -> boundary' against Haller et al. (2016) LAVD vortices
(outermost closed convex LAVD contour around each maximum).  Extraction never uses the reference."""
import json, pickle, time, numpy as np, contourpy, cases, lhdx as X, afrc_engine as AF
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from figs3 import project

def convex_vortices(F, xg, yg, min_area, cd_max=0.02, nlev=60, peak_q=0.90, nbhd=9, maxv=8):
    gen = contourpy.contour_generator(xg, yg, np.nan_to_num(F))
    mx = (F == ndi.maximum_filter(np.nan_to_num(F), size=nbhd)) & (F > np.nanquantile(F, peak_q))
    iy, ix = np.where(mx); order = np.argsort(-F[iy, ix]); polys = []
    area = lambda p: 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))
    for k in order:
        x0, y0, f0 = xg[ix[k]], yg[iy[k]], F[iy[k], ix[k]]
        if any(Path(p).contains_point((x0, y0)) for p in polys): continue
        best = None
        for lev in np.linspace(f0, np.nanquantile(F, 0.05), nlev)[1:]:
            found = None
            for ln in gen.lines(lev):
                if len(ln) < 8 or np.linalg.norm(ln[0] - ln[-1]) > 1e-9: continue
                if Path(ln).contains_point((x0, y0)): found = ln; break
            if found is None: break
            a = area(found)
            try: ah = ConvexHull(found).volume
            except Exception: break
            if (ah - a) / max(a, 1e-30) > cd_max: break
            best = found
        if best is not None and area(best) >= min_area: polys.append(best)
        if len(polys) >= maxv: break
    return polys

def mask_of(polys, XX, YY):
    m = np.zeros(XX.shape, bool); pts = np.c_[XX.ravel(), YY.ravel()]
    for p in polys: m |= Path(p).contains_points(pts).reshape(XX.shape)
    return m

def analyse(name, P, lavd, weight, afrcE, ref_none=False):
    n01 = lambda v: np.clip((v - np.nanquantile(v, 0.005)) / max(np.nanquantile(v, 0.995) - np.nanquantile(v, 0.005), 1e-30), 0, 1)
    bbox = (P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max())
    FL, ext = project(P, lavd, bbox=bbox, nx=260); FW, _ = project(P, n01(weight), bbox=bbox, nx=260); FE, _ = project(P, n01(afrcE), bbox=bbox, nx=260)
    xg, yg = np.linspace(ext[0], ext[1], FL.shape[1]), np.linspace(ext[2], ext[3], FL.shape[0]); XX, YY = np.meshgrid(xg, yg)
    cell = (xg[1] - xg[0]) * (yg[1] - yg[0]); dom = np.isfinite(FL).sum() * cell
    polys = [] if ref_none else convex_vortices(FL, xg, yg, min_area=0.002 * dom)
    # proposed extraction: cores = local maxima of the weight; region = filled AFRC_E>0.5 component holding the core
    W0 = np.nan_to_num(FW); E0 = np.nan_to_num(FE)
    cores = (W0 == ndi.maximum_filter(W0, size=11)) & (W0 > 0.5)
    lab, nl = ndi.label(ndi.binary_fill_holes(E0 > 0.5))
    regs = []
    for jy, jx in zip(*np.where(cores)):
        l = lab[jy, jx]
        if l > 0 and l not in [r[0] for r in regs]:
            regs.append((l, (xg[jx], yg[jy])))
    MR = np.isin(lab, [r[0] for r in regs]) if regs else np.zeros_like(W0, bool)
    ML = mask_of(polys, XX, YY)
    res = dict(n_lavd_vortices=len(polys), n_detected=len(regs), detected_area_frac=float(MR.sum() * cell / dom))
    if polys:
        res['IoU_proposed'] = float((MR & ML).sum() / max((MR | ML).sum(), 1))
        mw = ndi.binary_fill_holes(W0 > 0.5); res['IoU_weight_only'] = float((mw & ML).sum() / max((mw | ML).sum(), 1))
        me = E0 > 0.5; res['IoU_afrcE_only'] = float((me & ML).sum() / max((me | ML).sum(), 1))
        dw, de = [], []
        for p in polys:
            m = mask_of([p], XX, YY); req = np.sqrt(m.sum() * cell / np.pi)
            iL = np.nanargmax(np.where(m, FL, -np.inf)); iW = np.nanargmax(np.where(m, W0, -np.inf)); iE = np.nanargmax(np.where(m, E0, -np.inf))
            cL = np.array([XX.ravel()[iL], YY.ravel()[iL]])
            dw.append(np.linalg.norm(np.array([XX.ravel()[iW], YY.ravel()[iW]]) - cL) / req)
            de.append(np.linalg.norm(np.array([XX.ravel()[iE], YY.ravel()[iE]]) - cL) / req)
        res['centre_offset_weight'] = float(np.median(dw)); res['centre_offset_afrcE'] = float(np.median(de))
    print(name, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}), flush=True)
    return res, dict(FL=FL, FW=FW, FE=FE, ext=ext, polys=polys, MR=MR, cores=[r[1] for r in regs])

def afrcE_with_weight(fr):
    AF._CAP_W = []; F = AF.afrc_fields(np.ascontiguousarray(np.transpose(fr, (1, 0, 2))))
    w = np.nanmean(np.stack(AF._CAP_W), 0); AF._CAP_W = None
    return F['core_med'], w

R, G = {}, {}
S = pickle.load(open('res_afrcE_study.pkl', 'rb'))
for nm in ('LO_Rd6.5', 'LO_a1.4_Rd5.6', 'DG', 'SB'):
    R[nm], G[nm] = analyse(nm, S[nm]['P'], S[nm]['lavd'], S[nm]['weight'], S[nm]['AFRC_E_median'])
D = pickle.load(open('sv_dns.pkl', 'rb'))
Wst = np.array([X.star_area(D['fr_in'][s]) for s in range(10, 31)])
lavd_sv = X.lavd_from_series(D['Ev']['om'][10:31], D['dt'], A_weights=Wst)
for mode in ('in', 'off'):
    fr = D['fr_' + mode][10:31]; E_, w_ = afrcE_with_weight(fr)
    R['SV_' + mode], G['SV_' + mode] = analyse('SV_' + mode, fr[0], lavd_sv, w_, E_)
c = cases.case_blast(); E_, w_ = afrcE_with_weight(c['frames'])
R['BLAST'], G['BLAST'] = analyse('BLAST (irrotational: should detect nothing)', c['frames'][0], np.zeros(len(E_)) + 1e-9 * np.random.default_rng(0).random(len(E_)), w_, E_, ref_none=True)
json.dump(R, open('res_core_boundary.json', 'w'), indent=1); pickle.dump(G, open('res_core_boundary.pkl', 'wb'))
