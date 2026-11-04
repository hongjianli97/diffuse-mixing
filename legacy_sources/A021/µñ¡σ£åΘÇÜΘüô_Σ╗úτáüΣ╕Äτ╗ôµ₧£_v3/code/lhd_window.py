"""
lhd_window.py -- per-window LHD driver on the Part-1 outputs (Part-1/Part-2 code untouched).

    from lhd_window import run_window, kelvin_budget
    s = run_window(".../tc0.90_dE0.150/data/tracking_layers_tc0.90_dE0.150.npz", layer="fwd",
                   euler_npz=".../euler_context_tc0.90.npz", max_n=160_000)
    # -> lhd_fwd.npz next to the tracking file

Keys read (Part-1 §SAVE-v2): {ln}__traj, {ln}__dt_total, {ln}__oob, {ln}__objrot__vd (optional;
recomputed with mean references if absent), {ln}__ev_pidx, {ln}__ev_step, {ln}__ev_r_ratio
(optional).  euler_context*.npz: x, y, rho at t_c (any layout; 4-NN inverse-distance lookup).
Pass sel= (indices into the full tracer set) to reuse the FRC6 subsample, pairs= to reuse E.
"""
import os, time
import numpy as np
from scipy.spatial import cKDTree
import lhd2 as lhd


def load_layer(npz_path, layer="fwd", sel=None, max_n=None, seed=17):
    z = np.load(npz_path, allow_pickle=False)
    traj_all = z[f"{layer}__traj"]
    N_all, T = traj_all.shape[:2]
    dt = float(z[f"{layer}__dt_total"]) / max(T - 1, 1)              # signed: bwd < 0
    oob = (np.asarray(z[f"{layer}__oob"], bool) if f"{layer}__oob" in z.files
           else np.zeros(N_all, bool))
    good = ~oob & np.isfinite(traj_all).all(axis=(1, 2))
    if sel is None:
        idx = np.flatnonzero(good)
        if max_n is not None and idx.size > max_n:
            idx = np.sort(np.random.default_rng(seed).choice(idx, int(max_n), replace=False))
    else:
        idx = np.asarray(sel, np.int64)
        idx = idx[good[idx]]
    traj = np.asarray(traj_all[idx], np.float64)
    vk = f"{layer}__objrot__vd"
    vd = np.asarray(z[vk][idx], np.float64) if vk in z.files else None
    ev = {}
    if f"{layer}__ev_pidx" in z.files:
        lut = np.full(N_all, -1, np.int64)
        lut[idx] = np.arange(idx.size)
        p = np.asarray(z[f"{layer}__ev_pidx"], np.int64)
        ok = (p >= 0) & (p < N_all)
        loc = np.full(p.size, -1, np.int64)
        loc[ok] = lut[p[ok]]
        keep = loc >= 0
        ev["pidx"] = loc[keep]
        ev["step"] = np.asarray(z[f"{layer}__ev_step"], np.int64)[keep]
        for k in ("r_ratio", "t", "drho", "du_n", "ds_RH"):
            if f"{layer}__ev_{k}" in z.files:
                ev[k] = np.asarray(z[f"{layer}__ev_{k}"])[keep]
    meta = {k[len("__meta__"):]: z[k].item() for k in z.files if k.startswith("__meta__")}
    return dict(traj=traj, dt=dt, vd=vd, idx=idx, ev=ev, meta=meta, N_all=N_all)


def rho_at(euler_npz, P, k=4):
    """Density at t_c at tracer positions from euler_context*.npz (layout-agnostic)."""
    z = np.load(euler_npz, allow_pickle=False)
    x, y, rho = (np.asarray(z[n], np.float64) for n in ("x", "y", "rho"))
    if x.size == rho.size and y.size == rho.size:
        pts, vals = np.column_stack([x.ravel(), y.ravel()]), rho.ravel()
    elif x.ndim == 1 and y.ndim == 1 and rho.size == x.size * y.size:
        X2, Y2 = (np.meshgrid(x, y) if rho.shape == (y.size, x.size)
                  else np.meshgrid(x, y, indexing="ij"))
        pts, vals = np.column_stack([X2.ravel(), Y2.ravel()]), rho.ravel()
    else:
        raise ValueError(f"unrecognised euler_context layout x{x.shape} y{y.shape} rho{rho.shape}")
    fin = np.isfinite(vals) & np.isfinite(pts).all(1)
    d, j = cKDTree(pts[fin]).query(P, k=k)
    w = 1.0 / np.maximum(d, 1e-12)
    return (vals[fin][j] * w).sum(1) / w.sum(1)


def _last_per_group(group, key, value, n):
    o = np.lexsort((key, group))
    last = np.r_[group[o][1:] != group[o][:-1], True]
    out = np.full(n, np.nan)
    out[group[o][last]] = value[o][last]
    return out


def run_window(npz_path, layer="fwd", euler_npz=None, max_n=160_000, sel=None, seed=17,
               pairs=None, k_pairs=8, block=5, halo=1, max_len_fac=3.0, local_k=None,
               out_path=None, verbose=True):
    """Per-tracer and per-pair LHD fields for one layer of one window (see module doc)."""
    t_all = time.time()
    Ld = load_layer(npz_path, layer, sel=sel, max_n=max_n, seed=seed)
    X, dt = Ld["traj"], Ld["dt"]
    N, T = X.shape[:2]
    V = Ld["vd"]
    if V is None:
        Vr = lhd.velocity_from_positions(X, dt)
        V = np.stack([lhd.relative_velocity(X[:, t], Vr[:, t]) for t in range(T)], 1)
    ev = Ld["ev"]
    badv = lhd.vertex_bad_mask(ev.get("pidx", []), ev.get("step", []), N, T, halo)
    pairs = (lhd.mutual_knn_pairs([X[:, 0], X[:, -1]], k_pairs) if pairs is None
             else np.asarray(pairs, np.int64))
    nP, f32 = len(pairs), np.float32
    div = np.full((N, T), np.nan, f32); vort = np.full((N, T), np.nan, f32)
    badn = np.zeros((N, T), bool)
    St = np.full((nP, T), np.nan, f32); Sg = np.full((nP, T), np.nan, f32); Sc = np.full((nP, T), np.nan, f32)
    Efrac, tsnap = np.zeros((T, 3)), []
    for t in range(T):
        t1 = time.time()
        P = X[:, t]
        cx = lhd.complex_peeled(P)                                   # v2: no internal holes
        f = lhd.edge_cochain(P, V[:, t], cx["edges"])
        dec = lhd.galerkin_hodge(P, cx, V[:, t])                       # v2: Galerkin P1 / Crouzeix-Raviart
        tb = badv[:, t][cx["tri"]].any(1)
        badn[:, t] = lhd.nodes_of(cx, tb) | badv[:, t]
        dv = lhd.galerkin_div(P, cx, V[:, t]); dv[lhd.open_star_nodes(cx)] = np.nan
        div[:, t] = dv
        vort[:, t] = lhd.node_vorticity(P, cx, f, tri_ok=~tb)
        Ug, Uc = lhd.face_to_node(P, cx, dec["ug"]), lhd.face_to_node(P, cx, dec["uc"])
        St[:, t] = lhd.stretch_rate(P, V[:, t], pairs)
        Sg[:, t] = lhd.stretch_rate(P, Ug, pairs)
        Sc[:, t] = lhd.stretch_rate(P, Uc, pairs)
        Efrac[t] = np.array([dec["Eg"], dec["Ec"], dec["Eh"]])
        tsnap.append(time.time() - t1)
        if verbose and (t % 10 == 0 or t == T - 1):
            print(f"  [lhd {layer}] snapshot {t + 1}/{T}  {tsnap[-1]:.1f}s")
    adt = abs(dt)
    with np.errstate(all="ignore"):
        clean = ~badn & np.isfinite(vort)
        wbar = np.nanmedian(np.where(clean, vort, np.nan), axis=0)       # ensemble de-bias
        dev = np.where(clean, np.abs(vort - wbar[None, :]), 0.0)
        ncl = clean.sum(1)
        lacd = np.where(ncl > 0, dev.sum(1) / np.maximum(ncl, 1) * (T - 1) * adt, np.nan)
        valid = np.isfinite(St) & np.isfinite(Sg) & np.isfinite(Sc)
        pb = badn[pairs[:, 0]] | badn[pairs[:, 1]] | ~valid
        net = lambda S, m: np.where(m, 0, S).sum(1) * dt
        net_t, net_g, net_c = net(St, pb), net(Sg, pb), net(Sc, pb)       # censored (= AFRC_H input)
        raw_g, raw_c = net(Sg, ~valid), net(Sc, ~valid)                   # all steps (incl. crossings)
        share_g = np.abs(net_g) / (np.abs(net_g) + np.abs(net_c) + 1e-30)
        share_g_raw = np.abs(raw_g) / (np.abs(raw_g) + np.abs(raw_c) + 1e-30)
        med_t = np.nanmedian(np.where(pb, np.nan, St), axis=1)
        med_g = np.nanmedian(np.where(pb, np.nan, Sg), axis=1)
        med_c = np.nanmedian(np.where(pb, np.nan, Sc), axis=1)
    gi, a2 = np.r_[pairs[:, 0], pairs[:, 1]], np.r_[np.abs(net_t), np.abs(net_t)]
    node_share = _last_per_group(gi, a2, np.r_[share_g, share_g], N)
    a2r = np.r_[np.abs(raw_g + raw_c), np.abs(raw_g + raw_c)]
    node_share_raw = _last_per_group(gi, a2r, np.r_[share_g_raw, share_g_raw], N)
    lnJ = lhd.chained_lnJ_v2(np.ascontiguousarray(np.transpose(X, (1, 0, 2))), badv, block=block).T   # v2
    star0 = lhd._star(X[:, 0], lhd.complex_peeled(X[:, 0])["tri"], N) / 3.0
    lnJ_tv = np.abs(np.diff(lnJ, axis=1)).sum(1)                         # Lagrangian compression dose
    rho_tc = rho_at(euler_npz, X[:, 0]) if euler_npz else np.full(N, np.nan)
    rho_T, mass = rho_tc * np.exp(-lnJ[:, -1]), rho_tc * star0
    rh = None
    if len(ev.get("pidx", [])) and "r_ratio" in ev:
        p, s = ev["pidx"], ev["step"]
        lo, hi = s - halo - 1, s + 2 + halo
        ok = (lo >= 0) & (hi <= T - 1)
        dJ = lnJ[p[ok], hi[ok]] - lnJ[p[ok], lo[ok]]
        r = np.asarray(ev["r_ratio"], np.float64)[ok]
        g = np.isfinite(dJ) & np.isfinite(r) & (r > 0)
        q = np.exp(np.abs(dJ[g])) / r[g]
        if q.size:
            rh = dict(n=int(q.size), median=round(float(np.median(q)), 4),
                      q25=round(float(np.quantile(q, .25)), 4), q75=round(float(np.quantile(q, .75)), 4))
    out_path = out_path or os.path.join(os.path.dirname(os.path.abspath(npz_path)), f"lhd_{layer}.npz")
    np.savez_compressed(
        out_path, idx=Ld["idx"], layer=np.asarray(layer), dt=np.asarray(dt),
        x0=X[:, 0].astype(f32), xT=X[:, -1].astype(f32), lnJ=lnJ.astype(f32), lnJ_tv=lnJ_tv.astype(f32),
        rho_tc=rho_tc.astype(f32), rho_T=rho_T.astype(f32), mass=mass.astype(f32),
        div=div, vort=vort, bad=badn, lacd=lacd.astype(f32),
        node_share_g=node_share.astype(f32), node_share_g_raw=node_share_raw.astype(f32),
        pairs=pairs.astype(np.int32), pair_net=net_t.astype(f32), pair_net_g=net_g.astype(f32),
        pair_net_c=net_c.astype(f32), pair_share_g=share_g.astype(f32),
        pair_net_g_raw=raw_g.astype(f32), pair_net_c_raw=raw_c.astype(f32),
        pair_share_g_raw=share_g_raw.astype(f32), pair_med=med_t.astype(f32),
        pair_med_g=med_g.astype(f32), pair_med_c=med_c.astype(f32), Efrac=Efrac.astype(f32))
    summ = dict(path=out_path, N=N, T=T, n_pairs=nP, sec_per_snapshot=float(np.mean(tsnap)),
                total_sec=time.time() - t_all, rh=rh)
    if verbose:
        print(f"[lhd {layer}] N={N} T={T} pairs={nP}  {summ['sec_per_snapshot']:.2f}s/snapshot  "
              f"total {summ['total_sec']:.0f}s  RH={rh}  -> {out_path}")
    return summ


def kelvin_budget(X, V, masks, badv=None, max_len_fac=3.0):
    """Circulation of material regions (tracer sets, e.g. AFRC_E cores at t_c) on fresh
    Delaunay complexes.  Use the INERTIAL-frame velocity here (Kelvin holds there), not v_d.
    Returns {name: dict(Gamma, perimeter, area, n_tri, clean)} over snapshots; pass
    Gamma[None], ~clean[None] to lhd.censored_increments for shock-safe increments."""
    T = X.shape[1]
    out = {k: dict(Gamma=np.zeros(T), perimeter=np.zeros(T), area=np.zeros(T),
                   n_tri=np.zeros(T, int), clean=np.ones(T, bool)) for k in masks}
    for t in range(T):
        P = X[:, t]
        cx = lhd.complex_peeled(P)
        f = lhd.edge_cochain(P, V[:, t], cx["edges"])
        tb = badv[:, t][cx["tri"]].any(1) if badv is not None else None
        for k, m in masks.items():
            o = out[k]
            o["Gamma"][t], o["perimeter"][t], o["area"][t], o["n_tri"][t] = \
                lhd.region_circulation(P, f, cx, m, max_len_fac)
            if tb is not None:
                o["clean"][t] = not np.any(tb & m[cx["tri"]].all(1))
    return out


def masks_from_labels(labels, min_size=10):
    """{label: bool mask} from an integer label per tracer (-1 = none), e.g. AFRC_E core components."""
    labels = np.asarray(labels)
    return {int(l): labels == l for l in np.unique(labels) if l >= 0 and (labels == l).sum() >= min_size}
