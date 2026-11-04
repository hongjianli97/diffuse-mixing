"""Per-case transport evolution: label point plots, density and diffusing-scalar clouds at several times; time series of
one-way mass transfer, mass-weighted scalar variance, interface stretch/compression, density error; per-window metrics."""
import json, numpy as np, scipy.sparse as sp, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.ndimage import map_coordinates
from scipy.fft import dctn, idctn
import lhdx as X, netdiff as ND, afrc_h_engine as AH
from trajnet import TrajectoryNetwork as TN
from figs3 import show

def eul2d(vel, box, n, T, dt, c0fun, kappa, keep):
    x0, x1, y0, y1 = box; dxg = (x1 - x0) / n[0]; xs = x0 + (np.arange(n[0]) + .5) * dxg; ys = y0 + (np.arange(n[1]) + .5) * dxg; XX, YY = np.meshgrid(xs, ys)
    c = c0fun(XX, YY); kx = np.pi * np.arange(n[0]) / (x1 - x0); ky = np.pi * np.arange(n[1]) / (y1 - y0)
    damp = 1 / (1 + kappa * dt * (kx[None, :] ** 2 + ky[:, None] ** 2)); P = np.c_[XX.ravel(), YY.ravel()]; t = 0.0; out = {}
    for _ in range(int(round(T / dt))):
        um = vel(P - .5 * dt * vel(P, t + .5 * dt), t + .5 * dt); Pd = P - dt * um
        c = map_coordinates(c, [(Pd[:, 1] - y0) / dxg - .5, (Pd[:, 0] - x0) / dxg - .5], order=3, mode='nearest').reshape(n[1], n[0])
        c = idctn(dctn(c, norm='ortho') * damp, norm='ortho'); t += dt
        for k in keep:
            if abs(t - k) < 1e-6: out[k] = c.copy()
    return out, (XX, YY)

def lag1d(dmu, cond_face_fn, c0, ts, kappa, rec):
    c = c0.copy(); n = len(c); out = {0: c.copy()}
    for q, (a, b) in enumerate(zip(ts[:-1], ts[1:])):
        k = cond_face_fn(0.5 * (a + b)) * kappa / (0.5 * (dmu[1:] + dmu[:-1])); dt = b - a
        main = np.ones(n); lo = np.zeros(n - 1); up = np.zeros(n - 1)
        main[:-1] += dt * k / dmu[:-1]; main[1:] += dt * k / dmu[1:]; up -= dt * k / dmu[:-1]; lo -= dt * k / dmu[1:]
        c = spsolve(sp.diags([lo, main, up], [-1, 0, 1]).tocsc(), c)
        if q + 1 in rec: out[q + 1] = c.copy()
    return out

def wvar(c, w): mu = np.sum(w * c) / np.sum(w); return float(np.sum(w * (c - mu) ** 2) / np.sum(w))

def coherence(P0, P1, w, nb=(16, 16)):
    def boxes(P):
        x0, x1, y0, y1 = P[:, 0].min(), P[:, 0].max() + 1e-9, P[:, 1].min(), P[:, 1].max() + 1e-9
        return np.minimum((nb[1] * (P[:, 1] - y0) / (y1 - y0)).astype(int), nb[1] - 1) * nb[0] + np.minimum((nb[0] * (P[:, 0] - x0) / (x1 - x0)).astype(int), nb[0] - 1)
    b0, b1 = boxes(P0), boxes(P1); n = nb[0] * nb[1]; M = np.zeros((n, n)); np.add.at(M, (b0, b1), w)
    pr, qc = M.sum(1), M.sum(0); r_, c_ = pr > 0, qc > 0
    sv = np.linalg.svd(M[np.ix_(r_, c_)] / np.sqrt(np.outer(pr[r_], qc[c_])), compute_uv=False); return float(sv[1])

def interface_edges(P0, idx, tangent, k=12, alpha=np.pi / 8):
    E = AH._mutual_knn(P0, k); E = np.r_[E, E[:, ::-1]]; E = E[np.isin(E[:, 0], idx)]
    r0 = P0[E[:, 1]] - P0[E[:, 0]]; ok = np.abs((r0 * tangent[E[:, 0]]).sum(1)) / np.linalg.norm(r0, axis=1) > np.cos(alpha)
    return E[ok]

def interface_stretch(fr, s, E, idx, w):
    ls = np.log(np.linalg.norm(fr[s][E[:, 1]] - fr[s][E[:, 0]], axis=1) / np.linalg.norm(fr[0][E[:, 1]] - fr[0][E[:, 0]], axis=1))
    o = np.argsort(E[:, 0]); e0 = E[o, 0]; ls = ls[o]; u, st = np.unique(e0, return_index=True); med = np.array([np.median(g) for g in np.split(ls, st[1:])])
    ww = w[u]; return float(np.sum(ww * med) / np.sum(ww))

def run_case(cfg):
    """cfg: name, fr (S,N,2), t (S,), tidx (4), rho0 (N,), compressible, kappa, c0 (N,), label (N bool), label_names,
    side (fn P->bool), iface=(idx, tangent), refs: dict with optional 'flux'=(t, val), 'var'=(t, val), 'stretch'=(t, val),
    'comp'=(t, val), 'rho'=(S,N) array; extra_density=(title, P, rho) for one additional panel; coh_nb."""
    name, fr, t, tidx = cfg['name'], cfg['fr'], cfg['t'], cfg['tidx']; S, N = fr.shape[:2]; refs = cfg.get('refs', {})
    bad = None
    if cfg['compressible']:
        net = TN(fr, t[1] - t[0]); net.events(); bad = net.bad
    lnJ = X.chained_lnJ_v2(fr, bad); rho = cfg['rho0'][None, :] * np.exp(-lnJ); m = cfg['rho0'] * X.star_area(fr[0]) / 3.0
    dts = np.diff(t); cser = {0: cfg['c0'].copy()}; c = cfg['c0'].copy()
    for s in range(S - 1):                                   # implicit Euler, mass form, per snapshot interval
        out = ND.run(fr[s:s + 2], dts[s], c, cfg['kappa'], 'mass', m=m, rho=rho[s:s + 2]); c = out[1]; cser[s + 1] = c.copy()
    var = np.array([wvar(cser[s], m) for s in range(S)])
    lab = cfg['label']; side = cfg['side']; a0 = side(fr[0])
    wf = m * cfg.get('flux_weight', 1.0); flux = np.array([float(np.sum(wf * (~a0 & side(fr[s])))) for s in range(S)])
    idx, tang = cfg['iface']; E = interface_edges(fr[0], idx, tang)
    ss = np.unique(np.r_[np.linspace(0, S - 1, 11).astype(int), tidx]); stretch = np.array([interface_stretch(fr, s, E, idx, m) for s in ss])
    comp = np.array([float(np.sum(m[idx] * -lnJ[s][idx]) / np.sum(m[idx])) for s in ss])
    rerr = None
    if 'rho' in refs:
        R_ = refs['rho']; rerr = np.array([float(np.median(np.abs(rho[s] / R_[s] - 1))) for s in range(S)])
    win = []
    for a, b in zip(tidx[:-1], tidx[1:]):
        aa, bb = side(fr[a]), side(fr[b]); sa = ss.tolist()
        win.append(dict(window='[%.3g, %.3g]' % (t[a], t[b]), mass_into_side=float(np.sum(m * (~aa & bb))), mass_out_of_side=float(np.sum(m * (aa & ~bb))),
                        sigma2_mass=coherence(fr[a], fr[b], m, cfg.get('coh_nb', (16, 16))),
                        variance_ratio=float(var[b] / var[a]) if var[a] > 0 else None,
                        d_stretch=float(stretch[sa.index(b)] - stretch[sa.index(a)]), d_compression=float(comp[sa.index(b)] - comp[sa.index(a)])))
    # ---------------- figure
    fig = plt.figure(figsize=(16, 15)); g = fig.add_gridspec(4, 4)
    rlo, rhi = np.nanquantile(rho[list(tidx)], [0.01, 0.99]);
    if rhi - rlo < 0.05: rlo, rhi = rlo - 0.05, rhi + 0.05
    for j, s in enumerate(tidx):
        ax = fig.add_subplot(g[0, j]); P = fr[s]
        ax.scatter(P[~lab, 0], P[~lab, 1], s=0.6, c='#9ecae1', lw=0, label=cfg['label_names'][0]); ax.scatter(P[lab, 0], P[lab, 1], s=0.6, c='#de2d26', lw=0, label=cfg['label_names'][1])
        ax.scatter(P[idx, 0], P[idx, 1], s=1.5, c='k', lw=0); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title('t = %.3g: material labels' % t[s], fontsize=9)
        if j == 0: ax.legend(fontsize=6, markerscale=8, loc='upper right')
        for row, val, cm, lo_, hi_, ttl in ((1, rho[s], 'magma', rlo, rhi, 'density from positions'), (2, cser[s], 'viridis', 0, 1, 'diffusing scalar (mass form)')):
            if cfg.get('render') == 'scatter':
                ax = fig.add_subplot(g[row, j]); sc = ax.scatter(P[:, 0], P[:, 1], c=val, s=2.0, lw=0, cmap=cm, vmin=lo_, vmax=hi_)
                ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title('%s, t = %.3g' % (ttl, t[s]), fontsize=9)
                if j == 3: plt.colorbar(sc, ax=ax, fraction=0.046)
            else:
                show(fig.add_subplot(g[row, j]), P, val, cmap=cm, vmin=lo_, vmax=hi_, title='%s, t = %.3g' % (ttl, t[s]), cb=(j == 3))
    ax = fig.add_subplot(g[3, 0]); ax.plot(t, flux, 'C3-', label='network')
    if 'flux' in refs: ax.plot(*refs['flux'], 'k--', lw=1, label='reference')
    ax.set_xlabel('t'); ax.set_title('one-way mass transfer into ' + cfg['side_name'], fontsize=9); ax.legend(fontsize=7)
    ax = fig.add_subplot(g[3, 1]); ax.plot(t, var / var[0], 'C3-', label='network (mass form)')
    if 'var' in refs: ax.plot(refs['var'][0], np.array(refs['var'][1]) / refs['var'][1][0], 'k--', lw=1, label='reference')
    ax.set_xlabel('t'); ax.set_title('mass-weighted scalar variance', fontsize=9); ax.legend(fontsize=7)
    ax = fig.add_subplot(g[3, 2]); ax.plot(t[ss], stretch, 'C0o-', ms=3, label='tangential stretch (network)'); ax.plot(t[ss], comp, 'C3s-', ms=3, label='compression $-\\ln J$ (network)')
    if 'stretch' in refs: ax.plot(*refs['stretch'], 'C0--', lw=1, label='stretch, reference')
    if 'comp' in refs: ax.plot(*refs['comp'], 'C3--', lw=1, label='compression, reference')
    ax.set_xlabel('t'); ax.set_title('interface: $\\ln|\\nabla c|$ growth = stretch + compression', fontsize=9); ax.legend(fontsize=6)
    ax = fig.add_subplot(g[3, 3])
    if rerr is not None:
        ax.semilogy(t, np.maximum(rerr, 1e-16), 'C3-', label='positions-only density')
        if 'rho_extra' in refs: ax.semilogy(t, np.maximum(refs['rho_extra'][1], 1e-16), 'C1-', label=refs['rho_extra'][0])
        ax.set_title('median density error vs reference', fontsize=9); ax.legend(fontsize=7)
    else:
        ax.bar(np.arange(len(win)), [w['sigma2_mass'] for w in win], color='C2'); ax.set_ylim(min(w['sigma2_mass'] for w in win) - 0.02, 1.0)
        ax.set_xticks(np.arange(len(win))); ax.set_xticklabels([w['window'] for w in win], fontsize=7); ax.set_title('per-window coherence $\\sigma_2$ (mass-weighted)', fontsize=9)
    ax.set_xlabel('t' if rerr is not None else 'window')
    fig.suptitle(cfg['title'], fontsize=11); fig.tight_layout(); fig.savefig('fig_evo_%s.png' % name, dpi=110); plt.close(fig)
    res = dict(windows=win, var_final_ratio=float(var[-1] / var[0]), flux_final=float(flux[-1]), stretch=dict(zip(map(float, t[ss]), map(float, stretch))),
               compression=dict(zip(map(float, t[ss]), map(float, comp))), density_err_final=None if rerr is None else float(rerr[-1]))
    if 'flux' in refs: res['flux_final_reference'] = float(refs['flux'][1][-1])
    if 'var' in refs: res['var_final_ratio_reference'] = float(refs['var'][1][-1] / refs['var'][1][0])
    json.dump(res, open('res_evo_%s.json' % name, 'w'), indent=1)
    return res, dict(rho=rho, lnJ=lnJ, m=m, censor=(net.censor if cfg['compressible'] else None))
