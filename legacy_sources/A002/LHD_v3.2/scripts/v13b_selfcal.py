"""Self-calibration of a static (lab-frame) calibration distortion d(x) from positions only.  The distortion enters ln J
as the change of g = div d along the trajectory.  On segments over which a tracer's star moves by a pure translation
(least-squares affine map of the star within tol of the identity, no hot step), the true volume change is zero, so the
measured star-area log ratio equals g(x(t2)) - g(x(t1)).  g is fitted on a bilinear grid (spacing 4h, gradient
regularisation) by sparse least squares, and the ledger is corrected by g(x(T)) - g(x(0)).  Same distortion field as
scripts/v13_ptv_synth.py (calib).  usage: python scripts/v13b_selfcal.py"""
import sys, json, time, numpy as np
sys.path[:0] = ["lhd", "scripts"]
import lhdx as X, ledger_v4 as L4
from vcases import load, st
from ledger_ub import star_nb, _affine_F, bad_from_hot
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr
t0 = time.time(); fr0, rho0, ref, m = load("sv"); d, _ = cKDTree(fr0[0]).query(fr0[0], k=2); h = float(np.median(d[:, 1])); sig = 0.02 * h
r2 = np.random.default_rng(5); k = r2.normal(size=(16, 2)) / (10 * h); ph = r2.uniform(0, 2 * np.pi, (2, 16)); a = r2.normal(size=(2, 16))
dfun = lambda P: 5 * sig * np.sqrt(2.0 / 16) * np.stack([np.cos(P @ k.T + ph[c]) @ a[c] for c in range(2)], 1)
S = len(fr0); fr = np.stack([fr0[s] + dfun(fr0[s]) for s in range(S)]); N = fr.shape[1]
hot = L4.hot_steps_geo(fr)[0]; bad = bad_from_hot(hot); r4 = L4.ledger_v4(fr, rho0=rho0, hot=hot)
E = lambda lj: rho0 * np.exp(-lj) / ref - 1
lo = fr.reshape(-1, 2).min(0) - 2 * h; hg = 4 * h; nx, ny = (np.ceil((fr.reshape(-1, 2).max(0) + 2 * h - lo) / hg).astype(int) + 1)
def W(P):                                  # bilinear weights -> sparse rows
    f = (P - lo) / hg; i = np.floor(f).astype(int); w = f - i; rows = np.arange(len(P))
    cols, vals = [], []
    for di, dj, ww in ((0, 0, (1 - w[:, 0]) * (1 - w[:, 1])), (1, 0, w[:, 0] * (1 - w[:, 1])), (0, 1, (1 - w[:, 0]) * w[:, 1]), (1, 1, w[:, 0] * w[:, 1])):
        cols.append((i[:, 1] + dj) * nx + i[:, 0] + di); vals.append(ww)
    return coo_matrix((np.concatenate(vals), (np.tile(rows, 4), np.concatenate(cols))), shape=(len(P), nx * ny)).tocsr()
lam_l = [float(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['0.3'])]; tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
rows, rhs, tdg = [], [], []; L = 5; nseg = 0
for s in range(0, S - L, 2):
    tri = X.complex_peeled(fr[s])["tri"]; nb = star_nb(tri, N); A1 = X._star(fr[s], tri, N); A2 = X._star(fr[s + L], tri, N)
    clean = ~bad[:, s:s + L + 1].any(1); clean &= ~np.where(nb >= 0, bad[np.maximum(nb, 0), s:s + L + 1].any(2), False).any(1)
    moved = np.linalg.norm(fr[s + L] - fr[s], axis=1) > 2 * h; idx = np.where(clean & moved & (A1 > 0) & (A2 > 0))[0]
    if not len(idx): continue
    F = _affine_F(fr[s], fr[s + L], idx, nb[idx]); dev = np.linalg.norm(F - np.eye(2), axis=(1, 2)); ok = idx[dev < tol]
    if not len(ok): continue
    rows.append(W(fr[s + L][ok]) - W(fr[s][ok])); rhs.append(np.log(A2[ok] / A1[ok])); nseg += len(ok)
    tdg.append((s, ok))
Am = vstack(rows); b = np.concatenate(rhs)
# gradient regularisation between neighbouring grid nodes
I = np.arange(nx * ny).reshape(ny, nx)
def D(a_, b_): n = a_.size; return coo_matrix((np.r_[np.ones(n), -np.ones(n)], (np.r_[np.arange(n), np.arange(n)], np.r_[a_.ravel(), b_.ravel()])), shape=(n, nx * ny))
def divd(P, e=1e-6):
    return ((dfun(P + [e, 0])[:, 0] - dfun(P - [e, 0])[:, 0]) + (dfun(P + [0, e])[:, 1] - dfun(P - [0, e])[:, 1])) / (2 * e)
tb = np.concatenate([divd(fr0[s_ + L][ok_]) - divd(fr0[s_][ok_]) for s_, ok_ in tdg])
ceq = float(np.corrcoef(b, tb)[0, 1]); print("data equations: corr(measured, true dg) = %.3f, rms measured %.4f true %.4f" % (ceq, b.std(), tb.std()), flush=True)
true_corr = divd(fr0[-1]) - divd(fr0[0]); best = None
for lam in lam_l:
    R = vstack([D(I[:, 1:], I[:, :-1]), D(I[1:, :], I[:-1, :])]) * lam
    g = lsqr(vstack([Am, R]).tocsr(), np.r_[b, np.zeros(R.shape[0])], atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    corr = (W(fr[-1]) - W(fr[0])) @ g; e = float(np.sqrt(np.mean((corr - true_corr)[m] ** 2)))
    q = st(E(r4["lnJ_v4"] - corr), m); print("  lam %.3g: v4 %.2f%%/%.2f%%  rms(corr-true) %.4f" % (lam, 100 * q["med"], 100 * q["p95"], e), flush=True)
    if best is None or e < best[0]: best = (e, lam, corr)
corr = best[2]
o = dict(segments=nseg, grid=[int(nx), int(ny)], grid_spacing_over_h=4, tol=tol, lam=best[1], corr_equations=ceq,
         before=dict(poly=st(E(r4["lnJ_poly"]), m), v4=st(E(r4["lnJ_v4"]), m)),
         after=dict(poly=st(E(r4["lnJ_poly"] - corr), m), v4=st(E(r4["lnJ_v4"] - corr), m)),
         oracle=dict(v4=st(E(r4["lnJ_v4"] - true_corr), m)),
         corr_vs_true_rms=best[0], true_corr_rms=float(np.sqrt(np.mean(true_corr[m] ** 2))))
json.dump(o, open("results/res_v13b_selfcal.json", "w"), indent=1)
f = lambda q: "%.2f%%/%.2f%%" % (100 * q["med"], 100 * q["p95"])
print("segments %d grid %dx%d | before v4 %s | self-calibrated v4 %s | oracle (true div d) %s | rms(corr-true) %.4f of rms(true) %.4f  %.0fs" % (
    nseg, nx, ny, f(o["before"]["v4"]), f(o["after"]["v4"]), f(o["oracle"]["v4"]), o["corr_vs_true_rms"], o["true_corr_rms"], time.time() - t0))
