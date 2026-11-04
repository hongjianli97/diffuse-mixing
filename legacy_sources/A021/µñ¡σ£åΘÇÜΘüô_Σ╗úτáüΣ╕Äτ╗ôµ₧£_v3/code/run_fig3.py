"""Paper Fig. 3/4 reproduced exactly (grid proxy + ridge strength), then the SAME display and metric applied to the
actual AFRC_H curvature engine (Sec. II.D) on the same trajectories."""
import json, time, numpy as np, afrc_h_engine as AH
from dgbench import make_dg, rk4_traj, ftle_from_positions, AFRC_H_primitive, tse_bar_grid, ridge_strength, _dgb_nz
T, dt, nx, ny, TRIM = 15.0, 0.1, 200, 100, 5
vel = make_dg(); xa, ya = np.linspace(0, 2, nx), np.linspace(0, 1, ny); dx, dy = xa[1] - xa[0], ya[1] - ya[0]
GX, GY = np.meshgrid(xa, ya); P0 = np.column_stack([GX.ravel(), GY.ravel()])
interior = lambda a: a[TRIM:-TRIM, TRIM:-TRIM]
def cor(a, b):
    a = _dgb_nz(interior(a)).ravel(); b = _dgb_nz(interior(b)).ravel(); return float(np.corrcoef(a, b)[0, 1])
def engine(tr, ks=(12, 16, 24), Tw=T):
    per = []
    for k in ks:
        B, _, _ = AH.afrc_h_field(tr, k=k, dt=Tw); B = np.maximum(B - np.median(B), 0); pos = B[B > 0]
        per.append(B / (np.quantile(pos, 0.95) if pos.size else 1.0))
    return np.max(np.vstack(per), 0)
R = {}; OUT = {}
for dr in ('fwd', 'bwd'):
    t0 = time.time(); tr = rk4_traj(vel, P0, T, dt, dr); trg = tr.reshape(ny, nx, -1, 2)
    F = dict(proxy=AFRC_H_primitive(trg), FTLE=ftle_from_positions(trg[:, :, -1, :], dx, dy, T), TSE=tse_bar_grid(trg, dt))
    F['engine'] = engine(np.ascontiguousarray(tr[:, ::2, :])).reshape(ny, nx)          # 76 snapshots
    RS = {k: ridge_strength(v) for k, v in F.items()}
    R[dr] = {'RS_corr_vs_FTLE': {k: cor(RS[k], RS['FTLE']) for k in ('proxy', 'engine', 'TSE')},
             'raw_field_corr_vs_FTLE': {k: cor(F[k], F['FTLE']) for k in ('proxy', 'engine', 'TSE')}}
    OUT[dr] = dict(F=F, RS=RS); print(dr, json.dumps(R[dr]), '%.0fs' % (time.time() - t0), flush=True)
json.dump(R, open('res_fig3.json', 'w'), indent=1)
import pickle; pickle.dump(OUT, open('res_fig3.pkl', 'wb'))
