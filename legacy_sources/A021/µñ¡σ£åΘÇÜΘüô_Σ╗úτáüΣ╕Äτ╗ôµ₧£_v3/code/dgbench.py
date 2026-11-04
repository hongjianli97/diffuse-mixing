"""verbatim from routeA_H_validation_ridge.ipynb cell 77 (paper Fig. 3 machinery)."""
import numpy as np
from scipy.ndimage import gaussian_filter
def make_dg(eps=0.25, A=0.1, om=2*np.pi/10):
    def vel(P, t=0.0):
        x, y = P[:, 0], P[:, 1]; a = eps*np.sin(om*t); f = a*x*x+(1-2*a)*x; fx = 2*a*x+(1-2*a)
        return np.column_stack([-np.pi*A*np.sin(np.pi*f)*np.cos(np.pi*y), np.pi*A*np.cos(np.pi*f)*np.sin(np.pi*y)*fx])
    return vel
def rk4_traj(vel, P0, T, dt, direction="fwd"):
    sgn = 1.0 if direction == "fwd" else -1.0; nt = int(round(T/dt))+1
    tr = np.empty((P0.shape[0], nt, 2)); P = P0.copy(); h = sgn*dt
    for n in range(nt):
        tr[:, n, :] = P; t0 = sgn*n*dt
        k1 = vel(P, t0); k2 = vel(P+.5*h*k1, t0+.5*h); k3 = vel(P+.5*h*k2, t0+.5*h); k4 = vel(P+h*k3, t0+h)
        P = P+(h/6)*(k1+2*k2+2*k3+k4)
    return tr
def ftle_from_positions(P1, dx, dy, T):
    a11 = np.gradient(P1[..., 0], dx, axis=1); a12 = np.gradient(P1[..., 0], dy, axis=0)
    a21 = np.gradient(P1[..., 1], dx, axis=1); a22 = np.gradient(P1[..., 1], dy, axis=0)
    C00 = a11*a11+a21*a21; C01 = a11*a12+a21*a22; C11 = a12*a12+a22*a22
    tr = C00+C11; det = C00*C11-C01*C01; disc = np.sqrt(np.maximum(tr*tr/4-det, 0.0)); l2 = tr/2+disc
    return np.log(np.maximum(l2, 1e-30))/(2*abs(T))
def AFRC_H_primitive(trg):
    """Grid-native AFRC_H proxy: accumulated |Δ log(edge length)| over the 8-neighbour
       stencil — the same stretching-bottleneck the Forman/AFRC edge weight measures,
       evaluated on the regular grid (no kNN graph), i.e. the lightweight benchmark
       twin of the notebook's full multi-scale AFRC engine."""
    ny, nx, T, _ = trg.shape; acc = np.zeros((ny, nx)); cnt = np.zeros((ny, nx))
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0: continue
            ia0, ia1 = max(0, di), ny+min(0, di); ja0, ja1 = max(0, dj), nx+min(0, dj)
            a = trg[ia0:ia1, ja0:ja1]; b = trg[max(0, -di):ny+min(0, -di), max(0, -dj):nx+min(0, -dj)]
            d = np.linalg.norm(a-b, axis=-1)+1e-12
            st = np.sum(np.abs(np.diff(np.log(d), axis=-1)), axis=-1)/max(T-1, 1)
            acc[ia0:ia1, ja0:ja1] += st; cnt[ia0:ia1, ja0:ja1] += 1
    return acc/np.maximum(cnt, 1)
def tse_bar_grid(trg, dt):
    v = np.gradient(trg, dt, axis=2); sp = np.linalg.norm(v, axis=-1)+1e-12
    return np.sum(np.abs(np.diff(np.log(sp), axis=-1)), axis=-1)/max(trg.shape[2]-1, 1)
def _dgb_nz(a): a = np.nan_to_num(a); a = a-a.min(); return a/(a.max()+1e-12)
def ridge_strength(field, smooth=1.8):
    f = gaussian_filter(_dgb_nz(field), smooth)
    fy, fx = np.gradient(f); fyy, _ = np.gradient(fy); _, fxx = np.gradient(fx); fxy = np.gradient(fx, axis=0)
    tr = fxx+fyy; det = fxx*fyy-fxy*fxy; disc = np.sqrt(np.maximum(tr*tr/4-det, 0)); lam = tr/2-disc
    return np.maximum(-lam, 0.0)

