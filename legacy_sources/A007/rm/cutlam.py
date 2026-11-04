"""network-native local stretch from the cut (Voronoi boundary) length, binned by the reference label of the interface."""
import numpy as np, pickle, sys
sys.path.insert(0, '.'); from marker_ref import marker_metrics, ref_curve
from scipy.spatial import cKDTree
def proj_label(P0):
    yy = np.linspace(0, 0.5, 20001); tr = cKDTree(ref_curve(yy)); _, j = tr.query(P0); return yy[j]
def cut_bins(net, fr, N, lab, ylab, s, dbin):
    E, w = net['E'][s], net['w'][s]; k0 = (E < N).all(1); E, w = E[k0], w[k0]; k = lab[E[:, 0]] != lab[E[:, 1]]; E, w = E[k], w[k]
    P = fr[s]; l = w * np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    nb = int(round(0.5 / dbin)); b = np.zeros(nb)
    for j in range(2):
        idx = np.clip((ylab[E[:, j]] / dbin).astype(int), 0, nb - 1); np.add.at(b, idx, 0.5 * l)
    return b
if __name__ == '__main__':
    case = sys.argv[1]; b = np.load('base_%s.npz' % case); net = pickle.load(open('net_%s.pkl' % case, 'rb')); N = int(b['N'])
    sel = b['sel']; fr = b['fr'][sel]; t = b['t'][sel]; phi0 = b['phi0']; lab = (phi0 > 0); ylab = proj_label(fr[0])
    mk = pickle.load(open('marker_%s.pkl' % case, 'rb'))
    for dbin in (0.0125, 0.025):
        b0 = cut_bins(net, fr, N, lab, ylab, 0, dbin); r0 = ref_curve(np.arange(0, 0.5 + dbin / 2, dbin)); ds0 = np.linalg.norm(np.diff(r0, axis=0), axis=1)
        for tt in (1, 2, 3, 4, 6, 8, 12):
            i = np.argmin(abs(t - tt))
            if abs(t[i] - tt) > 1e-6: continue
            lamn = cut_bins(net, fr, N, lab, ylab, i, dbin) / b0
            mm = marker_metrics(*mk[sel[i]], dbin=dbin)
            lamm = mm['lam']
            print('dbin %.4f t=%g  net ln<lam2> %.3f  marker %.3f | net <lnlam> %.3f marker %.3f | corr %.2f' % (dbin, tt, np.log((lamn ** 2 * ds0).sum() / ds0.sum()), np.log((lamm ** 2 * ds0).sum() / ds0.sum()),
                  (np.log(lamn) * ds0).sum() / ds0.sum(), (np.log(lamm) * ds0).sum() / ds0.sum(), np.corrcoef(np.log(lamn), np.log(lamm))[0, 1]))
