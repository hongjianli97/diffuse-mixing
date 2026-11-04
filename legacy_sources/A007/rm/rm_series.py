"""stage 2: network exchange potential split (stretch x compression), marker reference, cut length, and the
network counterfactual diffusions (full / frozen / compression-only / stretch-only) for the takeover criteria."""
import numpy as np, pickle, sys, time, json
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import netmix as NM, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
case = sys.argv[1]; Gam = float(sys.argv[2]); do_series = len(sys.argv) > 3
b = np.load('base_%s.npz' % case); d = pickle.load(open('net_%s.pkl' % case, 'rb'))
N = int(b['N']); sel = b['sel']; t = b['t'][sel]; S = len(sel); phi0 = b['phi0']; lnJ = b['lnJ'][sel]
net = NM.Net.__new__(NM.Net); net.tri, net.E, net.w, net.m = d['tri'], d['E'], d['w'], d['m']; net.N = len(net.m); net.S = S; net.dt = float(t[1] - t[0])
net.L = [NM.laplacian(net.N, net.E[s], net.w[s]) for s in range(S)]
rep = lambda v: np.r_[v, v, v]
w0 = 0.02; c0 = rep(0.5 * (1 + np.tanh(phi0 / (w0 / 2)))); lab = rep((phi0 > 0).astype(float)); J = np.exp(np.c_[lnJ, lnJ, lnJ])
m = net.m; out = dict(t=t.tolist(), case=case)
if do_series:
    q0 = net.node_dirichlet(0, c0); W0 = (m * q0).sum(); wq = m * q0
    ser = {k: [] for k in ('lnP', 'lnS', 'lnC1', 'lnC0', 'logmean_stretch2', 'logmean_comp', 'lnLnet')}
    keep_maps = {'single': (40, 80, 120), 'reshock': (20, 40, 80, 120)}[case]; maps = {}
    Q = np.zeros(net.N); x0 = lab[net.E[0][:, 0]] != lab[net.E[0][:, 1]]
    def Lnet(s):
        E, w = net.E[s], net.w[s]; k = (E < N).all(1) & (lab[E[:, 0]] != lab[E[:, 1]])
        P = d_fr[s]; return float((w[k] * np.linalg.norm(P[E[k, 0]] - P[E[k, 1]], axis=1)).sum())
    d_fr = b['fr'][sel]
    L0 = Lnet(0); qprev = q0
    for s in range(S):
        q = net.node_dirichlet(s, c0); Js = J[s]
        ser['lnP'].append(float(np.log((m * q).sum() / W0)))
        ser['lnS'].append(float(np.log((m * q * Js).sum() / W0)))
        ser['lnC1'].append(float(np.log((wq / Js).sum() / W0)))
        ser['lnC0'].append(float(np.log((wq / Js ** 2).sum() / W0)))
        ok = (q0 > 1e-8 * q0.max()) & (q > 0)
        ser['logmean_stretch2'].append(float(np.average(np.log(q[ok] * Js[ok] / q0[ok]), weights=wq[ok])))
        ser['logmean_comp'].append(float(np.average(-np.log(Js[ok]), weights=wq[ok])))
        ser['lnLnet'].append(float(np.log(Lnet(s) / L0)))
        if s > 0: Q += 0.5 * net.dt * (qprev + q)
        qprev = q
        if s in keep_maps:
            band = np.abs(phi0) < w0
            maps['lnstretch2_%d' % s] = np.where(band, np.log(np.maximum(q[:N] * Js[:N], 1e-300) / np.maximum(q0[:N], 1e-300)), np.nan)
            maps['clock_%d' % s] = np.where(band, np.log(np.maximum(Q[:N], 1e-300) / (t[s] * np.maximum(q0[:N], 1e-300))), np.nan)
    out['series'] = ser
    # marker reference
    mk = pickle.load(open('marker_%s.pkl' % case, 'rb')); xi = lambda y: 0.5 + 0.1 * np.cos(2 * np.pi * y)
    ref = dict(lnL=[], ln_lam2=[], mean_lnlam=[])
    for s in sel:
        M, lb = mk[s]; r0 = np.c_[xi(lb), lb]; ds0 = np.linalg.norm(np.diff(r0, axis=0), axis=1); ds = np.linalg.norm(np.diff(M, axis=0), axis=1)
        ok = ds0 > 0; lam = ds[ok] / ds0[ok]; w = ds0[ok]
        ref['lnL'].append(float(np.log(ds.sum() / ds0.sum()))); ref['ln_lam2'].append(float(np.log(np.average(lam ** 2, weights=w))))
        ref['mean_lnlam'].append(float(np.average(np.log(lam), weights=w)))
    out['marker'] = ref
    np.savez_compressed('maps_series_%s.npz' % case, **maps)
    print('series done', flush=True)
# counterfactual diffusions
t0 = time.time(); E0, w0e = net.E[0], net.w[0]
def Je(s, E): return 0.5 * (J[s][E[:, 0]] + J[s][E[:, 1]])
ops = {'full': lambda s: net.L[s], 'frozen': lambda s: net.L[0],
       'comp1': lambda s: NM.laplacian(net.N, E0, w0e / Je(s, E0)),
       'stretch': lambda s: NM.laplacian(net.N, net.E[s], net.w[s] * Je(s, net.E[s])),
       'full_a0': lambda s: NM.laplacian(net.N, net.E[s], net.w[s] / Je(s, net.E[s])),
       'comp0': lambda s: NM.laplacian(net.N, E0, w0e / Je(s, E0) ** 2)}
Mm = sp.diags(m); mix = {}; cmaps = {}
keep_maps = {'single': (40, 80, 120), 'reshock': (20, 40, 80, 120)}[case]
for name, op in ops.items():
    c = lab.copy(); Lp = op(0); mf = [0.0]
    for s in range(S - 1):
        Ln = op(s + 1); c = spsolve((Mm + net.dt * Gam * 0.5 * (Lp + Ln)).tocsc(), m * c); Lp = Ln
        mf.append(NM.mixed_fraction(m[:N], c[:N], lab[:N]))
        if (s + 1) in keep_maps and name in ('full', 'frozen'): cmaps['c_%s_%d' % (name, s + 1)] = c[:N].copy()
    mix[name] = mf; print(name, '%.0fs' % (time.time() - t0), 'final', mf[-1], flush=True)
out['mix'] = mix; out['Gamma'] = Gam
json.dump(out, open('res_%s_G%g%s.json' % (case, Gam, '_ser' if do_series else ''), 'w'))
np.savez_compressed('maps_c_%s_G%g.npz' % (case, Gam), **cmaps)
