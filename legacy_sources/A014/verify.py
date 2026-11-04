"""Verification runs on the viscous shock-bubble, all through ltn_dyn.LTN.
variants: base | dt2 (frame interval 0.04) | subA (every other tracer, spacing sqrt2 h) | noiseX (Gaussian position noise, X in units of 1e-5)
metrics at t = 0.6, 1.0, 1.2: mu_E, Phi (three-point maps; pair strain) / DNS, pressure error, enstrophy."""
import numpy as np, pickle, json, sys, time
sys.path.insert(0, '/home/claude/w/ltn'); import ltn_dyn as LD
var = sys.argv[1]; t0 = time.time(); MU = 5e-4; h = 0.005
R = pickle.load(open('/home/claude/w/visc/sbv_ck.pkl', 'rb'))['R']; fr0 = np.array(R['frames']); Ed = {k: np.array(R[k]) for k in ('rho', 'p', 'Phi')}
sel = np.arange(fr0.shape[1]); dt = 0.02; stride = 1
if var == 'dt2': stride = 2; dt = 0.04
if var == 'subA':
    col = sel % 190; row = sel // 190; sel = sel[(col + row) % 2 == 0]
fr = fr0[::stride][:, sel].copy()
if var.startswith('noise'):
    sig = float(var[5:]) * 1e-5; fr += np.random.default_rng(11).normal(0, sig, fr.shape); fr[..., 1] = np.abs(fr[..., 1])
G = LD.LTN(fr, dt, Ed['rho'][0][sel], k=12, mirror_y=True); N0, N = G.N0, G.N
print(var, 'built %.0fs  N=%d' % (time.time() - t0, N), flush=True); out = {}
for tq in (0.6, 1.0, 1.2):
    s = int(round(tq / dt)); sd = s * stride
    ch = G.channels(s); c = ch['c']; hot = G.bad[:, s]
    ntri = np.asarray(abs(c['B2']).sum(1)).ravel(); bnode = np.zeros(N, bool); bnode[c['E'][ntri == 1].ravel()] = True
    smooth = ~hot & ~bnode; okT = ~hot[c['tri']].any(1) & ~bnode[c['tri']].all(1); org = np.arange(N) < N0
    pd = np.r_[Ed['p'][sd][sel], Ed['p'][sd][sel]]; p = ch['p'] + (pd[org] - ch['p'][org]).mean()
    Phi_d = float((np.r_[Ed['Phi'][sd][sel], Ed['Phi'][sd][sel]] * c['Ast'])[smooth].sum())
    ps = G.pair_strain(s, rmax=2.5 * h * (np.sqrt(2) if var == 'subA' else 1)); okN = smooth & np.isfinite(ps['phi'])
    r = dict(t=tq, mu_E=ch['mu_E'], enst=ch['enst'], tpm_over_dns=float(MU * (ch['phi_face'] * c['A'])[okT].sum()) / Phi_d,
             pair_over_dns=float(MU * (ps['phi'] * c['Ast'])[okN].sum()) / Phi_d,
             p_relrms_smooth=float(np.sqrt(((p - pd)[org & smooth] ** 2).mean()) / pd[org & smooth].std()))
    out[tq] = r; print(var, {k: '%.4g' % v for k, v in r.items()}, flush=True)
json.dump(out, open('res_verify_%s.json' % var, 'w'), indent=1)
