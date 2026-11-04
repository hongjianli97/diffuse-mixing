"""single-mode RM on one LTN: kinematic viscosity, dissipation (two activations), activation statistics.
usage: python run_ltn_rm.py <inv|visc>"""
import numpy as np, pickle, json, time, sys
sys.path.insert(0, '/home/claude/w/ltn'); import ltn_dyn as LD
case = sys.argv[1]; t0 = time.time()
ck = {'inv': '/home/claude/w/rm/rm_single_ck.pkl', 'visc': '/home/claude/w/rm/rmv_single_ck.pkl', 'visc2': '/home/claude/w/rm/rmv2_single_ck.pkl'}[case]
rec = pickle.load(open(ck, 'rb'))['rec']; fr0 = np.array(rec['frames']); rho = np.array(rec['rho'])
G = LD.LTN(fr0, 0.05, rho[0], k=12, mirror_band=0.5); N0, N = G.N0, G.N; h = 0.005
print(case, 'LTN |V|=%d |E_G|=%d (%.0fs)' % (N, len(G.keys), time.time() - t0), flush=True)
out = {}
for s in (10, 20, 30, 40, 60, 80, 100, 119):
    ch = G.channels(s); c = ch['c']; hot = G.bad[:, s]
    ntri = np.asarray(abs(c['B2']).sum(1)).ravel(); bnode = np.zeros(N, bool); bnode[c['E'][ntri == 1].ravel()] = True
    okT = ~hot[c['tri']].any(1) & ~bnode[c['tri']].all(1); orgT = (c['tri'] < N0).all(1); smooth = ~hot & ~bnode
    ps = G.pair_strain(s, rmax=2.5 * h); okN = smooth & np.isfinite(ps['phi']) & (np.arange(N) < N0)
    ea = G.edge_ages(s); Eb = G.E[ea['born']]; d0 = np.linalg.norm(G.fr[0][Eb[:, 1]] - G.fr[0][Eb[:, 0]], axis=1)
    r = dict(t=s * 0.05, mu_E=ch['mu_E'], enst=ch['enst'] / 3, phi_tpm=float((ch['phi_face'] * c['A'])[okT & orgT].sum()),
             phi_pair=float((ps['phi'] * c['Ast'])[okN].sum()), P_curl=ch['P_curl'] / 3, P_grad=ch['P_grad'] / 3,
             births=float(ea['born'].sum() / N / 0.05), births_far=float((d0 > 3 * h).mean()) if len(d0) else 0.0)
    out[s] = r; print(s, {k: '%.4g' % v for k, v in r.items()}, '%.0fs' % (time.time() - t0), flush=True)
json.dump(out, open('/home/claude/w/ltn/res_ltn_rm_%s.json' % case, 'w'), indent=1)
