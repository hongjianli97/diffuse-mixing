"""All channels on ONE LTN (viscous shock-bubble, symmetric extension).  DNS fields are loaded only to score.
Regression against the earlier per-script results + two LTN-native additions:
  (i) dissipation from the dynamic activation (pair stretch rates of G edges)  vs  geometric activation (three-point maps)
  (ii) activation statistics of G: edge births per unit time, split by reference separation (affine flips vs new contacts)."""
import numpy as np, pickle, json, time, sys
sys.path.insert(0, '/home/claude/w/ltn'); import ltn_dyn as LD
t0 = time.time(); R = pickle.load(open('/home/claude/w/visc/sbv_ck.pkl', 'rb'))['R']
fr0 = np.array(R['frames']); Ed = {k: np.array(R[k]) for k in ('rho', 'p', 'Phi', 'om')}; MU = 5e-4; h = 0.005
G = LD.LTN(fr0, 0.02, Ed['rho'][0], k=12, mirror_y=True); N0, N, S = G.N0, G.N, G.S
print('LTN: |V| = %d (with mirror images), |E_G| = %d, |E_afrc| = %d, mean |K(t)| edges = %.0f  (%.0fs)' % (N, len(G.keys), len(G.E_afrc), np.mean([len(k) for k in G.kidx]), time.time() - t0), flush=True)
old = json.load(open('/home/claude/w/visc/res_visc_hodge_visc.json')); oldD = json.load(open('/home/claude/w/visc/res_dissip_tpm.json'))
out = dict(summary=dict(V=int(N), E_G=int(len(G.keys)), E_afrc=int(len(G.E_afrc)), K_edges_mean=float(np.mean([len(k) for k in G.kidx]))), frames={})
org = np.arange(N) < N0
for s in (15, 30, 50, 59, 69):
    ch = G.channels(s); c = ch['c']; hot = G.bad[:, s]
    ntri = np.asarray(abs(c['B2']).sum(1)).ravel(); bnode = np.zeros(N, bool); bnode[c['E'][ntri == 1].ravel()] = True
    smooth = ~hot & ~bnode; okT = ~hot[c['tri']].any(1) & ~bnode[c['tri']].all(1)
    pd = np.r_[Ed['p'][s], Ed['p'][s]]; p = ch['p'] + (pd[org] - ch['p'][org]).mean()
    r = dict(t=s * 0.02, mu_E=ch['mu_E'], enst=ch['enst'], p_relrms_smooth=float(np.sqrt(((p - pd)[org & smooth] ** 2).mean()) / pd[org & smooth].std()),
             Phi_tpm_smooth=float(MU * (ch['phi_face'] * c['A'])[okT].sum()), Phi_dns_smooth=float((np.r_[Ed['Phi'][s], Ed['Phi'][s]] * c['Ast'])[smooth].sum()))
    ps = G.pair_strain(s, rmax=2.5 * h)
    okN = smooth & np.isfinite(ps['phi']); r['Phi_pair_smooth'] = float(MU * (ps['phi'] * c['Ast'])[okN].sum()); r['n_pair_edges'] = ps['n_edges']
    # strain from the two activations, node by node (three-point faces averaged to nodes)
    Sf = 0.5 * (ch['Lg'] + ch['Lg'].transpose(0, 2, 1)); Sn = np.zeros((N, 3)); cn = np.zeros(N)
    for k in range(3):
        np.add.at(Sn, c['tri'][:, k], np.c_[Sf[:, 0, 0], Sf[:, 0, 1], Sf[:, 1, 1]] * c['A'][:, None]); np.add.at(cn, c['tri'][:, k], c['A'])
    Sn /= np.maximum(cn, 1e-30)[:, None]; m2 = okN & org
    r['strain_corr_pair_vs_tpm'] = float(np.corrcoef(ps['S'][m2].ravel(), Sn[m2].ravel())[0, 1])
    # activation statistics on G
    ea = G.edge_ages(s); Eb = G.E[ea['born']]; d0 = np.linalg.norm(G.fr[0][Eb[:, 1]] - G.fr[0][Eb[:, 0]], axis=1)
    r['births_per_node_per_time'] = float(ea['born'].sum() / N / 0.02); r['births_far_frac'] = float((d0 > 3 * h).mean()) if len(d0) else 0.0
    r['active_age_median'] = float(np.median(ea['age'][G.act[:, s]]))
    so = old.get(str(s)); sd = oldD.get(str(s))
    if so: r['old_mu_E'] = so['mu_E']; r['old_enst'] = so['enstrophy']; r['old_p_relrms_smooth'] = so['p_relrms_smooth']
    if sd: r['old_Phi_tpm_smooth'] = sd['Phi_net_smooth']; r['old_Phi_dns_smooth'] = sd['Phi_dns_smooth']
    out['frames'][s] = r; print(s, {k: ('%.4g' % v if isinstance(v, float) else v) for k, v in r.items()}, '%.0fs' % (time.time() - t0), flush=True)
json.dump(out, open('/home/claude/w/ltn/res_ltn_sb.json', 'w'), indent=1)
act_frac = G.act.mean(1); out['summary']['edges_never_active_frac'] = float((act_frac == 0).mean())
print(out['summary'])
