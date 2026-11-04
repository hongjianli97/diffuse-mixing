import sys, json, numpy as np, tcx, tcx_cases as TC
name = sys.argv[1]; c = TC.load(name); fr = c['fr']; m = c['inner']; ref, _ = TC.phi_reference(c['om'], fr, c['dt'])
r, _ = tcx.mu_threshold(fr); out = {}
def st(x):
    if np.std(ref[m]) < 1e-12: return dict(p99=float(np.quantile(np.abs(x[m]), .99)))
    return dict(pearson=float(np.corrcoef(x[m], ref[m])[0, 1]), relrms=float(np.sqrt(np.mean((x[m]-ref[m])**2))/np.std(ref[m])))
cache = {}
for lab, kw in (('V0_nomerge', dict(mu_star=np.inf)), ('V1_cancel', dict(mu_star=r)), ('V2_always', dict(mu_star=r, cancel=0.0))):
    ch = tcx.run_chain(fr, cplx_cache=cache, **kw); phi, _, _, _ = tcx.rotation_fields(ch); out[lab] = st(phi); out[lab]['hot'] = ch['hot_frac']
    print(name, lab, out[lab], flush=True)
json.dump(out, open(f'rules_{name}.json', 'w'), indent=1)
