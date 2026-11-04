"""S-16 / S-20: the two sets of numbers that the interface-stretch report quotes from diag.py's terminal output.
  ledger_noise : interface-band compression -ln J from the material ledger, from the tensor ledger -ln det F
                 (chained F, same anchoring), and the reference, under position noise (Sod, blast, shock-bubble)
  comp_bands   : shock-bubble, ledger vs DNS-interpolated ln(rho(t)/rho(0)) in bands of initial signed distance phi0
Writes res_stretch_extra.json and report/tables_extra.tex (%% TABLE ledger_noise, %% TABLE comp_bands)."""
import json, os, numpy as np, cases_mix as CM, stretch_est as SE

def detF_comp(c, frames):
    FF = SE.chained_F(c['frames'], c['h'], bad=c['bad'], block=5, keep=frames, idx=c['band'])
    return {s: float(np.mean(-np.log(np.abs(np.linalg.det(FF[s][0]))))) for s in frames}

R = dict(ledger_noise={}, comp_bands={}); rows = []
specs = (('sod', CM.setup_sod, [0.0, 0.1], [40]), ('blast', CM.setup_blast, [0.0, 0.1], [15, 60]),
         ('sb', lambda **k: CM.setup_dns('sb', **k), [0.0, 0.05, 0.1], [30, 70]))
for nm, fn, noises, frames in specs:
    for nz in noises:
        c = fn(noise=nz); dF = detF_comp(c, frames)
        for s in frames:
            ref = float(np.mean(c['comp_per'][s])) if 'comp_per' in c else float(c['comp_exact'](s))
            led = float(np.mean(-c['lnJ'][s][c['band']]))
            key = '%s|noise=%.2fh|s=%d' % (nm, nz, s)
            R['ledger_noise'][key] = dict(t=float(c['t'][s]), ledger=led, detF=dF[s], ref=ref)
            rows.append('%s & %.2f$h$ & %.2f & %.3f & %.3f & %.3f \\\\' % (nm, nz, c['t'][s], led, dF[s], ref))
            print(key, R['ledger_noise'][key], flush=True)
tex = ['%% TABLE ledger_noise', *rows]

c = CM.setup_dns('sb'); rho = c['D']['rho'].astype(float); rows = []
for lo, hi in ((-0.06, -0.03), (-0.03, -0.015), (-0.015, -0.005), (-0.003, 0.003), (0.005, 0.015), (0.015, 0.03), (0.03, 0.06)):
    idx = np.where((c['phi0'] > lo) & (c['phi0'] < hi))[0]; key = 'phi0=%.3f..%.3f' % (lo, hi)
    R['comp_bands'][key] = dict(n=int(len(idx)))
    vals = []
    for s in (30, 70):
        a = float(np.mean(-c['lnJ'][s][idx])); b = float(np.mean(np.log(rho[s][idx] / rho[0][idx])))
        R['comp_bands'][key]['t=%.1f' % c['t'][s]] = dict(ledger=a, dns=b); vals += [a, b]
    rows.append('$[%.3f,%.3f]$ & %.3f & %.3f & %.3f & %.3f \\\\' % (lo, hi, *vals))
    print(key, R['comp_bands'][key], flush=True)
tex += ['%% TABLE comp_bands', *rows]
json.dump(R, open('res_stretch_extra.json', 'w'), indent=1)
os.makedirs('report', exist_ok=True); open('report/tables_extra.tex', 'w').write('\n'.join(tex) + '\n')
print('wrote res_stretch_extra.json, report/tables_extra.tex')
