import numpy as np, json, glob, sys
sys.path.insert(0, '.'); from rm_take import cross
out = {}
for fn in sorted(glob.glob('res_*_G*.json')):
    r = json.load(open(fn)); t = np.array(r['t']); mx = r['mix']; fz = np.maximum(np.array(mx['frozen']), 1e-12)
    Fs, F1, F0 = np.array(mx['stretch']) / fz, np.array(mx['comp1']) / fz, np.array(mx['comp0']) / fz
    key = '%s_G%g' % (r['case'], r['Gamma'])
    out[key] = dict(a1=cross(t, Fs, F1), a0=cross(t, Fs, F0), final=dict(F_stretch=float(Fs[-1]), F_comp1=float(F1[-1]), F_comp0=float(F0[-1]),
                    F_full=float(mx['full'][-1] / fz[-1]), F_full_a0=float(mx['full_a0'][-1] / fz[-1])))
print(json.dumps(out, indent=1)); json.dump(out, open('res_mixtake.json', 'w'), indent=1)
