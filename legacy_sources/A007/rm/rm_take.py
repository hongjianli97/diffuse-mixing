import numpy as np, json, sys
def cross(t, a, b, tmin=0.3):
    a, b = np.asarray(a), np.asarray(b); d = a - b; k = np.where((t > tmin) & (d > 0))[0]
    if len(k) == 0: return None
    i = k[0]
    # require it to stay above for the rest (first sustained crossing)
    for i in k:
        if (d[i:] > 0).all() or i == k[-1]: break
    if i == 0: return float(t[0])
    return float(t[i - 1] + (t[i] - t[i - 1]) * (-d[i - 1]) / (d[i] - d[i - 1]))
def takeover(fn):
    r = json.load(open(fn)); t = np.array(r['t']); s = r['series']; mk = r['marker']; mx = r['mix']
    comp = np.array(s['logmean_comp']); out = {}
    out['net_rate_a1 (ln<lam2> vs ln<1/J>)'] = cross(t, s['lnS'], s['lnC1'])
    out['net_rate_a0 (ln<lam2> vs ln<1/J2>)'] = cross(t, s['lnS'], s['lnC0'])
    out['net_logmean_clock_a1 (2<lnlam> vs <-lnJ>)'] = cross(t, s['logmean_stretch2'], comp)
    out['net_logmean_geom (<lnlam> vs <-lnJ>)'] = cross(t, np.array(s['logmean_stretch2']) / 2, comp)
    out['net_cutlength_geom (lnL vs <-lnJ>)'] = cross(t, s['lnLnet'], comp)
    out['net_cutlength_clock_a1 (2lnL vs <-lnJ>)'] = cross(t, 2 * np.array(s['lnLnet']), comp)
    out['marker_geom (lnL vs <-lnJ>)'] = cross(t, mk['lnL'], comp)
    out['marker_clock_a1 (2lnL vs <-lnJ>)'] = cross(t, 2 * np.array(mk['lnL']), comp)
    out['marker_rate_a1 (ln<lam2>_marker vs ln<1/J>)'] = cross(t, mk['ln_lam2'], s['lnC1'])
    out['marker_rate_a0'] = cross(t, mk['ln_lam2'], s['lnC0'])
    fz = np.maximum(np.array(mx['frozen']), 1e-12)
    out['net_mixedmass_a1 (F_stretch vs F_comp1)'] = cross(t, np.array(mx['stretch']) / fz, np.array(mx['comp1']) / fz)
    out['net_mixedmass_a0 (F_stretch vs F_comp0)'] = cross(t, np.array(mx['stretch']) / fz, np.array(mx['comp0']) / fz)
    return out, r
if __name__ == '__main__':
    out, r = takeover(sys.argv[1]); print(json.dumps(out, indent=1))
    t = np.array(r['t']); s = r['series']; mk = r['marker']
    for tt in (1, 2, 3, 4, 6, 8, 10, 12):
        i = np.argmin(abs(t - tt))
        if abs(t[i] - tt) > 1e-6: continue
        print('t=%g  lnS %.3f  marker ln<lam2> %.3f | lnLnet %.3f marker lnL %.3f | comp %.3f lnC1 %.3f lnC0 %.3f | lnP %.3f' % (tt, s['lnS'][i], mk['ln_lam2'][i], s['lnLnet'][i], mk['lnL'][i], s['logmean_comp'][i], s['lnC1'][i], s['lnC0'][i], s['lnP'][i]))
