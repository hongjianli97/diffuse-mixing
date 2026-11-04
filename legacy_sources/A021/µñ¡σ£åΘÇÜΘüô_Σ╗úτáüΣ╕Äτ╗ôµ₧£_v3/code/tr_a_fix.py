import json, numpy as np, cases, lhdx as X
R = json.load(open('res_tr_a.json'))
def oneway(fr, m, side):
    a, b = side(fr[0]), side(fr[-1]); return float((m * (~a & b)).sum())
out = {}
dg = {n: cases.case_dg(nx=n, ny=n // 2, S=26) for n in (80, 160, 320, 640)}
mid = lambda P: P[:, 0] > 1.0; ref = oneway(dg[640]['frames'], X.star_area(dg[640]['frames'][0]) / 3.0, mid)
out['DG_oneway_left_to_right'] = {'reference_640x320': ref, **{('N=%dx%d' % (n, n // 2)): oneway(dg[n]['frames'], X.star_area(dg[n]['frames'][0]) / 3.0, mid) / ref - 1 for n in (80, 160, 320)}}
del dg
lo = {N: cases.case_lo(a=1.0, Rd=6.5, N=N, S=21) for N in (2000, 8000, 32000, 128000)}
side = lambda P: P[:, 0] > 0.0; ref = oneway(lo[128000]['frames'], X.star_area(lo[128000]['frames'][0]) / 3.0, side)
out['LO_oneway_left_to_right'] = {'reference_N128000': ref, **{('N=%d' % N): oneway(lo[N]['frames'], X.star_area(lo[N]['frames'][0]) / 3.0, side) / ref - 1 for N in (2000, 8000, 32000)}}
R['B']['DG_midline_flux'] = out['DG_oneway_left_to_right']; R['B']['LO_centreline_flux'] = out['LO_oneway_left_to_right']
json.dump(R, open('res_tr_a.json', 'w'), indent=1); print(json.dumps(out))
