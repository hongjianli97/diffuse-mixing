"""Double gyre: AFRC_H continuity vs tracer density, with post-processing strength matched to density."""
import pickle, time, numpy as np, cases, lhdx as X, paper_post as PP
from afrcH_paper import afrcH_paper
def light_post(c, P, lam):
    PP.FORMAN_DESPECKLE_ITERS = 1
    c = np.maximum(PP.graph_despeckle(c.astype(np.float64), P, verbose=False), 0.0)
    return np.maximum(PP.graph_tv_denoise(c.astype(np.float64), P, lam=lam, iters=200, k=8, verbose=False), 0.0)
out = {}
O = pickle.load(open('res_afrcH_paper.pkl', 'rb'))['DG']
out['N12800'] = dict(P=O['P'], raw=O['raw'], light=light_post(O['raw'], O['P'], 0.02), ftle=O['ftle'])
t0 = time.time(); c = cases.case_dg(nx=320, ny=160); fr = c['frames']; P0 = fr[0]
traj = np.ascontiguousarray(np.transpose(fr, (1, 0, 2))); del fr
raw = afrcH_paper(traj, None, c['t'][-1], P0, post=False)
out['N51200'] = dict(P=P0, raw=raw, light=light_post(raw, P0, 0.02), ftle=c['ftle_ex'])
inner = (P0[:, 0] > .05) & (P0[:, 0] < 1.95) & (P0[:, 1] > .05) & (P0[:, 1] < .95)
for k in ('raw', 'light'):
    print('N51200', k, 'Spearman vs exact FTLE %.3f' % X.spearman(out['N51200'][k], c['ftle_ex'], inner), flush=True)
print('time %.0fs' % (time.time() - t0))
pickle.dump(out, open('res_afrcH_density.pkl', 'wb'))
