"""Replicate the paper's double-gyre setting: 200x100 REGULAR grid, T=15 (three forcing periods), image display."""
import pickle, time, numpy as np, cases, lhdx as X, paper_post as PP
from afrcH_paper import afrcH_paper
t0 = time.time()
c = cases.case_dg(nx=200, ny=100, T=15.0, S=76, jitter=0.0); fr = c['frames']; P0 = fr[0]
traj = np.ascontiguousarray(np.transpose(fr, (1, 0, 2)))
raw = afrcH_paper(traj, None, 15.0, P0, post=False)
PP.FORMAN_DESPECKLE_ITERS = 1
lt = np.maximum(PP.graph_tv_denoise(np.maximum(PP.graph_despeckle(raw.astype(float), P0, verbose=False), 0), P0, lam=0.02, iters=200, k=8, verbose=False), 0)
inner = (P0[:, 0] > .05) & (P0[:, 0] < 1.95) & (P0[:, 1] > .05) & (P0[:, 1] < .95)
print('replica: Spearman raw %.3f  light-post %.3f  (%.0fs)' % (X.spearman(raw, c['ftle_ex'], inner), X.spearman(lt, c['ftle_ex'], inner), time.time() - t0))
pickle.dump(dict(raw=raw, light=lt, ftle=c['ftle_ex'], nx=200, ny=100), open('res_afrcH_replica.pkl', 'wb'))
