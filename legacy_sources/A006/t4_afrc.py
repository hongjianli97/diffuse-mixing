import numpy as np, sys, time
sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import trajnet as TN
z = np.load('sb_lite.npz'); fr = z['frames']
out = {}
for s1 in (30, 70):
    t0 = time.time(); tn = TN.TrajectoryNetwork(fr[:s1 + 1], float(z['dt']), k=12); tn.events()
    out['barrier_s%d' % s1] = tn.robust_stretch(); print(s1, time.time() - t0, flush=True)
np.savez_compressed('out/maps_afrc.npz', **out)
