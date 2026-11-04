import numpy as np, json, pickle, time, sys
import cases
from common import *

def block_signature(fr, tau=0.05, gap=2):
    """fixed material triangles of frame 0; merge hot steps (|ln J_k|>tau) into blocks; return block signatures
    and the per-step pseudo-rotation sum vs the block rotation"""
    tri = X.complex_peeled(fr[0])["tri"]; S = fr.shape[0]
    P, Qs, LJ = [], [], []
    for s in range(S-1):
        F, A0 = tri_F(fr[s], fr[s+1], tri); p, q = pq(F); P.append(p); Qs.append(q); LJ.append(np.log(np.abs(np.linalg.det(F))))
    P, Qs, LJ = np.array(P), np.array(Qs), np.array(LJ)
    hot = np.abs(LJ) > tau
    sig = []; ps_step = []; ps_block = []
    for k in range(tri.shape[0]):
        hs = np.where(hot[:, k])[0]
        if hs.size == 0: continue
        # group with gaps
        groups = [[hs[0]]]
        for s in hs[1:]:
            if s-groups[-1][-1] <= gap+1: groups[-1].append(s)
            else: groups.append([s])
        for g in groups:
            a, b = g[0], g[-1]
            pB, qB = 1+0j, 0j
            for s in range(a, b+1): pB, qB = compose(P[s, k], Qs[s, k], pB, qB)
            J = abs(pB)**2-abs(qB)**2
            sig.append((-np.log(J), abs(qB/pB), np.angle(pB)))
            ps_step.append(np.abs(np.angle(P[a:b+1, k])).sum()); ps_block.append(abs(np.angle(pB)))
    sig = np.array(sig)
    return sig, np.array(ps_step), np.array(ps_block)

