"""Position-only thermodynamics for the Gamma(T) closure: isentropic between hot-step blocks, Rankine-Hugoniot across them,
using only the ledger (lnJ) and the hot-step labels; compared with DNS p and T at the tracers (inviscid SBI)."""
import numpy as np, sbcore as C
G = 1.4
D = C.load('inv005.pkl'); idx = C.subset(D, 2); pr = C.Prep(D, idx)
p0 = D['p'][0][idx].astype(float); r0 = pr.rho0
lnJ = pr.lnJ; S, N = lnJ.shape; hot = pr.hot                                 # hot: (N, S-1)
lnp = np.zeros((S, N)); lnp[0] = np.log(p0); s_jump = np.zeros(N)
inblk = np.zeros(N, bool); blk_dl = np.zeros(N)
for n in range(S - 1):
    dl = lnJ[n + 1] - lnJ[n]                                                 # = -dln rho over the step
    h = hot[:, n]
    blk_dl = np.where(h, blk_dl + dl, blk_dl)
    ending = inblk & ~h                                                      # block closed at step n: apply RH entropy jump
    eta = np.exp(-blk_dl)
    P = ((G + 1) * eta - (G - 1)) / ((G + 1) - (G - 1) * eta)
    jump = np.where(ending & (eta > 1), np.log(P) - G * np.log(eta), 0.0)
    s_jump += jump; blk_dl = np.where(ending, 0.0, blk_dl); inblk = h
    lnp[n + 1] = np.log(p0) + G * (-(lnJ[n + 1])) + s_jump                   # p = p0 (rho/rho0)^G exp(entropy jumps)
# blocks still open at the end: apply jump at the last frame
eta = np.exp(-blk_dl); P = ((G + 1) * eta - (G - 1)) / ((G + 1) - (G - 1) * eta)
lnp[-1] += np.where(inblk & (eta > 1), np.log(P) - G * np.log(eta), 0.0)
rho_l = r0[None] * np.exp(-lnJ)
d0 = pr.fr[0] - np.array([D['xb'], 0.0]); phi = np.hypot(*d0.T) - D['R']
for s in (15, 30, 50, 70):
    pd = D['p'][s][idx].astype(float); rd = D['rho'][s][idx].astype(float)
    ep = np.abs(np.exp(lnp[s]) / pd - 1); eT = np.abs(np.exp(lnp[s]) / rho_l[s] / (pd / rd) - 1)
    far = np.abs(phi) > 0.03
    print('t=%.1f  |p err| median %.3f (away from interface %.3f) 95%% %.3f   |T err| median %.3f (away %.3f)' % (
        s * 0.02, np.median(ep), np.median(ep[far]), np.percentile(ep, 95), np.median(eT), np.median(eT[far])))
