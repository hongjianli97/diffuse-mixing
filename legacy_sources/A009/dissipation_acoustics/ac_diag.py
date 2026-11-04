import numpy as np
src = open('ac_dirac.py').read(); exec(src[:src.index('res = dict(frames={})')])
for s in (10, 20, 40):
    A_ = (fr[s + 1] - 2 * fr[s] + fr[s - 1]) / dt ** 2; ae = coch(fr[s], A_)
    rhoe = 0.5 * (z['rho'][s][E[:, 0]] + z['rho'][s][E[:, 1]])
    for tag, g in (('ledger lnrho', B1.T @ np.log(rho_led[s])), ('DNS lnrho', B1.T @ np.log(z['rho'][s])), ('DNS p/rho', (B1.T @ z['p'][s]) / rhoe)):
        print(s, tag, 'slope', -(w * ae * g).sum() / (w * g * g).sum(), 'corr', -(w * ae * g).sum() / np.sqrt((w * g * g).sum() * (w * ae * ae).sum()))
    gL, gD = B1.T @ np.log(rho_led[s]), B1.T @ np.log(z['rho'][s])
    print(s, 'ledger-vs-DNS lnrho gradient slope', (w * gL * gD).sum() / (w * gD * gD).sum(), ' DNS dp/drho', ((B1.T @ z['p'][s]) * gD * w).sum() / (w * gD * gD).sum())
