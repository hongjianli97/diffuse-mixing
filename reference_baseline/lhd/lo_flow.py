import numpy as np
GAM, A_, D_ = 2*np.pi, 1.0, 6.0
WR = GAM*(1-np.exp(-D_**2/A_**2))/(np.pi*D_**2)
cen = lambda t: [0.5*D_*np.array([np.cos(WR*t), np.sin(WR*t)]), -0.5*D_*np.array([np.cos(WR*t), np.sin(WR*t)])]
def vel(Q, t):
    u = np.zeros_like(Q)
    for c in cen(t):
        r = Q-c; r2 = (r*r).sum(1)+1e-300
        u += (GAM/(2*np.pi*r2)*(1-np.exp(-r2/A_**2)))[:, None]*np.c_[-r[:, 1], r[:, 0]]
    return u
def vort(Q, t): return sum(GAM/(np.pi*A_**2)*np.exp(-((Q-c)**2).sum(1)/A_**2) for c in cen(t))
