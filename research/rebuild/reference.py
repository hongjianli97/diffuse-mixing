"""Independent fine uniform-mass conservative scalar reference.

Midface point coefficient, centered flux; midpoint-time Crank--Nicolson.
Does not import the candidate FV solver or resistance quadrature.
"""
import numpy as np
from scipy.linalg import solve_banded
from becker import initial


def solve(flow,cells,steps,T=.1,D=.02):
    edges=np.linspace(0,flow.total,cells+1);dm=flow.total/cells;dt=T/steps
    c0=initial(edges);c=c0.copy();mass0=dm*c.sum();worst=0.;lo=c.min();hi=c.max()
    band=np.zeros((3,cells))
    for k in range(steps):
        g=flow.rho_mu(edges[1:-1],(k+.5)*dt)**2*D/dm
        half=.5*dt*g
        diagonal=np.full(cells,dm);diagonal[:-1]+=half;diagonal[1:]+=half
        rhs=dm*c
        jump=half*np.diff(c);rhs[:-1]+=jump;rhs[1:]-=jump
        band[1]=diagonal;band[0,1:]=-half;band[2,:-1]=-half
        c=solve_banded((1,1),band,rhs,check_finite=False)
        if not np.isfinite(c).all():raise RuntimeError('reference nonfinite')
        worst=max(worst,abs(dm*c.sum()-mass0)/abs(mass0))
        lo=min(lo,float(c.min()));hi=max(hi,float(c.max()))
    return edges,c0,c,{'max_relative_mass_drift':worst,'min_all_times':lo,'max_all_times':hi,
                     'initial_min':float(c0.min()),'initial_max':float(c0.max())}


def project(edges,c,target):
    """Integrate piecewise-constant cell averages conservatively in mass."""
    primitive=np.r_[0.,np.cumsum(np.diff(edges)*c)]
    return np.diff(np.interp(target,edges,primitive))/np.diff(target)
