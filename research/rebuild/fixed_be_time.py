"""Cached SPD tridiagonal backward Euler for a fixed positive graph.

Identical equation to the original band solve, not a new time integrator.
Factorizations and states are checked locally. No scalar or probability
clipping/renormalization is used.
"""
import numpy as np
from scipy.linalg.lapack import dpttrf,dpttrs


class FixedBackwardEuler:
    def __init__(self,mass,g,dt):
        self.mass=np.asarray(mass,dtype=float).copy();self.g=np.asarray(g,dtype=float).copy();self.dt=float(dt)
        m=self.mass;g=self.g
        if m.ndim!=1 or len(m)<2 or g.shape!=(len(m)-1,) or not np.isfinite(m).all() or not np.isfinite(g).all():
            raise ValueError('finite compatible mass/conductance vectors required')
        if np.any(m<=0) or np.any(g<0) or not np.isfinite(dt) or dt<=0:
            raise ValueError('positive masses/dt, nonnegative g required')
        diag=m.copy();diag[:-1]+=dt*g;diag[1:]+=dt*g
        self.d,self.e,info=dpttrf(diag,-dt*g)
        if info!=0:raise RuntimeError(f'SPD factorization failed: {info}')

    def advance(self,c):
        c=np.asarray(c,dtype=float)
        if c.shape!=self.mass.shape or not np.isfinite(c).all():raise ValueError('invalid scalar state')
        rhs=np.asfortranarray((self.mass*c)[:,None])
        result,info=dpttrs(self.d,self.e,rhs,overwrite_b=1)
        if info!=0:raise RuntimeError(f'SPD solve failed: {info}')
        return result[:,0]


def integrate(c0,m,g,T,steps):
    if not isinstance(steps,(int,np.integer)) or steps<1 or not np.isfinite(T) or T<=0:
        raise ValueError('positive time and integer steps required')
    c=np.asarray(c0,dtype=float).copy();m=np.asarray(m);g=np.asarray(g);dt=T/steps
    solver=FixedBackwardEuler(m,g,dt);mean=float(m@c/m.sum());mass0=float(m@c)
    tiny=np.finfo(float).tiny;scale=max(abs(mass0),float(m@abs(c)),tiny)
    drift=residual=balance=0.;lo=float(c.min());hi=float(c.max())
    initial_min,initial_max=lo,hi
    for _ in range(steps):
        nxt=solver.advance(c)
        d=nxt-c;flux=dt*g*np.diff(nxt);r=m*d;r[:-1]-=flux;r[1:]+=flux
        residual=max(residual,float(np.linalg.norm(r)/max(np.linalg.norm(m*c),tiny)))
        drift=max(drift,abs(float(m@(nxt-c0)))/scale)
        E=lambda z:.5*float(m@((z-mean)**2))
        identity=E(nxt)-E(c)+dt*float(g@(np.diff(nxt)**2))+.5*float(m@(d*d))
        balance=max(balance,abs(identity)/max(float(m@(c*c)),tiny))
        lo=min(lo,float(nxt.min()));hi=max(hi,float(nxt.max()));c=nxt
    if max(drift,residual,balance)>1e-10 or lo<initial_min-1e-10 or hi>initial_max+1e-10:
        raise RuntimeError('fixed BE invariant/solve failure')
    return c,{'steps':steps,'dt':dt,'max_relative_mass_drift':drift,'max_relative_solve_residual':residual,
              'max_relative_energy_balance_residual':balance,'min_all_steps':lo,'max_all_steps':hi}


def spectral_backward_euler(eigenvalues,U,m,c0,T,steps):
    """Exact fixed-BE transfer factors and stable BE-minus-exponential vector.

    No extrapolation or correction of the BE state. The difference formula
    avoids cancellation and overflow in stiff, heavily damped modes.
    """
    if steps<1 or int(steps)!=steps or T<=0:raise ValueError('positive T/steps required')
    dt=T/steps
    if np.any(1+dt*eigenvalues<=0):raise ValueError('invalid modal BE denominator')
    log_be=-steps*np.log1p(dt*eigenvalues);log_exp=-T*eigenvalues
    top=np.maximum(log_be,log_exp);bottom=np.minimum(log_be,log_exp)
    difference=np.sign(log_be-log_exp)*np.exp(top)*(-np.expm1(bottom-top))
    root=np.sqrt(m);coeff=U.T@(root*c0)
    state=(U@(np.exp(log_be)*coeff))/root
    delta=(U@(difference*coeff))/root
    return state,delta
