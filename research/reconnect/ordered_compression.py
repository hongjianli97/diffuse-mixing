"""Chronological frozen-midpoint exponential vector actions; never full P."""
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import expm_multiply


def interval_grid(intervals,T):
    if not isinstance(intervals,(int,np.integer)) or intervals<2 or intervals%2 or not np.isfinite(T) or T<=0:
        raise ValueError('even number of at least two intervals and positive duration required')
    # Boundary and midpoint samples, in one strictly ordered trajectory record.
    samples=np.linspace(0.,T,2*intervals+1)
    return samples[::2],samples[1::2],samples


def frozen_exponential_step(mass,L,state,dt):
    m=np.asarray(mass,dtype=float);c=np.asarray(state,dtype=float)
    if m.ndim!=1 or len(m)<2 or c.shape!=m.shape or np.any(m<=0) or not np.isfinite(m).all() or not np.isfinite(c).all() or not issparse(L) or L.shape!=(len(m),len(m)) or not np.isfinite(L.data).all():
        raise ValueError('matching finite scalar, positive fixed mass and sparse L required')
    if not np.isfinite(dt) or dt<=0:raise ValueError('positive finite step duration required')
    Q=(-L.multiply((1/m)[:,None])).tocsr();trace=float(Q.diagonal().sum())
    path=expm_multiply(Q,c,start=0.,stop=float(dt),num=3,endpoint=True,traceA=trace)
    direct=expm_multiply(dt*Q,c,traceA=dt*trace)
    half=expm_multiply((dt/2)*Q,c,traceA=(dt/2)*trace)
    two_half=expm_multiply((dt/2)*Q,half,traceA=(dt/2)*trace)
    # Main chronological state is the sampled raw exponential endpoint. Other
    # representations are checks and are never blended into or substituted for it.
    checks={'direct_endpoint':direct,'two_half_endpoint':two_half,
        'direct_minus_path':direct-path[-1],'two_half_minus_path':two_half-path[-1]}
    return path[-1],path[1],checks


def dissipation(L,state):
    c=np.asarray(state);return float(c@(L@c))
