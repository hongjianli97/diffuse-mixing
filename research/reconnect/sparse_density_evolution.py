"""Constant saved material-graph exponential actions; never construct full P."""
import numpy as np
from scipy.sparse import diags,issparse
from scipy.sparse.linalg import expm_multiply
from sparse_exchange_audit import sparse_max_abs


def evolve_saved(mass,L,c0,times):
    m=np.asarray(mass,dtype=float);c=np.asarray(c0,dtype=float);times=np.asarray(times,dtype=float)
    if m.ndim!=1 or c.shape!=m.shape or np.any(m<=0) or not np.isfinite(m).all() or not np.isfinite(c).all() or not issparse(L) or L.shape!=(len(m),len(m)) or not np.isfinite(L.data).all():raise ValueError('matching finite saved state, positive mass and sparse L required')
    if times.shape!=(3,) or times[0]!=0 or not np.isfinite(times).all() or times[1]<=0 or times[2]!=2*times[1]:raise ValueError('three equally spaced times starting at zero required for this fixed control')
    Q=(-diags(1/m)@L).tocsr();trace=float(Q.diagonal().sum());T=float(times[-1]);half=T/2
    states=expm_multiply(Q,c,start=0.,stop=T,num=3,endpoint=True,traceA=trace)
    full=expm_multiply(T*Q,c,traceA=T*trace)
    first=expm_multiply(half*Q,c,traceA=half*trace);second=expm_multiply(half*Q,first,traceA=half*trace)
    root=np.sqrt(m);S=(-diags(1/root)@L@diags(1/root)).tocsr()
    symmetric=expm_multiply(T*S,root*c,traceA=T*float(S.diagonal().sum()))/root
    controls={'full_vs_two_half_max_difference':float(np.max(abs(full-second))),
        'sampled_path_vs_full_endpoint_max_difference':float(np.max(abs(states[-1]-full))),
        'sampled_half_vs_independent_half_max_difference':float(np.max(abs(states[1]-first))),
        'raw_mass_symmetric_vs_Q_endpoint_max_difference':float(np.max(abs(symmetric-full))),
        'raw_mass_symmetric_matrix_symmetry_defect':sparse_max_abs(S-S.T),
        'no_state_clipping_mass_correction_or_matrix_repair':True,'full_propagation_matrix_constructed':False,
        'exponential_checks_are_not_BE_time_order_validation':True}
    return states,{'full_endpoint':full,'two_half_endpoint':second,'mass_symmetric_endpoint':symmetric},controls
