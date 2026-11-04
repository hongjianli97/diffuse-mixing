"""Vector-only time-order diagnostic; no matrix products or propagation."""
import numpy as np
from scipy.sparse import issparse,csr_matrix


def load_laplacian(saved):
    return csr_matrix((saved['L_data'],saved['L_indices'],saved['L_indptr']),shape=tuple(saved['L_shape']))


def commutator_action(mass,start_L,end_L,state):
    """Return [Q_end,Q_start]state, with Q=-M^-1 L and fixed masses.

    Intermediate vectors are retained for independently auditable units/order.
    This is an operator-action diagnostic, never an evolved scalar or error.
    """
    m=np.asarray(mass,dtype=float);c=np.asarray(state,dtype=float)
    if m.ndim!=1 or len(m)<2 or c.shape!=m.shape or not np.isfinite(m).all() or not np.isfinite(c).all() or np.any(m<=0):
        raise ValueError('matching finite state and fixed positive masses required')
    for L in [start_L,end_L]:
        if not issparse(L) or L.shape!=(len(m),len(m)) or not np.isfinite(L.data).all():
            raise ValueError('matching finite sparse endpoint Laplacians required')
    start=-(start_L@c)/m;end=-(end_L@c)/m
    end_start=-(end_L@start)/m;start_end=-(start_L@end)/m
    return {'start_rate':start,'end_rate':end,'end_after_start':end_start,
            'start_after_end':start_end,'commutator':end_start-start_end}


def pair_action(mass,pairs,conductances,state):
    """Independent shared-flux scatter for checking sparse nested actions."""
    i,j=np.asarray(pairs).T;g=np.asarray(conductances);c=np.asarray(state)
    numerator=np.zeros(len(mass));flux=g*(c[j]-c[i])
    np.add.at(numerator,i,flux);np.add.at(numerator,j,-flux)
    return numerator/np.asarray(mass)


def weighted_norm(mass,values):
    m=np.asarray(mass);v=np.asarray(values)
    return float(np.sqrt(m@(v*v)))
