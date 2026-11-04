"""Common material-cell comparison of signed rates; never renormalize mass."""
import numpy as np
from compression_midpoint import weighted_norm
from variable_parent_observation import observe_parent


def compare_parent_rates(parent_mass,parent_reference,coarse_rate,fine_mass,fine_rate):
    m=np.asarray(parent_mass,dtype=float);ref=np.asarray(parent_reference,dtype=float);old=np.asarray(coarse_rate,dtype=float)
    if ref.shape!=m.shape or old.shape!=m.shape or not np.isfinite(ref).all() or not np.isfinite(old).all():
        raise ValueError('matching finite saved parent rate and reference required')
    observed=observe_parent(fine_rate,fine_mass,m);norm=weighted_norm(m,ref)
    if norm<=0:raise ValueError('nonzero reference rate norm required for this comparison')
    old_error=weighted_norm(m,old-ref)/norm;new_error=weighted_norm(m,observed-ref)/norm
    return observed,{'coarse_error_common_cells':old_error,'fine_error_common_cells':new_error,
        'absolute_error_reduction':old_error-new_error,
        'relative_error_reduction_fraction':(old_error-new_error)/old_error if old_error else None,
        'normalization':'same saved coarse full-PDE mass-weighted rate norm'}
