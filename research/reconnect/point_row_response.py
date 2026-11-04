"""Read-only diagnostics for one existing L row; never repair its row sum."""
import numpy as np


def row_response(L_values,columns,normal_centers,target,mass,ell,origin=0.):
    L=np.asarray(L_values,dtype=float);cols=np.asarray(columns);x=np.asarray(normal_centers,dtype=float)
    if L.ndim!=1 or not len(L) or cols.shape!=L.shape or x.shape!=L.shape or cols.dtype.kind not in 'iu' or np.any(np.diff(cols)<=0):
        raise ValueError('one nonempty sorted unique CSR row and matching centers required')
    if not np.isfinite(L).all() or not np.isfinite(x).all() or any(not np.isscalar(v) or not np.isfinite(v) for v in [mass,ell,origin]) or mass<=0 or ell<=0:
        raise ValueError('finite raw row, positive original mass/side and finite origin required')
    diagonal=np.flatnonzero(cols==target)
    if len(diagonal)!=1:raise ValueError('one saved diagonal entry required')
    position=int(diagonal[0]);center=float(x[position]);dx=x-center
    probe=(x-origin)**2/2+ell**2/24;target_probe=float(probe[position]);ones=np.ones(len(L))
    raw=-float(L@probe)/mass
    centered=-float(L@(probe-target_probe))/mass
    constant=-float(L@ones)/mass
    first=-float(L@dx)/mass
    half_second=-float(L@(dx*dx))/(2*mass)
    reconstruction=target_probe*constant+(center-origin)*first+half_second
    off=cols!=target
    arrays={'row_columns':cols,'saved_L_row':L,'normal_centers':x,'normal_displacement':dx,
        'probe_quadratic_mass_means':probe,'derived_Q_row_per_s':-L/mass,
        'raw_CSR_action_terms_per_s':-L*probe/mass,'centered_action_terms_per_s':-L*(probe-target_probe)/mass}
    metrics={'row_entries':len(L),'offdiagonal_entries':int(off.sum()),'target_diagonal_position':position,
        'original_mass_kg':mass,'target_normal_center_m':center,'probe_origin_m':origin,'target_probe_mass_average':target_probe,
        'raw_CSR_quadratic_response_per_s':raw,'centered_difference_response_per_s':centered,
        'raw_constant_response_per_s':constant,'raw_L_row_sum_kg_per_s':float(L.sum()),
        'first_normal_moment_m_per_s':first,'half_second_normal_moment_m2_per_s':half_second,
        'moment_reconstructed_response_per_s':reconstruction,'raw_minus_moment_reconstruction_per_s':raw-reconstruction,
        'raw_minus_centered_per_s':raw-centered,'target_value_times_constant_response_per_s':target_probe*constant,
        'raw_centered_identity_residual_per_s':(raw-centered)-target_probe*constant,
        'term_sum_minus_dot_response_per_s':float(arrays['raw_CSR_action_terms_per_s'].sum())-raw,
        'saved_L_diagonal_kg_per_s':float(L[position]),
        'minimum_L_offdiagonal_kg_per_s':float(np.min(L[off])) if off.any() else None,
        'maximum_L_offdiagonal_kg_per_s':float(np.max(L[off])) if off.any() else None,
        'main_value':'raw CSR action, fixed before inspection','row_sum_or_moments_repaired':False,'new_edge_weights':0}
    return arrays,metrics


def signed_budget(point,ideal,pde):
    if any(not np.isscalar(v) or not np.isfinite(v) for v in [point,ideal,pde]) or pde<=0:
        raise ValueError('finite point/ideal responses and positive PDE comparison required')
    a=point-ideal;b=ideal-pde;total=point-pde
    return {'point_minus_ideal_per_s':a,'ideal_minus_PDE_per_s':b,'point_minus_PDE_per_s':total,
        'point_minus_ideal_relative_to_PDE':a/pde,'ideal_minus_PDE_relative_to_PDE':b/pde,'point_minus_PDE_relative_to_PDE':total/pde,
        'signed_sum_per_s':a+b,'signed_identity_residual_per_s':(a+b)-total,
        'terms_have_opposite_sign':bool(a*b<0),'norms_are_additive_causal_fractions':False}
