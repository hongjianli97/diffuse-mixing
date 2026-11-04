"""Exact D=0 material-transport acceptance; no tolerance-based identity repair."""
from fractions import Fraction as F
import math
import numpy as np
from scipy.sparse import csr_matrix,eye
from voronoi_time_action import metrics


def bitwise_same(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes(order='C')==b.tobytes(order='C')


def action_is_zero_identity(saved,n):
    shape=tuple(saved['shape'])
    if shape!=(n,n):return False
    A=csr_matrix((saved['symmetric_data'],saved['symmetric_indices'],saved['symmetric_indptr']),shape=shape)
    P=csr_matrix((saved['uniformization_data'],saved['uniformization_indices'],saved['uniformization_indptr']),shape=shape)
    return bool(np.count_nonzero(A.data)==0 and np.count_nonzero((P-eye(n,format='csr')).data)==0
      and np.count_nonzero(saved['g_float'])==0 and bitwise_same(saved['weights'],np.array([1.])))


def evaluate(data,run,path,packets,actions):
    m,c0,ids=data['mass'],data['c0'],data['ids'];K=len(run['steps']);n=len(m)
    if len(packets)!=K or len(actions)!=K:raise ValueError('all original geometry/action intervals required')
    initial=metrics(m,c0,c0);sigma=math.sqrt(initial['variance']);expected=np.broadcast_to(c0,(K+1,n))
    state_checks={k:bitwise_same(path[k],expected) for k in ('states','independent_states')}
    state_checks['exponential_half_states']=bitwise_same(path['exponential_half_states'],np.broadcast_to(c0,(K,n)))
    operators=[]
    for k,(packet,saved,row) in enumerate(zip(packets,actions,run['steps'])):
        coefficient=F(packet['D_exact'])==0 and packet['ids']==list(map(int,ids)) and all(F(g)==0 for g in packet['conductances_exact'])
        zero=action_is_zero_identity(saved,n)
        branch=(row['operator']['Lambda_exact_integer']==0 and row['operator']['symmetric_generator_trace_raw']==0
          and row['poisson']['M']==0 and F(row['poisson']['z_exact'])==0 and F(row['poisson']['tail_upper_exact'])==0
          and row['poisson']['raw_weight_sum']==1. and row['poisson']['weights_or_state_normalized'] is False)
        raw=(row['state_metrics']==initial and row['independent_metrics']==initial and row['chi_start_half_end']==[0.,0.,0.]
          and row['raw_mass_change_from_c0']==0 and row['independent_raw_mass_change_from_c0']==0
          and row['primary_control_mass_norm_difference']==0 and row['half_variance_loss']==0
          and row['Simpson_chi_integral']==0 and row['raw_energy_balance_difference']==0)
        operators.append({'step':k,'exact_zero_conductances':coefficient,'raw_A_zero_P_identity_and_weight_one':zero,
          'original_zero_action_branch':branch,'reported_raw_metrics_and_energy_unchanged':raw})
    observations=[]
    for k in range(K+1):
        c=path['states'][k];v=path['independent_states'][k];met=metrics(m,c,c0);imet=metrics(m,v,c0)
        delta=float(np.sqrt(m@((c-c0)**2)));other=float(np.sqrt(m@((v-c0)**2)))
        observations.append({'step':k,'time':float(F(k,20*K)),'state_reference_difference_relative':delta/sigma,
          'independent_reference_difference_relative':other/sigma,'mass_inventory':float(m@c),'raw_mass_change':float(m@(c-c0)),
          'mean':met['mean'],'variance':met['variance'],'I':met['I'],'H':met['H'],
          'actual_and_independent_metrics_unchanged':met==initial and imet==initial,
          'actual_bitwise_initial':bitwise_same(c,c0),'independent_bitwise_initial':bitwise_same(v,c0)})
    operator_ok=all(all(v[k] for k in ('exact_zero_conductances','raw_A_zero_P_identity_and_weight_one','original_zero_action_branch','reported_raw_metrics_and_energy_unchanged')) for v in operators)
    states_ok=all(state_checks.values()) and all(v['actual_and_independent_metrics_unchanged'] for v in observations)
    passed=operator_ok and states_ok
    return {'initial_mass_fluctuation_norm':sigma,'initial_metrics':initial,'state_arrays_bitwise_initial':state_checks,
      'interval_zero_checks':operators,'observations':observations,'all_zero_operator_checks_passed':operator_ok,'all_material_identity_checks_passed':states_ok,
      'T2_registered_D0_case_passed':passed,'comparison_decision':'T2_D0_nonrigid_material_transport_passed' if passed else 'T2_D0_identity_failed',
      'positiveD_reference_used_as_D0_target':False,'acceptance_uses_state_or_coefficient_tolerance':False,
      'rigid_rotation_tested':False,'kappa_computed':False,'general_T2_admitted':False,'general_smooth_flow_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
