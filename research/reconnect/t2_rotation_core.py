"""Strict exact-boundary covariance comparison against unchanged static material paths."""
from fractions import Fraction as F
import math
import numpy as np
from voronoi_time_action import metrics
from t2_d0_core import bitwise_same


def compare(data,old_run,old_path,new_run,new_path,packets,static_packet,actions,static_actions):
    m,c0=data['mass'],data['c0'];sigma=math.sqrt(metrics(m,c0,c0)['variance']);checks=[]
    for i,(packet,action,old_action) in enumerate(zip(packets,actions,static_actions)):
        coefficients=(packet['ids']==static_packet['ids'] and packet['edges']==static_packet['edges']
          and packet['image_shifts']==static_packet['image_shifts']
          and list(map(F,packet['conductances_exact']))==list(map(F,static_packet['conductances_exact'])))
        raw=(set(action)==set(old_action) and all(bitwise_same(v,old_action[k]) for k,v in action.items()))
        checks.append({'step':i,'same_ID_edges_images_exact_conductances':coefficients,'raw_sparse_actions_and_weights_bitwise_static':raw})
    path_checks={k:bitwise_same(v,old_path[k]) for k,v in new_path.items()}
    rows=[]
    norm=lambda v:float(np.sqrt(m@(v*v)))
    for k,t in enumerate((0.,.025,.05)):
        i=8*k;c=new_path['states'][i];other=new_path['independent_states'][i];old=old_path['states'][i]
        H=metrics(m,c,c0)['H'];oldH=metrics(m,old,c0)['H']
        a=0. if not i else new_run['steps'][i-1]['cumulative_machine_indicator_absolute']+new_run['steps'][i-1]['cumulative_tail_indicator_absolute']
        b=0. if not i else old_run['steps'][i-1]['cumulative_machine_indicator_absolute']+old_run['steps'][i-1]['cumulative_tail_indicator_absolute']
        rows.append({'time':t,'state_difference_relative':norm(c-old)/sigma,'H_difference':abs(H-oldH),
          'rotating_H':H,'static_H':oldH,'reference_H':metrics(m,data['reference'][k],c0)['H'],
          'rotating_reference_error_relative':norm(c-data['reference'][k])/sigma,'static_reference_error_relative':norm(old-data['reference'][k])/sigma,
          'rotating_raw_mass_change':float(m@(c-c0)),'static_raw_mass_change':float(m@(old-c0)),
          'original_rotating_machine_tail_indicator_absolute':a,'original_static_machine_tail_indicator_absolute':b,
          'rotating_primary_independent_difference_absolute':norm(c-other),'rotating_complete_sequences_controlled':norm(c-other)<=a})
    controls=all(v['rotating_complete_sequences_controlled'] for v in rows)
    positive=(any(F(g)>0 for g in static_packet['conductances_exact']) and not bitwise_same(new_path['states'][-1],c0) and rows[-1]['rotating_H']>0)
    passed=all(v['same_ID_edges_images_exact_conductances'] and v['raw_sparse_actions_and_weights_bitwise_static'] for v in checks) and all(path_checks.values()) and controls and positive
    return {'initial_mass_fluctuation_norm':sigma,'interval_covariance_checks':checks,'path_arrays_bitwise_static':path_checks,'observations':rows,
      'all_original_numerical_controls_passed':controls,'positive_diffusion_and_nonzero_mixing_preserved':positive,
      'maximum_saved_midpoint_label_difference':max(abs(a['midpoint_time_saved']-b['midpoint_time_saved']) for a,b in zip(new_run['steps'],old_run['steps'])),
      'T2_registered_rotation_case_passed':passed,'comparison_decision':'T2_exact_boundary_continuous_rotation_passed' if passed else 'T2_rotation_covariance_failed',
      'supported_mode':'given_exact_proper_oriented_rectangle_and_lossless_local_float_coordinates',
      'arbitrary_float_world_inputs_or_nonorthogonal_lattices_admitted':False,'constant_angular_velocity_tested':False,
      'general_T2_admitted':False,'general_smooth_flow_admitted':False,'kappa_computed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
