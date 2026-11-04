"""Exact shared-coefficient perturbation and fixed T2 translation controls."""
from fractions import Fraction as F
import math
import numpy as np
from voronoi_time_action import metrics
from t1_uniform_core import h_indicator


def upper_float(q):
    q=F(q)
    if q<0:raise ValueError('nonnegative bound required')
    x=float(q)
    while F(x)<q:x=math.nextafter(x,math.inf)
    return x


def exact_fluctuation_upper(mass,c0):
    m=tuple(F(float(v)) for v in mass);c=tuple(F(float(v)) for v in c0)
    total=sum(m,F(0));mu=sum((a*b for a,b in zip(m,c)),F(0))/total
    variance=sum((a*(b-mu)**2 for a,b in zip(m,c)),F(0))
    x=math.sqrt(float(variance))
    while F(x)**2<variance:x=math.nextafter(x,math.inf)
    return variance,x


def edge_map(packet):
    result={}
    if len(packet['edges'])!=len(packet['conductances_exact']):raise ValueError('complete shared coefficients required')
    for pair,value in zip(packet['edges'],packet['conductances_exact']):
        i,j=map(int,pair);g=F(value)
        if i==j or g<0:raise ValueError('nonnegative off-diagonal shared edges required')
        key=tuple(sorted((i,j)));result[key]=result.get(key,F(0))+g
    return result


def coefficient_perturbation(mass,static,moving,tolerance=1e-10):
    if static['ids']!=moving['ids']:raise ValueError('saved material identity order changed')
    m=tuple(F(float(v)) for v in mass)
    if len(m)!=len(static['ids']) or min(m)<=0:raise ValueError('original positive capacities required')
    a,b=edge_map(static),edge_map(moving);rows=[F(0)]*len(m);changed=0;relative=F(0)
    for i,j in a.keys()|b.keys():
        delta=abs(b.get((i,j),F(0))-a.get((i,j),F(0)))
        rows[i]+=delta;rows[j]+=delta;changed+=int(delta!=0)
        if a.get((i,j),F(0))>0:relative=max(relative,delta/a[i,j])
    rate=2*max(r/v for r,v in zip(rows,m));same=set(a)==set(b)
    return {'delta_generator_mass_norm_upper_exact':str(rate),
      'maximum_relative_static_face_difference_exact':str(relative),'changed_shared_edges':changed,
      'introduced_shared_edges':len(b.keys()-a.keys()),'removed_shared_edges':len(a.keys()-b.keys()),
      'same_shared_edge_topology':same,'original_algebra_scale_passed':same and relative<=F(float(tolerance)),
      'original_algebra_tolerance':tolerance}


def compare(data,old_run,old_path,new_run,new_path,perturbations):
    m,c0=data['mass'],data['c0'];sigma=math.sqrt(metrics(m,c0,c0)['variance'])
    variance,upper=exact_fluctuation_upper(m,c0);dt=F(1,320)
    norm=lambda v:float(np.sqrt(m@(v*v)))
    ref=data['reference'];R=data['reference_evaluation_indicator_absolute'];rows=[]
    for k,t in enumerate((0.,.025,.05)):
        i=8*k;a=old_path['states'][i];b=new_path['states'][i]
        exact_rate=sum((F(v['delta_generator_mass_norm_upper_exact'])*dt for v in perturbations[:i]),F(0))
        exact_duhamel=exact_rate*F(upper);duhamel=upper_float(exact_duhamel)
        allowance=lambda r:0. if i==0 else r['steps'][i-1]['cumulative_machine_indicator_absolute']+r['steps'][i-1]['cumulative_tail_indicator_absolute']
        parts={'exact_coefficient_Duhamel':duhamel,'static_machine_and_tails':allowance(old_run),
          'moving_machine_and_tails':allowance(new_run),'static_observed_complete_sequence':norm(a-old_path['independent_states'][i]),
          'moving_observed_complete_sequence':norm(b-new_path['independent_states'][i]),'twice_reference_evaluation':2*R,
          'raw_initial_bridges':norm(new_path['states'][0]-old_path['states'][0])+norm(old_path['states'][0]-c0)+norm(ref[0]-c0)}
        budget=sum(parts.values());delta=norm(b-a);ha=metrics(m,a,c0)['H'];hb=metrics(m,b,c0)['H']
        hbudget=h_indicator(m,c0,budget);hdiff=abs(hb-ha)
        static_control=parts['static_observed_complete_sequence']<=allowance(old_run)
        moving_control=parts['moving_observed_complete_sequence']<=allowance(new_run)
        rows.append({'time':t,'checkpoint_index':i,'state_difference_absolute':delta,'state_difference_relative':delta/sigma,
          'state_indicator_components_absolute':parts,'state_indicator_absolute':budget,'state_indicator_relative':budget/sigma,
          'Duhamel_integrated_rate_exact':str(exact_rate),'Duhamel_absolute_upper_exact':str(exact_duhamel),
          'state_within_indicator':delta<=budget,'static_sequence_within_original_allowance':static_control,
          'moving_sequence_within_original_allowance':moving_control,
          'static_H':ha,'moving_H':hb,'H_difference':hdiff,'H_indicator':hbudget,'H_within_indicator':hdiff<=hbudget,
          'static_reference_error_relative':norm(a-ref[k])/sigma,'moving_reference_error_relative':norm(b-ref[k])/sigma,
          'moving_raw_mass_change':float(m@(b-c0)),'static_raw_mass_change':float(m@(a-c0))})
    geometry=all(v['original_algebra_scale_passed'] for v in perturbations)
    objective=all(v['state_within_indicator'] and v['H_within_indicator'] and v['static_sequence_within_original_allowance']
      and v['moving_sequence_within_original_allowance'] for v in rows[1:])
    passed=geometry and objective
    return {'initial_mass_fluctuation_norm':sigma,'initial_variance_exact':str(variance),'initial_fluctuation_float_upper':upper,
      'coefficient_perturbations':perturbations,'geometry_within_original_algebra_scale':geometry,'objectivity_within_fixed_indicators':objective,
      'observations':rows,'T2_registered_translation_case_passed':passed,
      'comparison_decision':'T2_rigid_translation_control_passed' if passed else 'T2_translation_failed_or_unresolved',
      'exact_Duhamel_coefficient_bound_is_total_numeric_or_PDE_error_certificate':False,
      'rigid_rotation_tested':False,'zero_diffusion_arbitrary_motion_tested':False,'kappa_computed':False,
      'general_T2_admitted':False,'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
