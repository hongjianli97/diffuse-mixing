"""Predeclared static-contact three-level material observations and positive budgets."""
from fractions import Fraction as F
import math
import numpy as np
from contact_oracle_reporting import evaluate_saved
from material_union import group_sum,observe_path
from voronoi_time_action import metrics
from t1_uniform_core import h_indicator
from shear_union64_study import classify


def compare(data,meta,runs,paths,controls,cm,base=16):
    levels=sorted(data)
    if levels!=[base,2*base,4*base]:raise ValueError('three nested aligned material grids required')
    mp,cp=data[base]['mass'],data[base]['c0'];norm=lambda m,v:float(np.sqrt(m@(v*v)))
    sigma=math.sqrt(metrics(mp,cp,cp)['variance']);H=lambda c:metrics(mp,c,cp)['H']
    common={};per={};factors={};capacities=[];obs={}
    for N in levels:
        d=data[N];m,c0=d['mass'],d['c0'];ratio=N//base
        if m.shape!=(N*N,) or c0.shape!=m.shape or not np.isfinite(m).all() or not np.isfinite(c0).all() or np.any(m<=0):raise ValueError('complete positive original material data required')
        sums=group_sum(m,base,ratio);defect=float(np.max(abs(sums/mp-1)));factor=math.sqrt(float(np.max(sums/mp)));factors[N]=factor
        capacities.append({'N':N,'maximum_original_parent_capacity_relative_defect':defect,'mass_norm_observation_factor':factor,
          'original_capacity_guard_passed':defect<=1e-10,'mass_reweighted':False})
        per[str(N)]=evaluate_saved(d,meta[N],runs[N],paths[N],controls[N],cm[N])
        project=lambda v:observe_path(v,m,mp,side=base,ratio=ratio)
        common[f'n{N}_c0']=project(c0[None,:])[0];common[f'n{N}_reference']=project(d['reference'])
        for field in ('states','independent_states'):common[f'n{N}_{field}']=project(paths[N][field])
        common[f'n{N}_Cartesian']=project(controls[N]['states'])
        native_sigma=math.sqrt(metrics(m,c0,c0)['variance']);rows=[]
        for j,t in enumerate((0.,.025,.05)):
            c=paths[N]['states'][8*j];p=common[f'n{N}_states'][8*j];ref=d['reference'][j];rp=data[base]['reference'][j]
            rows.append({'time':t,'native_error_common_denominator':norm(m,c-ref)/sigma,'native_error_native_denominator':norm(m,c-ref)/native_sigma,
              'common_error':norm(mp,p-rp)/sigma,'native_H':metrics(m,c,c0)['H'],'native_reference_H':metrics(m,ref,c0)['H'],
              'common_H':H(p),'common_reference_H':H(rp),'raw_native_mass_change':float(m@(c-c0)),
              'raw_common_mass_change':float(mp@(p-cp))})
        obs[str(N)]=rows
    comparisons=[]
    for C,N in zip(levels[:-1],levels[1:]):
        for j,t in enumerate((0.,.025,.05)):
            native={};projected={};nativeH={};projectedH={};nums={}
            refbase=data[base]['reference'][j]
            for Q in (C,N):
                d=data[Q];m,c0=d['mass'],d['c0'];q=per[str(Q)];rm=meta[Q]
                # Copy the parent's original action components, including its
                # sequential NumPy scalar summation; no budget retuning.
                parts=q['all_endpoint_action_controls'][8*j]['components_absolute'];nums[Q]=float(q['all_endpoint_action_controls'][8*j]['fixed_indicator_absolute'])
                for key,value in parts.items():native[f'n{Q}_{key}']=float(value)
                tc=rm['tail_certificates_by_time'][j];trunc=0. if tc is None else float(F(tc['material_mass_norm_tail_bound_exact']))
                refparts={'reference_evaluation':float(rm['reference_evaluation_indicator_absolute']),
                  'reference_float_serialization':float(rm['float_serialization_mass_norms'][j]),'reference_truncation':trunc,
                  'initial_reference_bridge':float(rm['initial_serialization_bridge_absolute'])}
                native.update({f'n{Q}_{key}':v for key,v in refparts.items()})
                # Each named part remains nonnegative. Canonical native Python
                # float evaluation is fixed before scientific inputs.
                total=sum(float(v) for v in parts.values())+sum(refparts.values())
                projected[f'n{Q}_numeric_reference']=factors[Q]*total
                projected[f'n{Q}_reference_projection']=norm(mp,common[f'n{Q}_reference'][j]-refbase)
                projected[f'n{Q}_initial_projection_bridge']=norm(mp,common[f'n{Q}_c0']-cp)+norm(mp,common[f'n{Q}_reference'][0]-common[f'n{Q}_c0'])
                nativeH[f'n{Q}_numeric_reference']=float(h_indicator(m,c0,total))
                nativeH[f'n{Q}_initial_H_bridge']=abs(metrics(m,d['reference'][0],c0)['H']-metrics(m,c0,c0)['H'])
                projectedH[f'n{Q}_numeric_reference']=float(h_indicator(mp,cp,factors[Q]*total))
                projectedH[f'n{Q}_reference_projection']=abs(H(common[f'n{Q}_reference'][j])-H(refbase))
                projectedH[f'n{Q}_initial_projection_bridge']=abs(H(common[f'n{Q}_c0'])-H(cp))+abs(H(common[f'n{Q}_reference'][0])-H(common[f'n{Q}_c0']))
            projected['base_initial_reference_bridge']=norm(mp,data[base]['reference'][0]-cp)
            projectedH['base_initial_reference_bridge']=abs(H(data[base]['reference'][0])-H(cp))
            oc,of=obs[str(C)][j],obs[str(N)][j]
            gains={'native':oc['native_error_common_denominator']-of['native_error_common_denominator'],
              'common':oc['common_error']-of['common_error'],
              'native_H':abs(oc['native_H']-oc['native_reference_H'])-abs(of['native_H']-of['native_reference_H']),
              'common_H':abs(oc['common_H']-oc['common_reference_H'])-abs(of['common_H']-of['common_reference_H'])}
            budgets={'native':sum(native.values())/sigma,'common':sum(projected.values())/sigma,
              'native_H':sum(nativeH.values()),'common_H':sum(projectedH.values())}
            comparisons.append({'coarse_N':C,'fine_N':N,'time':t,'gains':gains,'fixed_indicators':budgets,
              'decisions':{key:classify(value,budgets[key]) for key,value in gains.items()},
              'native_indicator_components_absolute':native,'common_indicator_components_absolute':projected,
              'native_H_indicator_components':nativeH,'common_H_indicator_components':projectedH,
              'signed_cancellation_or_extrapolated_state_used':False})
    spatial=all(v=='fine_better_resolvably' for row in comparisons if row['time']>0 for v in row['decisions'].values())
    grid=all(v['registered_same_backend_contact_oracle_control_passed'] for v in per.values())
    capacity=all(v['original_capacity_guard_passed'] for v in capacities);passed=spatial and grid and capacity
    return {'common_initial_fluctuation_norm':sigma,'common_material_observation_N':base,'per_grid_controls':per,
      'per_grid_observations':obs,'original_parent_capacity_controls':capacities,'adjacent_spatial_comparisons':comparisons,
      'all_native_common_and_H_spatial_gains_resolved':spatial,'all_grid_action_reference_and_H_controls_passed':grid,
      'original_parent_capacities_preserved':capacity,'registered_three_space_level_contact_oracle_control_passed':passed,
      'comparison_decision':'T4_three_space_level_static_contact_oracle_control_passed' if passed else 'T4_contact_spatial_control_failed_or_unresolved',
      'general_spatial_order_claimed':False,'indicators_are_certified_total_error_bounds':False,'static_time_freezing_error_zero':True,
      'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False},common
