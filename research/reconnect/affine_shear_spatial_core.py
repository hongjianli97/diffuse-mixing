"""Registered affine-shear space comparison; raw fixed-parent mass observations."""
import math
import numpy as np
from affine_shear_core import COUNTS,evaluate
from material_union import group_sum,observe_path
from voronoi_time_action import metrics
from t1_uniform_core import h_indicator
from shear_union64_study import classify

NUMERIC=('machine_tail','observed_complete_sequence','integrated_mode_constant_residues',
 'twice_reference_evaluation','initial_two_mode_bridge')


def compare(data,labels,worlds,runs,paths,residues,base=16):
    levels=sorted(data)
    if levels!=[base,2*base,4*base]:raise ValueError('three nested original material levels required')
    norm=lambda m,v:float(np.sqrt(m@(v*v)))
    mp,cp=data[base]['mass'],data[base]['c0'];sigma=math.sqrt(metrics(mp,cp,cp)['variance'])
    H=lambda c:metrics(mp,c,cp)['H']
    per={};bundles={};common={};factors={};capacity=[];observations={}
    for N in levels:
        m,c0=data[N]['mass'],data[N]['c0'];ratio=N//base
        if m.shape!=(N*N,) or c0.shape!=m.shape or not np.isfinite(m).all() or not np.isfinite(c0).all() or np.any(m<=0):raise ValueError('complete original positive material inputs required')
        sums=group_sum(m,base,ratio);factor=math.sqrt(float(np.max(sums/mp)));factors[N]=factor
        defect=float(np.max(abs(sums/mp-1)))
        capacity.append({'N':N,'maximum_original_parent_capacity_relative_defect':defect,'observation_mass_norm_factor':factor,
          'original_algebra_scale_passed':defect<=1e-10,'masses_reweighted':False})
        rr={K:runs[N,K] for K in COUNTS};pp={K:paths[N,K] for K in COUNTS};ee={K:residues[N,K] for K in COUNTS}
        per[N],bundles[N]=evaluate(data[N],labels[N],worlds[N],rr,pp,ee,N=N)
        project=lambda v:observe_path(v,m,mp,side=base,ratio=ratio)
        common[f'n{N}_c0']=project(c0[None,:])[0]
        for field in ('reference','continuous_grid_symbol'):common[f'n{N}_{field}']=project(bundles[N][field])
        common[f'n{N}_frozen16']=project(bundles[N]['frozen_symbols'][16])
        for K in COUNTS:
            for field in ('states','independent_states'):common[f'n{N}_k{K}_{field}']=project(paths[N,K][field])
        rows=[];native_sigma=math.sqrt(metrics(m,c0,c0)['variance'])
        for k,t in enumerate((0.,.025,.05)):
            c=paths[N,16]['states'][8*k];p=common[f'n{N}_k16_states'][8*k];ref=bundles[N]['reference'][k];refbase=bundles[base]['reference'][k]
            rows.append({'time':t,'native_error_common_denominator':norm(m,c-ref)/sigma,'native_error_native_denominator':norm(m,c-ref)/native_sigma,
              'common_error':norm(mp,p-refbase)/sigma,'native_H':metrics(m,c,c0)['H'],'native_reference_H':metrics(m,ref,c0)['H'],
              'common_H':H(p),'common_reference_H':H(refbase),'raw_native_mass_change':float(m@(c-c0))})
        observations[str(N)]=rows
    comparisons=[]
    for C,N in zip(levels[:-1],levels[1:]):
        for k,t in enumerate((0.,.025,.05)):
            common_parts={};Hparts={};native_parts={};nativeHparts={};num_native={}
            refbase=bundles[base]['reference'][k]
            for Q in (C,N):
                m,c0=data[Q]['mass'],data[Q]['c0'];hn=lambda c:metrics(m,c,c0)['H'];b=bundles[Q]
                a=paths[Q,8]['states'][4*k];c=paths[Q,16]['states'][8*k]
                num=sum(per[Q]['reference_error_components'][str(K)][k]['components_absolute'][key] for K in (8,16) for key in NUMERIC)
                num_native[Q]=num;temporal=norm(m,b['frozen_symbols'][16][k]-b['continuous_grid_symbol'][k]);observed=norm(m,a-c)
                p=common[f'n{Q}_k16_states'][8*k];p8=common[f'n{Q}_k8_states'][4*k]
                pc=common[f'n{Q}_c0'];pref0=common[f'n{Q}_reference'][0];pref=common[f'n{Q}_reference'][k]
                common_parts[f'n{Q}_observed8_16']=norm(mp,p8-p)
                common_parts[f'n{Q}_analytic_time16']=norm(mp,common[f'n{Q}_frozen16'][k]-common[f'n{Q}_continuous_grid_symbol'][k])
                common_parts[f'n{Q}_numeric_reference']=factors[Q]*num
                common_parts[f'n{Q}_reference_projection']=norm(mp,pref-refbase)
                common_parts[f'n{Q}_initial_projection_bridge']=norm(mp,pc-cp)+norm(mp,pref0-pc)
                Hparts[f'n{Q}_observed8_16']=abs(H(p8)-H(p))
                Hparts[f'n{Q}_analytic_time16']=abs(H(common[f'n{Q}_frozen16'][k])-H(common[f'n{Q}_continuous_grid_symbol'][k]))
                Hparts[f'n{Q}_reference_projection']=abs(H(pref)-H(refbase))
                Hparts[f'n{Q}_initial_projection_bridge']=abs(H(pc)-H(cp))+abs(H(pref0)-H(pc))
                native_parts[f'n{Q}_observed8_16']=observed;native_parts[f'n{Q}_analytic_time16']=temporal;native_parts[f'n{Q}_numeric_reference']=num
                nativeHparts[f'n{Q}_observed8_16']=abs(hn(a)-hn(c))
                nativeHparts[f'n{Q}_analytic_time16']=abs(hn(b['frozen_symbols'][16][k])-hn(b['continuous_grid_symbol'][k]))
                nativeHparts[f'n{Q}_numeric_reference']=h_indicator(m,c0,num)
                nativeHparts[f'n{Q}_initial_bridge']=abs(hn(b['reference'][0])-hn(c0))
            common_parts['base_initial_reference_bridge']=norm(mp,bundles[base]['reference'][0]-cp)
            Hparts['base_initial_reference_bridge']=abs(H(bundles[base]['reference'][0])-H(cp))
            Hparts['numeric_reference_original_H_indicator']=h_indicator(mp,cp,sum(factors[Q]*num_native[Q] for Q in (C,N)))
            oc=observations[str(C)][k];of=observations[str(N)][k]
            gains={'native':oc['native_error_common_denominator']-of['native_error_common_denominator'],'common':oc['common_error']-of['common_error'],
              'native_H':abs(oc['native_H']-oc['native_reference_H'])-abs(of['native_H']-of['native_reference_H']),
              'common_H':abs(oc['common_H']-oc['common_reference_H'])-abs(of['common_H']-of['common_reference_H'])}
            budgets={'native':sum(native_parts.values())/sigma,'common':sum(common_parts.values())/sigma,
              'native_H':sum(nativeHparts.values()),'common_H':sum(Hparts.values())}
            comparisons.append({'coarse_N':C,'fine_N':N,'time':t,'gains':gains,'fixed_indicators':budgets,
              'decisions':{name:classify(gains[name],budgets[name]) for name in gains},
              'native_indicator_components_absolute':native_parts,'common_indicator_components_absolute':common_parts,
              'native_H_indicator_components':nativeHparts,'common_H_indicator_components':Hparts,'signed_cancellation_or_extrapolated_state_used':False})
    spatial=all(v=='fine_better_resolvably' for row in comparisons if row['time']>0 for v in row['decisions'].values())
    grid=all(v['all_frozen_symbol_and_sequence_controls_passed'] and v['all_reference_decompositions_passed'] and v['time_decreases_resolved'] for v in per.values())
    fine=per[levels[-1]]['two_tilted_mode_direction_response_resolved'];capacities=all(r['original_algebra_scale_passed'] for r in capacity)
    passed=spatial and grid and fine and capacities
    return {'common_initial_fluctuation_norm':sigma,'common_material_observation_N':base,'per_grid_controls':{str(N):v for N,v in per.items()},
      'original_parent_capacity_controls':capacity,'per_grid_observations':observations,'adjacent_spatial_comparisons':comparisons,
      'all_native_common_and_H_spatial_gains_resolved':spatial,'all_grid_symbol_reference_and_time_controls_passed':grid,
      'fine_grid_direction_response_resolved':fine,'original_parent_capacities_preserved':capacities,
      'registered_three_space_level_affine_shear_control_passed':passed,
      'comparison_decision':'T3_three_space_level_affine_shear_control_passed' if passed else 'T3_three_space_level_affine_shear_control_failed_or_unresolved',
      'general_spatial_order_claimed':False,'indicators_are_certified_total_error_bounds':False,'general_T3_admitted':False,
      'general_nonorthogonal_domains_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False},common,bundles
