"""Three original material space levels, positive comparison budgets, unchanged T3 controls."""
from fractions import Fraction as F
import math
import numpy as np
from scipy.sparse import csr_matrix
from affine_compression_inputs import MODELS,COUNTS
from t3_affine_core import evaluate
from t1_uniform_core import h_indicator
from voronoi_time_action import metrics
from material_union import group_sum,observe_path
from shear_union64_study import classify

NUMERIC=('machine_tail','observed_complete_sequence','integrated_mode_and_constant_residues',
 'twice_analytic_evaluation','initial_bridge','input_rounding_Duhamel')


def modal_certificate(data,frame,action,N,step):
    m,c0=data['mass'],data['c0'];root=np.sqrt(m);phi=c0-.5;a=F(frame['a_exact'])
    qh=4*N*N*math.sin(math.pi/N)**2;lam=qh*float(F(frame['Gamma_actual_exact'])*(a+1/a))
    A=csr_matrix((action['symmetric_data'],action['symmetric_indices'],action['symmetric_indptr']),shape=tuple(action['shape']))
    norm=lambda v:float(np.sqrt(m@(v*v)))
    eigen=norm(A@(root*phi)/root+lam*phi);constant=norm(.5*(A@root)/root);limit=1e-10*max(1.,lam*norm(phi))
    if eigen+constant>limit:raise ArithmeticError('original modal algebra guard failed; no inflated residual budget')
    return {'step':step,'declared_frozen_eigenvalue':lam,'mode_residue_absolute':eigen,'constant_residue_absolute':constant,
      'original_algebra_scale':limit,'original_algebra_scale_passed':True}


def compare(data,worlds,runs,paths,residues,base=16):
    levels=sorted(data)
    if levels!=[base,2*base,4*base]:raise ValueError('three nested original material levels required')
    norm=lambda m,v:float(np.sqrt(m@(v*v)))
    mp,cp=data[base]['mass'],data[base]['c0'];sigma=math.sqrt(metrics(mp,cp,cp)['variance'])
    H=lambda c:metrics(mp,c,cp)['H']
    per={};bundles={};common={};factors={};capacity=[];observations={}
    for N in levels:
        m,c0=data[N]['mass'],data[N]['c0'];ratio=N//base
        if m.shape!=(N*N,) or c0.shape!=m.shape or not np.isfinite(m).all() or not np.isfinite(c0).all() or np.any(m<=0):raise ValueError('original complete positive material data required')
        sums=group_sum(m,base,ratio);factor=math.sqrt(float(np.max(sums/mp)));factors[N]=factor
        defect=float(np.max(abs(sums/mp-1)));capacity.append({'N':N,'maximum_original_parent_capacity_relative_defect':defect,
          'observation_mass_norm_factor':factor,'original_algebra_scale_passed':defect<=1e-10,'masses_reweighted':False})
        rr={key[1:]:value for key,value in runs.items() if key[0]==N};pp={key[1:]:value for key,value in paths.items() if key[0]==N};ee={key[1:]:value for key,value in residues.items() if key[0]==N}
        per[N],bundles[N]=evaluate(data[N],worlds[N],rr,pp,ee,N=N)
        project=lambda v:observe_path(v,m,mp,side=base,ratio=ratio)
        common[f'n{N}_c0']=project(c0[None,:])[0]
        for model in MODELS:
            common[f'n{N}_{model}_reference']=project(bundles[N]['reference'][model])
            common[f'n{N}_{model}_continuous_grid_symbol']=project(bundles[N]['continuous_grid_symbol'][model])
            common[f'n{N}_{model}_frozen_nominal16']=project(bundles[N]['frozen_nominal_symbol'][model,16])
            for K in COUNTS:
                for field in ('states','independent_states'):common[f'n{N}_{model}_k{K}_{field}']=project(paths[N,model,K][field])
            rows=[]
            for k,t in enumerate((0.,.025,.05)):
                c=paths[N,model,16]['states'][8*k];p=common[f'n{N}_{model}_k16_states'][8*k];ref=bundles[N]['reference'][model][k];refbase=bundles[base]['reference'][model][k]
                native_sigma=math.sqrt(metrics(m,c0,c0)['variance']);e=norm(m,c-ref)
                rows.append({'time':t,'native_error_common_denominator':e/sigma,'native_error_native_denominator':e/native_sigma,
                  'common_error':norm(mp,p-refbase)/sigma,'native_H':metrics(m,c,c0)['H'],'native_reference_H':metrics(m,ref,c0)['H'],
                  'common_H':H(p),'common_reference_H':H(refbase),'raw_native_mass_change':float(m@(c-c0))})
            observations[f'n{N}_{model}']=rows
    comparisons=[]
    for model in MODELS:
        for C,N in zip(levels[:-1],levels[1:]):
            for k,t in enumerate((0.,.025,.05)):
                abs_native={};num_native={};time_native={};obs_native={};timeH_native={};obsH_native={};bridgeH_native={}
                common_parts={};Hparts={};native_parts={};nativeHparts={}
                refbase=bundles[base]['reference'][model][k]
                for Q in (C,N):
                    m,c0=data[Q]['mass'],data[Q]['c0'];hn=lambda c:metrics(m,c,c0)['H'];b=bundles[Q]
                    a=paths[Q,model,8]['states'][4*k];c=paths[Q,model,16]['states'][8*k]
                    num=sum(per[Q]['reference_error_components'][f'{model}-k{K}'][k]['components_absolute'][key] for K in (8,16) for key in NUMERIC)
                    temporal=norm(m,b['frozen_nominal_symbol'][model,16][k]-b['continuous_grid_symbol'][model][k]);observed=norm(m,a-c)
                    num_native[Q]=num;time_native[Q]=temporal;obs_native[Q]=observed;abs_native[Q]=num+temporal+observed
                    timeH_native[Q]=abs(hn(b['frozen_nominal_symbol'][model,16][k])-hn(b['continuous_grid_symbol'][model][k]));obsH_native[Q]=abs(hn(a)-hn(c))
                    bridgeH_native[Q]=abs(hn(b['reference'][model][0])-hn(c0))
                    projected=common[f'n{Q}_{model}_k16_states'][8*k];projected8=common[f'n{Q}_{model}_k8_states'][4*k]
                    pc=common[f'n{Q}_c0'];pref0=common[f'n{Q}_{model}_reference'][0];pref=common[f'n{Q}_{model}_reference'][k]
                    common_parts[f'n{Q}_observed8_16']=norm(mp,projected8-projected)
                    common_parts[f'n{Q}_analytic_time16']=norm(mp,common[f'n{Q}_{model}_frozen_nominal16'][k]-common[f'n{Q}_{model}_continuous_grid_symbol'][k])
                    common_parts[f'n{Q}_numeric_input_reference']=factors[Q]*num
                    common_parts[f'n{Q}_reference_projection']=norm(mp,pref-refbase)
                    common_parts[f'n{Q}_initial_projection_bridge']=norm(mp,pc-cp)+norm(mp,pref0-pc)
                    Hparts[f'n{Q}_observed8_16']=abs(H(projected8)-H(projected))
                    Hparts[f'n{Q}_analytic_time16']=abs(H(common[f'n{Q}_{model}_frozen_nominal16'][k])-H(common[f'n{Q}_{model}_continuous_grid_symbol'][k]))
                    Hparts[f'n{Q}_reference_projection']=abs(H(pref)-H(refbase))
                    Hparts[f'n{Q}_initial_projection_bridge']=abs(H(pc)-H(cp))+abs(H(pref0)-H(pc))
                    native_parts[f'n{Q}_observed8_16']=observed;native_parts[f'n{Q}_analytic_time16']=temporal;native_parts[f'n{Q}_numeric_input_reference']=num
                    nativeHparts[f'n{Q}_observed8_16']=obsH_native[Q];nativeHparts[f'n{Q}_analytic_time16']=timeH_native[Q]
                    nativeHparts[f'n{Q}_numeric_input_reference']=h_indicator(m,c0,num);nativeHparts[f'n{Q}_initial_bridge']=bridgeH_native[Q]
                common_parts['base_initial_reference_bridge']=norm(mp,bundles[base]['reference'][model][0]-cp)
                Hparts['base_initial_reference_bridge']=abs(H(bundles[base]['reference'][model][0])-H(cp))
                Hparts['numeric_input_reference_original_H_indicator']=h_indicator(mp,cp,sum(factors[Q]*num_native[Q] for Q in (C,N)))
                oc=observations[f'n{C}_{model}'][k];of=observations[f'n{N}_{model}'][k]
                gains={'native':oc['native_error_common_denominator']-of['native_error_common_denominator'],
                  'common':oc['common_error']-of['common_error'],
                  'native_H':abs(oc['native_H']-oc['native_reference_H'])-abs(of['native_H']-of['native_reference_H']),
                  'common_H':abs(oc['common_H']-oc['common_reference_H'])-abs(of['common_H']-of['common_reference_H'])}
                budgets={'native':sum(native_parts.values())/sigma,'common':sum(common_parts.values())/sigma,
                  'native_H':sum(nativeHparts.values()),'common_H':sum(Hparts.values())}
                row={'model':model,'coarse_N':C,'fine_N':N,'time':t,'gains':gains,'fixed_indicators':budgets,
                  'decisions':{name:classify(gains[name],budgets[name]) for name in gains},
                  'native_indicator_components_absolute':native_parts,'common_indicator_components_absolute':common_parts,
                  'native_H_indicator_components':nativeHparts,'common_H_indicator_components':Hparts,
                  'signed_cancellation_or_extrapolated_state_used':False}
                comparisons.append(row)
    spatial=all(value=='fine_better_resolvably' for row in comparisons if row['time']>0 for value in row['decisions'].values())
    grid=all(value['all_frozen_symbol_and_sequence_controls_passed'] and value['all_analytic_reference_error_decompositions_passed'] and value['both_models_time_decreases_resolved'] for value in per.values())
    fine=per[levels[-1]]['physical_model_response_resolved'];capacities=all(row['original_algebra_scale_passed'] for row in capacity)
    passed=spatial and grid and fine and capacities
    return {'common_initial_fluctuation_norm':sigma,'common_material_observation_N':base,'per_grid_controls':{str(N):v for N,v in per.items()},
      'original_parent_capacity_controls':capacity,'per_grid_observations':observations,'adjacent_spatial_comparisons':comparisons,
      'all_native_common_and_H_spatial_gains_resolved':spatial,'all_grid_symbol_reference_and_time_controls_passed':grid,
      'fine_grid_model_response_resolved':fine,'original_parent_capacities_preserved':capacities,
      'T3_registered_three_space_level_affine_control_passed':passed,
      'comparison_decision':'T3_three_space_level_affine_control_passed' if passed else 'T3_three_space_level_affine_control_failed_or_unresolved',
      'general_spatial_order_claimed':False,'indicators_are_certified_total_error_bounds':False,'general_T3_admitted':False,
      'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False},common,bundles
