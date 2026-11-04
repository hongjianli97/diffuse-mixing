"""Predetermined analytic clocks, raw-input Duhamel control and paired-model decisions."""
from fractions import Fraction as F
import math
import numpy as np
from affine_compression_inputs import PARAMETER,MODELS,COUNTS
from voronoi_time_action import metrics
from t1_uniform_core import h_indicator
from t2_translation_core import exact_fluctuation_upper,upper_float

TIMES=(0.,.025,.05)


def integrated_clocks(model,t):
    t=float(t);a=1-5*t;p=float(PARAMETER)
    if not 0<=t<=.05:raise ValueError('declared time required')
    if model=='constant_D':return p*t/a,p*t
    if model=='constant_Gamma':return .2*p*math.log(1/a),p*(t-2.5*t*t)
    raise ValueError('declared model required')


def midpoint_clocks(case,i,N=32):
    K=case['K'];dt=F(1,20*K);wire=nominal=rounding_rate=F(0)
    for step in range(i):
        frame=case['frames'][2*step+1];a=F(frame['a_exact']);factor=1/a+a
        wire+=dt*F(frame['Gamma_actual_exact'])*factor
        nominal+=dt*F(frame['Gamma_nominal_exact'])*factor
        rounding_rate+=dt*4*N**2*abs(F(frame['Gamma_difference_exact']))*factor
    return wire,nominal,rounding_rate


def reference_bundle(m,c0,N,world):
    sigma=math.sqrt(metrics(m,c0,c0)['variance']);_,upper=exact_fluctuation_upper(m,c0)
    qh=4*N*N*math.sin(math.pi/N)**2;qc=4*math.pi**2;R=float(64*np.finfo(float).eps*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0)))))
    refs={};symbols={};frozen={};nominal={};meta={}
    for model in MODELS:
        C=[sum(integrated_clocks(model,t)) for t in TIMES]
        refs[model]=np.array([.5+(c0-.5)*math.exp(-qc*x) for x in C]);symbols[model]=np.array([.5+(c0-.5)*math.exp(-qh*x) for x in C])
        meta[model]={'integrated_clocks':[list(integrated_clocks(model,t)) for t in TIMES],'continuous_label_clock':C,'cases':{}}
        for K in COUNTS:
            case=world['models'][model][str(K)];ww=[];nn=[];rr=[]
            for k in range(3):
                wire,ideal,bound=midpoint_clocks(case,k*K//2,N);ww.append(wire);nn.append(ideal);rr.append(bound)
            frozen[model,K]=np.array([.5+(c0-.5)*math.exp(-qh*float(v)) for v in ww])
            nominal[model,K]=np.array([.5+(c0-.5)*math.exp(-qh*float(v)) for v in nn])
            meta[model]['cases'][str(K)]={'wire_midpoint_clocks_exact':list(map(str,ww)),'nominal_midpoint_clocks_exact':list(map(str,nn)),
              'input_rounding_integrated_generator_bound_exact':list(map(str,rr)),
              'input_rounding_absolute_upper':[upper_float(q*F(upper)) for q in rr]}
    return {'reference':refs,'continuous_grid_symbol':symbols,'frozen_wire_symbol':frozen,'frozen_nominal_symbol':nominal,
      'meta':meta,'qh':qh,'qc':qc,'reference_evaluation_indicator_absolute':R,'initial_fluctuation_upper':upper,'initial_fluctuation_norm':sigma}


def evaluate(data,world,runs,paths,residues,N=32):
    m,c0=data['mass'],data['c0'];b=reference_bundle(m,c0,N,world);sigma=b['initial_fluctuation_norm'];R=b['reference_evaluation_indicator_absolute']
    norm=lambda v:float(np.sqrt(m@(v*v)));H=lambda c:metrics(m,c,c0)['H']
    observations={};symbol_rows=[];controls={};case_components={}
    for model in MODELS:
        observations[model]=[]
        for K in COUNTS:
            p=paths[model,K];run=runs[model,K];components=[]
            for k,t in enumerate(TIMES):
                i=k*K//2;c=p['states'][i];other=p['independent_states'][i]
                solver=0. if i==0 else run['steps'][i-1]['cumulative_machine_indicator_absolute']+run['steps'][i-1]['cumulative_tail_indicator_absolute']
                observed=norm(c-other);defect=sum(residues[model,K][:i])/(20*K)
                input_bound=b['meta'][model]['cases'][str(K)]['input_rounding_absolute_upper'][k]
                bridge=norm(b['reference'][model][0]-c0)
                numerical={'machine_tail':solver,'observed_complete_sequence':observed,'integrated_mode_and_constant_residues':defect,
                  'twice_analytic_evaluation':2*R,'initial_bridge':bridge}
                symbol_budget=sum(numerical.values());err=norm(c-b['frozen_wire_symbol'][model,K][k])
                Hdiff=abs(H(c)-H(b['frozen_wire_symbol'][model,K][k]));seq=observed<=solver
                symbol_rows.append({'model':model,'K':K,'time':t,'actual_vs_frozen_symbol_absolute':err,'fixed_indicator_absolute':symbol_budget,
                  'H_difference':Hdiff,'fixed_H_indicator':h_indicator(m,c0,symbol_budget),'symbol_within_indicator':err<=symbol_budget,
                  'H_within_indicator':Hdiff<=h_indicator(m,c0,symbol_budget),'complete_sequence_within_original_allowance':seq})
                time_abs=norm(b['frozen_nominal_symbol'][model,K][k]-b['continuous_grid_symbol'][model][k])
                spatial=norm(b['continuous_grid_symbol'][model][k]-b['reference'][model][k])
                parts={**numerical,'input_rounding_Duhamel':input_bound,'midpoint_time_error_symbol':time_abs,'spatial_symbol_error':spatial}
                budget=sum(parts.values());error=norm(c-b['reference'][model][k]);hb=h_indicator(m,c0,budget)
                components.append({'time':t,'components_absolute':parts,'total_absolute':budget,'reference_error_within_indicator':error<=budget,
                  'H_reference_error_within_indicator':abs(H(c)-H(b['reference'][model][k]))<=hb})
                if K==16:
                    observations[model].append({'time':t,'reference_error_relative':error/sigma,'H':H(c),'reference_H':H(b['reference'][model][k]),
                      'raw_mass_change':float(m@(c-c0)),'spatial_symbol_error_relative':spatial/sigma,'midpoint_time_error_relative':time_abs/sigma,
                      'input_rounding_bound_relative':input_bound/sigma,'fixed_total_indicator_relative':budget/sigma})
            case_components[model,K]=components
        rows=[]
        for k,t in enumerate(TIMES):
            errors=[];Herr=[];numeric=0.
            for A,B in ((4,8),(8,16)):
                a=paths[model,A]['states'][k*A//2];c=paths[model,B]['states'][k*B//2]
                errors.append(norm(a-c));Herr.append(abs(H(a)-H(c)))
            for K in COUNTS:
                parts=case_components[model,K][k]['components_absolute']
                numeric+=sum(parts[v] for v in ('machine_tail','observed_complete_sequence','integrated_mode_and_constant_residues','twice_analytic_evaluation','initial_bridge','input_rounding_Duhamel'))
            margin=errors[0]-errors[1];hm=Herr[0]-Herr[1];hb=h_indicator(m,c0,2*numeric)
            rows.append({'time':t,'state4_8_relative':errors[0]/sigma,'state8_16_relative':errors[1]/sigma,
              'H4_8':Herr[0],'H8_16':Herr[1],'state_decrease_margin_relative':margin/sigma,'fixed_state_decrease_indicator_relative':2*numeric/sigma,
              'H_decrease_margin':hm,'fixed_H_decrease_indicator':hb,'state_time_decrease_resolved':margin>2*numeric,
              'H_time_decrease_resolved':hm>hb})
        controls[model]=rows
    model_rows=[]
    for k,t in enumerate(TIMES):
        c=paths['constant_D',16]['states'][8*k];g=paths['constant_Gamma',16]['states'][8*k]
        parts={};budget=0.
        for model in MODELS:
            budget+=case_components[model,16][k]['total_absolute']
            # Also retain the observed8/16 difference; no assumed asymptotic cancellation.
            v=norm(paths[model,8]['states'][4*k]-paths[model,16]['states'][8*k]);budget+=v
            parts[model+'_reference_time_input_numerical_absolute']=case_components[model,16][k]['total_absolute']
            parts[model+'_observed8_16_absolute']=v
        delta=norm(c-g);hg=H(c)-H(g);hb=h_indicator(m,c0,budget)
        refdelta=b['reference']['constant_D'][k]-b['reference']['constant_Gamma'][k]
        model_rows.append({'time':t,'model_state_difference_relative':delta/sigma,'model_H_difference':hg,'fixed_state_indicator_relative':budget/sigma,
          'fixed_H_indicator':hb,'indicator_components_absolute':parts,'model_response_resolved':delta>budget and hg>hb,
          'model_state_response_resolved':delta>budget,'model_H_response_resolved':hg>hb,
          'model_difference_vs_analytic_difference_relative':norm((c-g)-refdelta)/sigma,
          'analytic_model_difference_consistent':norm((c-g)-refdelta)<=budget,
          'different_physical_PDEs_not_algorithm_accuracy_ranking':True})
    symbol_ok=all(v['symbol_within_indicator'] and v['H_within_indicator'] and v['complete_sequence_within_original_allowance'] for v in symbol_rows)
    ref_ok=all(v['reference_error_within_indicator'] and v['H_reference_error_within_indicator'] for vv in case_components.values() for v in vv)
    time_ok=all(v['state_time_decrease_resolved'] and v['H_time_decrease_resolved'] for vv in controls.values() for v in vv if v['time']>0)
    model_ok=all(v['model_response_resolved'] and v['analytic_model_difference_consistent'] for v in model_rows if v['time']>0)
    passed=symbol_ok and ref_ok and time_ok and model_ok
    result={'initial_mass_fluctuation_norm':sigma,'per_model_observations':observations,'symbol_controls':symbol_rows,
      'reference_error_components':{f'{M}-k{K}':v for (M,K),v in case_components.items()},'time_controls':controls,'model_comparisons':model_rows,
      'all_frozen_symbol_and_sequence_controls_passed':symbol_ok,'all_analytic_reference_error_decompositions_passed':ref_ok,
      'both_models_time_decreases_resolved':time_ok,'physical_model_response_resolved':model_ok,'T3_registered_affine_case_passed':passed,
      'comparison_decision':('T3_affine_two_constitutive_models_control_passed' if passed else
        ('T3_affine_model_response_inconclusive' if symbol_ok and ref_ok and time_ok else 'T3_affine_control_failed_or_unresolved')),
      'general_spatial_convergence_claimed':False,'general_T3_admitted':False,'general_smooth_flow_admitted':False,
      'indicators_are_certified_total_error_bounds':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    return result,b
