"""One original saved point graph, one T=1/20 endpoint, existing PDE reference."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import resource
import shutil
import signal
import time
import numpy as np
import scipy

from audit_route_decision import ROOT,read,save,digest
from saved_exchange_time import load_stencil,build_modal_blocks,modal_actions,stencil_action,uniformization,poisson_tail_bound

SOURCE=ROOT/'runs/reconnect-post-ideal-route'
SIGMA=0.19793792100879673


def record(a):
    return {'shape':list(a.shape),'dtype':a.dtype.str,'C_order_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def norm(m,v):return math.sqrt(math.fsum(float(a)*float(b)**2 for a,b in zip(m,np.asarray(v).ravel())))


def state_metrics(m,state,initial,reference,initial_E,initial_inventory):
    values=np.asarray(state).ravel();mass=list(map(lambda v:F(float(v)),m));u=list(map(lambda v:F(float(v)),values))
    total=sum(mass,F(0));inventory=sum((a*b for a,b in zip(mass,u)),F(0));mean=inventory/total
    energy=sum((a*(b-mean)**2 for a,b in zip(mass,u)),F(0))/2
    I=energy/initial_E;H=1-I
    return {'error_over_original_initial_fluctuation':norm(m,values-reference)/SIGMA,
        'minimum':float(values.min()),'maximum':float(values.max()),
        'inside_original_initial_extrema':bool(values.min()>=initial.min() and values.max()<=initial.max()),
        'inventory_exact_of_binary_output':str(inventory),'inventory_drift_exact':str(inventory-initial_inventory),
        'inventory_drift_display':float(inventory-initial_inventory),'raw_mass_mean_exact':str(mean),
        'resolved_material_variance_E_exact':str(energy),'resolved_material_variance_E':float(energy),
        'I':float(I),'H':float(H),'variance_uses_raw_endpoint_mean_not_repaired_inventory':True}


def solve(data,cache,spec,reference_results,progress=None,checkpoint=None):
    s=load_stencil(cache,data['mass'],data['ids'],data['positions']);N=s.N
    assert N==128
    expected=list(map(F,spec['uniformization']['outgoing_rates_exact']))
    assert list(s.outgoing)==expected
    modes=build_modal_blocks(s,progress)
    block_packet={'blocks':modes.blocks,'eigenvalues':modes.eigenvalues,'eigenvectors':modes.eigenvectors,
        'eigen_residuals':modes.eigen_residuals,'orthogonality_residuals':modes.orthogonality_residuals,
        'cosine_values':np.array(modes.cosine_values)}
    if checkpoint is not None:checkpoint('modal-blocks.npz',block_packet)
    c0=data['c0'];m=data['mass'];initial=modal_actions(modes,c0,0.)
    eps=float(np.finfo(float).eps);c_norm=norm(m,c0);rate_scale=2*float(max(s.outgoing))
    conversion=max(map(float,modes.conversion_bounds));phase=max(map(float,modes.phase_bounds))
    operator_error=conversion+phase
    transform_allowance=64*N*eps*max(1.,c_norm)
    action_allowance=(operator_error+64*N*eps*max(1.,rate_scale))*max(1.,c_norm)
    initial_action_difference=norm(m,initial['initial_action']-data['saved_initial_rate'])
    roundtrip=norm(m,initial['initial_roundtrip']-c0)
    imag_initial=norm(m,initial['imaginary_initial']);imag_action=norm(m,initial['imaginary_initial_action'])
    if initial_action_difference>action_allowance or roundtrip>transform_allowance or imag_initial>transform_allowance or imag_action>action_allowance:
        raise ArithmeticError('predeclared initial-action/transform control failed; no propagation')
    if progress is not None:progress('initial_action_checked',1,1)
    T_exact=F(spec['T_exact']);T=float(T_exact)
    primary=modal_actions(modes,c0,T)
    Lam=spec['uniformization']['uniformization_lambda_integer'];M=spec['uniformization']['Poisson_max_degree_predeclared']
    tail=poisson_tail_bound(F(Lam)*T_exact,M)
    assert str(tail)==spec['uniformization']['Poisson_tail_bound_exact']
    alternate=uniformization(s,c0,T,Lam,M,progress)
    eigen_res=float(modes.eigen_residuals.max());orth=float(modes.orthogonality_residuals.max())
    raw_largest=float(modes.eigenvalues.max());raw_smallest=float(modes.eigenvalues.min())
    growth=math.exp(T*(max(0.,raw_largest)+operator_error)) # Error estimate only; no eigenvalue is changed.
    q_bound=math.sqrt(1+orth)
    spectral_indicator=(orth+T*growth*(eigen_res*q_bound+operator_error*q_bound*q_bound))*max(1.,c_norm)
    fft_indicator=64*N*eps*growth*(q_bound*q_bound+1)*max(1.,c_norm)
    alternate_allowance=8*(2*len(s.offsets)+4)*(M+1)*eps*max(1.,c_norm)
    tail_mass=float(F(spec['uniformization']['mass_norm_tail_bound_exact']))
    T_gap=abs(F(T)-T_exact);time_label=float(T_gap)*norm(m,data['saved_initial_rate'])
    z_gap=abs(F(Lam*T)-F(Lam)*T_exact);z_indicator=float(z_gap)*2*max(1.,c_norm)
    imag_final=norm(m,primary['imaginary_T'])
    vector_difference=norm(m,primary['state']-alternate['state'])
    independent_allowance=spectral_indicator+fft_indicator+alternate_allowance+tail_mass+time_label+z_indicator
    if vector_difference>independent_allowance or imag_final>fft_indicator:
        raise ArithmeticError('independent time action/imaginary control failed; no repair or extra terms')
    reference_indicator=float(reference_results['reference_initial_bridge_serialization_and_precision_indicator_mass_norm'])
    solver_indicator=independent_allowance+vector_difference+roundtrip+imag_initial+imag_final
    tiny_modes=[k for k in range(N) if k not in [0,1,N-1]]
    coefficient_norm=float(np.linalg.norm(primary['full_Fourier_coefficients'][tiny_modes]))
    controls={'blocks':len(modes.blocks),'normal_block_dimension':N,'transverse_mode_slots_retained':N,
        'total_initial_discrete_coefficients_retained':N*N,'all_mode_slots_retained':True,
        'nonprincipal_transverse_coefficient_mass_norm':coefficient_norm,
        'raw_minimum_generator_eigenvalue':raw_smallest,'raw_maximum_generator_eigenvalue':raw_largest,
        'raw_zero_mode_largest_eigenvalue':float(modes.eigenvalues[0].max()),
        'maximum_block_conversion_row_bound_exact':str(max(modes.conversion_bounds)),
        'maximum_phase_vs_110_digit_row_indicator_exact':str(max(modes.phase_bounds)),
        'maximum_eigen_residual_frobenius':eigen_res,'maximum_orthogonality_residual_frobenius':orth,
        'initial_action_minus_saved_exact_rate_mass_norm':initial_action_difference,
        'initial_action_machine_bound':action_allowance,'initial_transform_roundtrip_mass_norm':roundtrip,
        'transform_machine_bound':transform_allowance,'imaginary_initial_mass_norm':imag_initial,
        'imaginary_initial_action_mass_norm':imag_action,'imaginary_final_mass_norm':imag_final,
        'modal_minus_uniformization_mass_norm':vector_difference,'independent_action_numerical_allowance_mass_norm':independent_allowance,
        'spectral_operator_indicator_mass_norm':spectral_indicator,'FFT_roundoff_allowance_mass_norm':fft_indicator,
        'uniformization_roundoff_allowance_mass_norm':alternate_allowance,
        'uniformization_Lambda':Lam,'uniformization_M':M,'uniformization_raw_weight_sum':alternate['raw_weight_sum'],
        'maximum_uniformization_row_sum_residual_unrepaired':float(np.max(abs(alternate['embedded_row_sum_minus_one']))),
        'minimum_raw_embedded_diagonal':alternate['raw_minimum_embedded_diagonal'],
        'Poisson_tail_bound_exact':str(tail),'Poisson_tail_mass_bound_exact':spec['uniformization']['mass_norm_tail_bound_exact'],
        'T_float_minus_exact_absolute_exact':str(T_gap),'T_float_label_indicator_mass_norm':time_label,
        'uniformization_z_float_gap_exact':str(z_gap),'uniformization_z_indicator_mass_norm':z_indicator,
        'reference_indicator_mass_norm':reference_indicator,'solver_indicator_mass_norm':solver_indicator,
        'combined_reference_solver_indicator_over_initial_fluctuation':(reference_indicator+solver_indicator)/SIGMA,
        'raw_eigenvalues_not_reset_weights_not_renormalized':True,
        'indicator_not_a_certified_total_physical_error_bound':True,
        'new_global_2D_matrices':0,'new_full_propagator_matrices':0,
        'full_replay_absolute_state_tolerance_fixed_formula':64*N*eps*max(1.,float(np.max(abs(c0))))}
    final_action=stencil_action(s,primary['state'])
    graph_chi=-math.fsum(float(a)*float(b)*float(c) for a,b,c in zip(m,primary['state'],final_action))
    chi_pair=0.
    field=primary['state'].reshape(N,N)
    for k,(dx,dy) in enumerate(s.offsets):
        g=np.array([float(s.conductances[s.coefficient_map[c][k]]) for c in range(N)])
        delta=np.roll(field,(-dy,-dx),(0,1))-field
        chi_pair+=float(np.sum(g[None,:]*delta*delta))
    controls['graph_chi_at_T_action']=graph_chi;controls['graph_chi_at_T_pair']=chi_pair
    controls['graph_chi_action_pair_difference_unrepaired']=graph_chi-chi_pair
    arrays={'point_semidiscrete_T':primary['state'],'independent_uniformization_T':alternate['state'],
        'initial_roundtrip':initial['initial_roundtrip'],'modal_initial_action':initial['initial_action'],
        'initial_full_Fourier_coefficients':primary['full_Fourier_coefficients'],
        'initial_full_eigen_coefficients':primary['full_eigen_coefficients'],
        'imaginary_initial':primary['imaginary_initial'],'imaginary_T':primary['imaginary_T'],
        'imaginary_initial_action':primary['imaginary_initial_action'],'uniformization_weights':alternate['weights'],
        'uniformization_row_sum_minus_one':alternate['embedded_row_sum_minus_one'],'graph_action_at_T':final_action}
    return arrays,controls,block_packet


def metrics(data,arrays,controls,fv_results):
    m=data['mass'];c0=data['c0'];reference=data['reference_cbar_T']
    mass=list(map(lambda v:F(float(v)),m));source=list(map(lambda v:F(float(v)),c0))
    total=sum(mass,F(0));inventory=sum((a*b for a,b in zip(mass,source)),F(0));mean=inventory/total
    E0=sum((a*(b-mean)**2 for a,b in zip(mass,source)),F(0))/2
    assert math.sqrt(float(2*E0))==SIGMA
    values={'point':arrays['point_semidiscrete_T'],'uniformization_control':arrays['independent_uniformization_T'],
        'PDE_reference':reference,'saved_FV_context':data['saved_FV_semidiscrete_T']}
    outputs={key:state_metrics(m,v,c0,reference,E0,inventory) for key,v in values.items()}
    point_error=outputs['point']['error_over_original_initial_fluctuation'];fv_error=fv_results['spatial_semidiscrete_relative_error']
    U=controls['combined_reference_solver_indicator_over_initial_fluctuation']
    comparison_U=U+fv_results['combined_reference_and_solver_indicator_over_initial_fluctuation']
    difference=point_error-fv_error
    decision='point_endpoint_worse_than_saved_FV_resolvably' if difference>comparison_U else ('point_endpoint_better_than_saved_FV_resolvably' if difference < -comparison_U else 'endpoint_comparison_inconclusive')
    error=arrays['point_semidiscrete_T']-reference;regions={}
    for name in ['near_contact','far_from_contact']:
        mask=data[name];value=norm(m[mask],error[mask])/SIGMA
        regions[name]={'cells':int(np.count_nonzero(mask)),'error_over_initial_fluctuation':value,
            'fraction_of_global_error_square':(value/point_error)**2}
    return {'N':128,'T_exact':'1/20','initial_fluctuation_norm_locked':SIGMA,'initial_inventory_exact':str(inventory),
        'initial_mean_exact':str(mean),'initial_resolved_variance_E_exact':str(E0),'initial_resolved_variance_E':float(E0),
        'outputs':outputs,'point_endpoint_relative_error':point_error,'saved_FV_endpoint_error_context':fv_error,
        'point_over_saved_FV_error_ratio':point_error/fv_error,'point_minus_saved_FV_error':difference,
        'candidate_combined_reference_solver_indicator':U,'candidate_plus_FV_comparison_indicator':comparison_U,
        'comparison_decision':decision,'point_error_is_resolvable_against_own_indicators':bool(point_error>U),
        'point_H_minus_PDE_H':outputs['point']['H']-outputs['PDE_reference']['H'],
        'FV_H_minus_PDE_H_context':outputs['saved_FV_context']['H']-outputs['PDE_reference']['H'],
        'same_original_regions':regions,'raw_graph_dissipation_at_T':controls['graph_chi_at_T_action'],
        'continuous_PDE_T_dissipation_not_evaluated':True,
        'no_old_endpoint_harmonic_T_baseline_no_improvement_claim_over_it':True,
        'initial_bridge_interpretation_requires_explicit_cellwise_constant_perturbation_lift':True,
        'all_variance_and_mixing_are_resolved_material_mean_observables':True,
        'no_new_absolute_physical_acceptance_threshold':True,'no_clipping_mass_or_probability_renormalization':True,
        'finite_node_candidate_admitted':False,'contact_transient_physically_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}


def main(output,tests,test_log):
    manifest=read(SOURCE/'sha256.json')
    for rel,h in manifest.items():assert digest(ROOT/rel)==h,rel
    spec=read(SOURCE/'next-gate-spec.json');assert not spec['already_executed'] and spec['T_exact']=='1/20'
    checks=read(tests);assert checks=={'successful':True,'checks_run':5,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',{'protocol':spec,'parent_gate_sha256':digest(SOURCE/'next-gate-spec.json'),
        'source_manifest_entries_verified':len(manifest),'primary_method':'all-mode raw symmetric eigensystem exponential, unchanged original state',
        'independent_method':'fixed Lambda168, M47 original-stencil uniformization, no weight renormalization',
        'arithmetic_bounds_fixed_formula':'64*N*epsilon transform/action, spectral residual/phase conversion and 8*(2*offsets+4)*(M+1)*epsilon uniformization allowance',
        'time_case_count':1,'new_candidate_weights':0,'no_new_PDE_reference':True})
    for name,path in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('tests.json',Path(tests)),('tests.log',Path(test_log)),
        ('source-coefficient-cache.json',ROOT/spec['source_coefficient_cache']),
        ('source-reference-results.json',ROOT/'runs/reconnect-contact-finite-time-reference/results.json'),
        ('source-FV-results.json',ROOT/'runs/reconnect-FV-contact-time/results.json')]:shutil.copyfile(path,out/name)
    with np.load(SOURCE/'matched-time-reference-inputs.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with np.load(ROOT/'runs/reconnect-reconstructed-global-frozen/global-action.npz',allow_pickle=False) as z:data['saved_initial_rate']=z['rate'].copy()
    records={k:record(v) for k,v in data.items()};np.savez_compressed(out/'original-inputs.npz',**data)
    started=time.monotonic();last=['start']
    def guard(stage,n,total):
        last[0]=f'{stage}-{n}/{total}';elapsed=time.monotonic()-started;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>600 or rss>512:raise RuntimeError(f'predeclared resource cap exceeded: {elapsed}s, {rss}MiB')
        if n%8==0 or n==total:print(f'{last[0]}; {elapsed:.3f}s; peak {rss:.2f} MiB',flush=True)
    def checkpoint(name,packet):np.savez_compressed(out/name,**packet)
    def timeout_handler(signum,frame):raise TimeoutError('predeclared 600 second limit reached')
    signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(600)
    try:
        arrays,controls,blocks=solve(data,read(out/'source-coefficient-cache.json'),spec,read(out/'source-reference-results.json'),guard,checkpoint)
        result=metrics(data,arrays,controls,read(out/'source-FV-results.json'))
        guard('completed',1,1);signal.alarm(0)
        assert records=={k:record(v) for k,v in data.items()}
        save(out/'solver-controls.json',controls);save(out/'results.json',result);np.savez_compressed(out/'point-T0.05.npz',**arrays)
        save(out/'input-provenance.json',{'parent_matched_inputs_sha256':digest(SOURCE/'matched-time-reference-inputs.npz'),
            'saved_initial_rate_artifact_sha256':digest(ROOT/'runs/reconnect-reconstructed-global-frozen/global-action.npz'),
            'source_cache_sha256':digest(ROOT/spec['source_coefficient_cache']),'copied_cache_sha256':digest(out/'source-coefficient-cache.json'),
            'input_artifact_sha256':digest(out/'original-inputs.npz'),'original_array_records':records,
            'all_original_mass_state_modes_and_coefficients_preserved':True})
        save(out/'stage3-status.json',{'updated':'2026-11-01','timezone':'Etc/UTC','gate_executed':True,
            'T_exact':'1/20','new_software_checks':5,'previous_recorded_checks':423,'recorded_checks_total':428,'previous_checks_rerun':0,
            'new_saved_point_graph_research_time_cases':1,'new_candidate_weights':0,'new_kernel_builds':0,'new_density_path_builds':0,
            'new_modal_normal_blocks':65,'all_discrete_transverse_modes_retained':128,'new_global_2D_matrices':0,'new_full_propagator_matrices':0,
            'new_PDE_reference_solves':0,'new_exact_algebra_certificate_groups':0,'comparison_decision':result['comparison_decision'],
            'finite_node_candidate_admitted':False,'contact_transient_physically_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
            'ordinary_central_resistance_FV_baseline_retained':True,'mass_average_resistance_remains_experimental':True,
            'next_work':'read-only interpretation of fixed endpoint state/mixing evidence; no time/grid/h/weight repair or new ORC',
            'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'first_attempt_passed':True,
            'source_manifest_entries_verified':len(manifest),'new_software_checks':5,'recorded_checks_total':428,'previous_checks_rerun':0,
            'elapsed_candidate_and_control_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'predeclared_resource_caps_passed':True,'wall_clock_cap_seconds':600,'peak_RSS_MiB_cap':512,
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'scoped_BLAS_threads':4,
            'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps({'point_endpoint_error':result['point_endpoint_relative_error'],
            'FV_context_error':result['saved_FV_endpoint_error_context'],'ratio':result['point_over_saved_FV_error_ratio'],
            'point_H':result['outputs']['point']['H'],'PDE_H':result['outputs']['PDE_reference']['H'],
            'comparison':result['comparison_decision'],'solver_controls':controls},indent=2),flush=True)
    except BaseException as error:
        signal.alarm(0)
        save(out/'failure.json',{'type':type(error).__name__,'reason':str(error),'last_stage':last[0],
            'elapsed_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'partial_evidence_preserved':True,'no_weight_state_threshold_or_rule_retuning':True,'fluid_orc_enabled':False})
        raise


def summary_replay(directory):
    out=Path(directory)
    with np.load(out/'original-inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(out/'point-T0.05.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    result=metrics(data,arrays,read(out/'solver-controls.json'),read(out/'source-FV-results.json'))
    assert result==read(out/'results.json')
    print('Saved endpoint error, raw inventory and resolved I/H metrics replay exactly; ORC disabled.',flush=True)


def replay(directory):
    out=Path(directory);p=read(out/'input-provenance.json')
    assert digest(out/'original-inputs.npz')==p['input_artifact_sha256']
    with np.load(out/'original-inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    arrays,controls,_=solve(data,read(out/'source-coefficient-cache.json'),read(out/'source-gate-spec.json'),read(out/'source-reference-results.json'))
    tolerance=read(out/'solver-controls.json')['full_replay_absolute_state_tolerance_fixed_formula']
    with np.load(out/'point-T0.05.npz',allow_pickle=False) as z:
        delta=max(float(np.max(abs(z[k]-arrays[k]))) for k in ['point_semidiscrete_T','independent_uniformization_T'])
        assert delta<=tolerance
    summary_replay(out)
    print(f'Same saved graph, all-mode time action and independent fixed uniformization replay; largest endpoint difference {delta}; no physical admission.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output');parser.add_argument('--tests');parser.add_argument('--test-log');parser.add_argument('--replay');parser.add_argument('--summary-replay');args=parser.parse_args()
    if args.summary_replay:summary_replay(args.summary_replay)
    elif args.replay:replay(args.replay)
    else:
        if not all([args.output,args.tests,args.test_log]):parser.error('--output, --tests and --test-log required')
        main(args.output,args.tests,args.test_log)
