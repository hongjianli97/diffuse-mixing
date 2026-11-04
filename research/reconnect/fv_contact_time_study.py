"""One fixed-time ordinary FV comparison with separate space/time controls.

Solver replay needs NumPy. Metric replay uses only the standard library and
does not rerun the solver or the PDE reference. All source c0 modes are kept.
"""
from decimal import Decimal as Dec,localcontext
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform

from audit_route_decision import ROOT,digest,read,save


SOURCE=ROOT/'runs/reconnect-contact-finite-time-reference'
STEPS=[512,1024]


def norm(m,vector):return math.sqrt(math.fsum(a*b*b for a,b in zip(m,vector)))


def difference(a,b):return [x-y for x,y in zip(a,b)]


def array_record(a):
    return {'shape':list(a.shape),'dtype':a.dtype.str,'C_order_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def solve(packet):
    import numpy as np
    from cartesian_fv_modes import build_modes,coefficients,reconstruct,propagate,direct_structured_action
    from contact_spectral_reference import TrigPi,dec
    source,model=packet['source_arrays'],packet['model'];N=model['N'];T=float(F(model['T_exact']))
    rho=np.asarray(source['rho']).reshape(N,N);mass=np.asarray(source['mass']).reshape(N,N)
    c0=np.asarray(source['c0']).reshape(N,N)
    modes=build_modes(rho,mass,float(F(model['D_exact'])),model['periods'],model_prior=model['model_prior'])
    coeff=coefficients(modes,c0);initial,imag0=reconstruct(modes,coeff)
    direct,imag_action=direct_structured_action(modes,c0)
    lam=modes.transverse_eigenvalues[:,None]+modes.normal_eigenvalues[None,:]
    eigen_action,imag_eigen_action=reconstruct(modes,-lam*coeff)
    states={};imaginary={}
    for label,steps in [('semidiscrete',None),('BE512',512),('BE1024',1024)]:
        value,imaginary[label]=propagate(modes,coeff,T,steps);states[label]=value.ravel().tolist()
    with localcontext() as context:
        context.prec=90;trig=TrigPi();dy=F(model['periods'][1])/N;D=F(model['D_exact'])
        y_differences=[abs(dec(float(modes.transverse_eigenvalues[k]))-
            4*dec(D)/dec(dy)**2*trig.sc(F(k,N))[0]**2) for k in range(N)]
        y_error=float(max(y_differences))
    m=source['mass'];c=source['c0'];initial_norm=norm(m,c)
    q_bound=math.sqrt(1+modes.orthogonality_residual_frobenius)
    eta=max(0.,-float(modes.normal_eigenvalues.min())) # Bound only; raw eigenvalues are unchanged.
    growth=math.exp(T*eta)
    B_error=float(modes.normal_matrix_rounding_row_bound_exact)
    residual=modes.eigen_residual_frobenius+B_error*q_bound
    residual_indicator=(modes.orthogonality_residual_frobenius+T*growth*(residual*q_bound+y_error*q_bound*q_bound))*initial_norm
    eps=float(np.finfo(float).eps)
    transform_indicator=32*N*eps*growth*(q_bound*q_bound+1)*initial_norm
    roundtrip=norm(m,difference(initial.ravel().tolist(),c))
    imaginary_indicator=max(norm(m,v.ravel().tolist()) for v in imaginary.values())
    frozen=list(map(lambda x:float(F(x)),packet['saved_frozen_exact_rates']))
    T_gap=abs(F(T)-F(model['T_exact']))
    time_label_indicator=float(T_gap)*norm(m,frozen)
    controls={
        'normal_ring_dimension':N,'transverse_modes_retained':N,'all_discrete_mode_slots_retained':N*N,
        'normal_face_law_evaluations':N,'full_2D_transport_matrices':0,
        'normal_g_exact':list(map(str,modes.normal_g_exact)),
        'normal_matrix_rounding_row_bound_exact':str(modes.normal_matrix_rounding_row_bound_exact),
        'raw_normal_minimum_eigenvalue':float(modes.normal_eigenvalues.min()),
        'raw_normal_maximum_eigenvalue':float(modes.normal_eigenvalues.max()),
        'raw_transverse_maximum_eigenvalue':float(modes.transverse_eigenvalues.max()),
        'normal_eigen_residual_frobenius':modes.eigen_residual_frobenius,
        'normal_orthogonality_residual_frobenius':modes.orthogonality_residual_frobenius,
        'transverse_eigenvalue_vs_90_digit_max_absolute_difference':y_error,
        'initial_transform_roundtrip_mass_norm':roundtrip,
        'structured_action_minus_saved_exact_FV_mass_norm':norm(m,difference(direct.ravel().tolist(),frozen)),
        'eigen_action_minus_structured_action_mass_norm':norm(m,difference(eigen_action.ravel().tolist(),direct.ravel().tolist())),
        'maximum_output_imaginary_mass_norm':imaginary_indicator,
        'maximum_output_imaginary_absolute_value':max(float(np.max(np.abs(v))) for v in imaginary.values()),
        'spectral_residual_and_operator_indicator_mass_norm':residual_indicator,
        'epsilon_based_transform_roundoff_allowance_mass_norm':transform_indicator,
        'time_float_label_minus_exact_T_absolute_exact':str(T_gap),
        'time_float_label_rounding_indicator_mass_norm':time_label_indicator,
        'source_nonprincipal_transverse_coefficient_mass_norm':float(math.sqrt(float(modes.area_exact))*
            np.linalg.norm(coeff[[k for k in range(N) if k not in [0,1,N-1]],:])),
        'no_mode_filter_no_eigenvalue_reset_no_mean_replacement':True,
        'solver_roundoff_indicator_not_a_certified_total_physical_error_bound':True,
        'solver_indicator_mass_norm':residual_indicator+transform_indicator+roundtrip+imaginary_indicator+time_label_indicator,
        'numerical_replay_absolute_tolerance_fixed_formula':64*N*eps*max(1.,max(map(abs,c))),
    }
    diagnostic={'normal_matrix':modes.normal_matrix,'normal_eigenvalues':modes.normal_eigenvalues,
        'normal_eigenvectors':modes.normal_eigenvectors,'transverse_eigenvalues':modes.transverse_eigenvalues,
        'initial_full_coefficients':coeff,'initial_reconstruction':initial.ravel(),
        'structured_initial_rate':direct.ravel(),'eigen_initial_rate':eigen_action.ravel()}
    return states,controls,diagnostic


def metrics(packet,states,controls):
    source=packet['source_arrays'];m=source['mass'];c0=source['c0'];reference=source['reference_cbar_T']
    total=sum(map(F,m),F(0));inventory=sum((F(a)*F(b) for a,b in zip(m,c0)),F(0));mean=inventory/total
    fluctuation_square=sum((F(a)*(F(b)-mean)**2 for a,b in zip(m,c0)),F(0))
    denominator=math.sqrt(float(fluctuation_square));relative=lambda v:norm(m,v)/denominator
    spatial=difference(states['semidiscrete'],reference);spatial_norm=relative(spatial)
    outputs={}
    for name,value in states.items():
        error=difference(value,reference);temporal=difference(value,states['semidiscrete'])
        temporal_norm=relative(temporal)
        cross=2*math.fsum(a*b*c for a,b,c in zip(m,spatial,temporal))/denominator**2
        e=relative(error)
        final_inventory=sum((F(a)*F(b) for a,b in zip(m,value)),F(0))
        regions={}
        for label,mask in [('near_contact',packet['original_masks']['near_contact']),('far_from_contact',packet['original_masks']['far_from_contact'])]:
            square=math.fsum(a*b*b for a,b,keep in zip(m,error,mask) if keep)/denominator**2
            regions[label]={'cells':sum(mask),'error_over_initial_fluctuation':math.sqrt(square),
                'fraction_of_method_global_error_square':square/(e*e)}
        outputs[name]={
            'relative_error_vs_PDE':e,'time_error_vs_semidiscrete':temporal_norm,
            'signed_spatial_time_cross_term':cross,
            'error_square_budget_residual':e*e-(spatial_norm**2+temporal_norm**2+cross),
            'minimum':min(value),'maximum':max(value),
            'inside_original_c0_extrema_at_T':min(value)>=min(c0) and max(value)<=max(c0),
            'inventory_minus_original_exact_kg':str(final_inventory-inventory),
            'inventory_minus_original_display_kg':float(final_inventory-inventory),
            'fluctuation_norm_about_original_mean':norm(m,[v-float(mean) for v in value]),
            'regional_comparison_with_original_masks':regions}
    reference_indicator=float(packet['reference_results']['reference_initial_bridge_serialization_and_precision_indicator_mass_norm'])
    solver_indicator=controls['solver_indicator_mass_norm'];indicator=(reference_indicator+solver_indicator)/denominator
    return {
        'scope':'one ordinary central-resistance FV fixed-time comparison on original material means',
        'N':128,'T_exact':'1/20','original_initial_inventory_exact':str(inventory),
        'original_initial_mass_mean_exact':str(mean),'initial_fluctuation_norm':denominator,
        'spatial_semidiscrete_relative_error':spatial_norm,'outputs':outputs,
        'time_error_ratio_512_over_1024':outputs['BE512']['time_error_vs_semidiscrete']/outputs['BE1024']['time_error_vs_semidiscrete'],
        'reference_indicator_over_initial_fluctuation':reference_indicator/denominator,
        'solver_indicator_over_initial_fluctuation':solver_indicator/denominator,
        'combined_reference_and_solver_indicator_over_initial_fluctuation':indicator,
        'spatial_error_divided_by_combined_indicator':spatial_norm/indicator,
        'BE1024_time_error_divided_by_combined_indicator':outputs['BE1024']['time_error_vs_semidiscrete']/indicator,
        'indicator_is_not_a_certified_total_error_bound_or_new_acceptance_threshold':True,
        'reference_bridge_remains_conditional_on_explicit_perturbation_lift':True,
        'kernel_finite_time_comparison_performed':False,
        'new_FV_baseline_time_cases':1,'FV_numerical_time_outputs':3,
        'implicit_Euler_controls_are_spectral_factors_not_512_or_1024_explicit_step_loops':True,
        'all_original_binary_c0_discrete_modes_retained':True,
        'new_point_kernel_builds':0,'new_point_candidate_transient_runs':0,
        'new_reference_time_solutions':0,'new_full_2D_transport_matrices':0,
        'new_normal_1D_operators':1,'new_software_checks':4,'previous_checks_rerun':0,
        'recorded_checks_total':400,'new_exact_algebra_certificate_groups':0,
        'four_mean_correction_installed':False,'clipping_or_inventory_renormalization':False,
        'finite_node_candidate_admitted':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}


def summary_replay(output):
    out=Path(output).resolve()
    result=metrics(read(out/'comparison-inputs.json'),read(out/'comparison-states.json'),read(out/'solver-controls.json'))
    assert result==read(out/'results.json')
    print(json.dumps({'standard_library_metric_replay_identical':True,'recorded_checks_total':400,
        'spatial_error':result['spatial_semidiscrete_relative_error'],'fluid_orc_enabled':False}),flush=True)


def replay(output):
    import numpy as np
    out=Path(output).resolve();packet=read(out/'comparison-inputs.json')
    states,controls,_=solve(packet);saved=read(out/'comparison-states.json')
    tolerance=64*packet['model']['N']*np.finfo(float).eps*max(1.,max(map(abs,packet['source_arrays']['c0'])))
    largest=max(max(abs(a-b) for a,b in zip(states[key],saved[key])) for key in states)
    assert largest<=tolerance
    summary_replay(out)
    print(json.dumps({'NumPy_solver_replay_within_epsilon_based_tolerance':True,
        'maximum_state_difference':largest,'arithmetic_replay_tolerance':tolerance,
        'replay_tolerance_is_not_a_physical_acceptance_threshold':True}),flush=True)


def main(output,tests,test_log):
    import numpy as np
    status_path=SOURCE/'stage3-status.json';manifest=read(SOURCE/'sha256.json')
    for rel,expected in manifest.items():assert digest(ROOT/rel)==expected,rel
    protocol=read(status_path)['next_ordinary_FV_fixed_time_predeclared']
    assert not protocol['already_executed'] and protocol['T_exact']=='1/20' and protocol['N']==128
    assert digest(ROOT/protocol['source_snapshot'])==protocol['source_snapshot_sha256']
    tests_result=read(tests);assert tests_result['successful'] and tests_result['checks_run']==4
    assert tests_result['errors']==tests_result['failures']==tests_result['skipped']==0
    ref_file=SOURCE/'reference-T0.05.npz';ref_provenance=read(SOURCE/'input-provenance.json')
    assert digest(ref_file)==ref_provenance['output_reference_NPZ_sha256']
    with np.load(ref_file,allow_pickle=False) as z:
        raw={key:z[key].copy() for key in ['ids','positions','rho','mass','c0','reference_cbar_T']}
    assert raw['ids'].tolist()==list(range(16384))
    assert raw['positions'].tolist()==[[(i+.5)/128,(j+.5)/128] for j in range(128) for i in range(128)]
    old_frozen=ROOT/'runs/reconnect-FV-global-frozen'
    old_packet=read(old_frozen/'frozen-inputs.json')
    packet={'model':{'N':128,'T_exact':'1/20','D_exact':read(SOURCE/'reference-inputs.json')['model']['D_exact'],
        'periods':[1.,1.],'model_prior':'static Cartesian material cells, supplied points at centers, constant density per cell'},
        'source_arrays':{key:value.tolist() for key,value in raw.items()},
        'saved_frozen_exact_rates':read(old_frozen/'exact-action.json')['rates_per_s'],
        'original_masks':{key:old_packet['saved_arrays'][key] for key in ['near_contact','far_from_contact']},
        'reference_results':read(SOURCE/'results.json')}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'comparison-inputs.json',packet)
    receipts={'source-stage3-status.json':status_path,'source-reference-results.json':SOURCE/'results.json',
        'tests.json':Path(tests),'tests.log':Path(test_log)}
    for name,path in receipts.items():
        with (out/name).open('xb') as f:f.write(path.read_bytes())
    save(out/'config.json',{'study':'one ordinary FV finite-time baseline comparison',
        'protocol':protocol,'protocol_source_sha256':digest(status_path),'source_manifest_entries_verified':len(manifest),
        'software_checks':tests_result,'steps':STEPS,'all_initial_modes_retained':True,
        'arithmetic_replay_tolerance_formula':'64*N*machine_epsilon*max(1,max(abs(original_c0)))',
        'solver_indicator_fixed_before_evaluation':'normal matrix exact-rounding bound, raw eigen residual/orthogonality, high-precision transverse eigenvalues, epsilon-based transform allowance, roundtrip, imaginary residue and time-label rounding',
        'solver_indicator_not_interval_or_physical_acceptance_threshold':True,
        'raw_eigenvalues_used_without_zeroing_or_clipping':True})
    states,controls,diagnostic=solve(packet);result=metrics(packet,states,controls)
    save(out/'comparison-states.json',states);save(out/'solver-controls.json',controls);save(out/'results.json',result)
    np.savez_compressed(out/'FV-T0.05.npz',**raw,**{key:np.asarray(value) for key,value in states.items()},**diagnostic)
    save(out/'input-provenance.json',{'original_snapshot':protocol['source_snapshot'],
        'original_snapshot_sha256':protocol['source_snapshot_sha256'],'reference_source':str(ref_file.relative_to(ROOT)),
        'reference_source_sha256':digest(ref_file),'original_source_member_records':{key:array_record(value) for key,value in raw.items()},
        'output_NPZ_sha256':digest(out/'FV-T0.05.npz'),
        'receipt_SHA256':{name:digest(path) for name,path in receipts.items()},
        'frozen_exact_action_sha256':digest(old_frozen/'exact-action.json'),
        'original_masks_source_sha256':digest(old_frozen/'frozen-inputs.json')})
    save(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,
        'thread_settings_for_execution':{'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'},
        'numerical_replay_requires_numpy':True,'metric_replay_standard_library_only':True})
    with (out/'requirements-solver.txt').open('x') as f:f.write('numpy=='+np.__version__+'\n')
    print(json.dumps({'results':result,'solver_controls':{key:value for key,value in controls.items() if key!='normal_g_exact'}},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--output');group.add_argument('--replay');group.add_argument('--summary-replay')
    parser.add_argument('--tests');parser.add_argument('--test-log');args=parser.parse_args()
    if args.summary_replay:summary_replay(args.summary_replay)
    elif args.replay:replay(args.replay)
    else:
        if not args.tests or not args.test_log:parser.error('--tests and --test-log required for new source run')
        main(args.output,args.tests,args.test_log)
