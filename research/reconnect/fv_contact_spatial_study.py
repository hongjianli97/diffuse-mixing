"""One N256 control, restricted onto the original N128 material observation."""
from decimal import Decimal as Dec,localcontext
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform

from audit_route_decision import ROOT,digest,read,save
from nested_material_observation import restrict_material_means


SOURCE=ROOT/'runs/reconnect-FV-contact-time'
REFERENCE=ROOT/'runs/reconnect-contact-finite-time-reference'


def norm(m,v):return math.sqrt(math.fsum(a*b*b for a,b in zip(m,v)))
def diff(a,b):return [x-y for x,y in zip(a,b)]
def exact_norm(m,v):return math.sqrt(float(sum((F(a)*b*b for a,b in zip(m,v)),F(0))))


def array_record(a):
    return {'shape':list(a.shape),'dtype':a.dtype.str,'C_order_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def derive_fine(packet):
    """New material integrals; no old c0 interpolation and no new projection fit."""
    from contact_spectral_reference import TrigPi,dec,material_factors,render
    N=packet['model']['fine_N'];case=packet['sealed_reference_case']
    assert case['K']==128 and case['requested_decimal_digits']==90
    with localcontext() as context:
        context.prec=120;trig=TrigPi()
        a=list(map(Dec,case['alpha']));b=list(map(Dec,case['beta']))
        normal,y,initial=material_factors(trig,a,b,N,dec(packet['model']['D_exact']),dec(packet['model']['T_exact']))
        factors=render({'normal_T':normal,'transverse_cell_means':y,'initial_normal_cell_means':initial},90)
        values0=[Dec('.5')+Dec(x)*Dec(z) for z in factors['transverse_cell_means'] for x in factors['initial_normal_cell_means']]
        valuesT=[Dec('.5')+Dec(x)*Dec(z) for z in factors['transverse_cell_means'] for x in factors['normal_T']]
        rho=[1. if i<N//2 else 4. for j in range(N) for i in range(N)];mass=[r/(N*N) for r in rho]
        c0=list(map(float,values0));ref=list(map(float,valuesT))
        initial_round=sum((dec(m)*(dec(v)-z)**2 for m,v,z in zip(mass,c0,values0)),Dec(0)).sqrt()
        reference_round=sum((dec(m)*(dec(v)-z)**2 for m,v,z in zip(mass,ref,valuesT)),Dec(0)).sqrt()
        # Factor nesting uses the phase-aligned, equal-width material cells.
        nested_T=[(normal[2*i]+normal[2*i+1])/2 for i in range(N//2)]
        nested_c0=[(initial[2*i]+initial[2*i+1])/2 for i in range(N//2)]
        nested_y=[(y[2*j]+y[2*j+1])/2 for j in range(N//2)]
        nesting=max([abs(v-Dec(w)) for v,w in zip(nested_T,case['normal_T'])]
            +[abs(v-Dec(w)) for v,w in zip(nested_c0,case['initial_normal_cell_means'])]
            +[abs(v-Dec(w)) for v,w in zip(nested_y,case['transverse_cell_means'])])
        generation={'fine_inputs_newly_derived_not_recovered_originals':True,
            'source_spectral_coefficients_reused_without_new_projection':True,
            'high_precision_factor_nesting_max_difference':str(nesting),
            'fine_initial_float_serialization_mass_norm':str(initial_round),
            'fine_reference_float_serialization_mass_norm':str(reference_round),
            'reused_mathematical_tail_bound_exact':case['tail_certificate']['material_mass_norm_tail_bound_exact'],
            'Decimal_generation_controls_are_not_interval_certification':True}
    return {'mass':mass,'rho':rho,'c0':c0,'reference_cbar_T':ref},factors,generation


def solve_fine(packet,fine):
    import numpy as np
    from cartesian_fv_modes import build_modes,coefficients,reconstruct,propagate,direct_structured_action
    from contact_spectral_reference import TrigPi,dec
    N=packet['model']['fine_N'];D=F(packet['model']['D_exact']);target=F(packet['model']['T_exact']);T=float(target)
    rho=np.asarray(fine['rho']).reshape(N,N);mass=np.asarray(fine['mass']).reshape(N,N);c0=np.asarray(fine['c0']).reshape(N,N)
    modes=build_modes(rho,mass,float(D),model_prior=packet['model']['model_prior'])
    coeff=coefficients(modes,c0);initial,imag0=reconstruct(modes,coeff)
    state,imag=propagate(modes,coeff,T)
    direct,imag_action=direct_structured_action(modes,c0)
    with localcontext() as context:
        context.prec=90;trig=TrigPi()
        y_error=float(max(abs(dec(float(modes.transverse_eigenvalues[k]))-
            4*dec(D)*N*N*trig.sc(F(k,N))[0]**2) for k in range(N)))
    qnorm=math.sqrt(1+modes.orthogonality_residual_frobenius)
    growth=math.exp(T*max(0.,-float(modes.normal_eigenvalues.min())))
    initial_norm=norm(fine['mass'],fine['c0'])
    residual=modes.eigen_residual_frobenius+float(modes.normal_matrix_rounding_row_bound_exact)*qnorm
    residual_indicator=(modes.orthogonality_residual_frobenius+T*growth*(residual*qnorm+y_error*qnorm*qnorm))*initial_norm
    eps=float(np.finfo(float).eps)
    transform_indicator=32*N*eps*growth*(qnorm*qnorm+1)*initial_norm
    roundtrip=norm(fine['mass'],diff(initial.ravel().tolist(),fine['c0']))
    imag_norm=norm(fine['mass'],imag.ravel().tolist())
    time_round=float(abs(F(T)-target))*norm(fine['mass'],direct.ravel().tolist())
    controls={'normal_ring_dimension':N,'normal_face_law_evaluations':N,
        'full_Cartesian_face_locations_represented':2*N*N,'all_discrete_mode_slots_retained':N*N,
        'raw_normal_minimum_eigenvalue':float(modes.normal_eigenvalues.min()),
        'raw_normal_maximum_eigenvalue':float(modes.normal_eigenvalues.max()),
        'normal_matrix_rounding_row_bound_exact':str(modes.normal_matrix_rounding_row_bound_exact),
        'normal_eigen_residual_frobenius':modes.eigen_residual_frobenius,
        'normal_orthogonality_residual_frobenius':modes.orthogonality_residual_frobenius,
        'transverse_eigenvalue_vs_90_digit_max_difference':y_error,
        'roundtrip_mass_norm':roundtrip,'output_imaginary_mass_norm':imag_norm,
        'spectral_residual_operator_indicator_mass_norm':residual_indicator,
        'epsilon_based_transform_allowance_mass_norm':transform_indicator,
        'time_label_rounding_indicator_mass_norm':time_round,
        'solver_indicator_mass_norm':residual_indicator+transform_indicator+roundtrip+imag_norm+time_round,
        'numerical_solver_replay_absolute_tolerance':64*N*eps*max(1.,max(map(abs,fine['c0']))),
        'no_mode_filter_no_eigenvalue_reset_no_inventory_repair':True}
    diagnostics={'normal_matrix':modes.normal_matrix,'normal_eigenvalues':modes.normal_eigenvalues,
        'normal_eigenvectors':modes.normal_eigenvectors,'transverse_eigenvalues':modes.transverse_eigenvalues,
        'fine_initial_full_coefficients':coeff,'fine_initial_structured_rate':direct.ravel()}
    return state.ravel().tolist(),controls,diagnostics


def metrics(packet,fine,state,controls,generation):
    coarse=packet['coarse_arrays'];N=packet['model']['coarse_N'];m=coarse['mass'];den=packet['coarse_results']['initial_fluctuation_norm']
    p0=restrict_material_means(fine['mass'],fine['c0'],m,N)
    pt=restrict_material_means(fine['mass'],state,m,N)
    pr=restrict_material_means(fine['mass'],fine['reference_cbar_T'],m,N)
    coarse_reference=list(map(F,coarse['reference_cbar_T']));coarse_c0=list(map(F,coarse['c0']))
    error=[a-b for a,b in zip(pt.means,coarse_reference)]
    native_error=[F(a)-F(b) for a,b in zip(state,fine['reference_cbar_T'])]
    projected_error=[a-b for a,b in zip(pt.means,pr.means)]
    ref_bridge=[a-b for a,b in zip(pr.means,coarse_reference)]
    initial_bridge=[a-b for a,b in zip(p0.means,coarse_c0)]
    assert all(e==p+b for e,p,b in zip(error,projected_error,ref_bridge))
    jensen_error=sum((F(a)*b*b for a,b in zip(fine['mass'],native_error)),F(0))-sum((F(a)*b*b for a,b in zip(m,projected_error)),F(0))
    assert jensen_error>=0
    e128=norm(m,diff(coarse['coarse_FV_T'],coarse['reference_cbar_T']))/den
    e256=exact_norm(m,error)/den
    coarse_diff=[a-F(b) for a,b in zip(pt.means,coarse['coarse_FV_T'])]
    diff_norm=exact_norm(m,coarse_diff)/den
    source_inventory=sum((F(a)*F(b) for a,b in zip(m,coarse['c0'])),F(0))
    fine_initial_inventory=p0.inventory;fine_final_inventory=pt.inventory
    fine_indicator=(controls['solver_indicator_mass_norm']
        +float(generation['fine_initial_float_serialization_mass_norm'])
        +float(generation['fine_reference_float_serialization_mass_norm']))/den
    old_indicator=packet['coarse_results']['combined_reference_and_solver_indicator_over_initial_fluctuation']
    initial_norm=exact_norm(m,initial_bridge)/den;ref_norm=exact_norm(m,ref_bridge)/den
    combined=old_indicator+fine_indicator+initial_norm+ref_norm
    result={'scope':'one N256 semidiscrete FV control under original N128 material observation',
        'T_exact':'1/20','coarse_N':128,'fine_N':256,'original_initial_fluctuation_norm':den,
        'coarse128_error':e128,'fine256_error_under_original128_observation':e256,
        'coarse_error_divided_by_observed_fine_error':e128/e256,
        'two_level_log2_ratio_observation_only':math.log2(e128/e256),
        'coarse_vs_observed_fine_state_difference':diff_norm,
        'native_fine256_error_same_original_denominator_diagnostic_only':exact_norm(fine['mass'],native_error)/den,
        'projected_native_error':exact_norm(m,projected_error)/den,
        'fine_reference_to_original_coarse_reference_bridge':ref_norm,
        'fine_initial_to_original_c0_bridge':initial_norm,
        'fine_initial_to_original_c0_bridge_max_absolute':float(max(map(abs,initial_bridge))),
        'fine_initial_inventory_minus_original_coarse_exact':str(fine_initial_inventory-source_inventory),
        'fine_final_inventory_minus_fine_initial_exact':str(fine_final_inventory-fine_initial_inventory),
        'fine_final_inventory_minus_fine_initial_display':float(fine_final_inventory-fine_initial_inventory),
        'restriction_inventory_residual_exact':str(pt.exact_inventory_residual),
        'restriction_reference_error_closure_max_exact':'0',
        'projected_error_Jensen_square_gap_exact':str(jensen_error),
        'fine_minimum':min(state),'fine_maximum':max(state),
        'fine_inside_own_initial_extrema_at_T':min(state)>=min(fine['c0']) and max(state)<=max(fine['c0']),
        'observed_fine_minimum':float(min(pt.means)),'observed_fine_maximum':float(max(pt.means)),
        'coarse_reference_solver_indicator':old_indicator,'fine_solver_and_serialization_indicator':fine_indicator,
        'combined_comparison_indicator':combined,
        'state_difference_divided_by_combined_indicator':diff_norm/combined,
        'error_reduction_divided_by_combined_indicator':(e128-e256)/combined,
        'indicator_is_not_certified_total_physical_bound_or_new_threshold':True,
        'fine_inputs_are_new_material_integrals_not_recovered_originals':True,
        'initial_and_reference_bridges_retained_not_repaired':True,
        'one_spatial_level_is_not_a_general_convergence_order_theorem':True,
        'new_fine_FV_time_cases':1,'new_fine_normal_1D_operators':1,
        'new_material_reference_average_evaluations':1,'new_PDE_time_solutions_or_projection_fits':0,
        'new_time_step_controls':0,'new_full_2D_transport_matrices':0,
        'new_point_kernel_builds':0,'new_point_candidate_transient_runs':0,
        'new_software_checks':3,'recorded_checks_total':403,'previous_checks_rerun':0,
        'new_exact_algebra_certificate_groups':0,'finite_node_candidate_admitted':False,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    observed={'fine_to_coarse_initial_exact':list(map(str,p0.means)),
        'fine_to_coarse_FV_T_exact':list(map(str,pt.means)),
        'fine_to_coarse_reference_exact':list(map(str,pr.means))}
    return result,observed


def summary_replay(output):
    out=Path(output).resolve()
    result,observed=metrics(read(out/'comparison-inputs.json'),read(out/'derived-fine-data.json'),
        read(out/'fine-state.json'),read(out/'solver-controls.json'),read(out/'generation-controls.json'))
    assert result==read(out/'results.json') and observed==read(out/'exact-coarse-observation.json')
    print(json.dumps({'standard_library_observation_and_metric_replay_identical':True,
        'coarse_error':result['coarse128_error'],'observed_fine_error':result['fine256_error_under_original128_observation'],
        'recorded_checks_total':403,'fluid_orc_enabled':False}),flush=True)


def replay(output):
    import numpy as np
    out=Path(output).resolve();packet=read(out/'comparison-inputs.json')
    fine,factors,generation=derive_fine(packet)
    assert fine==read(out/'derived-fine-data.json') and factors==read(out/'fine-reference-factors.json')
    assert generation==read(out/'generation-controls.json')
    state,controls,_=solve_fine(packet,fine);saved=read(out/'fine-state.json')
    tolerance=64*packet['model']['fine_N']*np.finfo(float).eps*max(1.,max(map(abs,fine['c0'])))
    difference=max(abs(a-b) for a,b in zip(state,saved));assert difference<=tolerance
    summary_replay(out)
    print(json.dumps({'NumPy_fine_solver_replay_within_epsilon_tolerance':True,'maximum_state_difference':difference,
        'arithmetic_replay_tolerance':tolerance,'tolerance_is_not_a_physical_acceptance_threshold':True}),flush=True)


def main(output,tests,test_log):
    import numpy as np
    manifest=read(SOURCE/'sha256.json')
    for rel,expected in manifest.items():assert digest(ROOT/rel)==expected,rel
    protocol_path=SOURCE/'stage3-status.json';protocol=read(protocol_path)['next_single_spatial_control_predeclared']
    assert not protocol['already_executed'] and protocol['fine_N']==256 and protocol['coarse_N']==128
    check=read(tests);assert check['successful'] and check['checks_run']==3
    assert check['failures']==check['errors']==check['skipped']==0
    previous=read(SOURCE/'comparison-inputs.json')
    coarse={**previous['source_arrays'],'coarse_FV_T':read(SOURCE/'comparison-states.json')['semidiscrete']}
    packet={'model':{'coarse_N':128,'fine_N':256,'T_exact':'1/20','D_exact':previous['model']['D_exact'],
        'model_prior':previous['model']['model_prior']},'coarse_arrays':coarse,
        'coarse_results':read(SOURCE/'results.json'),
        'sealed_reference_case':read(REFERENCE/'reference-K128-p90.json')}
    assert packet['sealed_reference_case']['K']==128
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'comparison-inputs.json',packet)
    receipts={'source-stage3-status.json':protocol_path,'source-coarse-results.json':SOURCE/'results.json',
        'source-reference-K128-p90.json':REFERENCE/'reference-K128-p90.json','tests.json':Path(tests),'tests.log':Path(test_log)}
    for name,path in receipts.items():
        with (out/name).open('xb') as f:f.write(path.read_bytes())
    save(out/'config.json',{'study':'one nested N256 material-observation FV control',
        'protocol':protocol,'protocol_source_sha256':digest(protocol_path),'source_manifest_entries_verified':len(manifest),
        'software_checks':check,'fine_reference_uses_sealed_coefficients_only':True,
        'restriction':'exact Fraction child mass sum equality before mass averaging; no normalization repair',
        'reference_and_initial_generation_working_digits':120,'stored_factor_digits':90,
        'comparison_indicator_fixed_before_run':'coarse saved reference/solver indicator plus fine solver/serialization controls and retained initial/reference nesting bridges',
        'no_new_physical_acceptance_threshold':True})
    fine,factors,generation=derive_fine(packet)
    save(out/'derived-fine-data.json',fine);save(out/'fine-reference-factors.json',factors);save(out/'generation-controls.json',generation)
    state,controls,diagnostic=solve_fine(packet,fine)
    save(out/'fine-state.json',state);save(out/'solver-controls.json',controls)
    result,observed=metrics(packet,fine,state,controls,generation)
    save(out/'results.json',result);save(out/'exact-coarse-observation.json',observed)
    N=256;positions=np.array([[(i+.5)/N,(j+.5)/N] for j in range(N) for i in range(N)])
    np.savez_compressed(out/'FV-N256-common-N128.npz',
        **{key:np.asarray(value) for key,value in coarse.items()},
        **{'fine_'+key:np.asarray(value) for key,value in fine.items()},
        fine_cell_ids=np.arange(N*N),fine_positions=positions,fine_FV_T=np.asarray(state),
        observed_fine_FV_T=np.array(list(map(lambda x:float(F(x)),observed['fine_to_coarse_FV_T_exact']))),**diagnostic)
    save(out/'input-provenance.json',{'coarse_source_NPZ':'runs/reconnect-FV-contact-time/FV-T0.05.npz',
        'coarse_source_NPZ_sha256':digest(SOURCE/'FV-T0.05.npz'),
        'source_reference_case_sha256':digest(REFERENCE/'reference-K128-p90.json'),
        'source_original_member_records':read(SOURCE/'input-provenance.json')['original_source_member_records'],
        'source_comparison_inputs_sha256':digest(SOURCE/'comparison-inputs.json'),
        'source_coarse_states_sha256':digest(SOURCE/'comparison-states.json'),
        'new_derived_fine_inputs_not_original_sample_records':True,
        'output_NPZ_sha256':digest(out/'FV-N256-common-N128.npz'),
        'receipt_SHA256':{name:digest(path) for name,path in receipts.items()}})
    save(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,
        'thread_settings':{'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'},
        'metric_replay_standard_library_only':True,'full_solver_replay_requires_numpy':True})
    with (out/'requirements-solver.txt').open('x') as f:f.write('numpy=='+np.__version__+'\n')
    print(json.dumps({'results':result,'solver_controls':controls,'generation_controls':generation},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--output');group.add_argument('--replay');group.add_argument('--summary-replay')
    parser.add_argument('--tests');parser.add_argument('--test-log');args=parser.parse_args()
    if args.summary_replay:summary_replay(args.summary_replay)
    elif args.replay:replay(args.replay)
    else:
        if not args.tests or not args.test_log:parser.error('--tests and --test-log required for a new run')
        main(args.output,args.tests,args.test_log)
