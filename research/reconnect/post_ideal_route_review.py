"""Read-only route decision after the original-c0 ideal pair reference.

Only saved JSON/NPZ are read. No kernel, graph action, Fourier transform,
eigensolver, old proof, quadrature or propagator is imported or executed.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import math
import shutil
import sys
import numpy as np

from audit_route_decision import ROOT,read,save,digest

SOURCE=ROOT/'runs/reconnect-ideal-original-c0-pair'
GLOBAL=ROOT/'runs/reconnect-reconstructed-global-frozen'
REFERENCE=ROOT/'runs/reconnect-contact-finite-time-reference'
FV=ROOT/'runs/reconnect-FV-contact-time'


def record(a):
    return {'shape':list(a.shape),'dtype':a.dtype.str,'C_order_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def saved_degree_plan(cache,mass,row_L1):
    """Derive exact outgoing rates from saved coefficients, not new weights."""
    N=cache['N'];assert N==128
    g=tuple(map(F,cache['conductances_exact']));mapping=cache['coefficient_map'];offsets=cache['offsets_positive_half']
    degrees=[]
    for c in range(N):
        weight=sum((g[mapping[c][k]]+g[mapping[(c-dx)%N][k]] for k,(dx,dy) in enumerate(offsets)),F(0))
        degrees.append(weight/F(float(mass[c])))
    assert np.array_equal(np.tile([float(2*v) for v in degrees],N),row_L1)
    maximum=max(degrees);rate=(maximum.numerator+maximum.denominator-1)//maximum.denominator
    assert F(rate)>=maximum>0
    z=F(rate,20);tolerance=F(1,10**16)
    M=0
    while True:
        if M+2>z:
            # exp(-z)<=1 and all subsequent ratios <=z/(M+2).
            tail=z**(M+1)/math.factorial(M+1)/(1-z/F(M+2))
            if tail<=tolerance:break
        M+=1
    return {'origin':'exact degree sum of unchanged saved conductance cache; no new edge coefficient',
        'outgoing_rates_exact':list(map(str,degrees)),'maximum_outgoing_rate_exact':str(maximum),
        'uniformization_lambda_integer':rate,'Lambda_T_exact':str(z),
        'Poisson_max_degree_predeclared':M,'Poisson_terms_0_through_M':M+1,
        'Poisson_tail_bound_exact':str(tail),'Poisson_tail_tolerance_absolute_concentration_exact':str(tolerance),
        'tail_derivation':'exp(-z)<=1; first omitted term times geometric ratio bound, provided M+2>z',
        'state_bound_for_tail':'original mass fractions in [0,1]; operator/Lambda/M independent of c0 or PDE response',
        'mass_norm_tail_bound_exact':str(F(8,5)*tail),
        'mass_norm_factor_reason':'sum original m=5/2 <(8/5)^2',
        'numerical_action_budget_not_a_physical_acceptance_threshold':True,
        'probabilities_or_Poisson_weights_not_renormalized':True,
        'primary_transverse_modes':128,'distinct_real_modal_blocks_at_most':65,
        'normal_block_dimension':128,'full_16384_squared_transport_matrix_required':False,
        'future_operator_actions_executed_here':0}


def gate_spec(cost):
    return {'name':'one_fixed_saved_point_graph_T0_05_research_diagnostic','already_executed':False,
        'research_only_explicit_new_gate_not_physical_admission':True,
        'scope':'same original stationary periodic contact, existing point-path exchange only, one fixed endpoint; no moving trajectory or ORC inference',
        'N':128,'T_exact':'1/20','D_exact':str(F(.02)),'h_exact':str(F(.04)),
        'source_coefficient_cache':'runs/reconnect-reconstructed-global-frozen/coefficient-cache.json',
        'source_original_state':'runs/reconnect-ideal-original-c0-pair/original-inputs.npz',
        'source_PDE_reference':'runs/reconnect-contact-finite-time-reference/reference-T0.05.npz',
        'source_FV_context':'runs/reconnect-FV-contact-time/FV-T0.05.npz',
        'state':'full unchanged original binary fixed-material mass means c0; all IDs and masses unchanged',
        'new_candidate_edge_weights':0,'rebuild_kernel_or_density_paths':False,
        'main_action':'mass-weighted real Fourier blocks and raw symmetric eigensystem exponential action at exact T; apply to every original discrete transverse Fourier coefficient',
        'all_128_transverse_modes_retained_including_roundoff_size_modes':True,
        'no_single_ideal_cosine_replacement':True,
        'matrix_contract':'assemble from original shared g and masses; verify self-adjoint structure and initial action against saved exact global rate before propagation; no symmetrization repair',
        'phase_contract':'exact integer phase/conjugate identities may reuse blocks; do not average unequal matrices to enforce symmetry',
        'raw_eigenvalues_including_roundoff_zero_mode_retained':True,
        'real_state_output':'retain imaginary transform remainder as a solver control; no silent removal of a failed complex control',
        'independent_action':'uniformization on the same full original stencil, using P_Lambda=I+Q/Lambda and predeclared Poisson terms',
        'uniformization':cost,
        'no_Poisson_weight_sum_renormalization_no_concentration_clipping':True,
        'time_grid_or_BE_scan':False,
        'mandatory_implementation_checks':['small-grid full shared-stencil action versus reduced Fourier action',
            'all-mode reconstruction and raw mass similarity without mean replacement',
            'small independent dense matrix exponential versus modal action and uniformization',
            'invalid input/cache rejection and Poisson omitted-tail formula'],
        'real_case_controls':['coefficient cache and original state hashes',
            'initial modal action versus saved exact full-domain rate',
            'raw block conversion, eigen residual, orthogonality and transform roundtrip/imaginary residual',
            'modal versus independently streamed uniformization endpoint vector',
            'unadjusted inventory, extrema and resolved-material variance identities'],
        'finite_time_error':'norm_m(c_graph(T)-saved_PDE_material_means(T))/saved original initial fluctuation norm',
        'initial_fluctuation_norm_locked':0.19793792100879673,
        'FV_semidiscrete_error_context_locked':2.724389373882817e-5,
        'old_endpoint_harmonic_point_graph_T_result':'not available in reviewed evidence; do not claim finite-time improvement over it',
        'mixing_outputs':['original inventory and raw endpoint inventory drift','raw concentration min/max',
            'resolved-material variance E(T), I=E(T)/E(0), H=1-I, compared with the saved PDE material means',
            'graph instantaneous dissipation at T as graph quantity only; no invented continuous PDE dissipation reference'],
        'reference_bridge':'reuse saved truncation/precision/serialization/initial-mean bridge indicators and explicit cellwise-constant perturbation-lift assumption; do not change c0 or subtract bridge',
        'no_new_PDE_solve_or_spectral_reference_evaluation':True,
        'regions':'reuse original byte-identical near_contact/far_from_contact masks for endpoint error contributions',
        'interpretation':'compare endpoint error and differences from FV context against saved reference plus solver/tail indicators; no new absolute physical pass threshold',
        'stop_rules':['unsupported saved symmetry/cache, invalid inputs, failed operator-action or solver controls: retain failure and stop',
            'time action/control cannot distinguish a claimed result: preserve inconclusive outcome; do not extend modes, Poisson degree, times or states',
            'point result worse than FV or inaccurate: retain it; no h/grid/weight/state/threshold retuning',
            'even a good endpoint supports only this research diagnostic; no production backend, general trajectory evolution, Stage3 pass or ORC admission'],
        'wall_clock_seconds_cap':600,'peak_RSS_MiB_cap':512,'resource_limits_not_physical_thresholds':True,
        'candidate_graph_physical_time_cases_max':1,'new_global_2D_matrices':0,'new_full_propagator_matrices':0,
        'fluid_orc_enabled':False,'finite_node_candidate_admitted':False,'contact_transient_physically_admitted':False}


def main(output):
    manifest=read(SOURCE/'sha256.json')
    for rel,h in manifest.items():assert digest(ROOT/rel)==h,rel
    parent=read(SOURCE/'stage3-status.json')
    assert parent['recorded_checks_total']==423 and parent['next_work'].startswith('read-only interpretation')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',{'scope':'read-only post-ideal route interpretation and independently predeclared finite-time research gate',
        'parent_status_sha256':digest(SOURCE/'stage3-status.json'),'parent_manifest_entries_verified':len(manifest),
        'old_theory_and_numerical_operators_not_rerun':True,'no_gate_propagation_this_review':True})
    shutil.copyfile(SOURCE/'stage3-status.json',out/'source-stage3-status.json')
    pair=read(SOURCE/'results.json');global_result=read(GLOBAL/'results.json')
    residual=read(ROOT/'runs/reconnect-global-residual-review/residual-budget.json')
    reference_results=read(REFERENCE/'results.json');fv_results=read(FV/'results.json')
    with np.load(SOURCE/'original-inputs.npz',allow_pickle=False) as z:
        data={k:z[k].copy() for k in ['ids','mass','c0','positions','rho','near_contact','far_from_contact']}
    matches=[]
    with np.load(REFERENCE/'reference-T0.05.npz',allow_pickle=False) as z:
        for key in ['ids','mass','c0','positions','rho']:
            assert record(data[key])==record(z[key]),key
            matches.append({'member':key,'original_and_reference_same_bytes':record(data[key])})
        data['reference_cbar_T']=z['reference_cbar_T'].copy()
    with np.load(FV/'FV-T0.05.npz',allow_pickle=False) as z:
        for key in ['ids','mass','c0','positions','rho','reference_cbar_T']:assert record(data[key])==record(z[key]),key
        data['saved_FV_semidiscrete_T']=z['semidiscrete'].copy()
    with np.load(GLOBAL/'global-action.npz',allow_pickle=False) as z:row_L1=z['row_L1'].copy()
    cost=saved_degree_plan(read(GLOBAL/'coefficient-cache.json'),data['mass'],row_L1)
    gate=gate_spec(cost)
    ledger=[
        {'route':'ordinary central resistance FV','decision':'retain limited geometry baseline; saved finite-time result available, not generic point backend'},
        {'route':'mass-average resistance FV flux','decision':'retain experimental status; ideal material-pair reference is a different object'},
        {'route':'current nearest-point path exchange','decision':'retain fixed research model and saved improvement; no local-curvature, arbitrary geometry or production admission inferred'},
        {'route':'ideal material-pair reference','decision':'close 8/16 two-row diagnostic; no higher order/global expansion or installation as candidate'},
        {'route':'fitted common cross-phase scalar / four-mean linear repair','decision':'do not reopen closed restricted families or install negative offdiagonals'},
        {'route':'generic positive exchange','decision':'not ruled out; no feasibility claim or new formula inferred from current two-row cancellation'},
        {'route':'finite-time point-kernel research diagnostic','decision':'select one separate unexecuted T=1/20 gate because byte-identical independent reference already exists; no physical transient admission'},
        {'route':'fluid ORC / general ordered trajectory prediction','decision':'keep disabled/unadmitted; stationary one-case diagnostic cannot certify them'},
    ]
    decision={'interpretation_completed':True,'ideal_pair_cancellation_scope':'original two rows and original c0 relative to this intermediate; no global causal fractions',
        'global_frozen_improvement_kept':global_result['reconstructed_global_error'],
        'point_vs_ideal_pair_ordering_kept':{k:pair[k] for k in ['same_pair_point_total_error','same_pair_ideal_total_error_context_only','point_quadrature_projection_correlation']},
        'global_combined_quadrature_projection_split':'still not measured; no claim filled by two rows',
        'old_conditional_weak_theory_not_fixed_h_or_evolution_acceptance':True,
        'new_route_step':'one independent finite-time research gate, not automatic promotion from frozen or ideal results',
        'ordinary_FV_retained':True,'candidate_coefficients_frozen':True,'finite_node_candidate_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    status={'updated':'2026-11-01','timezone':'Etc/UTC','route_review_completed':True,
        'new_software_checks':0,'recorded_checks_total':423,'previous_checks_rerun':0,
        'new_candidate_weights':0,'new_kernel_or_path_evaluations':0,'new_operator_actions':0,'new_quadrature_evaluations':0,
        'new_Fourier_transforms_or_eigensolves':0,'new_candidate_graphs':0,'new_numerical_or_flow_cases':0,
        'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,'new_exact_algebra_certificate_groups':0,
        'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'contact_transient_physically_admitted':False,'next_finite_time_research_gate_predeclared':gate,
        'point_kernel_finite_time_result':'not_computed','ordinary_central_resistance_FV_baseline_retained':True,
        'mass_average_resistance_remains_experimental':True,'pdf_generated':False,'software_release_created':False,'published':False}
    save(out/'route-decision.json',decision);save(out/'route-ledger.json',ledger);save(out/'saved-operator-cost.json',cost)
    save(out/'next-gate-spec.json',gate);save(out/'stage3-status.json',status)
    for name,value in [('source-pair-results.json',pair),('source-global-results.json',global_result),('source-residual-budget.json',residual),
        ('source-time-reference-results.json',reference_results),('source-FV-time-results.json',fv_results)]:save(out/name,value)
    np.savez_compressed(out/'matched-time-reference-inputs.npz',**data)
    save(out/'input-provenance.json',{'source_point_inputs_sha256':digest(SOURCE/'original-inputs.npz'),
        'source_PDE_T_reference_sha256':digest(REFERENCE/'reference-T0.05.npz'),'source_FV_T_sha256':digest(FV/'FV-T0.05.npz'),
        'source_coefficient_cache_sha256':digest(GLOBAL/'coefficient-cache.json'),
        'matched_original_members':matches,'reference_and_FV_T_same_bytes':record(data['reference_cbar_T']),
        'matched_input_artifact_sha256':digest(out/'matched-time-reference-inputs.npz'),
        'array_records':{k:record(v) for k,v in data.items()},'no_interpolation_state_replacement_or_renormalization':True})
    for name in ['scipy','cached_global_exchange','ideal_material_pair_reference','cartesian_fv_modes',
        'oracle_path_reference','contact_spectral_reference']:
        assert name not in sys.modules,name
    print({'review_complete':True,'matched_original_state_and_T_reference':True,
        'Lambda':cost['uniformization_lambda_integer'],'Poisson_M':cost['Poisson_max_degree_predeclared'],
        'future_time_gate_executed':False,'recorded_checks_total':423,'fluid_orc_enabled':False},flush=True)


def replay(directory):
    out=Path(directory);p=read(out/'input-provenance.json')
    assert digest(out/'matched-time-reference-inputs.npz')==p['matched_input_artifact_sha256']
    with np.load(out/'matched-time-reference-inputs.npz',allow_pickle=False) as z:
        for k in z.files:assert record(z[k])==p['array_records'][k],k
    cost=read(out/'saved-operator-cost.json');gate=read(out/'next-gate-spec.json')
    assert gate==gate_spec(cost) and not gate['already_executed']
    z=F(cost['Lambda_T_exact']);M=cost['Poisson_max_degree_predeclared']
    tail=z**(M+1)/math.factorial(M+1)/(1-z/F(M+2))
    assert tail==F(cost['Poisson_tail_bound_exact'])<=F(1,10**16)
    assert read(out/'stage3-status.json')['point_kernel_finite_time_result']=='not_computed'
    assert read(out/'route-decision.json')['finite_node_candidate_admitted'] is False
    print('Route declarations, matched saved reference records and a priori time-action cost/tail plan replay; no propagation or operator executed.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output');parser.add_argument('--replay');args=parser.parse_args()
    if args.replay:replay(args.replay)
    elif args.output:main(args.output)
    else:parser.error('--output or --replay required')
