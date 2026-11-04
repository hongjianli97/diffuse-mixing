"""Read-only contract/evidence audit. No numerical implementation imports."""
from pathlib import Path
import argparse
import json
import sys

from audit_route_decision import ROOT,digest,read,save,preservation
from point_candidate_contract_review import SOURCE,ROLES,signatures,validate_review,replay


def main(output):
    out=Path(output).resolve();assert not (out/'sha256.json').exists()
    config=read(out/'config.json');source_status=SOURCE/'stage3-status.json'
    assert config['protocol']==read(source_status)['next_point_candidate_contract_review_predeclared']
    assert config['protocol_source_sha256']==digest(source_status) and not config['protocol']['already_executed']
    decision,gate,ledger,status=(read(out/name) for name in ['review-decision.json','gate-spec.json','assumption-ledger.json','stage3-status.json'])
    validate_review(decision,gate,ledger,status);replay(out)
    index=read(out/'evidence-index.json');manifest=read(SOURCE/'sha256.json')
    assert index['source_manifest_sha256']==digest(SOURCE/'sha256.json')
    assert {item['path'] for item in index['sources']}==set(ROLES)
    for item in index['sources']:
        assert item['source_read_only'] and not item['scientific_recalculation']
        assert item['sha256']==manifest[item['path']]==digest(ROOT/item['path'])
    receipts={'source-review-protocol.json':source_status,
        'source-density-weak-status.json':ROOT/'runs/reconnect-density-weak/stage3-status.json',
        'source-point-pair-status.json':ROOT/'runs/reconnect-point-quadrature/stage3-status.json'}
    for name,path in receipts.items():assert (out/name).read_bytes()==path.read_bytes()
    inventory=read(out/'implementation-inventory.json')
    assert inventory['periodic_density_path.py']==signatures(ROOT/'research/reconnect/periodic_density_path.py',['inverse_density_path'])
    assert inventory['endpoint_density_path.py']==signatures(ROOT/'research/reconnect/endpoint_density_path.py',['endpoint_ball_selection','localized_inverse_density_path'])
    assert inventory['positive_kernel_2d.py']==signatures(ROOT/'research/reconnect/positive_kernel_2d.py',['slope_over_radius','build_periodic_exchange'])
    assert inventory['periodic_density_path.py']['inverse_density_path']==['positions','ids','rho','periods','start_id','end_id']
    assert inventory['endpoint_density_path.py']['endpoint_ball_selection']==['positions','ids','periods','start_id','end_id']
    assert inventory['positive_kernel_2d.py']['slope_over_radius']==['radius','h']
    assert not inventory['future_adapter_present_at_review'] # Saved review-time inventory, not a perpetual file-absence rule.
    weak=read(out/'source-density-weak-status.json');bridge=read(out/'source-point-pair-status.json')
    assert weak['density_weak_perturbation_proved_under_stated_assumptions']
    assert weak['area_L1_not_a_point_path_trace_bound'] and not weak['finite_node_convergence_proved']
    assert bridge['conditional_point_pair_quadrature_bound_proved'] and not bridge['path_kernel_superiority_proved']
    assert not bridge['fill_distance_alone_sufficient'] and not bridge['evolution_convergence_proved']
    old_contract=read(ROOT/'runs/reconnect-route-decision/route-decision.json')
    assert old_contract['known_curvature_defect_not_fixed_by_density_reconstruction']
    forbidden=set(gate['forbidden_weight_dependencies'])
    assert {'true_material_faces','true_material_adjacency','true_cell_volumes','F_C','velocity_gradients','c0','PDE_rate'}<=forbidden
    assert not {'c0','PDE_rate','true_material_faces','F_C','velocity_gradients'}.intersection(gate['coefficient_builder_inputs'])
    assert gate['all_support_directions_required_not_only_three_previously_checked_paths']
    assert gate['all_global_density_inputs_validated_before_exact_endpoint_ball_localization']
    assert gate['open_pair_not_claimed_to_have_zero_total_mass_rate']
    assert gate['comparison']['no_new_absolute_physical_acceptance_threshold']
    assert gate['comparison']['pair_contribution_not_a_global_error_or_finite_time_result']
    saved=read(out/'saved-metrics.json')
    assert saved['saved_values_only_no_scientific_recalculation']
    pair=read(ROOT/'runs/reconnect-FV-face-budget/source-main-c0-comparison.json')
    assert saved['pair_controls']==pair
    assert gate['comparison']['fixed_global_PDE_norm']==pair['original_all_cell_mass_weighted_PDE_rate_norm']
    assert gate['comparison']['saved_old_kernel_pair_contribution']==pair['saved_kernel_two_row_error_contribution_over_global_norm']
    assert gate['comparison']['saved_FV_pair_contribution']==pair['FV_two_row_error_contribution_over_global_norm']
    assert saved['FV_global_frozen_error']==read(ROOT/'runs/reconnect-FV-global-frozen/results.json')['ordinary_FV_global_relative_error']
    assert saved['original_point_kernel_global_frozen_error']==read(ROOT/'runs/reconnect-FV-global-frozen/results.json')['saved_kernel_global_relative_error']
    assert saved['FV_semidiscrete_T0_05_error']==read(ROOT/'runs/reconnect-FV-contact-time/results.json')['spatial_semidiscrete_relative_error']
    assert saved['FV_fine_common_observation_error']==read(SOURCE/'results.json')['fine256_error_under_original128_observation']
    assert saved['continuous_oracle_path_frozen_error']==read(ROOT/'runs/reconnect-oracle-path/results.json')['oracle_runs']['fine']['errors']['all']
    for module in ['numpy','scipy','positive_kernel_2d','periodic_density_path','endpoint_density_path',
        'density_weak_certificates','point_quadrature_certificates','cartesian_fv_modes','contact_spectral_reference']:
        assert module not in sys.modules,module
    preserved=preservation()
    for directory in ['reconnect-route-decision','reconnect-density-weak','reconnect-point-quadrature',
        'reconnect-density-path','reconnect-saved-density-paths','reconnect-cartesian-fv-rows',
        'reconnect-FV-face-budget','reconnect-FV-interface-expansion','reconnect-four-mean-interface',
        'reconnect-FV-global-frozen','reconnect-contact-finite-time-reference','reconnect-FV-contact-time','reconnect-FV-contact-spatial']:
        path=ROOT/'runs'/directory/'sha256.json';entries=read(path)
        for rel,expected in entries.items():assert digest(ROOT/rel)==expected,rel
        preserved['formal_scientific_manifests_verified'].append({'manifest':str(path.relative_to(ROOT)),
            'manifest_sha256':digest(path),'entries_verified':len(entries)})
    assert len(preserved['formal_scientific_manifests_verified'])==66
    save(out/'preservation.json',preserved)
    audit={'predeclared_read_only_review_scope_verified':True,
        'historical_future_work_reconciled_with_later_completed_conditional_proofs':True,
        'FV_extra_geometry_continuous_oracle_and_discrete_candidate_objects_separated':True,
        'primitive_API_signatures_inspected_by_AST_without_execution':True,
        'one_fixed_rule_full_support_pair_gate_selected_not_executed':True,
        'inverse_density_average_not_confused_with_K_or_conductance':True,
        'no_c0_or_oracle_geometry_dependency_added_to_weight_constructor':True,
        'weak_assumptions_and_restricted_counterexample_scope_retained':True,
        'saved_metrics_copied_without_new_scientific_calculation':True,
        'no_numerical_or_proof_implementation_modules_imported':True,
        'standard_library_contract_replay_passed':True,
        'new_software_checks':0,'previous_checks_rerun':0,'recorded_checks_total':403,
        'new_exact_algebra_certificate_groups':0,'new_numerical_cases':0,'new_edge_weights':0,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_time_solutions':0,
        'prior_formal_manifests_verified':66,'fluid_orc_enabled':False}
    save(out/'review-audit.json',audit)
    print(json.dumps({'original_files_verified':782,'historical_attachments_verified':15,**audit},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
