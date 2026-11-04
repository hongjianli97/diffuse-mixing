"""Source preservation and new exact algebra audit; no graph or old proof runs."""
from pathlib import Path
import argparse
import json
from audit_route_decision import ROOT, digest, read, save, preservation as earlier_preservation
from point_quadrature_certificates import certificates


SOURCE = ROOT / 'runs/reconnect-density-weak'


def main(output):
    out = Path(output).resolve()
    assert not (out / 'sha256.json').exists(), 'never rewrite sealed evidence'
    source_status = SOURCE / 'stage3-status.json'
    protocol = read(source_status)['next_point_pair_quadrature_bridge_review_predeclared']
    config = read(out / 'config.json')
    assert config['protocol'] == protocol and config['protocol_source_sha256'] == digest(source_status)
    assert config['true_cells_are_conditional_analysis_prior_not_candidate_input']
    result = read(out / 'proof-certificates.json')
    assert result == certificates() == read(out / 'standalone-certificates.json')
    assert result['verified_exact_algebra_groups'] == 3 and result['recorded_checks_total'] == 363
    assert result['conditional_point_pair_quadrature_bound_proved']
    assert result['conditional_fixed_test_discrete_weak_consistency_proved']
    assert not result['general_point_backend_admitted'] and not result['evolution_convergence_proved']
    for key in ['new_numerical_or_flow_tests', 'new_candidate_graphs', 'new_edge_weights',
                'new_operator_quadrature_evaluations', 'LP_solves',
                'new_candidate_transient_runs', 'new_reference_transient_runs']:
        assert result[key] == 0
    assert result['stage3_physical_acceptance'] == 'incomplete' and not result['fluid_orc_enabled']

    preserved = earlier_preservation()
    for source in [ROOT / 'runs/reconnect-route-decision', SOURCE]:
        entries = read(source / 'sha256.json')
        for rel, expected in entries.items():
            assert digest(ROOT / rel) == expected, rel
        preserved['formal_scientific_manifests_verified'].append({
            'manifest': str((source / 'sha256.json').relative_to(ROOT)),
            'manifest_sha256': digest(source / 'sha256.json'), 'entries_verified': len(entries),
        })
    assert len(preserved['formal_scientific_manifests_verified']) == 55
    save(out / 'preservation.json', preserved)
    roles = {
        'research/reconnect/CONDITIONAL_DENSITY_CLOSURE.md': 'unchanged point input and density closure contract',
        'research/reconnect/MASS_AVERAGE_PROJECTION_WEAK_BUDGET.md': 'existing projection bound cited, not rerun',
        'research/reconnect/DENSITY_RECONSTRUCTION_WEAK_BOUND.md': 'existing continuous density bound and trace limitation cited',
        'research/reconnect/ORACLE_WEAK_INTERFACE_CONSISTENCY.md': 'existing flat continuous weak budget cited',
        'research/reconnect/ORACLE_INTERFACE_SECOND_ORDER.md': 'strong local obstruction remains',
        'research/reconnect/positive_kernel_2d.py': 'unchanged original B_h definition; not executed',
        'runs/reconnect-density-weak/stage3-status.json': 'predeclared point-pair quadrature review',
    }
    entries = read(SOURCE / 'sha256.json')
    index = []
    for rel, role in roles.items():
        assert rel in entries and digest(ROOT / rel) == entries[rel]
        index.append({'path': rel, 'sha256': entries[rel], 'role': role})
    save(out / 'evidence-index.json', {
        'source_manifest': str((SOURCE / 'sha256.json').relative_to(ROOT)),
        'source_manifest_sha256': digest(SOURCE / 'sha256.json'), 'sources': index,
    })
    status = read(out / 'stage3-status.json')
    assert status['conditional_fixed_test_discrete_weak_consistency_proved']
    assert status['true_material_cell_locality_required_not_inferred_from_fill_distance']
    assert not status['path_kernel_superiority_proved'] and not status['finite_node_candidate_admitted']
    assert not status['fluid_orc_enabled'] and status['recorded_checks_total'] == 363
    assert status['exact_algebra_certificate_groups'] == 3
    next_plan = status['next_density_path_primitive_implementation_predeclared']
    assert not next_plan['primitive_already_implemented']
    assert not next_plan['new_graphs_or_electrical_conductances']
    assert {'mass', 'concentration', 'true_material_faces', 'F_C', 'reference_PDE'} <= set(next_plan['forbidden_primitive_inputs'])
    audit = {
        'predeclared_theory_only_bridge_scope_verified': True,
        'three_new_exact_algebra_groups_recomputed': True,
        'independent_standard_library_process_output_identical': True,
        'old_kernel_moments_projection_and_weak_proofs_not_executed': True,
        'material_average_state_not_replaced_by_point_concentration': True,
        'locality_is_explicit_conditional_prior_not_recovered_geometry': True,
        'weak_consistency_does_not_admit_backend_or_prove_path_superiority': True,
        'analytic_and_geometric_proofs_in_report_not_automatically_verified': True,
        'new_numerical_or_flow_tests': 0, 'exact_algebra_certificate_groups': 3,
        'recorded_checks_total': 363, 'prior_formal_manifests_verified': 55,
        'fluid_orc_enabled': False,
    }
    save(out / 'proof-audit.json', audit)
    print(json.dumps({'original_files_verified': 782, 'historical_attachments_verified': 15, **audit}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
