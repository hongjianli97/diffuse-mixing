"""Preservation and exact-coefficient audit for the new continuous density bound."""
from pathlib import Path
import argparse
import json
from fractions import Fraction
from audit_route_decision import ROOT, digest, read, save, preservation as earlier_preservation
from density_weak_certificates import certificates


SOURCE = ROOT / 'runs/reconnect-route-decision'


def main(output):
    out = Path(output).resolve()
    assert not (out / 'sha256.json').exists(), 'never rewrite sealed evidence'
    protocol_source = SOURCE / 'stage3-status.json'
    protocol = read(protocol_source)['next_density_closure_weak_perturbation_predeclared']
    config = read(out / 'config.json')
    assert config['protocol'] == protocol and config['protocol_source_sha256'] == digest(protocol_source)
    assert config['mass_measure_unchanged'] and config['rho_hat_used_only_for_path_resistance']
    result = read(out / 'proof-certificates.json')
    assert result == certificates() == read(out / 'standalone-certificates.json')
    assert result['verified_exact_algebra_groups'] == 3 and result['recorded_checks_total'] == 363
    assert result['new_numerical_or_flow_tests'] == 0
    assert all(result[key] == 0 for key in [
        'new_candidate_graphs', 'new_edge_weights', 'new_operator_quadrature_evaluations',
        'LP_solves', 'new_candidate_transient_runs', 'new_reference_transient_runs'])
    assert not result['fluid_orc_enabled'] and not result['finite_node_convergence_proved']
    # Cite the sealed moment value; do not execute its proof script.
    old_moment = read(ROOT / 'runs/reconnect-oracle-weak/proof-certificates.json')
    moment = old_moment['certificate_groups']['unchanged_kernel_radial_and_half_mixed_moments'][
        'radial_moments_after_dividing_h_power']['2']['exact']
    assert Fraction(moment) == 2
    assert result['certificate_groups']['original_flat_contact_new_bound_constants'][
        'cited_existing_kernel_second_moment']['exact'] == moment

    preserved = earlier_preservation()
    entries = read(SOURCE / 'sha256.json')
    for rel, expected in entries.items():
        assert digest(ROOT / rel) == expected, rel
    preserved['formal_scientific_manifests_verified'].append({
        'manifest': str((SOURCE / 'sha256.json').relative_to(ROOT)),
        'manifest_sha256': digest(SOURCE / 'sha256.json'), 'entries_verified': len(entries),
    })
    assert len(preserved['formal_scientific_manifests_verified']) == 54
    save(out / 'preservation.json', preserved)
    roles = {
        'research/reconnect/CANDIDATE_ROUTE_DECISION.md': 'route and scope; immutable prior decision',
        'research/reconnect/CONDITIONAL_DENSITY_CLOSURE.md': 'predeclared allowed input and model contract',
        'research/reconnect/ORACLE_WEAK_INTERFACE_CONSISTENCY.md': 'cited existing continuous flat-interface weak budget',
        'research/reconnect/MASS_AVERAGE_PROJECTION_WEAK_BUDGET.md': 'cited existing projection budget; not a new discrete convergence claim',
        'research/reconnect/ORACLE_INTERFACE_SECOND_ORDER.md': 'known strong curvature defect retained',
        'runs/reconnect-oracle-weak/proof-certificates.json': 'cited second radial kernel moment only',
        'runs/reconnect-route-decision/stage3-status.json': 'predeclared next theorem scope, pending in source',
    }
    indexed = []
    for rel, role in roles.items():
        assert rel in entries and digest(ROOT / rel) == entries[rel], rel
        indexed.append({'path': rel, 'sha256': entries[rel], 'role': role})
    save(out / 'evidence-index.json', {
        'source_manifest': str((SOURCE / 'sha256.json').relative_to(ROOT)),
        'source_manifest_sha256': digest(SOURCE / 'sha256.json'), 'sources': indexed,
    })
    status = read(out / 'stage3-status.json')
    assert status['density_weak_perturbation_proved_under_stated_assumptions']
    assert status['nearest_point_L1_bound_proved_under_stated_assumptions']
    assert status['combined_continuous_weak_bound_scope'] == 'existing flat phase-constant transmission model only'
    assert not status['finite_node_convergence_proved'] and not status['finite_node_candidate_admitted']
    assert status['recorded_checks_total'] == 363 and not status['fluid_orc_enabled']
    assert status['exact_algebra_certificate_groups'] == 3
    next_plan = status['next_point_pair_quadrature_bridge_review_predeclared']
    assert next_plan['theory_only'] and not next_plan['new_graphs_or_numerical_cases']
    assert not next_plan['proof_completed']
    audit = {
        'predeclared_density_weak_scope_and_input_contract_verified': True,
        'new_three_exact_algebra_groups_recomputed': True,
        'independent_standard_library_process_output_identical': True,
        'old_kernel_moment_cited_and_hash_verified_not_reproved': True,
        'old_interface_and_projection_proofs_not_executed': True,
        'original_mass_measure_and_state_retained': True,
        'continuous_bound_not_promoted_to_point_edge_or_transient_bound': True,
        'mathematical_integral_and_regularity_proofs_are_in_report_not_automatically_verified': True,
        'new_numerical_or_flow_tests': 0, 'exact_algebra_certificate_groups': 3,
        'recorded_checks_total': 363, 'prior_formal_manifests_verified': 54,
        'fluid_orc_enabled': False,
    }
    save(out / 'proof-audit.json', audit)
    print(json.dumps({'original_files_verified': 782,
        'historical_attachments_verified': 15, **audit}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
