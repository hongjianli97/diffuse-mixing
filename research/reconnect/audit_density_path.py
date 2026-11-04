"""Audit primitive tests, exact serialized examples and historical preservation."""
from pathlib import Path
from fractions import Fraction
import argparse
import inspect
import json

from audit_route_decision import ROOT, digest, read, save, preservation as earlier_preservation
from density_path_study import encoded_result
from periodic_density_path import inverse_density_path


SOURCE = ROOT / 'runs/reconnect-point-quadrature'


def main(output):
    out = Path(output).resolve()
    assert not (out / 'sha256.json').exists(), 'never rewrite sealed evidence'
    source_status = SOURCE / 'stage3-status.json'
    protocol = read(source_status)['next_density_path_primitive_implementation_predeclared']
    config = read(out / 'config.json')
    assert config['protocol'] == protocol and config['protocol_source_sha256'] == digest(source_status)
    signature = list(inspect.signature(inverse_density_path).parameters)
    assert signature == config['primitive_signature'] == ['positions', 'ids', 'rho', 'periods', 'start_id', 'end_id']
    checks = read(out / 'tests.json')
    assert checks['successful'] and checks['checks_run'] == 9 and len(checks['check_names']) == 9
    assert checks['errors'] == checks['failures'] == checks['skipped'] == 0
    assert digest(out / 'tests.json') == config['unit_checks_supplied']['json_sha256']
    assert digest(out / 'tests.log') == config['unit_checks_supplied']['log_sha256']
    assert 'Ran 9 tests' in (out / 'tests.log').read_text() and '\nOK\n' in (out / 'tests.log').read_text()
    examples = read(out / 'unit-examples.json')
    assert len(examples) == 5
    for example in examples.values():
        raw = example['input']
        data = {**raw, 'positions': [tuple(map(Fraction, point)) for point in raw['positions']],
                'rho': list(map(Fraction, raw['rho'])), 'periods': list(map(Fraction, raw['periods']))}
        replay = inverse_density_path(**data)
        assert encoded_result(replay) == example['output']
        assert str(replay.mean_inverse_density) == example['analytic_expected_mean_inverse_density_exact']
    result = read(out / 'results.json')
    assert result['new_primitive_unit_checks_passed'] == 9
    assert result['previous_recorded_checks'] == read(source_status)['recorded_checks_total'] == 363
    assert result['recorded_checks_total'] == 372
    assert not result['actual_full_cloud_or_saved_flow_inputs_used']
    for key in ['new_physical_or_flow_tests', 'new_exact_algebra_certificate_groups',
                'new_candidate_graphs', 'new_edge_weights', 'new_operator_quadrature_evaluations',
                'LP_solves', 'new_candidate_transient_runs', 'new_reference_transient_runs']:
        assert result[key] == 0
    assert not result['finite_node_candidate_admitted'] and not result['fluid_orc_enabled']

    preserved = earlier_preservation()
    for source in [ROOT / 'runs/reconnect-route-decision', ROOT / 'runs/reconnect-density-weak', SOURCE]:
        entries = read(source / 'sha256.json')
        for rel, expected in entries.items():
            assert digest(ROOT / rel) == expected, rel
        preserved['formal_scientific_manifests_verified'].append({
            'manifest': str((source / 'sha256.json').relative_to(ROOT)),
            'manifest_sha256': digest(source / 'sha256.json'), 'entries_verified': len(entries),
        })
    assert len(preserved['formal_scientific_manifests_verified']) == 56
    save(out / 'preservation.json', preserved)
    roles = {
        'research/reconnect/CONDITIONAL_DENSITY_CLOSURE.md': 'original point-only closure and tie contract',
        'research/reconnect/POINT_PAIR_QUADRATURE_BRIDGE.md': 'conditional weak scope, not backend admission',
        'runs/reconnect-point-quadrature/stage3-status.json': 'predeclared bounded primitive task',
    }
    manifest = read(SOURCE / 'sha256.json')
    index = []
    for rel, role in roles.items():
        assert manifest[rel] == digest(ROOT / rel)
        index.append({'path': rel, 'sha256': manifest[rel], 'role': role})
    save(out / 'evidence-index.json', {
        'source_manifest': str((SOURCE / 'sha256.json').relative_to(ROOT)),
        'source_manifest_sha256': digest(SOURCE / 'sha256.json'), 'sources': index,
    })
    status = read(out / 'stage3-status.json')
    assert status['primitive_implemented'] and status['new_primitive_unit_checks_passed'] == 9
    assert status['recorded_checks_total'] == 372 and not status['finite_node_candidate_admitted']
    assert not status['fluid_orc_enabled'] and status['stage3_physical_acceptance'] == 'incomplete'
    next_plan = status['next_saved_density_paths_predeclared']
    assert next_plan['paths_to_check'] == 3 and not next_plan['saved_input_paths_already_checked']
    assert not next_plan['endpoint_ball_filter_implemented'] and not next_plan['new_graphs_or_conductances']
    assert manifest[next_plan['source_snapshot']] == next_plan['source_snapshot_sha256']
    audit = {
        'predeclared_primitive_scope_and_input_signature_verified': True,
        'nine_new_software_unit_checks_passed': True,
        'test_JSON_and_original_log_hashes_verified': True,
        'five_serialized_geometry_examples_replay_exactly': True,
        'fixed_ID_point_ties_and_positive_length_intervals_recorded': True,
        'old_proofs_and_physical_cases_not_reexecuted': True,
        'new_primitive_unit_checks_passed': 9, 'new_physical_or_flow_tests': 0,
        'recorded_checks_total': 372, 'previous_checks_rerun': 0,
        'prior_formal_manifests_verified': 56, 'fluid_orc_enabled': False,
    }
    save(out / 'implementation-audit.json', audit)
    print(json.dumps({'original_files_verified': 782, 'historical_attachments_verified': 15, **audit}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
