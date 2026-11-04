"""Record the bounded point-only primitive and its passed geometry unit checks."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import argparse
import inspect
import json
import platform

from audit_route_decision import ROOT, digest, read, save
from periodic_density_path import inverse_density_path
from test_periodic_density_path import fixtures


def encoded_input(data):
    return {
        'positions': [[str(Fraction(value)) for value in point] for point in data['positions']],
        'ids': list(data['ids']), 'rho': [str(Fraction(value)) for value in data['rho']],
        'periods': [str(Fraction(value)) for value in data['periods']],
        'start_id': data['start_id'], 'end_id': data['end_id'],
    }


def encoded_result(result):
    return {
        'start_id': result.start_id, 'end_id': result.end_id,
        'displacement': [str(value) for value in result.displacement],
        'mean_inverse_density_exact': str(result.mean_inverse_density),
        'mean_inverse_density_decimal_for_reading_only': float(result.mean_inverse_density),
        'intervals': [{'start': str(part.start), 'stop': str(part.stop),
                       'source_id': part.source_id, 'density': str(part.density)} for part in result.intervals],
        'breakpoint_owners': [{'parameter': str(parameter), 'source_id': identity}
                              for parameter, identity in result.breakpoint_owners],
    }


def main(output, tests, test_log):
    source = ROOT / 'runs/reconnect-point-quadrature'
    manifest = read(source / 'sha256.json')
    for rel, expected in manifest.items():
        assert digest(ROOT / rel) == expected, rel
    status_path = source / 'stage3-status.json'
    protocol = read(status_path)['next_density_path_primitive_implementation_predeclared']
    assert not protocol['primitive_already_implemented']
    assert not protocol['new_graphs_or_electrical_conductances']
    checks = read(tests)
    assert checks['successful'] and checks['checks_run'] == 9
    assert checks['failures'] == checks['errors'] == checks['skipped'] == 0
    assert len(checks['check_names']) == 9
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'config.json', {
        'study': 'predeclared periodic nearest-point inverse-density path primitive',
        'protocol': protocol, 'protocol_source_sha256': digest(status_path),
        'source_manifest_entries_verified': len(manifest),
        'scope': 'small geometry unit inputs only; no physical case, candidate graph or conductance',
        'primitive_signature': list(inspect.signature(inverse_density_path).parameters),
        'exact_binary_float_interpretation': True,
        'intervals_are_density_interpolation_not_material_cells': True,
        'unit_checks_supplied': {'json_sha256': digest(tests), 'log_sha256': digest(test_log)},
    })
    with (out / 'tests.json').open('xb') as f:
        f.write(Path(tests).read_bytes())
    with (out / 'tests.log').open('xb') as f:
        f.write(Path(test_log).read_bytes())
    expected_means = {'collinear': Fraction(15, 32), 'off_axis': Fraction(37, 64),
                      'periodic_corner': Fraction(9, 16), 'interval_tie': Fraction(25, 96),
                      'point_tie': Fraction(5, 8)}
    examples = {}
    for label, data in fixtures().items():
        before = deepcopy(data)
        result = inverse_density_path(**data)
        assert data == before and result.mean_inverse_density == expected_means[label]
        examples[label] = {'input': encoded_input(data), 'output': encoded_result(result),
                           'analytic_expected_mean_inverse_density_exact': str(expected_means[label])}
    save(out / 'unit-examples.json', examples)
    previous = read(source / 'stage3-status.json')['recorded_checks_total']
    assert previous == 363
    result = {
        'primitive_implemented': True, 'new_primitive_unit_checks_passed': checks['checks_run'],
        'previous_recorded_checks': previous, 'recorded_checks_total': previous + checks['checks_run'],
        'previous_checks_rerun': 0, 'new_physical_or_flow_tests': 0,
        'new_exact_algebra_certificate_groups': 0,
        'unit_examples_recorded': len(examples),
        'independent_oracle': checks['independent_oracle'],
        'no_tolerance_or_interval_renormalization': True,
        'actual_full_cloud_or_saved_flow_inputs_used': False,
        'new_candidate_graphs': 0, 'new_edge_weights': 0, 'new_operator_quadrature_evaluations': 0,
        'LP_solves': 0, 'new_candidate_transient_runs': 0, 'new_reference_transient_runs': 0,
        'finite_node_candidate_admitted': False, 'fluid_orc_enabled': False,
        'stage3_physical_acceptance': 'incomplete',
    }
    save(out / 'results.json', result)
    save(out / 'environment.json', {'python': platform.python_version(), 'dependencies': 'standard library only',
        'network_used': False, 'environment_configuration_changed': False})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tests', required=True)
    parser.add_argument('--test-log', required=True)
    args = parser.parse_args()
    main(args.output, args.tests, args.test_log)
