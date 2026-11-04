"""Exactly three predeclared density-path observations from a sealed snapshot.

Loads only point positions, fixed IDs and given point density as primitive
inputs. The piecewise analytic density is an extra-information reference only.
No graph, conductance, PDE rate or evolution is computed.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np

from audit_route_decision import ROOT, digest, read, save
from density_path_study import encoded_result
from endpoint_density_path import localized_inverse_density_path


def reference_inverse_density(start_a, displacement_a):
    """Original analytic periodic rho=1 on [0,1/2), rho=4 on [1/2,1)."""
    cuts = {F(0), F(1)}
    if displacement_a:
        for integer in range(-2, 5):
            parameter = (F(integer, 2) - start_a) / displacement_a
            if 0 < parameter < 1:
                cuts.add(parameter)
    cuts = sorted(cuts)
    segments = []
    total = F(0)
    for left, right in zip(cuts, cuts[1:]):
        normal = (start_a + (left + right) * displacement_a / 2) % 1
        rho = F(1) if normal < F(1, 2) else F(4)
        total += (right - left) / rho
        segments.append({'start': str(left), 'stop': str(right), 'rho': str(rho)})
    return total, segments


def array_record(array):
    return {'shape': list(array.shape), 'dtype': array.dtype.str,
            'C_order_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def exact_integer_array(values, scale):
    result = []
    for value in values:
        scaled = value * scale
        if scaled.denominator != 1:
            raise ValueError('saved certificate encoding would round an exact distance')
        result.append(scaled.numerator)
    return np.asarray(result, dtype=np.int64)


def main(output, tests, test_log):
    source = ROOT / 'runs/reconnect-density-path'
    manifest = read(source / 'sha256.json')
    for rel, expected in manifest.items():
        assert digest(ROOT / rel) == expected, rel
    status_path = source / 'stage3-status.json'
    protocol = read(status_path)['next_saved_density_paths_predeclared']
    assert protocol['paths_to_check'] == 3 and not protocol['saved_input_paths_already_checked']
    assert protocol['normal_column_pairs'] == [[63, 64], [63, 65], [127, 0]]
    assert protocol['transverse_column'] == 0 and not protocol['new_graphs_or_conductances']
    snapshot = ROOT / protocol['source_snapshot']
    assert digest(snapshot) == protocol['source_snapshot_sha256']
    checks = read(tests)
    assert checks['successful'] and checks['checks_run'] == 6
    assert checks['failures'] == checks['errors'] == checks['skipped'] == 0
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'config.json', {
        'study': 'certified endpoint-ball localization and three predeclared saved paths',
        'protocol': protocol, 'protocol_source_sha256': digest(status_path),
        'source_manifest_entries_verified': len(manifest), 'frame': 0,
        'periods': [1, 1], 'primitive_inputs': ['positions', 'ids', 'rho', 'periods', 'start_id', 'end_id'],
        'analytic_density_is_reference_only': True,
        'new_graphs_or_conductances': False,
        'unit_checks_supplied': {'json_sha256': digest(tests), 'log_sha256': digest(test_log)},
    })
    for name, path in [('tests.json', tests), ('tests.log', test_log)]:
        with (out / name).open('xb') as f:
            f.write(Path(path).read_bytes())
    with np.load(snapshot, allow_pickle=False) as archive:
        positions = archive['positions'][0].copy()
        identities = archive['ids'].copy()
        rho = archive['rho'][0].copy()
    N = 128
    assert positions.shape == (N * N, 2) and identities.shape == rho.shape == (N * N,)
    originals = {key: array_record(array) for key, array in [('positions', positions), ('ids', identities), ('rho', rho)]}
    points = tuple(tuple(F(float(value)) for value in point) for point in positions)
    point_ids = tuple(int(value) for value in identities)
    point_rho = tuple(F(float(value)) for value in rho)
    lattice = 2 * N
    integer_positions = np.asarray([exact_integer_array(point, lattice) for point in points], dtype=np.int64)
    for index, point in enumerate(integer_positions):
        row, column = divmod(index, N)
        assert tuple(point) == (2 * column + 1, 2 * row + 1), 'predeclared point layout differs'
    assert all(value == (1 if point[0] < F(1, 2) else 4) for point, value in zip(points, point_rho))
    denominator = lattice**2
    saved = {'positions': positions, 'ids': identities, 'rho': rho, 'periods': np.array([1., 1.]),
             'position_lattice_numerators': integer_positions,
             'position_lattice_denominator': np.array(lattice, dtype=np.int64),
             'distance_squared_denominator': np.array(denominator, dtype=np.int64)}
    observations = []
    for path_index, (left_column, right_column) in enumerate(protocol['normal_column_pairs']):
        left = protocol['transverse_column'] * N + left_column
        right = protocol['transverse_column'] * N + right_column
        result = localized_inverse_density_path(points, point_ids, point_rho, (F(1), F(1)), point_ids[left], point_ids[right])
        selection, path = result.selection, result.path
        expected, reference_segments = reference_inverse_density(points[left][0], path.displacement[0])
        keep = np.zeros(len(point_ids), dtype=bool)
        keep[list(selection.retained_indices)] = True
        prefix = f'path{path_index}_'
        saved[prefix + 'keep'] = keep
        saved[prefix + 'distance2_start_numerators'] = exact_integer_array(selection.distance2_to_start, denominator)
        saved[prefix + 'distance2_end_numerators'] = exact_integer_array(selection.distance2_to_end, denominator)
        scaled_radius = selection.radius_squared * denominator
        assert scaled_radius.denominator == 1
        observations.append({
            'normal_columns': [left_column, right_column], 'transverse_column': 0,
            'start_input_index': left, 'end_input_index': right,
            'start_id': point_ids[left], 'end_id': point_ids[right],
            'start_position': [str(value) for value in points[left]],
            'end_position': [str(value) for value in points[right]],
            'global_nodes_checked': len(point_ids), 'retained_indices': list(selection.retained_indices),
            'retained_ids': list(selection.retained_ids), 'retained_nodes': len(selection.retained_indices),
            'radius_squared_exact': str(selection.radius_squared),
            'radius_squared_numerator': scaled_radius.numerator,
            'distance_squared_denominator': denominator,
            'primitive_output': encoded_result(path),
            'extra_information_analytic_reference_segments': reference_segments,
            'reference_mean_inverse_density_exact': str(expected),
            'primitive_minus_reference_exact': str(path.mean_inverse_density - expected),
            'matches_reference_exactly': path.mean_inverse_density == expected,
        })
        print(f'path {point_ids[left]} -> {point_ids[right]}: retained {len(selection.retained_indices)}/{len(point_ids)}, mean={path.mean_inverse_density}, reference={expected}', flush=True)
    assert len(observations) == 3 and all(item['matches_reference_exactly'] for item in observations)
    assert originals == {key: array_record(array) for key, array in [('positions', positions), ('ids', identities), ('rho', rho)]}
    assert digest(snapshot) == protocol['source_snapshot_sha256']
    np.savez_compressed(out / 'saved-point-inputs.npz', **saved)
    save(out / 'observations.json', observations)
    save(out / 'input-provenance.json', {
        'source_snapshot': protocol['source_snapshot'], 'source_snapshot_sha256_before_and_after': digest(snapshot),
        'source_frame': 0, 'source_members_copied_without_numeric_change': originals,
        'mass_c0_graph_and_PDE_arrays_not_passed_to_primitive': True,
        'saved_point_inputs_sha256': digest(out / 'saved-point-inputs.npz'),
    })
    previous = read(status_path)['recorded_checks_total']
    assert previous == 372
    result = {
        'endpoint_ball_filter_implemented': True, 'endpoint_ball_lemma_proved_in_report': True,
        'new_localization_unit_checks_passed': checks['checks_run'], 'previous_recorded_checks': previous,
        'recorded_checks_total': previous + checks['checks_run'], 'previous_checks_rerun': 0,
        'saved_path_observations': 3, 'saved_path_observations_counted_as_new_tests': False,
        'retained_node_counts': [item['retained_nodes'] for item in observations],
        'mean_inverse_density_exact': [item['primitive_output']['mean_inverse_density_exact'] for item in observations],
        'all_three_reference_integrals_match_exactly': True,
        'global_nearest_field_preserved_by_certified_exclusion': True,
        'full_saved_cloud_lower_envelope_runs': 0,
        'new_physical_or_flow_tests': 0, 'new_exact_algebra_certificate_groups': 0,
        'new_candidate_graphs': 0, 'new_edge_weights': 0, 'new_operator_quadrature_evaluations': 0,
        'LP_solves': 0, 'new_candidate_transient_runs': 0, 'new_reference_transient_runs': 0,
        'finite_node_candidate_admitted': False, 'fluid_orc_enabled': False,
        'stage3_physical_acceptance': 'incomplete',
    }
    save(out / 'results.json', result)
    save(out / 'environment.json', {'python': platform.python_version(), 'numpy': np.__version__,
        'primitive_and_filter_dependencies': 'standard library only', 'snapshot_IO_dependency': 'NumPy',
        'network_used': False, 'environment_configuration_changed': False})
    print(json.dumps(result, indent=2), flush=True)


def replay(directory):
    """Replay the three records from the small input-only artifact, no history."""
    directory = Path(directory)
    observations = read(directory / 'observations.json')
    provenance = read(directory / 'input-provenance.json')
    assert digest(directory / 'saved-point-inputs.npz') == provenance['saved_point_inputs_sha256']
    with np.load(directory / 'saved-point-inputs.npz', allow_pickle=False) as archive:
        positions, ids, rho, periods = archive['positions'], archive['ids'], archive['rho'], archive['periods']
    for item in observations:
        result = localized_inverse_density_path(positions, ids, rho, periods, item['start_id'], item['end_id'])
        assert encoded_result(result.path) == item['primitive_output']
        assert list(result.selection.retained_ids) == item['retained_ids']
    print('All three saved point-input path records replay exactly; this is not a physical backend acceptance.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--tests')
    parser.add_argument('--test-log')
    parser.add_argument('--replay')
    args = parser.parse_args()
    if args.replay:
        replay(args.replay)
    else:
        if not all([args.output, args.tests, args.test_log]):
            parser.error('--output, --tests and --test-log are required')
        main(args.output, args.tests, args.test_log)
