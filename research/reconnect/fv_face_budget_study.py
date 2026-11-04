"""Predeclared signed face budget using only immutable saved FV outputs.

NumPy is used only to copy the old NPZ face references in main(). The exact
bookkeeping and a standalone replay require the standard library alone.
"""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform

from audit_route_decision import ROOT, digest, read, save
from saved_fv_face_budget import signed_face_budget, FACES


def encode(value):
    if isinstance(value, F):
        return {'exact': str(value), 'decimal_for_reading_only': float(value)}
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def decode_parameters(raw):
    result = {}
    for key, value in raw.items():
        if key in ['fv_outward', 'reference_faces']:
            result[key] = [{name: F(number) for name, number in row.items()} for row in value]
        elif isinstance(value, list):
            result[key] = [F(number) for number in value]
        else:
            result[key] = F(value)
    return result


def array_record(array):
    return {'shape': list(array.shape), 'dtype': array.dtype.str,
            'C_order_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def summary_from_budget(budget):
    components = budget['signed_rate_components']
    matrix = budget['normalized_Gram_matrix']
    return {
        'common_interface_flux_defect_kg_per_s': float(budget['shared_interface']['common_interface_flux_defect']),
        'common_interface_flux_relative_error_percent': 100 * float(budget['shared_interface']['common_flux_relative_error']),
        'interface_rate_components_per_s': list(map(float, components['interface'])),
        'other_three_faces_rate_components_per_s': list(map(float, components['other_three_faces'])),
        'reference_rounding_components_per_s': list(map(float, components['reference_rounding'])),
        'display_bridge_components_per_s': list(map(float, components['display_bridge'])),
        'component_norms_over_original_global_PDE_norm': {
            label: math.sqrt(float(matrix[index][index])) for index, label in enumerate(budget['component_labels'])},
        'interface_other_cross_term_in_normalized_squared_error': float(budget['interface_and_other_cross_term']),
        'sum_all_signed_Gram_entries': float(budget['Gram_sum']),
        'original_published_pair_error_contribution': float(budget['published_pair_error_norm']),
        'exact_row_closures': [str(row['published_error_budget_closure']) for row in budget['rows']],
        'exact_Gram_closure': str(budget['exact_Gram_closure']),
        'component_norms_are_not_additive_error_percentages': True,
    }


def main(output, tests, test_log):
    import numpy as np
    source = ROOT / 'runs/reconnect-cartesian-fv-rows'
    manifest = read(source / 'sha256.json')
    for rel, expected in manifest.items():
        assert digest(ROOT / rel) == expected, rel
    status_path = source / 'stage3-status.json'
    protocol = read(status_path)['next_saved_FV_face_budget_predeclared']
    assert protocol['target_ids'] == [63, 64] and not protocol['already_executed']
    assert protocol['new_edge_weights'] == protocol['new_candidate_graphs'] == 0
    snapshot = ROOT / protocol['source_snapshot']
    assert digest(snapshot) == protocol['source_snapshot_sha256']
    tests_result = read(tests)
    assert tests_result['successful'] and tests_result['checks_run'] == 4
    assert tests_result['errors'] == tests_result['failures'] == tests_result['skipped'] == 0
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'config.json', {
        'study': 'signed saved FV face-flux error budget, no transport changes',
        'protocol': protocol, 'protocol_source_sha256': digest(status_path),
        'source_manifest_entries_verified': len(manifest),
        'arithmetic': 'Fraction exact values for all source binary inputs and saved exact flux fractions',
        'reference_faces_are_extra_information_not_runtime_input': True,
        'source_FV_model_or_weight_code_executed': False,
        'unit_checks_supplied': {'json_sha256': digest(tests), 'log_sha256': digest(test_log)},
    })
    for name, path in [('tests.json', tests), ('tests.log', test_log)]:
        with (out / name).open('xb') as f: f.write(Path(path).read_bytes())
    main_comparison = read(source / 'main-c0-comparison.json')
    fluxes = read(source / 'main-c0-local-fluxes.json')
    source_rows = main_comparison['rows']
    targets = protocol['target_ids']
    assert [row['target_id'] for row in source_rows] == [row['target_id'] for row in fluxes] == targets
    fields = {'right': 'face_right_q_integral', 'left': 'face_left_q_integral',
              'top': 'face_top_q_integral', 'bottom': 'face_bottom_q_integral'}
    with np.load(snapshot, allow_pickle=False) as z:
        ids = z['ids']
        indices = [int(np.flatnonzero(ids == target)[0]) for target in targets]
        raw_reference = {field: z[field][indices].copy() for field in protocol['reference_fields']}
        raw_reference.update({'mass': z['mass'][indices].copy(), 'c0': z['c0'][indices].copy(),
                              'reference_rate': z['reference_rate'][indices].copy(),
                              'ids': ids[indices].copy()})
    for index, row in enumerate(source_rows):
        for field, source_key in [('mass', 'original_mass_kg'), ('c0', 'original_c0_mass_mean'),
                                  ('reference_rate', 'reference_rate_per_s')]:
            assert np.float64(raw_reference[field][index]).tobytes() == np.float64(row[source_key]).tobytes()
    shared_reference_bytes_equal = raw_reference['face_right_q_integral'][0].tobytes() == raw_reference['face_left_q_integral'][1].tobytes()
    np.savez_compressed(out / 'selected-reference-faces.npz', **raw_reference)
    parameters = {
        'masses': [str(F(float(value))) for value in raw_reference['mass']],
        'fv_outward': [{item['direction']: item['mass_rate_exact_kg_per_s'] for item in row['outward_contributions']} for row in fluxes],
        'reference_faces': [{face: str(F(float(raw_reference[field][index]))) for face, field in fields.items()} for index in range(2)],
        'saved_reference_rates': [str(F(row['reference_rate_per_s'])) for row in source_rows],
        'published_FV_rates': [str(F(row['FV_rate_display_per_s'])) for row in source_rows],
        'published_errors': [str(F(row['FV_signed_error_per_s'])) for row in source_rows],
        'global_reference_norm': str(F(main_comparison['original_all_cell_mass_weighted_PDE_rate_norm'])),
        'published_pair_error_norm': str(F(main_comparison['FV_two_row_error_contribution_over_global_norm'])),
    }
    packet = {'target_ids': targets, 'parameters': parameters,
              'source_exact_FV_rates': [row['FV_exact_binary_flux_rate_exact'] for row in source_rows],
              'source_snapshot_sha256': digest(snapshot),
              'source_main_comparison_sha256': digest(source / 'main-c0-comparison.json'),
              'source_FV_fluxes_sha256': digest(source / 'main-c0-local-fluxes.json'),
              'shared_reference_bytes_equal': shared_reference_bytes_equal}
    save(out / 'budget-inputs.json', packet)
    args = decode_parameters(parameters)
    before = deepcopy(args)
    budget = signed_face_budget(**args)
    assert args == before
    assert shared_reference_bytes_equal and budget['shared_interface']['reference_shared_face_equal']
    assert budget['shared_interface']['FV_shared_face_equal']
    for row, expected in zip(budget['rows'], packet['source_exact_FV_rates']):
        assert row['FV_rate_exact_from_saved_fluxes'] == F(expected)
    save(out / 'exact-budget.json', encode(budget))
    save(out / 'budget-summary.json', summary_from_budget(budget))
    for original_name, saved_name in [('stencil.json', 'source-stencil.json'),
                                      ('main-c0-comparison.json', 'source-main-c0-comparison.json'),
                                      ('main-c0-local-fluxes.json', 'source-main-c0-local-fluxes.json')]:
        with (out / saved_name).open('xb') as f: f.write((source / original_name).read_bytes())
    save(out / 'input-provenance.json', {
        'source_snapshot': protocol['source_snapshot'], 'source_snapshot_sha256': digest(snapshot),
        'selected_source_member_records': {key: array_record(value) for key, value in raw_reference.items()},
        'selected_reference_NPZ_sha256': digest(out / 'selected-reference-faces.npz'),
        'source_stencil_sha256': digest(source / 'stencil.json'),
        'source_c0_comparison_sha256': digest(source / 'main-c0-comparison.json'),
        'source_FV_fluxes_sha256': digest(source / 'main-c0-local-fluxes.json'),
        'raw_signed_zero_bits_preserved_in_NPZ': True,
        'reference_face_values_averaged_or_repaired': False,
    })
    previous = read(status_path)['recorded_checks_total']
    assert previous == 384
    result = {
        'new_budget_software_checks_passed': 4, 'previous_recorded_checks': previous,
        'recorded_checks_total': previous + tests_result['checks_run'], 'previous_checks_rerun': 0,
        'existing_two_row_budget_diagnostics': 1,
        'shared_reference_face_byte_equality': shared_reference_bytes_equal,
        'exact_published_error_row_closures': [str(row['published_error_budget_closure']) for row in budget['rows']],
        'exact_Gram_closure': str(budget['exact_Gram_closure']),
        'common_interface_flux_defect_exact_kg_per_s': str(budget['shared_interface']['common_interface_flux_defect']),
        'component_norms_not_additive_causal_shares': True,
        'new_edge_weights': 0, 'new_candidate_graphs': 0, 'new_global_matrices': 0,
        'new_operator_quadrature_evaluations': 0, 'new_reference_integrations': 0, 'LP_solves': 0,
        'new_flow_solves': 0, 'new_reference_solves': 0,
        'new_candidate_transient_runs': 0, 'new_reference_transient_runs': 0,
        'new_exact_algebra_certificate_groups': 0, 'FV_weight_or_state_code_executed': False,
        'finite_node_candidate_admitted': False, 'fluid_orc_enabled': False,
        'stage3_physical_acceptance': 'incomplete',
    }
    save(out / 'results.json', result)
    save(out / 'environment.json', {'python': platform.python_version(), 'numpy_for_source_IO_only': np.__version__,
        'budget_and_replay_dependencies': 'standard library only', 'network_used': False,
        'environment_configuration_changed': False})
    print(json.dumps({'summary': summary_from_budget(budget), 'result': result}, indent=2), flush=True)


def replay(directory):
    directory = Path(directory)
    packet = read(directory / 'budget-inputs.json')
    budget = signed_face_budget(**decode_parameters(packet['parameters']))
    assert encode(budget) == read(directory / 'exact-budget.json')
    assert summary_from_budget(budget) == read(directory / 'budget-summary.json')
    for row, expected in zip(budget['rows'], packet['source_exact_FV_rates']):
        assert row['FV_rate_exact_from_saved_fluxes'] == F(expected)
    print('Saved two-row signed face budget and all exact closures replay identically; no transport rebuilt.', flush=True)


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
            parser.error('--output, --tests and --test-log required')
        main(args.output, args.tests, args.test_log)
