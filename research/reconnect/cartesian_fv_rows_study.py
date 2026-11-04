"""One predeclared two-row ordinary FV control on immutable saved states.

The original c0 is primary. Auxiliary material-mean quadratics and arithmetic
controls are recorded separately. No global matrix, time integration or new
reference integration is performed.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import numpy as np

from audit_route_decision import ROOT, digest, read, save
from cartesian_material_fv_rows import build_cartesian_material_rows, apply_material_rows


def array_record(array):
    return {'shape': list(array.shape), 'dtype': array.dtype.str,
            'C_order_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def norm(mass, vector):
    return float(np.sqrt(mass @ (vector * vector)))


def stencil_record(stencil):
    return {
        'point_count': stencil.point_count, 'shape': list(stencil.shape),
        'spacing_exact': list(map(str, stencil.spacing)), 'periods_exact': list(map(str, stencil.periods)),
        'thickness_exact': str(stencil.thickness), 'D_binary_input_exact': str(stencil.diffusivity),
        'model_prior': stencil.model_prior,
        'edges': [{'ids': [edge.left_id, edge.right_id], 'axis': edge.axis,
                   'conductance_exact': str(edge.conductance), 'conductance_display_kg_per_s': float(edge.conductance)}
                  for edge in stencil.edges],
        'rows': [{'target_id': row.target_id, 'mass_exact': str(row.mass),
                  'neighbors': [{'direction': direction, 'id': other, 'conductance_exact': str(g)}
                                for direction, other, g in row.neighbors]} for row in stencil.rows],
    }


def float_action(stencil, means):
    return np.array([math.fsum(float(g) * (float(means[other]) - float(means[row.target_id]))
                              for _, other, g in row.neighbors) / float(row.mass) for row in stencil.rows])


def evaluate(data, controls):
    targets = [63, 64]
    stencil = build_cartesian_material_rows(data['positions'], data['ids'], data['mass'], data['rho'],
                                            data['periods'], float(data['D']), targets)
    assert stencil.shape == (128, 128) and len(stencil.edges) == 7
    lookup = {int(identity): index for index, identity in enumerate(data['ids'])}
    target_indices = np.array([lookup[identity] for identity in targets], dtype=int)
    pair_mass = data['mass'][target_indices]
    local_ids = list(map(int, data['fv_local_ids']))
    initial = {identity: F(float(data['c0'][lookup[identity]])) for identity in local_ids}
    exact_initial_rates = apply_material_rows(stencil, initial)
    fv_rate = np.array([float(exact_initial_rates[identity]) for identity in targets])
    fv_float = float_action(stencil, initial)
    reference = data['reference_rate'][target_indices]
    kernel = data['kernel_rate'][target_indices]
    global_reference_norm = norm(data['mass'], data['reference_rate'])
    assert global_reference_norm > 0
    relative = lambda vector: norm(pair_mass, np.asarray(vector, dtype=float)) / global_reference_norm
    kernel_dot = []
    normal_kernel, transverse_kernel = [], []
    raw_row_norms = []
    for side, target in zip(['left', 'right'], target_indices):
        values = data[side + '_kernel_L_values']
        columns = data[side + '_kernel_columns']
        kernel_dot.append(-float(values @ data['c0'][columns]) / data['mass'][target])
        normal_kernel.append(float(data[side + '_saved_normal_kernel_rate']))
        transverse_kernel.append(-float(values @ data[side + '_transverse_mass_means']) / data['mass'][target])
        raw_row_norms.append(float(np.sum(np.abs(values))) / data['mass'][target])
    kernel_dot = np.array(kernel_dot)
    error_fv, error_kernel = fv_rate - reference, kernel - reference
    fv_error, kernel_error = relative(error_fv), relative(error_kernel)
    reference_face_vs_volume16 = relative(reference - data['volume16_rate'][target_indices])
    reference_volume8_vs_16 = relative(data['volume8_rate'][target_indices] - data['volume16_rate'][target_indices])
    fv_rounding = relative(fv_rate - fv_float)
    kernel_reduction = relative(kernel_dot - kernel)
    state_difference = float(controls['exact_c0_vs_Gauss16_max_absolute'])
    fv_l1 = np.array([float(2 * sum((g for _, _, g in row.neighbors), F(0)) / row.mass) for row in stencil.rows])
    input_FV_indicator = relative(fv_l1 * state_difference)
    input_kernel_indicator = relative(np.array(raw_row_norms) * state_difference)
    uncertainty_parts = {
        'saved_face_vs_volume16_pair_over_global_norm': reference_face_vs_volume16,
        'saved_volume8_vs_volume16_pair_over_global_norm': reference_volume8_vs_16,
        'FV_exact_binary_vs_floating_flux_pair_over_global_norm': fv_rounding,
        'saved_kernel_vs_cached_row_dot_pair_over_global_norm': kernel_reduction,
        'saved_mean_difference_FV_row_bound_indicator_over_global_norm': input_FV_indicator,
        'saved_mean_difference_kernel_row_bound_indicator_over_global_norm': input_kernel_indicator,
    }
    indicator = sum(uncertainty_parts.values())
    rows = []
    for i, identity in enumerate(targets):
        rows.append({'target_id': identity, 'original_mass_kg': float(pair_mass[i]),
                     'original_c0_mass_mean': float(data['c0'][target_indices[i]]),
                     'reference_rate_per_s': float(reference[i]),
                     'saved_kernel_rate_per_s': float(kernel[i]),
                     'FV_exact_binary_flux_rate_exact': str(exact_initial_rates[identity]),
                     'FV_rate_display_per_s': float(fv_rate[i]), 'FV_float_flux_control_per_s': float(fv_float[i]),
                     'FV_signed_error_per_s': float(error_fv[i]), 'kernel_signed_error_per_s': float(error_kernel[i]),
                     'FV_error_contribution_over_original_global_reference_norm': float(np.sqrt(pair_mass[i]) * abs(error_fv[i]) / global_reference_norm),
                     'kernel_error_contribution_over_original_global_reference_norm': float(np.sqrt(pair_mass[i]) * abs(error_kernel[i]) / global_reference_norm)})
    main = {
        'state': 'original saved c0; no point-state replacement', 'rows': rows,
        'original_all_cell_mass_weighted_PDE_rate_norm': global_reference_norm,
        'FV_two_row_error_contribution_over_global_norm': fv_error,
        'saved_kernel_two_row_error_contribution_over_global_norm': kernel_error,
        'error_contribution_reduction': kernel_error - fv_error,
        'error_contribution_reduction_percent_of_saved_kernel': 100 * (kernel_error - fv_error) / kernel_error,
        'FV_minus_kernel_action_pair_over_global_norm': relative(fv_rate - kernel),
        'these_are_not_all_row_global_error_norms': True,
        'empirical_uncertainty_components': uncertainty_parts,
        'sum_of_empirical_difference_indicators': indicator,
        'reduction_divided_by_empirical_indicator': (kernel_error - fv_error) / indicator if indicator else None,
        'indicator_is_not_a_certified_physical_error_bound_or_acceptance_threshold': True,
        'time_discretization_present': False,
    }
    auxiliary = {}
    for label, values, kernel_values in [
        ('normal', data['fv_normal_means_cached'], np.array(normal_kernel)),
        ('transverse', data['fv_transverse_mass_means'], np.array(transverse_kernel)),
    ]:
        state = {identity: F(float(value)) for identity, value in zip(local_ids, values)}
        exact_rates = apply_material_rows(stencil, state)
        rates = np.array([float(exact_rates[identity]) for identity in targets])
        physical_D = F(1, 50)
        reference_values = np.array([float(physical_D), float(physical_D)])
        exact_errors = [exact_rates[identity] - physical_D for identity in targets]
        denominator = norm(pair_mass, reference_values)
        auxiliary[label] = {
            'state_role': 'auxiliary true material-mean quadratic, never replaces c0',
            'FV_rates_exact_on_saved_binary_means': [str(exact_rates[identity]) for identity in targets],
            'FV_rates_display_per_s': rates.tolist(), 'kernel_raw_CSR_rates_per_s': kernel_values.tolist(),
            'PDE_rate_reference_exact': str(physical_D),
            'FV_exact_binary_minus_exact_PDE_rate_per_s': [
                {'exact': str(error), 'decimal_for_reading_only': float(error)} for error in exact_errors],
            'FV_pair_mass_weighted_relative_response_error': norm(pair_mass, np.array(list(map(float, exact_errors)))) / denominator,
            'FV_float_display_subtraction_relative_error_control': norm(pair_mass, rates - reference_values) / denominator,
            'kernel_pair_mass_weighted_relative_response_error': norm(pair_mass, kernel_values - reference_values) / denominator,
        }
    # Independent exact polynomial means: keep this control separate from cached binary means.
    dx, dy = stencil.spacing
    origin_y = F(float(data['positions'][target_indices[0], 1]))
    exact_controls = {}
    for label in ['normal', 'transverse']:
        state = {}
        for identity in local_ids:
            point = tuple(F(float(value)) for value in data['positions'][lookup[identity]])
            if label == 'normal':
                value = (point[0] - F(1, 2))**2 / 2 + dx**2 / 24
            else:
                delta = point[1] - origin_y
                if delta > F(1, 2): delta -= 1
                elif delta < -F(1, 2): delta += 1
                value = delta**2 / 2 + dy**2 / 24
            state[identity] = value
        rates = apply_material_rows(stencil, state)
        assert all(value == stencil.diffusivity for value in rates.values())
        exact_controls[label] = {'FV_exact_polynomial_mass_means_rates': [str(rates[identity]) for identity in targets],
                                 'equals_original_binary_D_exactly': True,
                                 'binary_D_minus_decimal_1_over_50_exact': str(stencil.diffusivity - F(1, 50))}
    main_fluxes = []
    for row in stencil.rows:
        main_fluxes.append({'target_id': row.target_id,
            'outward_contributions': [{'direction': direction, 'neighbor_id': other,
                'mass_rate_exact_kg_per_s': str(g * (initial[other] - initial[row.target_id]))}
                for direction, other, g in row.neighbors]})
    return stencil_record(stencil), main, auxiliary, exact_controls, main_fluxes


def main(output, tests, test_log):
    source = ROOT / 'runs/reconnect-saved-density-paths'
    entries = read(source / 'sha256.json')
    for rel, expected in entries.items():
        assert digest(ROOT / rel) == expected, rel
    source_status = source / 'stage3-status.json'
    protocol = read(source_status)['next_conditional_cartesian_FV_rows_predeclared']
    assert protocol['target_ids'] == [63, 64] and protocol['main_probe'] == 'original saved c0'
    assert not protocol['already_executed'] and protocol['new_global_graphs'] == 0
    snapshot = ROOT / protocol['source_snapshot']
    assert digest(snapshot) == protocol['source_snapshot_sha256']
    checks = read(tests)
    assert checks['successful'] and checks['checks_run'] == 6
    assert checks['failures'] == checks['errors'] == checks['skipped'] == 0
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'config.json', {
        'study': 'one conditional ordinary central-resistance FV two-row comparison',
        'protocol': protocol, 'protocol_source_sha256': digest(source_status),
        'source_manifest_entries_verified': len(entries),
        'main_FV_arithmetic_fixed_before_evaluation': 'exact Fraction edge action on original binary input D, masses, density and c0; float flux sum is control only',
        'main_kernel_value': 'original saved raw CSR kernel_rate; cached row dot is control only',
        'uncertainty_combination_fixed_before_evaluation': 'sum of saved two reference-rate differences, two arithmetic reductions, and two row-norm indicators times saved mean difference',
        'no_new_acceptance_threshold': True,
        'unit_checks_supplied': {'json_sha256': digest(tests), 'log_sha256': digest(test_log)},
    })
    for name, path in [('tests.json', tests), ('tests.log', test_log)]:
        with (out / name).open('xb') as f: f.write(Path(path).read_bytes())
    keys = ['ids', 'mass', 'c0', 'reference_rate', 'kernel_rate', 'volume8_rate', 'volume16_rate']
    with np.load(snapshot, allow_pickle=False) as z:
        data = {key: z[key].copy() for key in keys}
        data['positions'] = z['positions'][0].copy()
        data['rho'] = z['rho'][0].copy()
    original_records = {key: array_record(value) for key, value in data.items()}
    original_D = read(ROOT / 'runs/reconnect-contact-frozen/config.json')['protocol']['D_m2_per_s']
    assert original_D == .02
    data['D'], data['periods'] = np.array(original_D), np.array([1., 1.])
    stencil = build_cartesian_material_rows(data['positions'], data['ids'], data['mass'], data['rho'],
                                            data['periods'], original_D, [63, 64])
    assert stencil.spacing == (F(1, 128), F(1, 128)) and len(stencil.edges) == 7
    local_ids = sorted({row.target_id for row in stencil.rows} |
                       {other for row in stencil.rows for _, other, _ in row.neighbors})
    data['fv_local_ids'] = np.array(local_ids, dtype=np.int64)
    cache_path = ROOT / 'runs/reconnect-interface-pair/interface-pair.npz'
    cache_result = read(ROOT / 'runs/reconnect-interface-pair/results.json')
    normal_means = {}
    with np.load(cache_path, allow_pickle=False) as cache:
        for side in ['left', 'right']:
            columns = cache[side + '_row_columns'].copy()
            values = cache[side + '_saved_L_row'].copy()
            means = cache[side + '_probe_quadratic_mass_means'].copy()
            np.testing.assert_array_equal(cache[side + '_original_ids'], data['ids'][columns])
            np.testing.assert_array_equal(cache[side + '_original_c0'], data['c0'][columns])
            data[side + '_kernel_columns'], data[side + '_kernel_L_values'] = columns, values
            data[side + '_normal_means_cached'] = means
            data[side + '_saved_normal_kernel_rate'] = np.array(cache_result['rows'][side]['row']['raw_CSR_quadratic_response_per_s'])
            for identity, value in zip(data['ids'][columns], means):
                identity = int(identity)
                if identity in normal_means:
                    assert np.float64(normal_means[identity]).tobytes() == np.float64(value).tobytes()
                normal_means[identity] = value
    data['fv_normal_means_cached'] = np.array([normal_means[identity] for identity in local_ids])
    origin_y = F(float(data['positions'][63, 1]))
    def transverse_mean(point):
        delta = F(float(point[1])) - origin_y
        if delta > F(1, 2): delta -= 1
        elif delta < -F(1, 2): delta += 1
        return float(delta**2 / 2 + F(1, 128)**2 / 24)
    data['fv_transverse_mass_means'] = np.array([transverse_mean(data['positions'][identity]) for identity in local_ids])
    for side in ['left', 'right']:
        data[side + '_transverse_mass_means'] = np.array([transverse_mean(data['positions'][index]) for index in data[side + '_kernel_columns']])
    controls = read(ROOT / 'runs/reconnect-contact-frozen/results.json')['reference_controls']
    save(out / 'saved-reference-controls.json', controls)
    model, main_result, auxiliary, exact_controls, fluxes = evaluate(data, controls)
    np.savez_compressed(out / 'cartesian-fv-row-inputs.npz', **data)
    save(out / 'stencil.json', model)
    save(out / 'main-c0-comparison.json', main_result)
    save(out / 'auxiliary-comparison.json', auxiliary)
    save(out / 'exact-polynomial-controls.json', exact_controls)
    save(out / 'main-c0-local-fluxes.json', fluxes)
    save(out / 'input-provenance.json', {
        'source_snapshot': protocol['source_snapshot'], 'source_snapshot_sha256': digest(snapshot),
        'source_point_frame': 0, 'copied_source_members': original_records,
        'normal_probe_cache': str(cache_path.relative_to(ROOT)), 'normal_probe_cache_sha256': digest(cache_path),
        'normal_probe_results_sha256': digest(ROOT / 'runs/reconnect-interface-pair/results.json'),
        'current_input_artifact_sha256': digest(out / 'cartesian-fv-row-inputs.npz'),
    })
    assert original_records == {key: array_record(data[key]) for key in original_records}
    result = {
        'new_FV_row_unit_checks_passed': 6, 'previous_recorded_checks': 378, 'recorded_checks_total': 384,
        'previous_checks_rerun': 0, 'fixed_background_two_row_diagnostics': 1,
        'target_ids': [63, 64], 'main_state': 'original saved c0',
        'FV_two_row_error_contribution_over_original_global_norm': main_result['FV_two_row_error_contribution_over_global_norm'],
        'kernel_two_row_error_contribution_over_original_global_norm': main_result['saved_kernel_two_row_error_contribution_over_global_norm'],
        'main_error_reduction_percent_of_saved_kernel_pair_contribution': main_result['error_contribution_reduction_percent_of_saved_kernel'],
        'main_empirical_difference_indicator': main_result['sum_of_empirical_difference_indicators'],
        'new_local_FV_edge_conductances': len(stencil.edges), 'new_edge_weights': len(stencil.edges),
        'new_kernel_edge_weights': 0, 'new_candidate_graphs': 0, 'new_global_matrices': 0,
        'new_operator_quadrature_evaluations': 0, 'LP_solves': 0,
        'new_flow_solves': 0, 'new_reference_solves': 0,
        'new_candidate_transient_runs': 0, 'new_reference_transient_runs': 0,
        'new_exact_algebra_certificate_groups': 0,
        'finite_node_candidate_admitted': False, 'fluid_orc_enabled': False,
        'stage3_physical_acceptance': 'incomplete',
    }
    save(out / 'results.json', result)
    save(out / 'environment.json', {'python': platform.python_version(), 'numpy': np.__version__,
        'FV_adapter_dependencies': 'standard library only', 'IO_and_norms': 'NumPy',
        'network_used': False, 'environment_configuration_changed': False})
    print(json.dumps({'main': main_result, 'auxiliary': auxiliary, 'summary': result}, indent=2), flush=True)


def replay(directory):
    directory = Path(directory)
    provenance = read(directory / 'input-provenance.json')
    assert digest(directory / 'cartesian-fv-row-inputs.npz') == provenance['current_input_artifact_sha256']
    with np.load(directory / 'cartesian-fv-row-inputs.npz', allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    controls = read(directory / 'saved-reference-controls.json')
    current = evaluate(data, controls)
    names = ['stencil.json', 'main-c0-comparison.json', 'auxiliary-comparison.json',
             'exact-polynomial-controls.json', 'main-c0-local-fluxes.json']
    for value, name in zip(current, names):
        assert value == read(directory / name), name
    print('Two-row FV baseline, original c0 comparison and auxiliary controls replay exactly; no physical backend admission.', flush=True)


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
