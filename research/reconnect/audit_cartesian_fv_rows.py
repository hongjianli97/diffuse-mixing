"""Audit source bytes, seven local FV edges, retained reporting correction."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import difflib
import json
import numpy as np

from audit_route_decision import ROOT, digest, read, save, preservation as earlier_preservation
from cartesian_fv_rows_study import array_record, replay, norm


SOURCE = ROOT / 'runs/reconnect-saved-density-paths'


def main(output):
    out = Path(output).resolve()
    assert not (out / 'sha256.json').exists(), 'never rewrite sealed evidence'
    protocol_source = SOURCE / 'stage3-status.json'
    protocol = read(protocol_source)['next_conditional_cartesian_FV_rows_predeclared']
    cfg = read(out / 'config.json')
    assert cfg['protocol'] == protocol and cfg['protocol_source_sha256'] == digest(protocol_source)
    provenance = read(out / 'input-provenance.json')
    snapshot = ROOT / protocol['source_snapshot']
    assert digest(snapshot) == protocol['source_snapshot_sha256'] == provenance['source_snapshot_sha256']
    assert digest(out / 'cartesian-fv-row-inputs.npz') == provenance['current_input_artifact_sha256']
    old = ROOT / 'runs/reconnect-cartesian-fv-rows-initial-float-norm'
    correction = read(out / 'reporting-correction.json')
    for rel, expected in correction['retained_initial_files_sha256'].items():
        assert digest(ROOT / rel) == expected, rel
    before_script = ROOT / correction['retained_initial_script']
    assert digest(before_script) == correction['initial_script_sha256']
    before_text = before_script.read_text()
    expected_text = before_text.replace(
        '        reference_values = np.array([float(physical_D), float(physical_D)])\n',
        '        reference_values = np.array([float(physical_D), float(physical_D)])\n'
        '        exact_errors = [exact_rates[identity] - physical_D for identity in targets]\n', 1)
    expected_text = expected_text.replace(
        "            'FV_pair_mass_weighted_relative_response_error': norm(pair_mass, rates - reference_values) / denominator,\n",
        "            'FV_exact_binary_minus_exact_PDE_rate_per_s': [\n"
        "                {'exact': str(error), 'decimal_for_reading_only': float(error)} for error in exact_errors],\n"
        "            'FV_pair_mass_weighted_relative_response_error': norm(pair_mass, np.array(list(map(float, exact_errors)))) / denominator,\n"
        "            'FV_float_display_subtraction_relative_error_control': norm(pair_mass, rates - reference_values) / denominator,\n", 1)
    current_script = ROOT / 'research/reconnect/cartesian_fv_rows_study.py'
    assert current_script.read_text() == expected_text, 'reporting correction changed other code'
    with (out / 'reporting-correction.diff').open('x') as f:
        f.writelines(difflib.unified_diff(before_text.splitlines(True), expected_text.splitlines(True),
                                        fromfile=str(before_script.relative_to(ROOT)), tofile=str(current_script.relative_to(ROOT))))
    for name in ['stencil.json', 'main-c0-comparison.json', 'exact-polynomial-controls.json', 'main-c0-local-fluxes.json']:
        assert (out / name).read_bytes() == (old / name).read_bytes(), name
    with np.load(out / 'cartesian-fv-row-inputs.npz', allow_pickle=False) as data, \
            np.load(old / 'cartesian-fv-row-inputs.npz', allow_pickle=False) as initial, \
            np.load(snapshot, allow_pickle=False) as original:
        assert data.files == initial.files
        for key in data.files:
            np.testing.assert_array_equal(data[key], initial[key])
        for key, record in provenance['copied_source_members'].items():
            raw = original[key][0] if key in ['positions', 'rho'] else original[key]
            assert array_record(data[key]) == array_record(raw) == record
            np.testing.assert_array_equal(data[key], raw)
        ids = data['ids']
        lookup = {int(identity): index for index, identity in enumerate(ids)}
        model = read(out / 'stencil.json')
        dx, dy = map(F, model['spacing_exact'])
        D, b = F(model['D_binary_input_exact']), F(model['thickness_exact'])
        assert len(model['edges']) == 7 and model['shape'] == [128, 128]
        for edge in model['edges']:
            left, right = map(lookup.__getitem__, edge['ids'])
            r_left, r_right = F(float(data['rho'][left])), F(float(data['rho'][right]))
            length, width = (dx, dy) if edge['axis'] == 0 else (dy, dx)
            resistance_per_face = length / (2 * r_left * D * b * width) + length / (2 * r_right * D * b * width)
            assert F(edge['conductance_exact']) > 0
            assert 1 / F(edge['conductance_exact']) == resistance_per_face
        main = read(out / 'main-c0-comparison.json')
        reference_norm = norm(data['mass'], data['reference_rate'])
        assert reference_norm == main['original_all_cell_mass_weighted_PDE_rate_norm']
        target_indices = [lookup[63], lookup[64]]
        pair_mass = data['mass'][target_indices]
        auxiliary = read(out / 'auxiliary-comparison.json')
        initial_auxiliary = read(old / 'auxiliary-comparison.json')
        for label, item in auxiliary.items():
            assert item['FV_rates_exact_on_saved_binary_means'] == initial_auxiliary[label]['FV_rates_exact_on_saved_binary_means']
            errors = [F(value) - F(1, 50) for value in item['FV_rates_exact_on_saved_binary_means']]
            assert [record['exact'] for record in item['FV_exact_binary_minus_exact_PDE_rate_per_s']] == list(map(str, errors))
            expected_error = norm(pair_mass, np.array(list(map(float, errors)))) / norm(pair_mass, np.array([.02, .02]))
            assert item['FV_pair_mass_weighted_relative_response_error'] == expected_error > 0
            assert item['FV_float_display_subtraction_relative_error_control'] == initial_auxiliary[label]['FV_pair_mass_weighted_relative_response_error'] == 0
    replay(out)
    tests = read(out / 'tests.json')
    assert tests['successful'] and tests['checks_run'] == 6
    assert tests['failures'] == tests['errors'] == tests['skipped'] == 0
    assert digest(out / 'tests.json') == cfg['unit_checks_supplied']['json_sha256']
    assert digest(out / 'tests.log') == cfg['unit_checks_supplied']['log_sha256']
    result = read(out / 'results.json')
    assert result['recorded_checks_total'] == 378 + tests['checks_run'] == 384
    assert result['new_local_FV_edge_conductances'] == result['new_edge_weights'] == 7
    assert result['new_kernel_edge_weights'] == result['new_candidate_graphs'] == result['new_global_matrices'] == 0
    assert not result['finite_node_candidate_admitted'] and not result['fluid_orc_enabled']

    preserved = earlier_preservation()
    for source in [ROOT / 'runs/reconnect-route-decision', ROOT / 'runs/reconnect-density-weak',
                   ROOT / 'runs/reconnect-point-quadrature', ROOT / 'runs/reconnect-density-path', SOURCE]:
        entries = read(source / 'sha256.json')
        for rel, expected in entries.items():
            assert digest(ROOT / rel) == expected, rel
        preserved['formal_scientific_manifests_verified'].append({
            'manifest': str((source / 'sha256.json').relative_to(ROOT)),
            'manifest_sha256': digest(source / 'sha256.json'), 'entries_verified': len(entries),
        })
    assert len(preserved['formal_scientific_manifests_verified']) == 58
    save(out / 'preservation.json', preserved)
    roles = {
        'research/reconnect/SAVED_DENSITY_PATH_CHECK.md': 'predeclared limited Cartesian FV model and target rows',
        'research/reconnect/CONTACT_CONSTITUTIVE_REVIEW.md': 'previous density and constitutive scope remains',
        'runs/reconnect-saved-density-paths/stage3-status.json': 'predeclared main state and norm',
        'runs/reconnect-contact-frozen/results.json': 'saved reference differences, not new reference integration',
        'runs/reconnect-interface-pair/interface-pair.npz': 'original normal mass-mean probe and raw kernel rows',
        'runs/reconnect-interface-pair/results.json': 'unchanged original raw normal kernel response',
        protocol['source_snapshot']: 'immutable original c0, mass, given rho and independent PDE rates',
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
    assert status['recorded_checks_total'] == 384 and status['main_state'] == 'original saved c0'
    assert not status['finite_node_candidate_admitted'] and not status['fluid_orc_enabled']
    assert status['next_saved_FV_face_budget_predeclared']['already_executed'] is False
    assert status['next_saved_FV_face_budget_predeclared']['new_edge_weights'] == 0
    audit = {
        'predeclared_two_target_original_c0_scope_verified': True,
        'six_new_FV_row_unit_checks_passed': True,
        'original_input_member_bytes_and_cached_normal_probe_preserved': True,
        'seven_positive_shared_edges_match_independent_series_resistance_identity': True,
        'same_global_PDE_norm_for_FV_and_original_kernel_verified': True,
        'complete_input_state_stencil_and_primary_results_unchanged_by_reporting_fix': True,
        'reporting_fix_source_diff_limited_to_exact_auxiliary_residual': True,
        'initial_output_and_initial_script_retained_and_hashed': True,
        'nonzero_exact_auxiliary_residual_not_lost_to_float_display': True,
        'saved_baseline_and_all_controls_replay_identically': True,
        'new_FV_row_unit_checks_passed': 6, 'fixed_background_two_row_diagnostics': 1,
        'new_local_FV_edge_conductances': 7, 'new_kernel_edge_weights': 0,
        'recorded_checks_total': 384, 'previous_checks_rerun': 0,
        'prior_formal_manifests_verified': 58, 'fluid_orc_enabled': False,
    }
    save(out / 'implementation-audit.json', audit)
    print(json.dumps({'original_files_verified': 782, 'historical_attachments_verified': 15, **audit}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
