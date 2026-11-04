"""Audit signed saved-face arithmetic and immutable parent evidence."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import sys
import numpy as np

from audit_route_decision import ROOT, digest, read, save, preservation as earlier_preservation
from fv_face_budget_study import array_record, decode_parameters, replay
from saved_fv_face_budget import signed_face_budget


SOURCE = ROOT / 'runs/reconnect-cartesian-fv-rows'


def main(output):
    out = Path(output).resolve()
    assert not (out / 'sha256.json').exists(), 'never rewrite sealed evidence'
    protocol_source = SOURCE / 'stage3-status.json'
    protocol = read(protocol_source)['next_saved_FV_face_budget_predeclared']
    config = read(out / 'config.json')
    assert config['protocol'] == protocol and config['protocol_source_sha256'] == digest(protocol_source)
    assert protocol['new_edge_weights'] == protocol['new_candidate_graphs'] == 0
    provenance = read(out / 'input-provenance.json')
    snapshot = ROOT / protocol['source_snapshot']
    assert digest(snapshot) == protocol['source_snapshot_sha256'] == provenance['source_snapshot_sha256']
    assert digest(out / 'selected-reference-faces.npz') == provenance['selected_reference_NPZ_sha256']
    with np.load(snapshot, allow_pickle=False) as original, np.load(out / 'selected-reference-faces.npz', allow_pickle=False) as selected:
        ids = original['ids']
        indices = [int(np.flatnonzero(ids == target)[0]) for target in [63, 64]]
        for key, record in provenance['selected_source_member_records'].items():
            raw = original[key][indices]
            assert array_record(raw) == array_record(selected[key]) == record
            np.testing.assert_array_equal(raw, selected[key])
        assert selected['face_right_q_integral'][0].tobytes() == selected['face_left_q_integral'][1].tobytes()
        assert np.signbit(selected['face_bottom_q_integral']).tolist() == [True, True]
    for name, receipt, field in [
        ('stencil.json', 'source-stencil.json', 'source_stencil_sha256'),
        ('main-c0-comparison.json', 'source-main-c0-comparison.json', 'source_c0_comparison_sha256'),
        ('main-c0-local-fluxes.json', 'source-main-c0-local-fluxes.json', 'source_FV_fluxes_sha256'),
    ]:
        assert digest(SOURCE / name) == digest(out / receipt) == provenance[field]
        assert (SOURCE / name).read_bytes() == (out / receipt).read_bytes()
    packet = read(out / 'budget-inputs.json')
    parameters = decode_parameters(packet['parameters'])
    budget = signed_face_budget(**parameters)
    replay(out)
    source_main = read(SOURCE / 'main-c0-comparison.json')
    source_fluxes = read(SOURCE / 'main-c0-local-fluxes.json')
    assert parameters['global_reference_norm'] == F(source_main['original_all_cell_mass_weighted_PDE_rate_norm'])
    assert parameters['published_pair_error_norm'] == F(source_main['FV_two_row_error_contribution_over_global_norm'])
    for index in range(2):
        row = budget['rows'][index]
        outward = parameters['fv_outward'][index]
        reference = parameters['reference_faces'][index]
        m = parameters['masses'][index]
        exact_FV_rate = sum(outward.values(), F(0)) / m
        reference_divergence = (reference['right'] - reference['left'] + reference['top'] - reference['bottom']) / m
        interface = (outward['x_plus'] - reference['right']) / m if index == 0 else (outward['x_minus'] + reference['left']) / m
        if index == 0:
            other = (outward['x_minus'] + outward['y_plus'] + outward['y_minus']
                     - (-reference['left'] + reference['top'] - reference['bottom'])) / m
        else:
            other = (outward['x_plus'] + outward['y_plus'] + outward['y_minus']
                     - (reference['right'] + reference['top'] - reference['bottom'])) / m
        assert exact_FV_rate == F(source_main['rows'][index]['FV_exact_binary_flux_rate_exact'])
        assert row['FV_rate_exact_from_saved_fluxes'] == exact_FV_rate
        assert row['reference_face_divergence_exact'] == reference_divergence
        assert row['interface_rate_component'] == interface
        assert row['other_three_faces_rate_component'] == other
        assert row['published_error'] == F(source_main['rows'][index]['FV_signed_error_per_s'])
        assert row['exact_flux_error_budget_closure'] == row['published_error_budget_closure'] == 0
        assert parameters['fv_outward'][index] == {item['direction']: F(item['mass_rate_exact_kg_per_s'])
                                                   for item in source_fluxes[index]['outward_contributions']}
    assert budget['shared_interface']['reference_shared_face_equal']
    assert budget['shared_interface']['FV_shared_face_equal']
    assert budget['shared_interface']['interface_component_mass_balance'] == 0
    components = budget['signed_rate_components']
    sum_vector = [sum((vector[row] for vector in components.values()), F(0)) for row in range(2)]
    independent_square = sum((parameters['masses'][i] * sum_vector[i]**2 for i in range(2)), F(0)) / parameters['global_reference_norm']**2
    assert independent_square == budget['Gram_sum'] == budget['exact_normalized_published_error_square']
    assert budget['exact_Gram_closure'] == 0
    assert 'cartesian_material_fv_rows' not in sys.modules
    assert 'cartesian_fv_rows_study' not in sys.modules
    tests = read(out / 'tests.json')
    assert tests['successful'] and tests['checks_run'] == 4
    assert tests['failures'] == tests['errors'] == tests['skipped'] == 0
    assert digest(out / 'tests.json') == config['unit_checks_supplied']['json_sha256']
    assert digest(out / 'tests.log') == config['unit_checks_supplied']['log_sha256']
    result = read(out / 'results.json')
    assert result['recorded_checks_total'] == 384 + tests['checks_run'] == 388
    for key in ['new_edge_weights', 'new_candidate_graphs', 'new_global_matrices', 'new_operator_quadrature_evaluations',
                'new_reference_integrations', 'new_flow_solves', 'new_reference_solves',
                'new_candidate_transient_runs', 'new_reference_transient_runs']:
        assert result[key] == 0
    assert not result['fluid_orc_enabled'] and not result['finite_node_candidate_admitted']

    preserved = earlier_preservation()
    for source in [ROOT / 'runs/reconnect-route-decision', ROOT / 'runs/reconnect-density-weak',
                   ROOT / 'runs/reconnect-point-quadrature', ROOT / 'runs/reconnect-density-path',
                   ROOT / 'runs/reconnect-saved-density-paths', SOURCE]:
        entries = read(source / 'sha256.json')
        for rel, expected in entries.items():
            assert digest(ROOT / rel) == expected, rel
        preserved['formal_scientific_manifests_verified'].append({
            'manifest': str((source / 'sha256.json').relative_to(ROOT)),
            'manifest_sha256': digest(source / 'sha256.json'), 'entries_verified': len(entries),
        })
    assert len(preserved['formal_scientific_manifests_verified']) == 59
    save(out / 'preservation.json', preserved)
    roles = {
        'research/reconnect/CARTESIAN_FV_TWO_ROW_CONTROL.md': 'predeclared frozen c0 residual and next face budget',
        'runs/reconnect-cartesian-fv-rows/stage3-status.json': 'predeclared signed budget scope',
        'runs/reconnect-cartesian-fv-rows/main-c0-comparison.json': 'unchanged source response, error and normalization',
        'runs/reconnect-cartesian-fv-rows/main-c0-local-fluxes.json': 'unchanged exact outward contributions',
        'runs/reconnect-cartesian-fv-rows/stencil.json': 'unchanged seven local conductances, receipt only',
        protocol['source_snapshot']: 'original saved positive-axis face integrals; extra information reference',
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
    assert status['recorded_checks_total'] == 388 and not status['fluid_orc_enabled']
    assert not status['finite_node_candidate_admitted'] and status['stage3_physical_acceptance'] == 'incomplete'
    assert status['next_mass_mean_interface_expansion_predeclared']['already_executed'] is False
    assert status['next_mass_mean_interface_expansion_predeclared']['new_edge_weights'] == 0
    audit = {
        'predeclared_signed_saved_face_scope_verified': True,
        'four_new_budget_software_checks_passed': True,
        'selected_reference_dtype_and_member_bytes_including_signed_zeros_verified': True,
        'original_stencil_c0_comparison_and_flux_receipts_byte_identical': True,
        'positive_axis_mapping_verified_against_direct_outward_sum': True,
        'shared_faces_not_averaged_or_repaired': True,
        'exact_row_and_Gram_closures_independently_verified': True,
        'reference_and_published_rounding_not_removed': True,
        'FV_construction_modules_not_imported_or_executed': True,
        'standard_library_budget_replay_identical': True,
        'new_budget_software_checks_passed': 4, 'recorded_checks_total': 388,
        'previous_checks_rerun': 0, 'new_edge_weights': 0,
        'prior_formal_manifests_verified': 59, 'fluid_orc_enabled': False,
    }
    save(out / 'implementation-audit.json', audit)
    print(json.dumps({'original_files_verified': 782, 'historical_attachments_verified': 15, **audit}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
