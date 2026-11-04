"""Independent integer-image certificates, exact replay and preservation audit."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import json
import numpy as np

from audit_route_decision import ROOT, digest, read, save, preservation as earlier_preservation
from saved_density_paths_study import array_record, replay


SOURCE = ROOT / 'runs/reconnect-density-path'


def integer_image_distances(points, endpoint, period):
    distances = []
    for shift in product((-1, 0, 1), repeat=2):
        offset = points - endpoint + np.array(shift, dtype=np.int64) * period
        distances.append(np.sum(offset**2, axis=1))
    return np.min(np.stack(distances), axis=0)


def main(output):
    out = Path(output).resolve()
    assert not (out / 'sha256.json').exists(), 'never rewrite sealed evidence'
    protocol_source = SOURCE / 'stage3-status.json'
    protocol = read(protocol_source)['next_saved_density_paths_predeclared']
    config = read(out / 'config.json')
    assert config['protocol'] == protocol and config['protocol_source_sha256'] == digest(protocol_source)
    provenance = read(out / 'input-provenance.json')
    snapshot = ROOT / protocol['source_snapshot']
    assert digest(snapshot) == protocol['source_snapshot_sha256'] == provenance['source_snapshot_sha256_before_and_after']
    assert digest(out / 'saved-point-inputs.npz') == provenance['saved_point_inputs_sha256']
    observations = read(out / 'observations.json')
    assert len(observations) == protocol['paths_to_check'] == 3
    assert [item['normal_columns'] for item in observations] == protocol['normal_column_pairs']
    with np.load(out / 'saved-point-inputs.npz', allow_pickle=False) as small, np.load(snapshot, allow_pickle=False) as original:
        for key, source_array in [('positions', original['positions'][0]), ('ids', original['ids']), ('rho', original['rho'][0])]:
            assert array_record(small[key]) == array_record(source_array) == provenance['source_members_copied_without_numeric_change'][key]
            np.testing.assert_array_equal(small[key], source_array)
        scale = int(small['position_lattice_denominator'])
        denominator = int(small['distance_squared_denominator'])
        assert scale == 256 and denominator == scale**2
        integers = small['position_lattice_numerators']
        assert integers.dtype == np.dtype('int64') and integers.shape == (16384, 2)
        np.testing.assert_array_equal(small['positions'], integers / scale)
        for index, item in enumerate(observations):
            left, right = item['start_input_index'], item['end_input_index']
            assert int(small['ids'][left]) == item['start_id'] and int(small['ids'][right]) == item['end_id']
            a = integer_image_distances(integers, integers[left], scale)
            b = integer_image_distances(integers, integers[right], scale)
            np.testing.assert_array_equal(a, small[f'path{index}_distance2_start_numerators'])
            np.testing.assert_array_equal(b, small[f'path{index}_distance2_end_numerators'])
            radius = int(a[right])
            assert F(radius, denominator) == F(item['radius_squared_exact'])
            assert radius == item['radius_squared_numerator']
            mask = (a <= radius) & (b <= radius)
            np.testing.assert_array_equal(mask, small[f'path{index}_keep'])
            assert np.flatnonzero(mask).tolist() == item['retained_indices']
            assert small['ids'][mask].tolist() == item['retained_ids']
            assert int(mask.sum()) == item['retained_nodes']
    replay(out)
    tests = read(out / 'tests.json')
    assert tests['successful'] and tests['checks_run'] == 6
    assert tests['failures'] == tests['errors'] == tests['skipped'] == 0
    assert digest(out / 'tests.json') == config['unit_checks_supplied']['json_sha256']
    assert digest(out / 'tests.log') == config['unit_checks_supplied']['log_sha256']
    r = read(out / 'results.json')
    assert r['recorded_checks_total'] == 372 + tests['checks_run'] == 378
    assert r['saved_path_observations'] == 3 and not r['saved_path_observations_counted_as_new_tests']
    assert r['retained_node_counts'] == [2, 5, 2]
    assert r['mean_inverse_density_exact'] == ['5/8', '7/16', '5/8']
    assert r['full_saved_cloud_lower_envelope_runs'] == 0
    assert not r['finite_node_candidate_admitted'] and not r['fluid_orc_enabled']

    preserved = earlier_preservation()
    for source in [ROOT / 'runs/reconnect-route-decision', ROOT / 'runs/reconnect-density-weak',
                   ROOT / 'runs/reconnect-point-quadrature', SOURCE]:
        entries = read(source / 'sha256.json')
        for rel, expected in entries.items():
            assert digest(ROOT / rel) == expected, rel
        preserved['formal_scientific_manifests_verified'].append({
            'manifest': str((source / 'sha256.json').relative_to(ROOT)),
            'manifest_sha256': digest(source / 'sha256.json'), 'entries_verified': len(entries),
        })
    assert len(preserved['formal_scientific_manifests_verified']) == 57
    save(out / 'preservation.json', preserved)
    roles = {
        'research/reconnect/PERIODIC_DENSITY_PATH_PRIMITIVE.md': 'predeclared three paths and fixed-radius exclusion',
        'research/reconnect/periodic_density_path.py': 'unchanged exact primitive',
        'research/reconnect/CONDITIONAL_DENSITY_CLOSURE.md': 'point-only closure contract',
        'research/reconnect/contact_frozen_probe.py': 'original analytic density definition, read only as reference',
        'research/reconnect/ORACLE_INTERFACE_SECOND_ORDER.md': 'known curvature defect remains',
        'research/reconnect/MASS_AVERAGE_PROJECTION_WEAK_BUDGET.md': 'known projection conditions remain',
        'runs/reconnect-density-path/stage3-status.json': 'predeclared saved-path task',
        protocol['source_snapshot']: 'immutable original input snapshot',
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
    assert status['recorded_checks_total'] == 378 and not status['fluid_orc_enabled']
    assert not status['finite_node_candidate_admitted'] and status['stage3_physical_acceptance'] == 'incomplete'
    next_plan = status['next_conditional_cartesian_FV_rows_predeclared']
    assert next_plan['target_ids'] == [63, 64] and not next_plan['already_executed']
    assert next_plan['main_probe'] == 'original saved c0'
    assert next_plan['true_faces_F_C_or_velocity_gradients_as_runtime_inputs'] is False
    assert next_plan['shared_edge_law_not_fitted_to_state_or_PDE']
    audit = {
        'predeclared_three_path_scope_and_source_hash_verified': True,
        'six_new_localization_unit_checks_passed': True,
        'source_point_member_bytes_identical': True,
        'all_16384_nodes_per_path_distance_certificates_match_independent_integer_images': True,
        'closed_ball_masks_and_all_retained_IDs_verified': True,
        'three_saved_path_records_replay_exactly': True,
        'global_nearest_equivalence_uses_report_lemma_not_a_full_cloud_hull_run': True,
        'new_localization_unit_checks_passed': 6, 'saved_path_observations': 3,
        'saved_observations_not_added_to_unit_check_count': True,
        'recorded_checks_total': 378, 'previous_checks_rerun': 0,
        'prior_formal_manifests_verified': 57, 'fluid_orc_enabled': False,
    }
    save(out / 'implementation-audit.json', audit)
    print(json.dumps({'original_files_verified': 782, 'historical_attachments_verified': 15, **audit}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
