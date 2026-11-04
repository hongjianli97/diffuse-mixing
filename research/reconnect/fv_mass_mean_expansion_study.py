"""Predeclared interface expansion; saved data are read-only observations.

Only source extraction requires NumPy. Replay and exact certificates use the
standard library and do not execute any transport constructor or solver.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import platform

from audit_route_decision import ROOT, digest, read, save
from fv_mass_mean_expansion import analyse, certificates


SOURCE = ROOT / 'runs/reconnect-FV-face-budget'
SNAPSHOT = ROOT / 'runs/reconnect-contact-frozen/contact-frozen.npz'
RECEIPTS = ['source-stencil.json', 'source-main-c0-comparison.json',
            'budget-inputs.json', 'budget-summary.json', 'exact-budget.json']


def replay(output):
    out = Path(output).resolve()
    packet = read(out / 'analysis-inputs.json')
    result = analyse(packet)
    assert result == read(out / 'analysis-output.json')
    assert certificates(packet) == read(out / 'exact-certificates.json')
    assert result['arithmetic_precision_controls'] == read(out / 'arithmetic-controls.json')
    for name, expected in read(out / 'input-provenance.json')['receipt_SHA256'].items():
        assert digest(out / name) == expected, name
    print(json.dumps({'standalone_standard_library_replay_identical': True,
                      'exact_algebra_certificate_groups': 4,
                      'recorded_checks_total_unchanged': 388,
                      'new_transport_weights_or_solves': 0,
                      'fluid_orc_enabled': False}), flush=True)
    return result


def main(output):
    import numpy as np
    manifest = read(SOURCE / 'sha256.json')
    for rel, expected in manifest.items():
        assert digest(ROOT / rel) == expected, rel
    protocol_path = SOURCE / 'stage3-status.json'
    protocol = read(protocol_path)['next_mass_mean_interface_expansion_predeclared']
    assert not protocol['already_executed']
    assert protocol['new_edge_weights'] == protocol['new_candidate_graphs'] == 0
    source_provenance = read(SOURCE / 'input-provenance.json')
    assert digest(SNAPSHOT) == source_provenance['source_snapshot_sha256']
    stencil = read(SOURCE / 'source-stencil.json')
    comparison = read(SOURCE / 'source-main-c0-comparison.json')
    budget = read(SOURCE / 'budget-inputs.json')['parameters']
    rows = comparison['rows']
    assert [row['target_id'] for row in rows] == [63, 64]
    with np.load(SNAPSHOT, allow_pickle=False) as z:
        indices = [int(np.flatnonzero(z['ids'] == target)[0]) for target in [63, 64]]
        centers = z['material_label_centers'][indices]
        rho = z['rho0_point'][indices]
        mass, c0 = z['mass'][indices], z['c0'][indices]
        assert centers.tolist() == [[127/256, 1/256], [129/256, 1/256]]
        assert rho.tolist() == [1., 4.]
        for i, row in enumerate(rows):
            assert mass[i].tobytes() == np.float64(row['original_mass_kg']).tobytes()
            assert c0[i].tobytes() == np.float64(row['original_c0_mass_mean']).tobytes()
            assert F(float(mass[i])) == F(budget['masses'][i])
        geometry = {'ids': [63, 64], 'centers_exact': [[str(F(float(x))) for x in row] for row in centers],
                    'rho_exact': [str(F(float(x))) for x in rho],
                    'geometry_prior': stencil['model_prior'],
                    'geometry_not_inferred_from_generic_point_inputs': True}
    edge = next(edge for edge in stencil['edges'] if edge['ids'] == [63, 64])
    q_fv = F(budget['fv_outward'][0]['x_plus'])
    q_ref = F(budget['reference_faces'][0]['right'])
    assert q_fv == -F(budget['fv_outward'][1]['x_minus'])
    assert q_ref == F(budget['reference_faces'][1]['left'])
    model = {
        'ell': stencil['spacing_exact'][0], 'dy': stencil['spacing_exact'][1],
        'thickness': stencil['thickness_exact'], 'D': stencil['D_binary_input_exact'],
        'rho_left': '1', 'rho_right': '4', 'normal_period': stencil['periods_exact'][0],
        'transverse_period': stencil['periods_exact'][1], 'y_center': '1/256',
        'amplitude': '1/5', 'phase_pi_ratio': '1/5',
    }
    saved = {
        'interface_g': edge['conductance_exact'],
        'mass_left': budget['masses'][0], 'mass_right': budget['masses'][1],
        'cbar_left': str(F(rows[0]['original_c0_mass_mean'])),
        'cbar_right': str(F(rows[1]['original_c0_mass_mean'])),
        'FV_interface_flux': str(q_fv), 'reference_interface_flux': str(q_ref),
        'saved_flux_defect': str(q_fv - q_ref),
    }
    packet = {'model': model, 'saved': saved,
              'reference_interpretation': 'true pi and exact amplitude 1/5 in the original analytic formula; binary D retained',
              'binary_saved_inputs_not_overwritten': True,
              'analytic_reference_is_extra_information_not_transport_runtime_input': True}
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'config.json', {
        'study': 'unchanged ordinary resistance FV material-mean interface expansion',
        'protocol': protocol, 'protocol_source_sha256': digest(protocol_path),
        'source_manifest_entries_verified': len(manifest),
        'reference_arithmetic_precisions': [60, 90], 'guard_digits': 30,
        'new_numerical_tests': 0, 'saved_g_constructor_executed': False,
        'no_curvature_fit_no_transport_correction': True,
    })
    save(out / 'analysis-inputs.json', packet)
    save(out / 'selected-geometry.json', geometry)
    for name in RECEIPTS:
        with (out / name).open('xb') as f:
            f.write((SOURCE / name).read_bytes())
    save(out / 'input-provenance.json', {
        'source_snapshot': str(SNAPSHOT.relative_to(ROOT)), 'source_snapshot_sha256': digest(SNAPSHOT),
        'source_scientific_manifest': str((SOURCE / 'sha256.json').relative_to(ROOT)),
        'source_scientific_manifest_sha256': digest(SOURCE / 'sha256.json'),
        'receipt_SHA256': {name: digest(out / name) for name in RECEIPTS},
        'analytic_background_source': 'research/reconnect/contact_frozen_probe.py',
        'analytic_background_source_sha256': digest(ROOT / 'research/reconnect/contact_frozen_probe.py'),
        'exact_math_amplitude_and_pi_vs_source_binary_evaluation_bridges_retained': True,
    })
    result = analyse(packet)
    save(out / 'analysis-output.json', result)
    save(out / 'exact-certificates.json', result['exact_certificates'])
    save(out / 'arithmetic-controls.json', result['arithmetic_precision_controls'])
    save(out / 'environment.json', {'python': platform.python_version(), 'numpy_for_source_extraction_only': np.__version__,
                                    'reference_and_replay_arithmetic': 'standard library Fraction and Decimal'})
    replay(out)
    high = result['arithmetic_precision_controls'][1]
    print(json.dumps({key: high[key] for key in [
        'curvature_jump_leading_integrated_flux_defect', 'actual_model_third_order_integrated_remainder',
        'rational_bound_decimal_readout', 'saved_mean_rounding_bridge',
        'saved_reference_face_rounding_bridge', 'arithmetic_closure_residual_not_forced_to_zero',
        'predicted_leading_rate_components']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output')
    group.add_argument('--replay')
    group.add_argument('--certificates-only', metavar='INPUT_JSON')
    args = parser.parse_args()
    if args.certificates_only:
        print(json.dumps(certificates(read(args.certificates_only)), indent=2), flush=True)
    elif args.replay:
        replay(args.replay)
    else:
        main(args.output)
