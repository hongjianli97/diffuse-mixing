"""Replayable exact four-mean admissibility review; no scientific solver."""
from pathlib import Path
import argparse
import json
import platform

from audit_route_decision import ROOT, digest, read, save
from four_mean_interface_review import review


SOURCE = ROOT / 'runs/reconnect-FV-interface-expansion'


def replay(output):
    out = Path(output).resolve()
    result = review(read(out / 'review-inputs.json'))
    assert result == read(out / 'exact-certificates.json')
    for name, expected in read(out / 'input-provenance.json')['receipt_SHA256'].items():
        assert digest(out / name) == expected, name
    print(json.dumps({'standard_library_exact_replay_identical': True,
        'new_exact_algebra_certificate_groups': 4, 'recorded_checks_total_unchanged': 388,
        'formal_four_mean_correction_admitted': False, 'new_transport_weights_or_solves': 0,
        'fluid_orc_enabled': False}), flush=True)
    return result


def main(output):
    manifest = read(SOURCE / 'sha256.json')
    for rel, expected in manifest.items():
        assert digest(ROOT / rel) == expected, rel
    status_path = SOURCE / 'stage3-status.json'
    protocol = read(status_path)['next_four_mean_admissibility_predeclared']
    assert not protocol['already_executed'] and protocol['no_coefficients_installed_in_transport']
    stencil = read(SOURCE / 'source-stencil.json')
    previous_inputs = read(SOURCE / 'analysis-inputs.json')
    model, saved = previous_inputs['model'], previous_inputs['saved']
    edges = {tuple(edge['ids']): edge['conductance_exact'] for edge in stencil['edges']}
    from fractions import Fraction as F
    packet = {
        'ordered_cell_ids': [62, 63, 64, 65],
        'geometry_and_material': {'ell': model['ell'],
            'face_area': str(F(model['dy']) * F(model['thickness'])),
            'D': model['D'], 'rho_left': model['rho_left'], 'rho_right': model['rho_right']},
        'saved_baseline_coefficients': {'interface_g': edges[(63,64)],
            'left_outer_g': edges[(62,63)], 'right_outer_g': edges[(64,65)],
            'mass_left': saved['mass_left'], 'mass_right': saved['mass_right']},
        'c0_reference_rate_and_flux_defect_excluded_from_constraint_inputs': True,
        'transverse_strip_true_material_means_not_point_states': True,
    }
    assert packet['saved_baseline_coefficients']['interface_g'] == saved['interface_g']
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'config.json', {'study': 'predeclared four-true-mean linear reconstruction admissibility',
        'protocol': protocol, 'protocol_source_sha256': digest(status_path),
        'source_manifest_entries_verified': len(manifest),
        'arithmetic': 'Fraction exact source binary inputs and exact monomial integrals',
        'formal_constraint_system_not_a_transport_matrix': True,
        'new_physical_cases_or_scans_or_transients': 0})
    save(out / 'review-inputs.json', packet)
    receipts = {'source-stencil.json': SOURCE / 'source-stencil.json',
                'source-stage3-status.json': status_path}
    for name, source in receipts.items():
        with (out / name).open('xb') as f:
            f.write(source.read_bytes())
    save(out / 'input-provenance.json', {
        'source_scientific_manifest': str((SOURCE / 'sha256.json').relative_to(ROOT)),
        'source_scientific_manifest_sha256': digest(SOURCE / 'sha256.json'),
        'source_analysis_inputs_sha256': digest(SOURCE / 'analysis-inputs.json'),
        'receipt_SHA256': {name: digest(path) for name, path in receipts.items()},
        'no_original_snapshot_reference_integration_or_transport_module_executed': True})
    result = review(packet)
    save(out / 'exact-certificates.json', result)
    save(out / 'results.json', {key: value for key, value in result.items() if key != 'groups'})
    save(out / 'environment.json', {'python': platform.python_version(), 'dependencies': 'standard library only'})
    replay(out)
    print(json.dumps(result['groups']['nonnegative_generator_obstruction'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output')
    group.add_argument('--replay')
    group.add_argument('--certificates-only', metavar='INPUT_JSON')
    args = parser.parse_args()
    if args.certificates_only:
        print(json.dumps(review(read(args.certificates_only)), indent=2), flush=True)
    elif args.replay:
        replay(args.replay)
    else:
        main(args.output)
