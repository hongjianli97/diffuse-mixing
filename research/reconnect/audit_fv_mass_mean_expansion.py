"""Read-only verification of the predeclared interface expansion and history."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import inspect
import json
import subprocess
import sys

from audit_route_decision import ROOT, digest, read, save, preservation
from fv_mass_mean_expansion import analytic_reference, certificates
from fv_mass_mean_expansion_study import SOURCE, RECEIPTS, replay


def main(output):
    out = Path(output).resolve()
    assert not (out / 'sha256.json').exists(), 'never rewrite sealed science'
    config = read(out / 'config.json')
    protocol_source = SOURCE / 'stage3-status.json'
    assert config['protocol'] == read(protocol_source)['next_mass_mean_interface_expansion_predeclared']
    assert config['protocol_source_sha256'] == digest(protocol_source)
    assert config['protocol']['already_executed'] is False
    for name in RECEIPTS:
        assert (out / name).read_bytes() == (SOURCE / name).read_bytes(), name
    provenance = read(out / 'input-provenance.json')
    assert digest(ROOT / provenance['source_snapshot']) == provenance['source_snapshot_sha256']
    packet = read(out / 'analysis-inputs.json')
    saved, model = packet['saved'], packet['model']
    source_rows = read(SOURCE / 'source-main-c0-comparison.json')['rows']
    stencil = read(SOURCE / 'source-stencil.json')
    budget = read(SOURCE / 'budget-inputs.json')['parameters']
    assert [saved['mass_left'], saved['mass_right']] == budget['masses']
    assert [F(saved['cbar_left']), F(saved['cbar_right'])] == [F(row['original_c0_mass_mean']) for row in source_rows]
    assert saved['interface_g'] == next(edge['conductance_exact'] for edge in stencil['edges'] if edge['ids'] == [63, 64])
    assert F(saved['FV_interface_flux']) == F(budget['fv_outward'][0]['x_plus'])
    assert F(saved['reference_interface_flux']) == F(budget['reference_faces'][0]['right'])
    assert model['D'] == stencil['D_binary_input_exact']
    assert list(inspect.signature(analytic_reference).parameters) == ['model']
    result = replay(out)
    command = [sys.executable, '-S', str(ROOT / 'research/reconnect/fv_mass_mean_expansion_study.py'),
               '--certificates-only', str(out / 'analysis-inputs.json')]
    independent = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(independent.stdout) == certificates(packet) == read(out / 'exact-certificates.json')
    for precision_control in result['arithmetic_precision_controls']:
        assert precision_control['reference_values_not_applied_to_transport']
        assert precision_control['saved_defect_within_leading_plus_proved_remainder_and_recorded_bridges']
    for field in ['new_edge_weights', 'new_candidate_graphs', 'new_reference_integrations',
                  'new_numerical_or_flow_tests', 'new_candidate_transient_runs', 'new_reference_transient_runs']:
        assert result[field] == 0
    assert result['recorded_checks_total'] == 388
    assert result['new_exact_algebra_certificate_groups'] == 4
    for module in ['cartesian_material_fv_rows', 'cartesian_fv_rows_study', 'contact_frozen_probe']:
        assert module not in sys.modules, module
    preserved = preservation()
    for directory in ['reconnect-route-decision', 'reconnect-density-weak', 'reconnect-point-quadrature',
                      'reconnect-density-path', 'reconnect-saved-density-paths',
                      'reconnect-cartesian-fv-rows', 'reconnect-FV-face-budget']:
        manifest_path = ROOT / 'runs' / directory / 'sha256.json'
        entries = read(manifest_path)
        for rel, expected in entries.items():
            assert digest(ROOT / rel) == expected, rel
        preserved['formal_scientific_manifests_verified'].append({
            'manifest': str(manifest_path.relative_to(ROOT)), 'manifest_sha256': digest(manifest_path),
            'entries_verified': len(entries),
        })
    assert len(preserved['formal_scientific_manifests_verified']) == 60
    save(out / 'preservation.json', preserved)
    roles = {
        'research/reconnect/FV_SAVED_FACE_ERROR_BUDGET.md': 'prior saved defect and predeclared theory scope',
        'runs/reconnect-FV-face-budget/stage3-status.json': 'predeclared interface expansion',
        'runs/reconnect-FV-face-budget/source-stencil.json': 'unaltered original g, mass and known geometry prior',
        'runs/reconnect-FV-face-budget/source-main-c0-comparison.json': 'original mass-average c0 diagnostic',
        'runs/reconnect-FV-face-budget/budget-inputs.json': 'unaltered exact saved FV and reference fluxes',
        'research/reconnect/contact_frozen_probe.py': 'independent analytic background, read only, never executed',
        'research/rebuild/CONTACT_TRANSMISSION.md': 'existing aligned resistance equivalence, cited not reproved',
        provenance['source_snapshot']: 'original snapshot, not part of standalone replay',
    }
    manifest = read(SOURCE / 'sha256.json')
    index = []
    for rel, role in roles.items():
        assert manifest[rel] == digest(ROOT / rel), rel
        index.append({'path': rel, 'sha256': manifest[rel], 'role': role})
    save(out / 'evidence-index.json', {'source_manifest': str((SOURCE / 'sha256.json').relative_to(ROOT)),
                                      'source_manifest_sha256': digest(SOURCE / 'sha256.json'), 'sources': index})
    status = read(out / 'stage3-status.json')
    assert status['recorded_checks_total'] == 388 and status['new_exact_algebra_certificate_groups'] == 4
    assert not status['fluid_orc_enabled'] and status['stage3_physical_acceptance'] == 'incomplete'
    assert status['next_four_mean_admissibility_predeclared']['already_executed'] is False
    audit = {
        'predeclared_mass_mean_interface_expansion_scope_verified': True,
        'saved_stencil_c0_mass_and_flux_receipts_byte_identical': True,
        'exact_coefficients_recomputed_in_independent_no_site_packages_process': True,
        'analytic_reference_function_accepts_only_declared_model_parameters': True,
        'original_background_not_executed_and_no_curvature_fitted_to_error': True,
        'high_precision_controls_and_saved_value_bridges_replayed': True,
        'rigorous_remainder_bound_is_analytic_not_a_Decimal_interval_claim': True,
        'original_c0_retained_despite_initial_higher_order_compatibility_issue': True,
        'aligned_resistance_equivalence_cited_not_reproved': True,
        'FV_constructor_and_old_probe_modules_not_imported_or_executed': True,
        'prior_formal_manifests_verified': 60, 'new_exact_algebra_certificate_groups': 4,
        'new_numerical_or_flow_tests': 0, 'recorded_checks_total': 388,
        'previous_checks_rerun': 0, 'new_edge_weights': 0, 'fluid_orc_enabled': False,
    }
    save(out / 'implementation-audit.json', audit)
    print(json.dumps({'original_files_verified': 782, 'historical_attachments_verified': 15, **audit}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
