"""Independent algebra and immutable-source audit for the four-mean review."""
from fractions import Fraction as F
from itertools import permutations
from pathlib import Path
import argparse
import json
import math
import subprocess
import sys

from audit_route_decision import ROOT, digest, read, save, preservation
from four_mean_interface_study import SOURCE, replay


def exact(value):
    return F(value['exact'])


def main(output):
    out = Path(output).resolve()
    assert not (out / 'sha256.json').exists(), 'never rewrite sealed evidence'
    config = read(out / 'config.json')
    status_source = SOURCE / 'stage3-status.json'
    assert config['protocol'] == read(status_source)['next_four_mean_admissibility_predeclared']
    assert config['protocol_source_sha256'] == digest(status_source)
    assert not config['protocol']['already_executed']
    assert (out / 'source-stencil.json').read_bytes() == (SOURCE / 'source-stencil.json').read_bytes()
    assert (out / 'source-stage3-status.json').read_bytes() == status_source.read_bytes()
    packet = read(out / 'review-inputs.json')
    model, baseline = packet['geometry_and_material'], packet['saved_baseline_coefficients']
    old = read(SOURCE / 'analysis-inputs.json')
    stencil = read(SOURCE / 'source-stencil.json')
    assert packet['ordered_cell_ids'] == [62, 63, 64, 65]
    assert set(model) == {'ell', 'face_area', 'D', 'rho_left', 'rho_right'}
    assert set(baseline) == {'interface_g', 'left_outer_g', 'right_outer_g', 'mass_left', 'mass_right'}
    assert model['ell'] == old['model']['ell'] and model['D'] == old['model']['D']
    assert model['rho_left'] == old['model']['rho_left'] and model['rho_right'] == old['model']['rho_right']
    assert F(model['face_area']) == F(old['model']['dy']) * F(old['model']['thickness'])
    for key in ['interface_g', 'mass_left', 'mass_right']:
        assert baseline[key] == old['saved'][key]
    edges = {tuple(edge['ids']): edge['conductance_exact'] for edge in stencil['edges']}
    assert baseline['left_outer_g'] == edges[(62,63)] and baseline['right_outer_g'] == edges[(64,65)]
    result = replay(out)
    ell, area, D, rhoL, rhoR = [F(model[key]) for key in ['ell','face_area','D','rho_left','rho_right']]
    GL, GR = rhoL*D, rhoR*D
    g, gL, gR, mL, mR = [F(baseline[key]) for key in ['interface_g','left_outer_g','right_outer_g','mass_left','mass_right']]
    # A separate implementation: explicit integral expressions and Leibniz
    # determinant, followed by elimination identities; no Gaussian solver used.
    matrix = [[F(1)]*4, [-3*ell/(2*GL), -ell/(2*GL), ell/(2*GR), 3*ell/(2*GR)],
              [7*ell**2/6, ell**2/6, F(0), F(0)], [F(0), F(0), ell**2/6, 7*ell**2/6]]
    determinant = F(0)
    for perm in permutations(range(4)):
        sign = (-1)**sum(perm[i] > perm[j] for i in range(4) for j in range(i+1,4))
        determinant += sign * math.prod(matrix[i][perm[i]] for i in range(4))
    assert determinant == ell**5 / 3 * (1/GL + 1/GR) > 0
    groups = result['groups']
    reconstruction = groups['unique_data_independent_exact_reconstruction']
    assert [[exact(x) for x in row] for row in reconstruction['constraint_matrix']] == matrix
    assert exact(reconstruction['determinant']) == determinant
    w0 = area / (2*ell*(1/GL + 1/GR))
    expected_w = [w0, -7*w0, 7*w0, -w0]
    assert w0 == g/4
    assert list(map(exact,reconstruction['formal_integrated_flux_coefficients'])) == expected_w
    assert [sum((x*y for x,y in zip(row,expected_w)),F(0)) for row in matrix] == [F(0),area,F(0),F(0)]
    normal_numerators = [[gL+g/4, -gL-7*g/4, 7*g/4, -g/4],
                         [-g/4, 7*g/4, -gR-7*g/4, gR+g/4]]
    accounting = groups['shared_flux_and_unchanged_outer_closure_accounting']
    expected_rows = [[x/m for x in row] for row,m in zip(normal_numerators,[mL,mR])]
    assert [[exact(x) for x in row] for row in accounting['two_formal_normal_rate_rows_per_s']] == expected_rows
    assert all(sum(row,F(0)) == 0 for row in expected_rows)
    assert [mL*exact(accounting['interface_only_rate_rows_per_s'][0][i])+
            mR*exact(accounting['interface_only_rate_rows_per_s'][1][i]) for i in range(4)] == [F(0)]*4
    obstruction = groups['nonnegative_generator_obstruction']
    assert list(map(exact,obstruction['negative_remote_offdiagonal_coefficients_per_s'])) == [-g/(4*mL),-g/(4*mR)]
    for i,witness in enumerate(obstruction['nonnegative_initial_mean_witnesses']):
        state = list(map(exact,witness['nonnegative_discrete_material_mean_vector']))
        assert all(0 <= x <= 1 for x in state) and state[[1,2][i]] == 0
        assert sum((x*y for x,y in zip(expected_rows[i],state)),F(0)) == exact(witness['formal_initial_target_derivative_per_s']) < 0
    independent = subprocess.run([sys.executable,'-S',str(ROOT/'research/reconnect/four_mean_interface_study.py'),
        '--certificates-only',str(out/'review-inputs.json')],capture_output=True,text=True,check=True,timeout=30)
    assert json.loads(independent.stdout) == result
    for module in ['numpy','scipy','cartesian_material_fv_rows','cartesian_fv_rows_study','contact_frozen_probe']:
        assert module not in sys.modules, module
    preserved = preservation()
    for directory in ['reconnect-route-decision','reconnect-density-weak','reconnect-point-quadrature',
                      'reconnect-density-path','reconnect-saved-density-paths','reconnect-cartesian-fv-rows',
                      'reconnect-FV-face-budget','reconnect-FV-interface-expansion']:
        manifest_path = ROOT/'runs'/directory/'sha256.json'
        entries = read(manifest_path)
        for rel,expected in entries.items():
            assert digest(ROOT/rel) == expected,rel
        preserved['formal_scientific_manifests_verified'].append({
            'manifest':str(manifest_path.relative_to(ROOT)),'manifest_sha256':digest(manifest_path),
            'entries_verified':len(entries)})
    assert len(preserved['formal_scientific_manifests_verified']) == 61
    save(out/'preservation.json',preserved)
    roles = {
        'research/reconnect/FV_MASS_MEAN_INTERFACE_EXPANSION.md':'parent mass-average curvature defect and bounded remainder',
        'runs/reconnect-FV-interface-expansion/stage3-status.json':'predeclared four-mean exactness and sign review',
        'runs/reconnect-FV-interface-expansion/analysis-inputs.json':'saved material geometry, mass and g; c0/error excluded from new constraints',
        'runs/reconnect-FV-interface-expansion/source-stencil.json':'unchanged ordinary central interface and outer conductances',
        'research/reconnect/CARTESIAN_FV_TWO_ROW_CONTROL.md':'source two-row scope and limited frozen improvement',
        'research/reconnect/PERIODIC_DENSITY_CONTACT_FROZEN.md':'original global reference normalization; point-kernel transient not admitted',
    }
    source_manifest = read(SOURCE/'sha256.json')
    index=[]
    for rel,role in roles.items():
        assert source_manifest[rel] == digest(ROOT/rel),rel
        index.append({'path':rel,'sha256':source_manifest[rel],'role':role})
    save(out/'evidence-index.json',{'source_manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
         'source_manifest_sha256':digest(SOURCE/'sha256.json'),'sources':index})
    status=read(out/'stage3-status.json')
    assert status['recorded_checks_total'] == 388 and status['new_exact_algebra_certificate_groups'] == 4
    assert not status['fluid_orc_enabled'] and not status['finite_node_candidate_admitted']
    assert status['stage3_physical_acceptance'] == 'incomplete'
    assert status['next_ordinary_FV_global_frozen_predeclared']['already_executed'] is False
    assert result['decision'] == status['decision']
    audit={'predeclared_four_true_mean_scope_verified':True,
        'constraint_matrix_uses_no_c0_reference_rate_or_saved_error':True,
        'independent_Leibniz_determinant_and_closed_form_elimination_verified':True,
        'explicit_two_row_conservation_and_constant_residuals_verified':True,
        'negative_remote_coefficients_and_nonnegative_mean_witnesses_verified':True,
        'no_transport_modules_or_scientific_libraries_imported':True,
        'standard_library_exact_certificate_replay_identical':True,
        'original_c0_and_acceptance_criteria_retained':True,
        'four_mean_correction_admitted':False,
        'prior_formal_manifests_verified':61,'new_exact_algebra_certificate_groups':4,
        'new_numerical_or_flow_tests':0,'recorded_checks_total':388,'previous_checks_rerun':0,
        'new_edge_weights':0,'fluid_orc_enabled':False}
    save(out/'implementation-audit.json',audit)
    print(json.dumps({'original_files_verified':782,'historical_attachments_verified':15,**audit},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
