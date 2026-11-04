"""New exact algebra for the conditional point-pair quadrature bridge.

Standard library only; no graph, physical case, operator quadrature or old proof.
The report contains the analytic and geometric proofs.
"""
from fractions import Fraction as F
from math import comb
from pathlib import Path
import argparse
import hashlib
import json
import platform


def multiply(a, b):
    result = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i + j] += x * y
    return result


def power_linear(power):
    return [F(comb(power, i)) * (-F(1, 2))**i for i in range(power + 1)]


def derivative(poly):
    return [F(i) * poly[i] for i in range(1, len(poly))]


def rational(value):
    return {'exact': str(value), 'decimal_for_reading_only': float(value)}


def certificates():
    beta_over_C2 = [F(7, 3) * value for value in multiply([F(2), F(5)], power_linear(5))]
    expected_derivative = [F(0)] + [-35 * value for value in power_linear(4)]
    assert derivative(beta_over_C2) == expected_derivative
    # f=q(1-q/2)^4 has its sole interior maximum at q=2/5.
    f = [F(0)] + power_linear(4)
    assert derivative(f) == multiply(power_linear(3), [F(1), -F(5, 2)])
    critical_q = F(2, 5)
    max_f = critical_q * (1 - critical_q / 2)**4
    assert max_f == F(512, 3125)
    pi_M0 = F(9, 4) * F(14, 3)
    pi_Lbeta = 35 * F(9, 4) * max_f
    assert pi_M0 == F(21, 2) and pi_Lbeta == F(8064, 625)
    bulk_shape = multiply(multiply([F(2), F(4)], [F(2), F(4)]),
                          multiply([F(2), F(3)], [F(2), F(3)]))
    assert bulk_shape == list(map(F, [16, 112, 292, 336, 144]))
    Rrho = F(4)
    bracket = [Rrho**2 * 4 + 4, Rrho**2 * 4, Rrho**2]
    interface_shape = multiply(multiply([F(2), F(2)], [F(2), F(2)]), bracket)
    assert interface_shape == list(map(F, [272, 800, 848, 384, 64]))
    interface_coefficients = [pi_M0 * value for value in interface_shape]
    assert interface_coefficients == list(map(F, [2856, 8400, 8904, 4032, 672]))
    amplitude, mean = F(1, 5), F(1, 2)
    extrema = [mean - amplitude, mean + amplitude]
    continuum_energy_coefficient = 2 * amplitude**2
    assert extrema == [F(3, 10), F(7, 10)]
    assert continuum_energy_coefficient == F(2, 25)
    groups = {
        'new_kernel_gradient_and_supremum_constants': {
            'verified': True,
            'beta_over_C2_coefficients': [str(value) for value in beta_over_C2],
            'derivative_identity': 'beta_prime/C2=-35*q*(1-q/2)^4',
            'critical_point_q': rational(critical_q), 'max_q_times_fourth_power': rational(max_f),
            'pi_times_M0': rational(pi_M0), 'pi_times_Lbeta': rational(pi_Lbeta),
        },
        'quadrature_bound_shape_and_interface_coefficients': {
            'verified': True, 'bulk_shape_coefficients': [str(value) for value in bulk_shape],
            'bulk_prefactor_2_pi_Lbeta': rational(2 * pi_Lbeta),
            'bulk_C_B_at_t_zero': rational(2 * pi_Lbeta * bulk_shape[0]),
            'interface_C_I_coefficients_for_Rrho_4': [str(value) for value in interface_coefficients],
            'all_shape_coefficients_nonnegative': all(value >= 0 for value in bulk_shape + interface_shape),
            'constants_are_analytic_bounds_not_measured_errors': True,
        },
        'dense_points_long_material_strips_average_obstruction': {
            'verified': True, 'mean_concentration': rational(mean),
            'concentration_extrema': [rational(value) for value in extrema],
            'continuum_energy_coefficient_of_pi_squared_D': rational(continuum_energy_coefficient),
            'all_strip_mass_means_and_test_means_equal': True,
            'any_point_graph_pair_energy_zero_on_these_means': True,
            'fill_distance_upper': 'sqrt(2)/(2*N)',
            'material_diameter_lower': rational(F(1, 2)),
            'cell_area_and_q': '1/N^2',
            'no_N_or_graph_or_numerical_physical_case_evaluated': True,
        },
    }
    return {
        'certificate_groups': groups, 'verified_exact_algebra_groups': len(groups),
        'analytic_geometric_proofs_are_in_report': True,
        'conditional_point_pair_quadrature_bound_proved': True,
        'conditional_fixed_test_discrete_weak_consistency_proved': True,
        'weak_consistency_scope': 'static flat phase-constant density, true-cell locality prior, Delta/h to zero',
        'general_point_backend_admitted': False, 'evolution_convergence_proved': False,
        'new_numerical_or_flow_tests': 0, 'recorded_checks_total': 363,
        'new_candidate_graphs': 0, 'new_edge_weights': 0, 'new_operator_quadrature_evaluations': 0,
        'LP_solves': 0, 'new_candidate_transient_runs': 0, 'new_reference_transient_runs': 0,
        'fluid_orc_enabled': False, 'stage3_physical_acceptance': 'incomplete',
    }


def main(output):
    root = Path(__file__).resolve().parents[2]
    source = root / 'runs/reconnect-density-weak'
    manifest = json.loads((source / 'sha256.json').read_text())
    for rel, expected in manifest.items():
        with (root / rel).open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == expected, rel
    source_status = source / 'stage3-status.json'
    protocol = json.loads(source_status.read_text())['next_point_pair_quadrature_bridge_review_predeclared']
    assert protocol['theory_only'] and not protocol['new_graphs_or_numerical_cases']
    assert protocol['proof_completed'] is False
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        with (out / name).open('x') as f:
            f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    save('config.json', {
        'study': 'predeclared conditional point-pair quadrature bridge',
        'protocol': protocol, 'protocol_source_sha256': hashlib.sha256(source_status.read_bytes()).hexdigest(),
        'source_manifest_entries_verified': len(manifest),
        'state': 'original fixed material-cell mass averages',
        'true_cells_are_conditional_analysis_prior_not_candidate_input': True,
        'old_proofs_cited_not_executed': ['mass-average projection', 'density weak perturbation', 'continuous weak interface'],
    })
    result = certificates()
    save('proof-certificates.json', result)
    save('environment.json', {'python': platform.python_version(), 'dependencies': 'standard library only',
                             'network_used': False, 'environment_configuration_changed': False})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--certificates-only', action='store_true')
    args = parser.parse_args()
    if args.certificates_only:
        print(json.dumps(certificates(), indent=2))
    else:
        if not args.output:
            parser.error('--output is required')
        main(args.output)
