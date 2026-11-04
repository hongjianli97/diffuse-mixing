"""Exact algebra supporting the new density-reconstruction weak perturbation.

No old proof is executed. No graph, operator quadrature or numerical tests.
Integral/regularity arguments are in DENSITY_RECONSTRUCTION_WEAK_BOUND.md.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse
import hashlib
import json
import platform


def rational(value):
    return {'exact': str(value), 'decimal_for_reading_only': float(value)}


def multiply(left, right):
    result = {}
    for i, a in left.items():
        for j, b in right.items():
            exponent = tuple(x + y for x, y in zip(i, j))
            result[exponent] = result.get(exponent, F(0)) + a * b
    return {key: value for key, value in result.items() if value}


def certificates():
    # A=a_min+u, Ahat=a_min+v. This polynomial has nonnegative slack.
    product = multiply({(1, 0, 0): F(1), (0, 1, 0): F(1)},
                       {(1, 0, 0): F(1), (0, 0, 1): F(1)})
    expected = {(2, 0, 0): F(1), (1, 1, 0): F(1),
                (1, 0, 1): F(1), (0, 1, 1): F(1)}
    assert product == expected
    slack = {key: value for key, value in product.items() if key != (2, 0, 0)}
    assert len(slack) == 3 and all(value > 0 for value in slack.values())
    # Use the already established kernel second moment; do not reprove it.
    mu2 = F(2)
    lower, upper, D, tube_coefficient = F(1), F(4), F(1, 50), F(4)
    gamma_max = D * upper
    inverse_oscillation = 1 / (D * lower) - 1 / (D * upper)
    inverse_bound = mu2 * gamma_max**2 * tube_coefficient * inverse_oscillation
    assert gamma_max == F(2, 25) and inverse_oscillation == F(75, 2)
    assert inverse_bound == F(48, 25)
    density_prefactor = mu2 * D * (upper / lower)**2
    density_bound = density_prefactor * tube_coefficient * (upper - lower)
    assert density_prefactor == F(16, 25) and density_bound == F(192, 25)
    doubled_D_bound = mu2 * (2 * D * upper)**2 * tube_coefficient * (
        1 / (2 * D * lower) - 1 / (2 * D * upper))
    assert doubled_D_bound == 2 * inverse_bound
    # Dimensionless norm example, not a generated physical input or graph.
    strip_length = F(33, 64) - F(31, 64)
    segment_length = F(17, 32) - F(15, 32)
    fraction = strip_length / segment_length
    path_inverse_average = (1 - fraction) * F(1) + fraction * F(2)
    path_coefficient = 1 / path_inverse_average
    assert strip_length == F(1, 32) and fraction == F(1, 2)
    assert segment_length == F(1, 16) < F(2, 25)
    assert path_inverse_average == F(3, 2) and path_coefficient == F(2, 3)
    assert F(1) - path_coefficient == F(1, 3)
    groups = {
        'positive_inverse_average_sensitivity_denominator_slack': {
            'verified': True,
            'variables': ['a_min>0', 'u>=0', 'v>=0'],
            'identity': '(a_min+u)*(a_min+v)-a_min^2=a_min*u+a_min*v+u*v',
            'positive_slack_coefficients': {','.join(map(str, key)): str(value)
                                            for key, value in sorted(slack.items())},
            'result': '1/(A*Ahat)<=1/a_min^2=Gamma_max^2',
        },
        'original_flat_contact_new_bound_constants': {
            'verified': True,
            'cited_existing_kernel_second_moment': rational(mu2),
            'Gamma_max': rational(gamma_max), 'inverse_Gamma_oscillation': rational(inverse_oscillation),
            'tube_area_coefficient_two_unit_interfaces': rational(tube_coefficient),
            'coefficient_of_b_Lc_Lphi_epsilon': rational(inverse_bound),
            'coarser_density_L1_prefactor_excluding_b_Lc_Lphi': rational(density_prefactor),
            'coarser_density_coefficient_of_b_Lc_Lphi_epsilon': rational(density_bound),
            'doubling_D_doubles_bound': doubled_D_bound == 2 * inverse_bound,
            'no_sample_fill_distance_or_operator_response_evaluated': True,
        },
        'area_L1_does_not_bound_a_prescribed_path': {
            'verified': True,
            'L1_error_over_tau': rational(strip_length),
            'fixed_path_fraction_in_strip': rational(fraction),
            'path_inverse_average': rational(path_inverse_average),
            'path_coefficient': rational(path_coefficient),
            'fixed_path_coefficient_difference': rational(F(1) - path_coefficient),
            'scope': 'norm/trace distinction only; not a nearest-point or fixed-mass physical counterexample',
        },
    }
    return {
        'certificate_groups': groups, 'verified_exact_algebra_groups': len(groups),
        'analytic_proofs_are_in_report': True,
        'continuous_density_weak_perturbation_proved_under_stated_assumptions': True,
        'nearest_point_L1_bound_proved_under_stated_assumptions': True,
        'finite_node_convergence_proved': False,
        'new_numerical_or_flow_tests': 0, 'recorded_checks_total': 363,
        'new_candidate_graphs': 0, 'new_edge_weights': 0,
        'new_operator_quadrature_evaluations': 0, 'LP_solves': 0,
        'new_candidate_transient_runs': 0, 'new_reference_transient_runs': 0,
        'fluid_orc_enabled': False, 'stage3_physical_acceptance': 'incomplete',
    }


def main(output):
    root = Path(__file__).resolve().parents[2]
    source = root / 'runs/reconnect-route-decision'
    manifest = json.loads((source / 'sha256.json').read_text())
    # First verify sealed sources; no old proof/candidate code is imported.
    for rel, expected in manifest.items():
        with (root / rel).open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == expected, rel
    status_path = source / 'stage3-status.json'
    protocol = json.loads(status_path.read_text())['next_density_closure_weak_perturbation_predeclared']
    assert protocol['theory_only'] and not protocol['new_graphs_or_numerical_cases']
    assert protocol['proof_completed'] is False
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        with (out / name).open('x') as f:
            f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    save('config.json', {'study': 'predeclared density reconstruction continuous weak perturbation',
        'protocol': protocol, 'protocol_source_sha256': hashlib.sha256(status_path.read_bytes()).hexdigest(),
        'source_manifest_entries_verified': len(manifest),
        'state': 'original fixed material-cell mass averages; continuous fields used only in proof',
        'mass_measure_unchanged': True, 'rho_hat_used_only_for_path_resistance': True,
        'old_kernel_moments_weak_interface_and_projection_proofs_cited_not_rerun': True})
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
