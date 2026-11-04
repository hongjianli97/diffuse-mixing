"""Exact admissibility certificates for a formal four-material-mean flux.

Standard library only. This module builds a 4x4 constraint system, not a
transport graph or matrix. Its signed coefficients are never installed.
"""
from fractions import Fraction as F


def encode(value):
    if isinstance(value, F):
        return {'exact': str(value), 'decimal_for_reading_only': float(value)}
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def monomial_mean(lower, upper, degree):
    return (upper**(degree + 1) - lower**(degree + 1)) / ((degree + 1) * (upper - lower))


def solve_exact(matrix, rhs):
    """Rational elimination for the four predeclared equality constraints."""
    n = len(rhs)
    rows = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    determinant, swaps = F(1), 0
    for column in range(n):
        pivot_row = next((i for i in range(column, n) if rows[i][column]), None)
        if pivot_row is None:
            raise ValueError('singular four-mean exactness constraints')
        if pivot_row != column:
            rows[column], rows[pivot_row] = rows[pivot_row], rows[column]
            swaps += 1
        pivot = rows[column][column]
        determinant *= pivot
        rows[column] = [value / pivot for value in rows[column]]
        for i in range(n):
            if i != column:
                factor = rows[i][column]
                rows[i] = [left - factor * right for left, right in zip(rows[i], rows[column])]
    return [row[-1] for row in rows], determinant * (-1)**swaps


def review(packet):
    model, saved = packet['geometry_and_material'], packet['saved_baseline_coefficients']
    ell, area, D, rhoL, rhoR = (F(model[key]) for key in ['ell', 'face_area', 'D', 'rho_left', 'rho_right'])
    g, gL, gR, mL, mR = (F(saved[key]) for key in ['interface_g', 'left_outer_g', 'right_outer_g', 'mass_left', 'mass_right'])
    assert all(value > 0 for value in [ell, area, D, rhoL, rhoR, g, gL, gR, mL, mR])
    GL, GR = rhoL * D, rhoR * D
    assert g * ell / 2 * (1 / GL + 1 / GR) == area
    assert gL * ell == area * GL and gR * ell == area * GR
    assert mL == rhoL * area * ell and mR == rhoR * area * ell
    intervals = [(-2 * ell, -ell), (-ell, F(0)), (F(0), ell), (ell, 2 * ell)]
    means = [[monomial_mean(lo, hi, degree) for degree in range(3)] for lo, hi in intervals]
    assert means == [[F(1), -3*ell/2, 7*ell**2/3], [F(1), -ell/2, ell**2/3],
                     [F(1), ell/2, ell**2/3], [F(1), 3*ell/2, 7*ell**2/3]]
    basis = [
        [F(1)] * 4,
        [means[i][1] / (GL if i < 2 else GR) for i in range(4)],
        [means[0][2] / 2, means[1][2] / 2, F(0), F(0)],
        [F(0), F(0), means[2][2] / 2, means[3][2] / 2],
    ]
    rhs = [F(0), area, F(0), F(0)]
    w, determinant = solve_exact(basis, rhs)
    assert determinant != 0
    assert w == [g/4, -7*g/4, 7*g/4, -g/4]
    residual = [sum((a*b for a, b in zip(row, w)), F(0)) - value for row, value in zip(basis, rhs)]
    assert residual == [F(0)] * 4
    interface_rows = [[value / mL for value in w], [-value / mR for value in w]]
    # Two formal normal rows only; all other face closures remain unchanged.
    rows = [
        [(w[0]+gL)/mL, (w[1]-gL)/mL, w[2]/mL, w[3]/mL],
        [-w[0]/mR, -w[1]/mR, (-w[2]-gR)/mR, (-w[3]+gR)/mR],
    ]
    assert all(sum(row, F(0)) == 0 for row in rows)
    shared_mass_balance = [mL*interface_rows[0][i] + mR*interface_rows[1][i] for i in range(4)]
    assert shared_mass_balance == [F(0)] * 4
    assert rows[0][3] == -g/(4*mL) < 0
    assert rows[1][0] == -g/(4*mR) < 0
    witnesses = []
    for row_index, remote_index in [(0, 3), (1, 0)]:
        state = [F(0)] * 4
        state[remote_index] = F(1, 2)
        derivative = sum((a*b for a, b in zip(rows[row_index], state)), F(0))
        target_index = [1, 2][row_index]
        assert state[target_index] == 0 and all(0 <= value <= 1 for value in state)
        assert derivative < 0
        witnesses.append({'row_target_id': packet['ordered_cell_ids'][target_index],
                          'remote_nonzero_cell_id': packet['ordered_cell_ids'][remote_index],
                          'nonnegative_discrete_material_mean_vector': state,
                          'formal_initial_target_derivative_per_s': derivative,
                          'not_a_transient_simulation': True, 'not_the_saved_original_c0': True})
    groups = {
        'true_material_mean_basis_integrals': {'verified': True, 'intervals': intervals,
            'means_of_one_a_a_squared': means, 'phase_constant_rho_cancels_only_inside_each_material_cell': True},
        'unique_data_independent_exact_reconstruction': {'verified': True,
            'constraint_order': ['constant', 'matched_J', 'left_Q', 'right_Q'],
            'constraint_matrix': basis, 'rhs': rhs, 'determinant': determinant, 'rank': 4,
            'formal_integrated_flux_coefficients': w, 'exact_constraint_residuals': residual,
            'coefficient_formula': 'g/4 * [1,-7,7,-1]', 'saved_c0_or_flux_error_not_used': True},
        'shared_flux_and_unchanged_outer_closure_accounting': {'verified': True,
            'interface_only_rate_rows_per_s': interface_rows, 'two_formal_normal_rate_rows_per_s': rows,
            'row_constant_residuals': [sum(row, F(0)) for row in rows],
            'mass_weighted_interface_column_sums': shared_mass_balance,
            'open_pair_outer_fluxes_not_claimed_to_cancel': True,
            'unchanged_transverse_faces_do_not_couple_these_remote_normal_cells': True},
        'nonnegative_generator_obstruction': {'verified': True,
            'negative_remote_offdiagonal_coefficients_per_s': [rows[0][3], rows[1][0]],
            'nonnegative_initial_mean_witnesses': witnesses,
            'nonnegative_exchange_representation_with_unchanged_outer_faces': False,
            'scope': 'this four-mean linear shared interface closure with all stated exactness conditions and unchanged other faces',
            'not_a_no_go_for_all_positive_methods_or_weak_or_transient_convergence': True},
    }
    return encode({'groups': groups, 'verified_exact_algebra_groups': len(groups),
        'decision': 'reject_this_four_mean_linear_correction_for_nonnegative_exchange_installation',
        'formal_flux_coefficients_are_not_installed_transport_weights': True,
        'ordinary_central_resistance_FV_baseline_retained': True,
        'mass_average_resistance_remains_experimental': True,
        'original_c0_diagnostic_and_physical_acceptance_criteria_retained': True,
        'new_exact_algebra_certificate_groups': 4, 'new_numerical_or_flow_tests': 0,
        'recorded_checks_total': 388, 'previous_checks_rerun': 0,
        'new_edge_weights': 0, 'new_candidate_graphs': 0, 'new_global_matrices': 0,
        'new_reference_integrations': 0, 'LP_solves': 0,
        'new_candidate_transient_runs': 0, 'new_reference_transient_runs': 0,
        'finite_node_candidate_admitted': False, 'stage3_physical_acceptance': 'incomplete',
        'fluid_orc_enabled': False})
