"""Exact signed bookkeeping of existing FV fluxes and saved face references.

No conductance, state, geometry or physical reference is constructed here.
Reference q is +rho*D*grad(c), so its positive-axis divergence is the rate.
"""
from fractions import Fraction as F
from periodic_density_path import _fraction


FACES = ('right', 'left', 'top', 'bottom')
SIGNS = {'right': 1, 'left': -1, 'top': 1, 'bottom': -1}
DIRECTIONS = {'right': 'x_plus', 'left': 'x_minus', 'top': 'y_plus', 'bottom': 'y_minus'}


def _two(values, name):
    result = tuple(values)
    if len(result) != 2:
        raise ValueError(f'{name}: exactly two target rows required')
    return result


def signed_face_budget(masses, fv_outward, reference_faces, saved_reference_rates,
                       published_FV_rates, published_errors, global_reference_norm,
                       published_pair_error_norm):
    """Separate interface, other faces, reference and display roundoff exactly.

    Rows are ordered left target/right target at the same interface. If saved
    shared references differ, retain that mismatch and use per-side defects;
    never average or repair them into a shared scalar.
    """
    mass = tuple(_fraction(value, 'mass') for value in _two(masses, 'masses'))
    if any(value <= 0 for value in mass):
        raise ValueError('positive fixed masses required')
    norm = _fraction(global_reference_norm, 'global reference norm')
    pair_norm = _fraction(published_pair_error_norm, 'published pair norm')
    if norm <= 0 or pair_norm < 0:
        raise ValueError('positive reference norm and nonnegative pair norm required')
    rates = tuple(_fraction(value, 'saved PDE rate') for value in _two(saved_reference_rates, 'saved PDE rates'))
    published_rates = tuple(_fraction(value, 'published FV rate') for value in _two(published_FV_rates, 'published FV rates'))
    errors = tuple(_fraction(value, 'published error') for value in _two(published_errors, 'published errors'))
    outward, references = _two(fv_outward, 'outward fluxes'), _two(reference_faces, 'reference faces')
    rows = []
    components = {'interface': [], 'other_three_faces': [], 'reference_rounding': [], 'display_bridge': []}
    for index in range(2):
        if set(outward[index]) != set(DIRECTIONS.values()) or set(references[index]) != set(FACES):
            raise ValueError('exactly four named outward contributions and positive-axis reference faces required')
        fv = {face: SIGNS[face] * _fraction(outward[index][DIRECTIONS[face]], 'outward FV flux') for face in FACES}
        ref = {face: _fraction(references[index][face], 'reference flux') for face in FACES}
        difference = {face: fv[face] - ref[face] for face in FACES}
        contribution = {face: SIGNS[face] * difference[face] / mass[index] for face in FACES}
        interface_face = 'right' if index == 0 else 'left'
        interface = contribution[interface_face]
        other = sum((value for face, value in contribution.items() if face != interface_face), F(0))
        fv_rate = sum((SIGNS[face] * fv[face] for face in FACES), F(0)) / mass[index]
        reference_divergence = sum((SIGNS[face] * ref[face] for face in FACES), F(0)) / mass[index]
        sigma = rates[index] - reference_divergence
        exact_error = fv_rate - rates[index]
        FV_display_rounding = published_rates[index] - fv_rate
        subtraction_rounding = errors[index] - (published_rates[index] - rates[index])
        display_bridge = FV_display_rounding + subtraction_rounding
        assert exact_error == interface + other - sigma
        assert errors[index] == interface + other - sigma + display_bridge
        components['interface'].append(interface)
        components['other_three_faces'].append(other)
        components['reference_rounding'].append(-sigma)
        components['display_bridge'].append(display_bridge)
        rows.append({
            'mass': mass[index], 'FV_positive_axis_faces': fv, 'reference_positive_axis_faces': ref,
            'FV_minus_reference_face_fluxes': difference, 'signed_face_rate_contributions': contribution,
            'interface_face': interface_face, 'interface_rate_component': interface,
            'other_three_faces_rate_component': other,
            'FV_rate_exact_from_saved_fluxes': fv_rate,
            'saved_reference_rate': rates[index], 'reference_face_divergence_exact': reference_divergence,
            'saved_PDE_minus_reference_face_divergence': sigma,
            'exact_FV_minus_saved_PDE_rate': exact_error,
            'published_error': errors[index], 'FV_display_rounding': FV_display_rounding,
            'published_subtraction_rounding': subtraction_rounding,
            'display_bridge': display_bridge,
            'exact_flux_error_budget_closure': exact_error - (interface + other - sigma),
            'published_error_budget_closure': errors[index] - (interface + other - sigma + display_bridge),
        })
    same_fv = rows[0]['FV_positive_axis_faces']['right'] == rows[1]['FV_positive_axis_faces']['left']
    same_ref = rows[0]['reference_positive_axis_faces']['right'] == rows[1]['reference_positive_axis_faces']['left']
    common_defect = rows[0]['FV_minus_reference_face_fluxes']['right'] if same_fv and same_ref else None
    shared = {
        'FV_shared_face_equal': same_fv, 'reference_shared_face_equal': same_ref,
        'FV_left_right_values': [rows[0]['FV_positive_axis_faces']['right'], rows[1]['FV_positive_axis_faces']['left']],
        'reference_left_right_values': [rows[0]['reference_positive_axis_faces']['right'], rows[1]['reference_positive_axis_faces']['left']],
        'reference_right_copy_minus_left_copy': rows[1]['reference_positive_axis_faces']['left'] - rows[0]['reference_positive_axis_faces']['right'],
        'common_interface_flux_defect': common_defect,
        'interface_component_mass_balance': sum((mass[i] * components['interface'][i] for i in range(2)), F(0)),
        'reference_values_averaged_or_repaired': False,
    }
    if common_defect is not None and shared['reference_left_right_values'][0] != 0:
        shared['common_flux_relative_error'] = common_defect / shared['reference_left_right_values'][0]
    else:
        shared['common_flux_relative_error'] = None
    labels = list(components)
    gram = [[sum((mass[row] * components[a][row] * components[b][row] for row in range(2)), F(0)) / norm**2
             for b in labels] for a in labels]
    error_square = sum((mass[row] * errors[row]**2 for row in range(2)), F(0)) / norm**2
    gram_square = sum((sum(line, F(0)) for line in gram), F(0))
    assert gram_square == error_square
    return {
        'rows': rows, 'shared_interface': shared, 'component_labels': labels,
        'signed_rate_components': components, 'normalized_Gram_matrix': gram,
        'Gram_sum': gram_square, 'exact_normalized_published_error_square': error_square,
        'exact_Gram_closure': gram_square - error_square,
        'exact_norm_square_minus_square_of_published_float_norm': error_square - pair_norm**2,
        'published_pair_error_norm': pair_norm,
        'original_global_reference_norm': norm,
        'interface_and_other_cross_term': 2 * gram[0][1],
        'total_pair_inventory_error': sum((mass[row] * errors[row] for row in range(2)), F(0)),
        'norms_are_not_additive_causal_shares': True,
    }
