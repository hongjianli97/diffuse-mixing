"""Theory/reference arithmetic for the unchanged two-cell material FV flux.

Standard library only. Saved transport values are observations, not updated.
The report supplies the calculus proof; certificates check exact coefficients.
"""
from decimal import Decimal as Dec, getcontext, localcontext
from fractions import Fraction as F
from math import factorial


def rational(value):
    return {'exact': str(value), 'decimal_for_reading_only': float(value)}


def decimal(value):
    value = F(value)
    return Dec(value.numerator) / Dec(value.denominator)


def certificates(packet):
    model, saved = packet['model'], packet['saved']
    ell, dy, b, D = (F(model[key]) for key in ['ell', 'dy', 'thickness', 'D'])
    rhoL, rhoR = F(model['rho_left']), F(model['rho_right'])
    area = b * dy
    GL, GR = rhoL * D, rhoR * D
    g, mL, mR = (F(saved[key]) for key in ['interface_g', 'mass_left', 'mass_right'])
    assert all(value > 0 for value in [ell, dy, b, D, rhoL, rhoR, g, mL, mR])
    assert mL == rhoL * area * ell and mR == rhoR * area * ell
    left = [F((-1)**order, (order + 1) * factorial(order)) for order in range(4)]
    right = [F(1, (order + 1) * factorial(order)) for order in range(4)]
    assert left == [F(1), -F(1, 2), F(1, 6), -F(1, 24)]
    assert right == [F(1), F(1, 2), F(1, 6), F(1, 24)]
    assert g * ell / 2 * (1 / GL + 1 / GR) == area
    leading_coefficient = g * ell**2 / 6
    remainder_coefficient = g * ell**3 / 24
    assert leading_coefficient == area * GL * GR * ell / (3 * (GL + GR))
    assert remainder_coefficient == area * GL * GR * ell**2 / (12 * (GL + GR))
    rate_coefficients = [leading_coefficient / mL, -leading_coefficient / mR]
    assert rate_coefficients == [GL * GR / (3 * rhoL * (GL + GR)), -GL * GR / (3 * rhoR * (GL + GR))]
    assert mL * rate_coefficients[0] + mR * rate_coefficients[1] == 0
    # Exact polynomial corollaries, not physical experiments or new fluxes.
    assert leading_coefficient * (F(3) - F(3)) == 0
    assert leading_coefficient * (F(1) - F(0)) > 0
    period = F(model['normal_period'])
    S = period / 2 * (1 / rhoL + 1 / rhoR)
    slopes = [1 / (rhoL * S), 1 / (rhoR * S)]
    assert slopes == [F(8, 5), F(2, 5)]
    amplitude = abs(F(model['amplitude']))
    M3_upper = [amplitude * (2 * F(22, 7) * slope)**3 for slope in slopes]
    face_bound = remainder_coefficient * sum(M3_upper, F(0))
    assert face_bound > 0
    assert F(saved['FV_interface_flux']) == g * (F(saved['cbar_right']) - F(saved['cbar_left']))
    assert F(saved['saved_flux_defect']) == F(saved['FV_interface_flux']) - F(saved['reference_interface_flux'])
    groups = {
        'one_sided_true_average_polynomial_coefficients': {
            'verified': True, 'left_Taylor_coefficients': list(map(str, left)),
            'right_Taylor_coefficients': list(map(str, right)),
            'third_derivative_mean_remainder_coefficient': '1/24',
        },
        'unchanged_saved_series_resistance_and_units_coefficients': {
            'verified': True, 'saved_g': rational(g),
            'constant_flux_coefficient_equals_face_area': rational(area),
            'integrated_curvature_jump_coefficient': rational(leading_coefficient),
            'integrated_third_derivative_remainder_coefficient': rational(remainder_coefficient),
            'one_sided_rate_coefficients': list(map(rational, rate_coefficients)),
            'mass_weighted_interface_error_cancels': True,
        },
        'common_curvature_and_independent_curvature_corollaries': {
            'verified': True, 'common_curvature_leading_term_zero': True,
            'independent_quadratic_curvature_jump_has_nonzero_exact_face_defect': True,
            'scope': 'unchanged two-point law; not a no-go for all positive discretizations or transient convergence',
        },
        'rational_third_derivative_bound_for_original_background': {
            'verified': True, 'normal_phase_coordinate_slopes': list(map(str, slopes)),
            'pi_upper_bound_cited': '22/7', 'amplitude_upper_bound': str(amplitude),
            'M3_left_right_upper_bounds': list(map(rational, M3_upper)),
            'integrated_face_remainder_bound': rational(face_bound),
            'one_sided_rate_remainder_bounds': [rational(face_bound / mL), rational(face_bound / mR)],
            'uses_no_saved_flux_defect_to_choose_bound': True,
        },
    }
    return {'groups': groups, 'verified_exact_algebra_groups': len(groups),
            'proof_of_integral_and_regularity_statements_in_report': True,
            'new_numerical_or_flow_tests': 0, 'recorded_checks_total': 388,
            'new_edge_weights': 0, 'new_candidate_graphs': 0, 'fluid_orc_enabled': False}


def _atan(value):
    term, total, index = value, value, 1
    epsilon = Dec(1).scaleb(-getcontext().prec - 5)
    while True:
        term *= -value * value
        contribution = term / (2 * index + 1)
        total += contribution
        if abs(contribution) < epsilon:
            return total
        index += 1
        if index > 10000:
            raise ArithmeticError('arctangent series did not converge')


def _pi_machin():
    return 16 * _atan(Dec(1) / 5) - 4 * _atan(Dec(1) / 239)


def _pi_agm():
    a, b, t, multiplier = Dec(1), Dec(1) / Dec(2).sqrt(), Dec(1) / 4, Dec(1)
    for _ in range(20):
        previous = a
        a, b = (a + b) / 2, (a * b).sqrt()
        t -= multiplier * (previous - a)**2
        multiplier *= 2
        if abs(a - b) < Dec(1).scaleb(-getcontext().prec + 3):
            return (a + b)**2 / (4 * t)
    raise ArithmeticError('AGM pi iteration did not converge')


def _sin_cos(value):
    sine_term, cosine_term = value, Dec(1)
    sine, cosine = sine_term, cosine_term
    epsilon = Dec(1).scaleb(-getcontext().prec - 5)
    for index in range(1, 10000):
        sine_term *= -value**2 / ((2 * index) * (2 * index + 1))
        cosine_term *= -value**2 / ((2 * index - 1) * (2 * index))
        sine += sine_term
        cosine += cosine_term
        if max(abs(sine_term), abs(cosine_term)) < epsilon:
            return sine, cosine
    raise ArithmeticError('trigonometric series did not converge')


def _sinc(value):
    return Dec(1) if value == 0 else _sin_cos(value)[0] / value


def analytic_reference(model):
    """Only declared model parameters; no saved error, state or conductance."""
    ell, dy, Ly, y0 = (decimal(model[key]) for key in ['ell', 'dy', 'transverse_period', 'y_center'])
    rhoL, rhoR, D, amplitude = (decimal(model[key]) for key in ['rho_left', 'rho_right', 'D', 'amplitude'])
    Lx = decimal(model['normal_period'])
    pi = _pi_machin()
    S = Lx / 2 * (1 / rhoL + 1 / rhoR)
    slopes = [1 / (rhoL * S), 1 / (rhoR * S)]
    xi_face = (Lx / (2 * rhoL)) / S
    phi = 2 * pi * xi_face + pi * decimal(model['phase_pi_ratio'])
    sine, cosine = _sin_cos(phi)
    transverse_sinc = _sinc(pi * dy / Ly)
    transverse_cosine = _sin_cos(2 * pi * y0 / Ly)[1]
    amplitude_strip = amplitude * transverse_sinc * transverse_cosine
    wave_numbers = [2 * pi * slope for slope in slopes]
    slopes_C = [-amplitude_strip * wave * sine for wave in wave_numbers]
    curvatures = [-amplitude_strip * wave**2 * cosine for wave in wave_numbers]
    face_state = Dec(1) / 2 + amplitude_strip * cosine
    meanL = Dec(1) / 2 + amplitude_strip * _sinc(wave_numbers[0] * ell / 2) * _sin_cos(phi - wave_numbers[0] * ell / 2)[1]
    meanR = Dec(1) / 2 + amplitude_strip * _sinc(wave_numbers[1] * ell / 2) * _sin_cos(phi + wave_numbers[1] * ell / 2)[1]
    remainders = [meanL - (face_state - slopes_C[0] * ell / 2 + curvatures[0] * ell**2 / 6),
                  meanR - (face_state + slopes_C[1] * ell / 2 + curvatures[1] * ell**2 / 6)]
    # Independent representations at the prescribed original phase/strip.
    assert F(model['phase_pi_ratio']) == F(1, 5) and F(model['rho_left']) == 1 and F(model['rho_right']) == 4
    assert F(model['dy']) / F(model['transverse_period']) == F(1, 128)
    assert F(model['y_center']) == F(model['dy']) / 2
    radical_cosine = (1 + Dec(5).sqrt()) / 4
    radical_sine = -(10 - 2 * Dec(5).sqrt()).sqrt() / 4
    half_cosine = Dec(0)  # pi/2, halved five times to pi/64
    for _ in range(5):
        half_sine = ((1 - half_cosine) / 2).sqrt()
        half_cosine = ((1 + half_cosine) / 2).sqrt()
    direct_strip_amplitude = amplitude * _sinc(2 * pi * dy / Ly)
    return {
        'pi': pi, 'strip_amplitude': amplitude_strip, 'phase_sine': sine, 'phase_cosine': cosine,
        'left_right_curvatures': curvatures, 'face_state': face_state,
        'left_right_means': [meanL, meanR], 'left_right_mean_Taylor_remainders': remainders,
        'matched_J_left': rhoL * D * slopes_C[0], 'matched_J_right': rhoR * D * slopes_C[1],
        'normal_wave_numbers': wave_numbers,
        'independent_arithmetic_differences': {
            'pi_Machin_minus_AGM': pi - _pi_agm(),
            'phase_cosine_minus_radical': cosine - radical_cosine,
            'phase_sine_minus_radical': sine - radical_sine,
            'strip_amplitude_product_minus_direct': amplitude_strip - direct_strip_amplitude,
            'transverse_sine_minus_half_angle': _sin_cos(2 * pi * dy / Ly)[0] - half_sine,
        },
    }


def _render(value, precision):
    if isinstance(value, Dec):
        with localcontext() as context:
            context.prec = precision
            return str(+value)
    if isinstance(value, dict):
        return {key: _render(item, precision) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_render(item, precision) for item in value]
    return value


def reference_comparison(packet, precision):
    model, saved = packet['model'], packet['saved']
    cert = certificates(packet)
    with localcontext() as context:
        context.prec = precision + 30
        ref = analytic_reference(model)
        ell, dy, b = (decimal(model[key]) for key in ['ell', 'dy', 'thickness'])
        g = decimal(saved['interface_g'])
        qL, qR = ref['left_right_curvatures']
        leading = g * (qR - qL) * ell**2 / 6
        remainder = g * (ref['left_right_mean_Taylor_remainders'][1] - ref['left_right_mean_Taylor_remainders'][0])
        mean_bridge = g * ((decimal(saved['cbar_right']) - ref['left_right_means'][1])
                           - (decimal(saved['cbar_left']) - ref['left_right_means'][0]))
        reference_bridge = b * dy * ref['matched_J_left'] - decimal(saved['reference_interface_flux'])
        observed = decimal(saved['saved_flux_defect'])
        closure = observed - (leading + remainder + mean_bridge + reference_bridge)
        upper_exact = cert['groups']['rational_third_derivative_bound_for_original_background']['integrated_face_remainder_bound']['exact']
        bound = decimal(upper_exact)
        arithmetic_tolerance = Dec(1).scaleb(-precision + 5)
        differences = ref['independent_arithmetic_differences']
        assert all(abs(value) < arithmetic_tolerance for value in differences.values())
        assert abs(closure) < arithmetic_tolerance
        assert abs(remainder) <= bound
        assert abs(observed - leading) <= bound + abs(mean_bridge) + abs(reference_bridge) + arithmetic_tolerance
        mL, mR = decimal(saved['mass_left']), decimal(saved['mass_right'])
        result = {
            'requested_decimal_digits': precision, 'working_decimal_digits': context.prec,
            'analytic_reference': ref,
            'saved_integrated_flux_defect': observed,
            'curvature_jump_leading_integrated_flux_defect': leading,
            'actual_model_third_order_integrated_remainder': remainder,
            'rational_rigorous_face_remainder_upper_bound_exact': upper_exact,
            'rational_bound_decimal_readout': bound,
            'saved_minus_leading': observed - leading,
            'saved_mean_rounding_bridge': mean_bridge,
            'saved_reference_face_rounding_bridge': reference_bridge,
            'arithmetic_closure_residual_not_forced_to_zero': closure,
            'predicted_leading_rate_components': [leading / mL, -leading / mR],
            'model_remainder_rate_components': [remainder / mL, -remainder / mR],
            'remainder_bound_rate_components_abs': [bound / mL, bound / mR],
            'leading_over_saved_defect_diagnostic_only': leading / observed,
            'saved_defect_within_leading_plus_proved_remainder_and_recorded_bridges': True,
            'arithmetic_tolerance_is_not_physical_acceptance_threshold': True,
            'reference_values_not_applied_to_transport': True,
        }
        return _render(result, precision)


def analyse(packet):
    cert = certificates(packet)
    controls = [reference_comparison(packet, precision) for precision in [60, 90]]
    with localcontext() as context:
        context.prec = 100
        keys = ['curvature_jump_leading_integrated_flux_defect', 'actual_model_third_order_integrated_remainder',
                'saved_mean_rounding_bridge', 'saved_reference_face_rounding_bridge']
        precision_difference = {key: str(Dec(controls[1][key]) - Dec(controls[0][key])) for key in keys}
    return {'exact_certificates': cert, 'arithmetic_precision_controls': controls,
            'ninety_minus_sixty_digit_differences': precision_difference,
            'new_numerical_or_flow_tests': 0, 'new_exact_algebra_certificate_groups': 4,
            'recorded_checks_total': 388, 'new_edge_weights': 0, 'new_candidate_graphs': 0,
            'new_reference_integrations': 0, 'new_candidate_transient_runs': 0,
            'new_reference_transient_runs': 0, 'fluid_orc_enabled': False,
            'stage3_physical_acceptance': 'incomplete'}
