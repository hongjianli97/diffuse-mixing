"""One saved contact operator: error forcing, Duhamel response and averages.

No backend or physical parameter changes. All candidate states are loaded,
and forcing responses are diagnostic signed fields, not corrected solutions.
"""
from pathlib import Path
import argparse, hashlib, json, platform, time
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.sparse import diags, load_npz
from scipy.sparse.linalg import expm_multiply
from fixed_width_kernel import build_constant_exchange
from interface_error_forcing import scaled_forcing
from kernel_joint_study import verify
from positive_kernel_1d import slope_over_radius
from spatial_study import normalized_error
from step_image_profile import HeatStepImages


def affine_weak_moments(h, F, contrast):
    # Independent continuous half-line quadrature; the residual is even.
    def residual(u):
        return -2*F*contrast*quad(lambda v: float(slope_over_radius(u+v, h))*(v-u),
                                0., 2*h-u, epsabs=1e-13)[0]
    values = {}
    for power in (0, 2):
        values[str(power)] = 2*quad(lambda u: u**power*residual(u), 0., 2*h, epsabs=1e-13)[0]
    return {'h': h, 'flux': F, 'contrast': contrast,
            'weighted_trace_residual': residual(0.),
            'mass_moment': values['0'], 'first_moment_by_even_symmetry': 0.,
            'second_moment': values['2'], 'second_moment_exact': (12/55)*F*contrast*h*h,
            'weak_leading_coefficient': (6/55)*F*contrast*h*h,
            'scope': 'local affine transmission patch only, not the full transient error'}


def main(source, output):
    root = Path(__file__).resolve().parents[2]
    source = Path(source).resolve(); output = Path(output).resolve(); verify(root, source)
    cfg0 = json.loads((source/'config.json').read_text()); physical = cfg0['physical_configuration']
    n = max(cfg0['cells']); T = cfg0['time']
    with np.load(source/f'physical-N{n}.npz') as z:
        faces = z['material_faces']; m = z['mass']; V = z['volume']; x = z['positions']
        c0 = z['c0']; candidate = z['kernel_c']; exact = z['reference']; fv = z['FV_c']
        I = z['pair_i']; J = z['pair_j']; g = z['conductance']; h = float(z['h'])
    parent_n = min(cfg0['cells'])
    with np.load(source/f'physical-N{parent_n}.npz') as z:
        pm = z['mass']; pc0 = z['c0']; pref = z['reference']; pf = z['material_faces']
    if n % parent_n or not np.array_equal(faces[::n//parent_n], pf):
        raise RuntimeError('saved observable grid is not a nested material partition')
    index = np.repeat(np.arange(parent_n), n//parent_n)
    L = load_npz(source/f'operator-N{n}.npz'); Q = -diags(1/m)@L
    L0, _, _ = build_constant_exchange(x, V, V, np.ones_like(x), physical['D'], h, (-1., 1.))
    Q0 = -diags(1/V)@L0
    images = HeatStepImages(D=physical['D'], half_length=physical['half_length'], images=2)
    output.mkdir(parents=True, exist_ok=False)
    cfg = {'study': 'same saved N2048,T0.1,h0.0078125 operator; exact error-response decomposition',
           'source': str(source), 'physical_configuration': physical, 'cells': n, 'time': T, 'h': h,
           'observable_parent_cells': parent_n, 'reference_images': images.images,
           'reference_time_coordinate': 'tau=sqrt(t); analytic integrable endpoint',
           'quadrature_nodes': [128, 256], 'diagnostic_absolute_tolerance': 1e-10,
           'tolerance_scope': 'numerical decomposition check only, no physical acceptance threshold',
           'split': 'phasewise homogeneous heat-profile continuation plus exact density-interface cross-pair remainder',
           'no_fitted_coefficients': True, 'no_candidate_correction': True,
           'no_new_physical_evolution': True, 'physical_acceptance': False, 'fluid_orc_enabled': False}
    (output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    kwargs = {'positions': x, 'pair_i': I, 'pair_j': J, 'conductance': g,
              'rho_left': physical['rho_left'], 'rho_right': physical['rho_right'],
              'left': physical['initial_left'], 'right': physical['initial_right']}
    error = candidate-exact
    def norm(v):
        return normalized_error(m, c0, v)
    def aggregate(v):
        return np.bincount(index, weights=m*v)/pm
    chiT, derivativeT = images.average_and_tau_derivative(faces, np.sqrt(T))
    forcingT, image_ref, diagT = scaled_forcing(Q, Q0, m, chiT, derivativeT, np.sqrt(T), **kwargs)
    refcheck = float(np.max(abs(image_ref-exact)))
    if refcheck > 1e-12 or np.max(abs(aggregate(exact)-pref)) > 1e-12:
        raise RuntimeError('independent reference or material aggregation mismatch')
    records = []; responses = []
    for nodes in cfg['quadrature_nodes']:
        tic = time.monotonic(); z, w = np.polynomial.legendre.leggauss(nodes)
        taus = (z+1)*np.sqrt(T)/2; weights = w*np.sqrt(T)/2
        response = np.zeros((n, 2)); algebra = 0.; imass = bmass = 0.
        for k, (tau, weight) in enumerate(zip(taus, weights)):
            chi, ctau = images.average_and_tau_derivative(faces, tau)
            forcing, _, diagnostics = scaled_forcing(Q, Q0, m, chi, ctau, tau, **kwargs)
            algebra = max(algebra, diagnostics['algebra_closure_max_abs'])
            imass = max(imass, abs(diagnostics['interface_mass_forcing']))
            bmass = max(bmass, abs(diagnostics['bulk_mass_forcing']))
            lag = T-tau*tau
            propagated = expm_multiply(lag*Q, forcing, traceA=float(lag*Q.diagonal().sum()))
            response += weight*propagated
            if (k+1) % 64 == 0:
                print('Duhamel nodes', nodes, 'evaluated', k+1, flush=True)
        closure = response.sum(axis=1)-error
        record = {'nodes': nodes, 'closure_normalized_error': norm(closure),
                  'closure_max_abs': float(np.max(abs(closure))),
                  'interface_response_error_norm': norm(response[:, 0]),
                  'homogeneous_response_error_norm': norm(response[:, 1]),
                  'forcing_algebra_max_abs_residual': algebra,
                  'interface_forcing_mass_max_abs': imass, 'homogeneous_forcing_mass_max_abs': bmass,
                  'wall_seconds': time.monotonic()-tic}
        records.append(record); responses.append(response)
        np.savez_compressed(output/f'response-Gauss{nodes}.npz', tau_nodes=taus, tau_weights=weights,
                            interface_response=response[:, 0], homogeneous_response=response[:, 1], closure=closure)
        print(json.dumps(record), flush=True)
    fine = responses[-1]; coarse = responses[0]
    delta = [norm(fine[:, k]-coarse[:, k]) for k in range(2)]
    right = x > 0
    prior = json.loads((source/'results.json').read_text())['physical_runs'][-1]
    invref = prior['reference_right_inventory']
    variance = float(m@((c0-m@c0/m.sum())**2))
    observed = {'complete_error': norm(error), 'interface_response_norm': norm(fine[:, 0]),
                'homogeneous_response_norm': norm(fine[:, 1]),
                'normalized_cross_term': float(2*m@(fine[:, 0]*fine[:, 1])/variance),
                'interface_inventory_relative_error_contribution': float(m[right]@fine[right, 0])/invref,
                'homogeneous_inventory_relative_error_contribution': float(m[right]@fine[right, 1])/invref,
                'total_inventory_signed_relative_error': prior['kernel_inventory_signed_relative_error'],
                'total_error_squared_fraction_within_2h': float(m[abs(x)<2*h]@(error[abs(x)<2*h]**2)/(m@(error*error))),
                'parent_128_total_error': normalized_error(pm, pc0, aggregate(error)),
                'parent_128_interface_norm': normalized_error(pm, pc0, aggregate(fine[:, 0])),
                'parent_128_homogeneous_norm': normalized_error(pm, pc0, aggregate(fine[:, 1])),
                'parent_128_FV_error': normalized_error(pm, pc0, aggregate(fv-exact))}
    resolved = max(delta+[records[-1]['closure_max_abs'], records[-1]['closure_normalized_error']]) <= cfg['diagnostic_absolute_tolerance']
    result = {'responses': records, 'quadrature_response_changes': delta,
              'independent_image_reference_max_abs_difference': refcheck,
              'image_log_concentration_tail_bound_at_T': images.log_concentration_tail_bound(T),
              'observed': observed, 'numerical_decomposition_resolved': resolved,
              'physical_acceptance': False,
              'limits': ['one fixed saved operator','interface source is density-specific, homogeneous source includes finite h and node quadrature',
                         'forcing responses are not corrected physical solutions','point counterexample is not proof of weak inconsistency']}
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    moments = affine_weak_moments(h, .001, (physical['rho_left']-physical['rho_right'])/(physical['rho_left']+physical['rho_right']))
    (output/'affine-weak-moments.json').write_text(json.dumps(moments, indent=2)+'\n')
    np.savez_compressed(output/'decomposition.npz', positions=x, mass=m, c0=c0, candidate=candidate,
                        reference=exact, total_error=error, interface_response=fine[:, 0], homogeneous_response=fine[:, 1],
                        scaled_interface_forcing_at_T=forcingT[:, 0], scaled_homogeneous_forcing_at_T=forcingT[:, 1],
                        h=np.array(h), parent_mass=pm, parent_reference=pref, parent_c0=pc0, parent_index=index,
                        parent_total_error=aggregate(error), parent_interface_response=aggregate(fine[:, 0]),
                        parent_homogeneous_response=aggregate(fine[:, 1]))
    (output/'environment.json').write_text(json.dumps({'python': platform.python_version(), 'numpy': np.__version__,
                                                      'scipy': scipy.__version__, 'BLAS_threads_command': 4}, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('--output', required=True)
    a = p.parse_args(); main(a.source, a.output)
