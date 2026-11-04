"""Independent steady inner layer and unfitted transient predictions.

Reuses saved physical states and one saved forcing response. Only one earlier
width receives an additional interface-forcing quadrature; no new physical
candidate evolution, width selection or coefficient fit is performed.
"""
from pathlib import Path
import argparse, json, platform
import numpy as np
import scipy
from scipy.sparse import diags, load_npz
from scipy.sparse.linalg import expm_multiply
from scipy.special import erfc
from contact import Contact
from contact_step_reference import StepReference
from inner_cell_problem import solve_inner
from kernel_joint_study import verify
from spatial_study import normalized_error
from step_image_profile import HeatStepImages


def gaussian_average(faces, D, T):
    lo = np.minimum(abs(faces[:-1]), abs(faces[1:]))
    hi = np.maximum(abs(faces[:-1]), abs(faces[1:]))
    w = 2*np.sqrt(D*T)
    return np.sqrt(np.pi*D*T)*(erfc(lo/w)-erfc(hi/w))/np.diff(faces)


def inner_average(arrays, target_faces, region=4.):
    """Exact averaging of the computed piecewise-constant analysis field."""
    sf = arrays['faces']; psi = arrays['localized_corrector']
    chosen = (target_faces[:-1] >= -region) & (target_faces[1:] <= region)
    indices = np.flatnonzero(chosen); values = []
    for i in indices:
        a, b = target_faces[i:i+2]
        ia = np.searchsorted(sf, a); ib = np.searchsorted(sf, b)
        if ia >= len(sf) or ib >= len(sf) or sf[ia] != a or sf[ib] != b:
            raise ValueError('physical observable must align with inner analysis cells')
        values.append(np.dot(np.diff(sf[ia:ib+1]), psi[ia:ib])/(b-a))
    return indices, np.asarray(values)


def interface_response(joint, n, physical, T, output):
    with np.load(joint/f'physical-N{n}.npz') as z:
        faces = z['material_faces']; m = z['mass']; c0 = z['c0']; x = z['positions']
        I = z['pair_i']; J = z['pair_j']; g = z['conductance']
    Q = -diags(1/m)@load_npz(joint/f'operator-N{n}.npz')
    cross = (x[I] < 0)&(x[J] > 0); i = I[cross]; j = J[cross]; g = g[cross]
    beta = (physical['rho_left']-physical['rho_right'])/(physical['rho_left']+physical['rho_right'])
    jump = physical['initial_left']-physical['initial_right']
    images = HeatStepImages(D=physical['D'], half_length=physical['half_length'], images=2)
    rows = []; responses = []
    for count in (128, 256):
        z, w = np.polynomial.legendre.leggauss(count)
        taus = (z+1)*np.sqrt(T)/2; weights = w*np.sqrt(T)/2
        response = np.zeros(n)
        for tau, weight in zip(taus, weights):
            chi, _ = images.average_and_tau_derivative(faces, tau)
            value = tau*jump*beta*g*(chi[i]+chi[j])
            force = (np.bincount(j, weights=value, minlength=n)-np.bincount(i, weights=value, minlength=n))/m
            lag = T-tau*tau
            response += weight*expm_multiply(lag*Q, force, traceA=float(lag*Q.diagonal().sum()))
        rows.append({'nodes': count, 'response_norm': normalized_error(m, c0, response),
                     'response_mass': float(m@response)})
        responses.append(response)
        np.savez_compressed(output/f'interface-N{n}-Gauss{count}.npz', response=response, tau_nodes=taus, tau_weights=weights)
        print('additional saved width', n, 'interface Gauss nodes', count, flush=True)
    difference = normalized_error(m, c0, responses[1]-responses[0])
    return responses[-1], {'quadratures': rows, 'normalized_change': difference,
                            'numerically_resolved': difference <= 1e-10}


def main(source, output):
    root = Path(__file__).resolve().parents[2]
    source = Path(source).resolve(); output = Path(output).resolve(); verify(root, source)
    src = json.loads((source/'config.json').read_text())
    joint = root/'runs'/Path(src['source']).name; verify(root, joint)
    physics = src['physical_configuration']; T = src['time']
    output.mkdir(parents=True, exist_ok=False)
    config = {'study': 'unit-flux continuous interface layer, independent coefficient and saved-width comparison',
              'source': str(source), 'physical_configuration': physics, 'time': T,
              'inner_unit_width': 1., 'inner_unit_D': 1., 'prescribed_unit_flux': 1.,
              'cell_problem_controls': [{'half_window':8.,'spacing':1/16},
                                        {'half_window':8.,'spacing':1/32},
                                        {'half_window':16.,'spacing':1/32}],
              'quadrature': 'exact polynomial cell-pair integrals and affine residual, positive Gauss weights',
              'unknown_far_offsets': 'solve two constants with unit continuous flux and bL+bR=0 gauge',
              'no_weight_normalization': True, 'no_fitting_saved_errors': True,
              'prediction_region_in_inner_units':4.,
              'saved_width_cells':[512,2048], 'additional_interface_only_quadrature_nodes':[128,256],
              'inner_flux_scale':'independent local reference F(T), not saved numerical flux',
              'model_scope':'formal fixed-positive-time matched asymptotic prediction; not a full transient theorem',
              'physical_candidate_modified':False,'physical_acceptance':False,'fluid_orc_enabled':False}
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    cell_rows=[]; cell_arrays=[]
    for k, p in enumerate(config['cell_problem_controls']):
        ar,d=solve_inner(physics['rho_left'],physics['rho_right'],**p)
        cell_rows.append(d);cell_arrays.append(ar)
        np.savez_compressed(output/f'inner-control{k}.npz',**ar)
        print('inner problem',p,'kappa',d['kappa'],flush=True)
    fine=cell_arrays[-1]; kappa=cell_rows[-1]['kappa']
    # Freeze the coefficient before loading any measured interface errors.
    coefficient={'kappa':kappa,'quadrature_kappa_change':abs(cell_rows[1]['kappa']-cell_rows[0]['kappa']),
                 'window_kappa_change':abs(cell_rows[2]['kappa']-cell_rows[1]['kappa']),
                 'no_response_data_in_cell_problem':True}
    (output/'independent-coefficient.json').write_text(json.dumps(coefficient,indent=2)+'\n')
    flow=Contact(rho_left=physics['rho_left'],rho_right=physics['rho_right'],D=physics['D'],
                 half_length=physics['half_length'],speed=physics['speed'])
    ref=StepReference(flow,left=physics['initial_left'],right=physics['initial_right'])
    F=float(ref.flux(np.array([0.]),T)[0]); invref=ref.right_inventory(0.,T)[0]
    rhoL,rhoR=flow.rho_left,flow.rho_right
    bL=-rhoR/(rhoL+rhoR)*kappa; bR=rhoL/(rhoL+rhoR)*kappa
    comparisons=[]
    for n in config['saved_width_cells']:
        with np.load(joint/f'physical-N{n}.npz') as z:
            faces=z['material_faces'];m=z['mass'];c0=z['c0'];x=z['positions'];h=float(z['h'])
            actual_error=z['kernel_c']-z['reference']
        if n==src['cells']:
            with np.load(source/'decomposition.npz') as z:response=z['interface_response']
            response_check={'reuse':'prior validated Gauss256 response'}
        else:
            response,response_check=interface_response(joint,n,physics,T,output)
        gauss=gaussian_average(faces,flow.D,T); b=np.where(x<0,bL,bR)
        outer=F*h/flow.D*b*gauss
        selected,psibar=inner_average(fine,faces/h,config['prediction_region_in_inner_units'])
        predicted=F*h/flow.D*(psibar+b[selected]*gauss[selected])
        local_rel=float(np.sqrt(np.dot(m[selected],(predicted-response[selected])**2)
                                     /np.dot(m[selected],response[selected]**2)))
        observed_inventory=float(m[x>0]@response[x>0])
        predicted_inventory=h*kappa*(rhoL*rhoR/(rhoL+rhoR))**2*(physics['initial_left']-physics['initial_right'])
        row={'cells':n,'h':h,'h_over_sqrt_DT':h/np.sqrt(flow.D*T),
             'interface_inventory_observed':observed_inventory,
             'interface_inventory_leading_prediction':predicted_inventory,
             'observed_over_leading_inventory':observed_inventory/predicted_inventory,
             'inventory_prediction_relative_discrepancy':(predicted_inventory-observed_inventory)/observed_inventory,
             'observed_inventory_per_h':observed_inventory/h,
             'interface_response_norm':normalized_error(m,c0,response),
             'unchanged_total_kernel_error':normalized_error(m,c0,actual_error),
             'inner_profile_relative_discrepancy_on_fixed_region':local_rel,
             'interface_inventory_observed_relative_to_reference':observed_inventory/invref,
             'interface_inventory_prediction_relative_to_reference':predicted_inventory/invref,
             'response_quadrature':response_check}
        comparisons.append(row)
        np.savez_compressed(output/f'prediction-N{n}.npz',material_faces=faces,mass=m,c0=c0,positions=x,h=np.array(h),
                            observed_interface_response=response,unchanged_total_error=actual_error,
                            outer_prediction=outer,inner_selected_indices=selected,
                            inner_composite_prediction=predicted,inner_localized_average=psibar)
        print('unfitted prediction',n,'inventory ratio',row['observed_over_leading_inventory'],
              'local profile relative discrepancy',local_rel,flush=True)
    # State the inner corrector's moments, including its O(h^2) composite
    # mass defect. No uncomputed outer second-order completion is imposed.
    mask=fine['interior_mask'];z=fine['positions'];psi=fine['localized_corrector'];delta=cell_rows[-1]['spacing']
    moments={'localized_left_integral':float(delta*psi[mask&(z<0)].sum()),
             'localized_right_integral':float(delta*psi[mask&(z>0)].sum()),
             'localized_mass_weighted_squared_norm':float(delta*np.dot(fine['rho'][mask],psi[mask]**2)),
             'matching_b_left':bL,'matching_b_right':bR}
    result={'cell_problem_controls':cell_rows,'independent_coefficient':coefficient,'exact_reference_flux_at_T':F,
            'inner_moments':moments,'saved_width_comparisons':comparisons,
            'formal_orders':{'outer_concentration_and_fixed_scale_average':'O(h) when kappa is nonzero',
                             'localized_inner_L2':'O(h^(3/2)) at fixed positive time',
                             'localized_contribution_to_fixed_width_average':'O(h^2 / cell_width)'},
            'physical_acceptance':False,
            'limits':['formal matching, not a full time-dependent error proof','finite h and finite cross-pair quadrature remain',
                      'no second-order mass completion or candidate correction','two saved widths only, not a fitted convergence order']}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,
                            'scipy':scipy.__version__,'BLAS_threads_command':4},indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True)
    a=p.parse_args();main(a.source,a.output)
