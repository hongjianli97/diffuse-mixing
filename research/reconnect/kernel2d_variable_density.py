"""One predeclared variable-density point-kernel action; no transient or tuning."""
from pathlib import Path
import argparse,hashlib,json,platform,resource
import numpy as np
import scipy
from smooth_density_probe import make_samples,gauss_reference
from positive_kernel_2d import build_periodic_exchange
from sparse_exchange_audit import audit_laplacian,sparse_laplacian,sparse_max_abs
from chain_study import verify


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-kernel2d-refinement';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_probe_predeclared']
    if (protocol['N_per_direction'],protocol['h_m'],protocol['D_m2_per_s'],protocol['density_amplitude'])!=(64,.06,.02,.25):
        raise RuntimeError('predeclared variable-density parameters changed')
    if protocol['rho_material_formula']!='1+0.25*cos(2*pi*a_x)' or protocol['scalar_formula']!='.5+.2*cos(2*pi*a_x)*cos(2*pi*a_y)':
        raise RuntimeError('predeclared analytic state changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg={'study':'one smooth variable-density oracle snapshot, unchanged 2D positive point kernel','source':str(source),
        'predeclared_protocol':protocol,'protocol_source_sha256':hashlib.sha256((source/'stage3-status.json').read_bytes()).hexdigest(),
        'state_and_mass':'exact rho-weighted material-square integrals; point density distinct from mean density',
        'reference':'direct analytic divergence integral plus independent Gauss8/16 rho*PDE quadrature',
        'full_point_PDE':'D*laplacian(c)+D*gradient(log rho).gradient(c)',
        'density_drift_omitted_reference':'negative reference diagnostic only, not changed PDE or candidate',
        'candidate_backend_unchanged':True,'candidate_true_faces_F_C_or_probe_inputs':False,
        'quadrature_area':'m/(oracle point rho*unit thickness); retain raw discrepancy, no normalization',
        'candidate_coefficient_or_state_changes_after_observation':False,'parameter_or_mesh_search':False,
        'sparse_only':True,'new_fluid_transient_runs':0,'new_propagators_constructed':0,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    N=protocol['N_per_direction'];D=protocol['D_m2_per_s'];epsilon=protocol['density_amplitude'];h=protocol['h_m']
    data,ref=make_samples(N,epsilon,D);checks={order:gauss_reference(N,epsilon,D,ref['labels'],order) for order in [8,16]}
    L,a,diag=build_periodic_exchange(data,protocol['frozen_frame'],D,h,protocol['periods_m']);rate=-(L@data.c0)/data.mass
    audit=audit_laplacian(data.mass,L);shared=sparse_laplacian(len(data.mass),a['pairs'],a['conductances'])
    norm=float(np.sqrt(data.mass@(ref['rate']**2)));error=lambda v:float(np.sqrt(data.mass@(v*v))/norm)
    i,j=a['pairs'].T;flux=a['conductances']*(data.c0[j]-data.c0[i]);pair_rate=np.zeros(len(data.mass))
    np.add.at(pair_rate,i,flux);np.add.at(pair_rate,j,-flux);pair_rate/=data.mass
    point_state=.5+.2*np.cos(2*np.pi*ref['labels'][:,0])*np.cos(2*np.pi*ref['labels'][:,1])
    area_error=a['quadrature_area']/ref['true_area']-1
    old=json.loads((source/'results.json').read_text())
    result={'operator_diagnostics':diag,'sparse_graph_layer_audit':audit,
        'relative_mass_weighted_instantaneous_rate_error':error(rate-ref['rate']),
        'maximum_absolute_rate_error_per_s':float(np.max(abs(rate-ref['rate']))),
        'missing_density_drift_reference_error':error(ref['laplacian_rate']-ref['rate']),
        'candidate_distance_to_laplacian_only_diagnostic':error(rate-ref['laplacian_rate']),
        'analytic_total_vs_laplacian_plus_drift_max_difference_per_s':float(np.max(abs(ref['rate']-ref['laplacian_rate']-ref['drift_rate']))),
        'analytic_vs_Gauss16_reference_max_difference_per_s':float(np.max(abs(ref['rate']-checks[16]['rate']))),
        'analytic_vs_Gauss16_reference_relative_difference':error(ref['rate']-checks[16]['rate']),
        'Gauss8_vs_16_reference_relative_difference':error(checks[8]['rate']-checks[16]['rate']),
        'analytic_vs_Gauss16_mass_max_relative_difference':float(np.max(abs(data.mass-checks[16]['mass'])/data.mass)),
        'analytic_vs_Gauss16_state_max_difference':float(np.max(abs(data.c0-checks[16]['state']))),
        'oracle_point_density_range_kg_per_m3':[float(ref['rho_point'].min()),float(ref['rho_point'].max())],
        'mean_density_range_kg_per_m3':[float(ref['rho_average'].min()),float(ref['rho_average'].max())],
        'point_density_vs_true_mean_max_relative_difference':float(np.max(abs(ref['rho_point']/ref['rho_average']-1))),
        'raw_quadrature_area_vs_true_area_max_relative_difference':float(np.max(abs(area_error))),
        'raw_quadrature_area_sum_m2':float(a['quadrature_area'].sum()),'exact_true_total_area_m2':1.,
        'max_point_state_vs_mass_average_difference':float(np.max(abs(point_state-data.c0))),
        'raw_weighted_rate_mass_residual_kg_per_s':float(data.mass@rate),'analytic_weighted_rate_mass_residual_kg_per_s':float(data.mass@ref['rate']),
        'pair_scatter_vs_sparse_action_max_difference_per_s':float(np.max(abs(pair_rate-rate))),
        'shared_sparse_graph_vs_candidate_L_max_difference':sparse_max_abs(shared-L),
        'saved_uniform_density_control_error_context_only':old['relative_mass_weighted_instantaneous_rate_error'],
        'comparison_to_uniform_is_same_physical_problem':False,
        'new_fluid_transient_runs':0,'new_propagators_constructed':0,'sparse_only':True,
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'not_claimed':['complete recovery of density drift for arbitrary data','separation of variable-coefficient finite-width and mass-average closure errors','variable-density convergence','shock/contact or wall acceptance','transient or DNS acceptance']}
    saved={'time':data.time,'positions':data.positions,'ids':data.ids,'mass':data.mass,'c0':data.c0,'rho':data.rho,
        'material_label_centers':ref['labels'],'true_cell_area':ref['true_area'],'rho_cell_average':ref['rho_average'],
        'reference_rate':ref['rate'],'reference_laplacian_rate':ref['laplacian_rate'],'reference_drift_rate':ref['drift_rate'],
        'kernel_rate':rate,'rate_error':rate-ref['rate'],'quadrature_area_relative_error':area_error,
        'L_data':L.data,'L_indices':L.indices,'L_indptr':L.indptr,'L_shape':np.array(L.shape),**a}
    for order,control in checks.items():
        for key,value in control.items():saved[f'gauss{order}_{key}']=value
    np.savez_compressed(out/'variable-density-kernel2d.npz',**saved)
    for name,value in [('results.json',result),('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
