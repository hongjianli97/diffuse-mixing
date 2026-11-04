"""One constitutive true-face baseline on the unchanged nonlinear no-flux probe."""
from pathlib import Path
import argparse,json,platform
import numpy as np
import scipy
from kernel_joint_study import verify
from noflux_probe_study import weighted_error
from hybrid_material_fv import build_hybrid_fv,hybrid_action


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve();verify(root,source)
    old=json.loads((source/'config.json').read_text());p=old['physical_parameters']
    with np.load(source/'noflux-probe.npz') as z:
        mass=z['mass'];c0=z['original_c0'];quads=z['quads'];centers=z['mass_centroids'];columns=z['columns'];rows=z['rows'];boundary=z['boundary_cell_mask']
        average=z['probe_average_order64'];ref_rate=z['face_rate_order64'];ref_flux=z['face_flux_order64'];old_rate=z['graph_rate']
        topology={k:z['topology_'+k] for k in ('positions','cell_vertices','edge_vertices','incidence','boundary_mask')}
    output.mkdir(parents=True,exist_ok=False)
    cfg={'study':'one mass-centroid hybrid finite-volume energy baseline, same frozen manufactured probe',
        'source':str(source),'physical_parameters':p,'time':old['time'],'state':'same true material-cell mass averages, fixed m and original c0 retained',
        'coefficient':'kappa_K=D*m_K/true_volume_K; explicit piecewise-constant mean-density closure, not point density',
        'gradient':'G_K=1/V_K sum_e A_e n_Ke (lambda_e-cbar_K)',
        'residual':'R_Ke=lambda_e-cbar_K-(x_e-b_K).G_K',
        'quadratic_potential':'0.5*kappa_K*(V_K*|G_K|^2+sum_e (A_e/d_Ke)*R_Ke^2)',
        'stabilization_factor':1.,'unknowns':'one mass-average cell state and one massless shared trace per real face',
        'boundary':'outward physical flux prescribed zero in the face equilibrium equations; raw computed boundary flux retained',
        'face_flux':'negative derivative of the local quadratic potential with respect to shared face trace; +incidence owner used without averaging',
        'probe_or_density_coefficient_fit':False,'parameter_or_mesh_search':False,'physical_model_rho_or_D_changed':False,
        'new_transient_run':False,'positivity_test':'unmodified Q offdiagonals and basis-state initial derivative, no time step',
        'physical_acceptance':False,'fluid_orc_enabled':False}
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    operator=build_hybrid_fv(topology,mass,centers,p['D']);action=hybrid_action(operator,average);Q=operator['Q'];L=operator['L']
    internal=~topology['boundary_mask'];area=operator['face_area'];delta=action['face_flux']-ref_flux
    face_error=float(np.sqrt(np.sum(delta[internal]**2/area[internal])/np.sum(ref_flux[internal]**2/area[internal])))
    off=Q.copy();np.fill_diagonal(off,np.inf);i,j=np.unravel_index(np.argmin(off),off.shape)
    basis=np.zeros(len(mass));basis[j]=1.;basis_action=hybrid_action(operator,basis)
    result={'hybrid_all_cells':weighted_error(mass,action['rate'],ref_rate,np.ones(len(mass),dtype=bool)),
        'hybrid_interior_cells':weighted_error(mass,action['rate'],ref_rate,~boundary),
        'hybrid_boundary_cells':weighted_error(mass,action['rate'],ref_rate,boundary),
        'saved_graph_all_cells':weighted_error(mass,old_rate,ref_rate,np.ones(len(mass),dtype=bool)),
        'internal_face_flux_density_relative_L2_error':face_error,'maximum_absolute_internal_face_flux_error_kg_per_s':float(np.max(abs(delta[internal]))),
        'maximum_computed_outer_flux_kg_per_s':float(np.max(abs(action['face_flux'][~internal]))),
        'probe_rate_vs_own_face_divergence_max_difference_per_s':float(np.max(abs(action['rate']+(topology['incidence']@action['face_flux'])/mass))),
        'max_constant_residual_per_s':float(np.max(abs(Q.sum(axis=1)))),'max_mass_residual':float(np.max(abs(mass@Q))),
        'raw_weighted_symmetry_defect':float(np.max(abs(L-L.T))),
        'largest_mass_weighted_energy_eigenvalue':float(np.linalg.eigvalsh(-(L+L.T)/2).max()),
        'minimum_raw_offdiagonal_rate_per_s':float(off[i,j]),
        'basis_initial_derivative_counterexample':{'initial_cell':int(j),'observed_cell':int(i),'initial_observed_value':0.,
            'raw_initial_derivative_per_s':float(basis_action['rate'][i])},
        'coefficient_density_range_kg_per_m3':[float((operator['kappa']/p['D']).min()),float((operator['kappa']/p['D']).max())],
        'hybrid_diagnostics':operator['diagnostics'],'physical_acceptance':False,'fluid_orc_enabled':False,'new_transient_run':False,
        'limits':['one frozen probe, no convergence or terminal-concentration claim','density projection and flux closure errors not separated',
            'energy stability and conservation do not imply a nonnegative generator','baseline is a constitutive reference, not an accepted ORC backend']}
    saved={'mass':mass,'original_c0':c0,'quads':quads,'mass_centroids':centers,'columns':columns,'rows':rows,'boundary_cell_mask':boundary,
        'probe_average':average,'physical_reference_rate':ref_rate,'physical_reference_flux':ref_flux,'saved_graph_rate':old_rate,
        'hybrid_rate':action['rate'],'hybrid_face_flux':action['face_flux'],'hybrid_trace':action['trace'],
        'basis_initial':basis,'basis_rate':basis_action['rate']}
    for key,value in topology.items():saved['topology_'+key]=value
    for key,value in operator.items():
        if isinstance(value,np.ndarray):saved['operator_'+key]=value
    np.savez_compressed(output/'hybrid-face-baseline.npz',**saved)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'BLAS_threads_command':4},indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
