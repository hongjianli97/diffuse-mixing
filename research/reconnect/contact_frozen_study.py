"""One predeclared periodic density-contact graph, no transient or repair."""
from pathlib import Path
import argparse,json,platform,resource
import numpy as np
import scipy
from chain_study import verify
from package_compression_midpoint import digest,dump
from contact_frozen_probe import contact_inputs,analytic_face_reference,volume_reference,contact_regions,regional_metrics
from positive_kernel_2d import build_periodic_exchange
from sparse_exchange_audit import audit_laplacian,sparse_laplacian,sparse_max_abs
from compression_midpoint import pair_action,weighted_norm


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-shear-transient-spatial';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_contact_frozen_predeclared']
    if not protocol or (protocol['N_per_direction'],protocol['h_m'],protocol['D_m2_per_s'])!=(128,.04,.02) or protocol['new_frozen_graphs']!=1 or protocol['new_candidate_transient_runs'] or protocol['fluid_orc_enabled']:
        raise RuntimeError('predeclared single density-contact graph changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    dump(out/'config.json',{'study':'one stationary periodic density-contact frozen-rate comparison',
        'source':str(source),'protocol':protocol,'protocol_source_sha256':digest(source/'stage3-status.json'),
        'state':'exact rho-weighted fixed-cell mass averages; no point-state substitution',
        'input_producer':'new specified contact problem, exact aligned material integrals; old shear inputs/results untouched',
        'candidate_inputs':'points, fixed mass and given point rho only; no reference faces, derivatives, J/F/C or fitted interface weights',
        'candidate_backend_unchanged':True,'same_two_stationary_sample_frames_are_schema_only':True,
        'reference':'analytic integrated four-face flux versus independent piecewise full-PDE Gauss8/16',
        'norm':'every region divided by the same all-cell sqrt(sum m*reference_rate**2)',
        'regions_fixed_before_graph':'distance to either periodic contact <=2h; complement; neither mask changes graph',
        'parameter_scan_clipping_fitting_or_normalization':False,'full_P_constructed':False,
        'new_candidate_transient_runs':0,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'})
    N=128;D=.02;h=.04;data,inputs=contact_inputs(N,D);m=data.mass;c0=data.c0;labels=inputs['material_label_centers']
    ref=analytic_face_reference(N,D,labels,m)
    volumes={order:volume_reference(N,D,labels,m,order) for order in [8,16]}
    regions=contact_regions(labels,h);norm=weighted_norm(m,ref['rate']);relative=lambda v:weighted_norm(m,v)/norm
    print('Building the single unchanged contact graph.',flush=True)
    L,a,diag=build_periodic_exchange(data,0,D,h,[1.,1.]);graph_audit=audit_laplacian(m,L)
    independent=sparse_laplacian(len(m),a['pairs'],a['conductances']);assembly=sparse_max_abs(independent-L);del independent
    rate=-(L@c0)/m;scatter=pair_action(m,a['pairs'],a['conductances'],c0)
    i,j=a['pairs'].T;chi_pair=float(a['conductances']@((c0[j]-c0[i])**2));chi_CSR=float(c0@(L@c0))
    physical_chi=float(volumes[16]['physical_dissipation'].sum());reference_projected_chi=-float(m@(c0*ref['rate']))
    result={'regional_comparison':regional_metrics(m,rate,ref['rate'],regions,ref['rate_a'],ref['rate_b']),
        'reference_controls':{'analytic_faces_vs_volume16_relative':relative(ref['rate']-volumes[16]['rate']),
            'volume8_vs_volume16_relative':relative(volumes[8]['rate']-volumes[16]['rate']),
            'exact_mass_vs_Gauss16_max_relative':float(np.max(abs(m-volumes[16]['mass'])/m)),
            'exact_c0_vs_Gauss16_max_absolute':float(np.max(abs(c0-volumes[16]['state']))),
            'physical_inventory_rate_faces_kg_per_s':float(m@ref['rate']),
            'physical_inventory_rate_volume16_kg_per_s':float(m@volumes[16]['rate']),
            'physical_energy_identity_volume16_residual_kg_per_s':float(volumes[16]['c_rate_integral'].sum()+physical_chi)},
        'graph_algebra':graph_audit,'operator_diagnostics':diag,
        'raw_graph_controls':{'pair_scatter_vs_CSR_relative':relative(scatter-rate),
            'pair_scatter_vs_CSR_max_absolute_per_s':float(np.max(abs(scatter-rate))),
            'shared_edge_assembly_max_difference':assembly,'raw_component_mass_rate_kg_per_s':float(m@rate),
            'chi_CSR_kg_per_s':chi_CSR,'chi_pair_kg_per_s':chi_pair,'chi_pair_vs_CSR_absolute_difference':abs(chi_pair-chi_CSR)},
        'physical_continuous_chi_kg_per_s':physical_chi,'projected_reference_chi_kg_per_s':reference_projected_chi,
        'projected_energy_is_continuous_energy':False,
        'state_mean':float(m@c0/m.sum()),'state_mass_kg':float(m@c0),'total_mass_kg':float(m.sum()),
        'state_extrema':[float(c0.min()),float(c0.max())],
        'mass_average_vs_point_scalar_max_difference':float(np.max(abs(c0-inputs['point_scalar_reference_only']))),
        'quadrature_area_vs_true_cell_area_max_difference':float(np.max(abs(a['quadrature_area']-inputs['true_cell_area_reference_only']))),
        'new_frozen_graphs':1,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'full_P_constructed':False,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'limits':['one frozen density contact, one N/h, prescribed oracle rho','strong local residual is not proof about a weak continuum limit',
            'no transient mixing/maximum-principle experiment, no spatial convergence order','no graph or interface-weight repair, no fluid ORC']}
    saved={**inputs,**regions,'time':data.time,'positions':data.positions,'rho':data.rho,
        'reference_rate':ref['rate'],'kernel_rate':rate,'rate_error':rate-ref['rate'],'pair_scatter_rate':scatter,
        'L_data':L.data,'L_indices':L.indices,'L_indptr':L.indptr,'L_shape':np.array(L.shape),**a}
    for key,value in ref.items():saved['face_'+key]=value
    for order,values in volumes.items():
        for key,value in values.items():saved[f'volume{order}_{key}']=value
    np.savez_compressed(out/'contact-frozen.npz',**saved)
    small={key:value for key,value in saved.items() if key not in ['L_data','L_indices','L_indptr','L_shape',*a.keys()]}
    np.savez_compressed(out/'contact-observations.npz',**small)
    dump(out/'results.json',result)
    dump(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
