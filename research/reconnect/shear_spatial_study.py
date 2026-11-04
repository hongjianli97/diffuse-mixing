"""One predeclared N256/h.03 frozen shear graph, common original cells."""
from pathlib import Path
import argparse,hashlib,json,platform,resource
import numpy as np
import scipy
from chain_study import verify
from positive_kernel_2d import build_periodic_exchange
from shear_probe import make_shear_samples,shear_gauss_reference,shear_face_reference,shear_geometry
from variable_parent_observation import observe_parent,parent_mass_audit
from shear_spatial_observation import compare_parent_rates
from compression_midpoint import pair_action,weighted_norm
from sparse_exchange_audit import audit_laplacian,sparse_laplacian,sparse_max_abs


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-shear-frozen';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_shear_spatial_control_predeclared']
    if protocol['one_new_level']!={'N_per_direction':256,'h_m':.03} or protocol['shear_amplitude_at_final']!=.25 or protocol['D_m2_per_s']!=.02 or protocol['new_frozen_graphs']!=1 or protocol['new_transient_runs'] or protocol['fluid_orc_enabled']:
        raise RuntimeError('predeclared single shear spatial control changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    fine_path=root/protocol['fine_mass_c0_ids_source'];coarse_path=root/protocol['coarse_mass_c0_ids_and_reference_source']
    cfg={'study':'one joint N256/h.03 frozen shear control on original common material cells','source':str(source),'protocol':protocol,
        'protocol_source_sha256':hashlib.sha256((source/'stage3-status.json').read_bytes()).hexdigest(),
        'input_sources_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [fine_path,coarse_path]},
        'candidate_backend_unchanged':True,'fine_input':'reuse saved fine mass/c0/ids and original parent map byte for byte',
        'candidate_geometry_inputs':'points, fixed initial mass, oracle point rho only; no J/F/C/true faces',
        'comparison':'raw 2x2 fine mass*rate divided by saved coarse mass, same saved coarse reference norm',
        'reference':'full Eulerian Gauss8/16 plus independent material-face Gauss8/16',
        'indicator_fixed_before_graph':'sum coarse/fine face-volume and rule differences, reference nesting difference, coarse/fine pair-vs-CSR action norms; empirical, not rigorous',
        'graph_storage':'complete raw shared pairs/g plus CSR and all point inputs; distances/displacements reconstructible from points and pairs',
        'new_frozen_graphs':1,'new_transient_runs':0,'coarse_graph_rebuilt':False,'parameter_or_mesh_matrix_scan':False,
        'clipping_fitting_renormalization_or_full_P':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    keys=['mass','c0','ids','material_label_centers','rho0_point','parent_index','parent_id_per_child']
    with np.load(fine_path) as z:fine={key:z[key] for key in keys}
    with np.load(coarse_path) as z:
        parent={key:z[key] for key in ['mass','c0','ids','material_label_centers','rho0_point','reference_rate','kernel_rate','face8_rate','face16_rate','gauss8_rate','gauss16_rate']}
        coarse_scatter=pair_action(parent['mass'],z['pairs'],z['conductances'],parent['c0'])
    m=fine['mass'];c0=fine['c0'];mp=parent['mass'];refp=parent['reference_rate'];old=parent['kernel_rate'];labels=fine['material_label_centers']
    np.testing.assert_array_equal(fine['parent_id_per_child'],parent['ids'][fine['parent_index']])
    child_sum,rounding,mass_audit=parent_mass_audit(m,mp)
    data,map_arrays,map_audit=make_shear_samples(labels,fine['ids'],m,c0,fine['rho0_point'],protocol['time_samples_s'])
    gauss={order:shear_gauss_reference(256,.25,.25,.02,labels,m,order) for order in [8,16]}
    face={order:shear_face_reference(256,.25,.25,.02,labels,m,order) for order in [8,16]}
    ref=face[16]['rate'];norm=weighted_norm(m,ref);normp=weighted_norm(mp,refp)
    rel=lambda v:weighted_norm(m,v)/norm;relp=lambda v:weighted_norm(mp,v)/normp
    print('Building the one predeclared fine shear graph.',flush=True)
    L,a,diag=build_periodic_exchange(data,1,.02,.03,[1.,1.]);graph_audit=audit_laplacian(m,L)
    independent=sparse_laplacian(len(m),a['pairs'],a['conductances']);shared=sparse_max_abs(independent-L);del independent
    rate=-(L@c0)/m;scatter=pair_action(m,a['pairs'],a['conductances'],c0)
    observed,comparison=compare_parent_rates(mp,refp,old,m,rate);R=lambda v:observe_parent(v,m,mp)
    controls={'coarse_face_vs_volume':relp(parent['face16_rate']-parent['gauss16_rate']),
        'coarse_face8_vs16':relp(parent['face8_rate']-parent['face16_rate']),
        'coarse_volume8_vs16':relp(parent['gauss8_rate']-parent['gauss16_rate']),
        'fine_face_vs_volume_observed':relp(R(ref-gauss[16]['rate'])),
        'fine_face8_vs16_observed':relp(R(face[8]['rate']-ref)),
        'fine_volume8_vs16_observed':relp(R(gauss[8]['rate']-gauss[16]['rate'])),
        'reference_nesting':relp(R(ref)-refp),
        'coarse_pair_vs_CSR':relp(coarse_scatter-old),'fine_pair_vs_CSR_observed':relp(R(scatter-rate))}
    indicator=sum(controls.values());comparison.update({'fine_error_own_cells':rel(rate-ref),'empirical_indicator_components':controls,
        'combined_empirical_indicator':indicator,'improvement_over_indicator':comparison['absolute_error_reduction']/indicator,
        'improvement_exceeds_empirical_indicator':bool(comparison['absolute_error_reduction']>indicator),
        'indicator_is_a_rigorous_bound':False,'two_joint_levels_prove_spatial_order':False})
    geometry=shear_geometry(256,labels,.25,.05)
    result={'comparison':comparison,'operator_diagnostics':diag,'sparse_graph_audit':graph_audit,'map_audit':map_audit,
        'parent_mass_roundoff_audit':mass_audit,'fine_face16_vs_full_Gauss16_relative_difference':rel(ref-gauss[16]['rate']),
        'fine_face8_vs16_relative_difference':rel(face[8]['rate']-ref),'fine_Gauss8_vs16_relative_difference':rel(gauss[8]['rate']-gauss[16]['rate']),
        'fine_saved_mass_vs_Gauss16_max_relative_difference':float(np.max(abs(m-gauss[16]['mass'])/m)),
        'fine_saved_c0_vs_Gauss16_max_difference':float(np.max(abs(c0-gauss[16]['state']))),
        'initial_c0_parent_projection_max_difference':float(np.max(abs(R(c0)-parent['c0']))),
        'reference_parent_projection_max_difference_per_s':float(np.max(abs(R(ref)-refp))),
        'raw_area_vs_true_area_max_relative_difference':float(np.max(abs(a['quadrature_area']/geometry['true_cell_area']-1))),
        'sum_raw_quadrature_area_m2':float(a['quadrature_area'].sum()),'maximum_node_area_centroid_offset_m':float(np.max(abs(geometry['node_minus_area_centroid_x']))),
        'raw_mass_rate_residual_kg_per_s':float(m@rate),'shared_edge_assembly_max_difference':shared,
        'pair_scatter_vs_CSR_max_difference_per_s':float(np.max(abs(scatter-rate))),
        'fine_mass_c0_ids_and_parent_map_reused':True,'new_frozen_graphs':1,'new_transient_runs':0,'coarse_graph_rebuilt':False,
        'full_P_constructed':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'limits':['two joint N/h levels, one prescribed shear frozen scalar','finite-width, point quadrature and mean-state closure remain combined',
                  'no shear actual scalar propagation, shock/wall/ledger/nonplanar/DNS or ORC acceptance']}
    saved={**fine,**{'parent_'+key:value for key,value in parent.items()},'time':data.time,'positions':data.positions,'rho':data.rho,
        'reference_rate':ref,'kernel_rate':rate,'rate_error':rate-ref,'pair_scatter_rate':scatter,'coarse_pair_scatter_rate':coarse_scatter,
        'common_kernel_rate':observed,'common_reference_rate':R(ref),'common_rate_error':observed-refp,
        'child_mass_sum':child_sum,'parent_mass_rounding_relative':rounding,
        'L_data':L.data,'L_indices':L.indices,'L_indptr':L.indptr,'L_shape':np.array(L.shape),
        **{key:a[key] for key in ['pairs','conductances','quadrature_area','point_rho']},**map_arrays,**geometry}
    for order in [8,16]:
        for prefix,control in [('gauss',gauss[order]),('face',face[order])]:
            for key,value in control.items():saved[f'{prefix}{order}_{key}']=value
    np.savez_compressed(out/'shear-spatial.npz',**saved)
    for name,value in [('results.json',result),('comparison.json',comparison),('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
