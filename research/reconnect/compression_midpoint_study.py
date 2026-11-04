"""Predeclared single midpoint snapshot plus reused endpoint vector actions."""
from pathlib import Path
import argparse,hashlib,json,platform,resource
import numpy as np
import scipy
from chain_study import verify
from positive_kernel_2d import build_periodic_exchange
from sparse_exchange_audit import audit_laplacian,sparse_laplacian,sparse_max_abs
from compression_probe import make_compression_samples,analytic_material_rate,compression_geometry,compression_gauss_reference
from compression_midpoint import load_laplacian,commutator_action,pair_action,weighted_norm


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-kernel2d-compression';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_midpoint_probe_predeclared']
    if (protocol['N_per_direction'],protocol['h_m'],protocol['D_m2_per_s'],protocol['time_s'],protocol['alpha'])!=(128,.04,.02,.025,.125):
        raise RuntimeError('predeclared midpoint changed')
    if protocol['new_frozen_graphs']!=1 or protocol['parameter_scan'] or protocol['new_transient_run'] or protocol['fluid_orc_enabled']:
        raise RuntimeError('single frozen/no propagation protocol changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    endpoint=protocol['endpoint_commutator_diagnostic'];start_path=root/endpoint['start_source'];end_path=root/endpoint['end_source']
    cfg={'study':'one predeclared compression midpoint and saved endpoint commutator on actual c0',
        'source':str(source),'protocol':protocol,'protocol_source_sha256':hashlib.sha256((source/'stage3-status.json').read_bytes()).hexdigest(),
        'endpoint_sources_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [start_path,end_path]},
        'candidate_backend_unchanged':True,'candidate_J_F_C_or_true_faces_inputs':False,
        'scalar_state':'saved initial material mass averages as a prescribed frozen state, not an evolved compression scalar',
        'reference':'unchanged full material divergence and independent Eulerian Laplacian plus density drift, Gauss8/16',
        'endpoint_start_reuse':'saved rigidly translated alpha=0 point geometry; periodic separations translation invariant',
        'new_frozen_graphs':1,'new_spatial_levels':0,'new_transient_runs':0,'full_P_constructed':False,
        'matrix_matrix_products':False,'clipping_fitting_or_normalization':False,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    with np.load(end_path) as z:
        m=z['mass'];c=z['c0'];ids=z['ids'];labels=z['material_label_centers'];rho0=z['rho0_point']
    N=128;epsilon=.25;D=.02;h=.04;times=np.array([0.,protocol['time_s']]);frame=1;alpha=protocol['alpha'];t=times[frame]
    data,map_arrays,map_audit=make_compression_samples(labels,ids,m,c,rho0,times,.25,.05)
    np.testing.assert_array_equal(map_arrays['alpha'],[0.,alpha])
    ref,split=analytic_material_rate(N,epsilon,alpha,D,m,c,labels)
    geometry=compression_geometry(N,labels,alpha,t)
    gaussian={order:compression_gauss_reference(N,epsilon,alpha,D,labels,m,order) for order in [8,16]}
    L,a,diag=build_periodic_exchange(data,frame,D,h,[1.,1.]);rate=-(L@c)/m;audit=audit_laplacian(m,L)
    independent=sparse_laplacian(len(m),a['pairs'],a['conductances']);shared_difference=sparse_max_abs(independent-L);del independent
    scatter=pair_action(m,a['pairs'],a['conductances'],c)
    norm=weighted_norm(m,ref);err=lambda v:weighted_norm(m,v)/norm
    drift=gaussian[16]['drift_rate'];lap=gaussian[16]['laplacian_rate']
    with np.load(start_path) as start,np.load(end_path) as end:
        for key,value in [('mass',m),('c0',c),('ids',ids),('material_label_centers',labels)]:
            np.testing.assert_array_equal(start[key],value);np.testing.assert_array_equal(end[key],value)
        np.testing.assert_array_equal(start['rho'][1],rho0)
        shift=start['time'][1]*np.array([.2,-.1])
        expected=np.mod(labels+shift,1.);position_difference=float(np.max(abs(start['positions'][1]-expected)))
        np.testing.assert_allclose(start['positions'][1],expected,atol=3e-16,rtol=0.)
        start_L=load_laplacian(start);end_L=load_laplacian(end)
        actions=commutator_action(m,start_L,end_L,c)
        np.testing.assert_array_equal(actions['start_rate'],start['kernel_rate'])
        np.testing.assert_array_equal(actions['end_rate'],end['kernel_rate'])
        q0=pair_action(m,start['pairs'],start['conductances'],c);q1=pair_action(m,end['pairs'],end['conductances'],c)
        pair_end_start=pair_action(m,end['pairs'],end['conductances'],q0)
        pair_start_end=pair_action(m,start['pairs'],start['conductances'],q1)
    del start_L,end_L
    pair_commutator=pair_end_start-pair_start_end;comm=actions['commutator'];comm_norm=weighted_norm(m,comm)
    fluct=c-(m@c)/m.sum();fluct_norm=weighted_norm(m,fluct)
    order_result={'definition':'Q_end(Q_start c0)-Q_start(Q_end c0)',
        'weighted_commutator_norm_kg_sqrt_per_s2':comm_norm,
        'commutator_norm_over_initial_fluctuation_norm_per_s2':comm_norm/fluct_norm,
        'maximum_absolute_commutator_per_s2':float(np.max(abs(comm))),
        'norm_over_sum_of_nested_action_norms':comm_norm/(weighted_norm(m,actions['end_after_start'])+weighted_norm(m,actions['start_after_end'])),
        'weighted_commutator_mass_residual_kg_per_s2':float(m@comm),
        'pair_scatter_vs_CSR_commutator_relative_difference':weighted_norm(m,pair_commutator-comm)/comm_norm,
        'pair_scatter_vs_CSR_commutator_max_difference_per_s2':float(np.max(abs(pair_commutator-comm))),
        'pair_scatter_vs_CSR_start_rate_max_difference_per_s':float(np.max(abs(q0-actions['start_rate']))),
        'pair_scatter_vs_CSR_end_rate_max_difference_per_s':float(np.max(abs(q1-actions['end_rate']))),
        'start_saved_frame_time_s':float(shift[0]/.2),'start_saved_rigid_translation_m':shift.tolist(),
        'start_position_vs_prescribed_translation_max_difference_m':position_difference,
        'is_a_finite_time_concentration_error_or_mixing_result':False,'full_matrix_products_or_propagation':False}
    result={'map_audit':map_audit,'operator_diagnostics':diag,'sparse_graph_audit':audit,
        'relative_mass_weighted_instantaneous_rate_error':err(rate-ref),
        'maximum_absolute_rate_error_per_s':float(np.max(abs(rate-ref))),
        'analytic_face_vs_Gauss16_reference_relative_difference':err(ref-gaussian[16]['rate']),
        'analytic_face_vs_Gauss16_reference_max_difference_per_s':float(np.max(abs(ref-gaussian[16]['rate']))),
        'Gauss8_vs_16_reference_relative_difference':err(gaussian[8]['rate']-gaussian[16]['rate']),
        'saved_mass_vs_Gauss16_mass_max_relative_difference':float(np.max(abs(m-gaussian[16]['mass'])/m)),
        'saved_c0_vs_Gauss16_mass_average_max_difference':float(np.max(abs(c-gaussian[16]['state']))),
        'frozen_point_density_range_kg_per_m3':[float(data.rho[frame].min()),float(data.rho[frame].max())],
        'density_drift_weighted_norm_relative_to_full_reference':err(drift),
        'maximum_absolute_density_drift_per_s':float(np.max(abs(drift))),
        'analytic_reference_vs_Laplacian_only_relative_difference':err(ref-lap),
        'raw_quadrature_area_vs_true_area_max_relative_difference':float(np.max(abs(a['quadrature_area']/geometry['true_cell_area_from_unwrapped_faces']-1))),
        'raw_quadrature_area_total_m2':float(a['quadrature_area'].sum()),'true_area_total_m2':float(geometry['true_cell_area_from_unwrapped_faces'].sum()),
        'maximum_node_geometric_center_offset_m':float(np.max(abs(geometry['node_minus_geometric_center_x']))),
        'true_area_face_vs_closed_form_max_relative_difference':float(np.max(abs(geometry['true_cell_area_from_unwrapped_faces']/geometry['closed_form_cell_area']-1))),
        'raw_weighted_rate_mass_residual_kg_per_s':float(m@rate),'reference_weighted_rate_mass_residual_kg_per_s':float(m@ref),
        'pair_scatter_vs_sparse_rate_max_difference_per_s':float(np.max(abs(scatter-rate))),
        'shared_sparse_graph_vs_candidate_L_max_difference':shared_difference,'endpoint_time_order_diagnostic':order_result,
        'same_saved_mass_c0_ids':True,'reference_geometry_used_to_change_candidate':False,
        'new_frozen_graphs':1,'new_spatial_levels':0,'new_transient_runs':0,'full_P_constructed':False,
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'limits':['one prescribed nonuniform geometry/density state at one N/h',
                  'finite-width, node quadrature and cell-average closure errors not separated',
                  'nonzero endpoint commutator is not a measured finite-time order error',
                  'no new compression-flow transient or fluid ORC acceptance']}
    arrays={'time':times,'positions':data.positions,'ids':ids,'mass':m,'c0':c,'rho':data.rho,'rho0_point':rho0,'material_label_centers':labels,
        'reference_rate':ref,'kernel_rate':rate,'rate_error':rate-ref,
        'L_data':L.data,'L_indices':L.indices,'L_indptr':L.indptr,'L_shape':np.array(L.shape),**a,**map_arrays,**geometry,**split}
    for order,control in gaussian.items():
        for key,value in control.items():arrays[f'gauss{order}_{key}']=value
    np.savez_compressed(out/'compression-midpoint.npz',**arrays)
    np.savez_compressed(out/'endpoint-actions.npz',mass=m,c0=c,ids=ids,pair_end_after_start=pair_end_start,
        pair_start_after_end=pair_start_end,pair_commutator=pair_commutator,**actions)
    for name,value in [('results.json',result),('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
