"""One predeclared periodic shear frozen point-kernel physical probe."""
from pathlib import Path
import argparse,hashlib,json,platform,resource
import numpy as np
import scipy
from chain_study import verify
from positive_kernel_2d import build_periodic_exchange
from sparse_exchange_audit import audit_laplacian,sparse_laplacian,sparse_max_abs
from shear_probe import make_shear_samples,shear_gauss_reference,shear_face_reference,shear_geometry
from compression_midpoint import pair_action,weighted_norm


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-compression-spatial';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_shear_frozen_predeclared']
    if not protocol or (protocol['N_per_direction'],protocol['h_m'],protocol['D_m2_per_s'],protocol['shear_amplitude_at_final'])!=(128,.04,.02,.25) or protocol['time_samples_s']!=[0.,.05] or protocol['new_midpoint_or_frozen_graphs']!=1 or protocol['fluid_orc_enabled']:
        raise RuntimeError('predeclared single shear frozen state changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    state_path=root/protocol['saved_mass_c0_ids_source']
    cfg={'study':'one genuinely two-dimensional periodic shear frozen physical control','source':str(source),'protocol':protocol,
        'protocol_source_sha256':hashlib.sha256((source/'stage3-status.json').read_bytes()).hexdigest(),
        'state_source':str(state_path),'state_source_sha256':hashlib.sha256(state_path.read_bytes()).hexdigest(),
        'candidate_backend_unchanged':True,'candidate_geometry_inputs':'points, given initial masses and oracle point density; no J/F/C/true faces',
        'scalar_state':'saved original material mass-average c0; frozen prescribed state, not evolved under shear',
        'reference':'full Eulerian Laplacian/density drift Gauss8/16 and independent material face-flux Gauss8/16',
        'new_frozen_graphs':1,'new_transient_runs':0,'new_spatial_levels':0,'full_P_constructed':False,
        'parameter_scan_clipping_fitting_or_normalization':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    with np.load(state_path) as z:
        m=z['mass'];c0=z['c0'];ids=z['ids'];labels=z['material_label_centers'];rho0=z['rho0_point']
    data,map_arrays,map_audit=make_shear_samples(labels,ids,m,c0,rho0,protocol['time_samples_s'])
    N=128;s=.25;D=.02;h=.04;frame=1
    gauss={order:shear_gauss_reference(N,.25,s,D,labels,m,order) for order in [8,16]}
    face={order:shear_face_reference(N,.25,s,D,labels,m,order) for order in [8,16]}
    reference=face[16]['rate'];norm=weighted_norm(m,reference);relative=lambda v:weighted_norm(m,v)/norm
    L,a,diag=build_periodic_exchange(data,frame,D,h,[1.,1.]);rate=-(L@c0)/m;graph_audit=audit_laplacian(m,L)
    independent=sparse_laplacian(len(m),a['pairs'],a['conductances']);shared=sparse_max_abs(independent-L);del independent
    scatter=pair_action(m,a['pairs'],a['conductances'],c0);geometry=shear_geometry(N,labels,s,.05)
    contributions={key:relative(gauss[16][key]) for key in ['base_laplacian','metric_correction','mixed_derivative','shear_gradient','density_drift']}
    result={'map_audit':map_audit,'operator_diagnostics':diag,'sparse_graph_audit':graph_audit,
        'relative_mass_weighted_instantaneous_rate_error':relative(rate-reference),
        'maximum_absolute_rate_error_per_s':float(np.max(abs(rate-reference))),
        'face16_vs_full_Eulerian_Gauss16_relative_difference':relative(reference-gauss[16]['rate']),
        'face8_vs_face16_relative_difference':relative(face[8]['rate']-reference),
        'Gauss8_vs_Gauss16_relative_difference':relative(gauss[8]['rate']-gauss[16]['rate']),
        'saved_mass_vs_Gauss16_max_relative_difference':float(np.max(abs(m-gauss[16]['mass'])/m)),
        'saved_c0_vs_Gauss16_max_difference':float(np.max(abs(c0-gauss[16]['state']))),
        'PDE_component_norms_over_full_rate_norm':contributions,
        'omitting_mixed_and_shear_gradient_terms_reference_relative_change':relative(gauss[16]['mixed_derivative']+gauss[16]['shear_gradient']),
        'components_are_reference_physics_not_candidate_error_attribution':True,
        'raw_quadrature_area_vs_true_area_max_relative_difference':float(np.max(abs(a['quadrature_area']/geometry['true_cell_area']-1))),
        'sum_raw_quadrature_area_m2':float(a['quadrature_area'].sum()),'sum_true_cell_area_m2':float(geometry['true_cell_area'].sum()),
        'maximum_node_area_centroid_offset_m':float(np.max(abs(geometry['node_minus_area_centroid_x']))),
        'raw_candidate_mass_rate_residual_kg_per_s':float(m@rate),'raw_reference_mass_rate_residual_kg_per_s':float(m@reference),
        'shared_edge_vs_candidate_L_max_difference':shared,'pair_scatter_vs_CSR_rate_max_difference_per_s':float(np.max(abs(scatter-rate))),
        'given_oracle_rho_range_kg_per_m3':[float(data.rho[frame].min()),float(data.rho[frame].max())],
        'same_saved_mass_c0_ids':True,'new_frozen_graphs':1,'new_transient_runs':0,'full_P_constructed':False,
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'limits':['one s/N/h, prescribed frozen scalar rather than evolved shear transient','finite-width, quadrature and mass-average closure errors remain combined',
                  'no shear spatial convergence or finite-time order validation','no shock/wall/ledger/nonplanar/DNS or fluid ORC acceptance']}
    saved={'time':data.time,'positions':data.positions,'ids':ids,'mass':m,'c0':c0,'rho':data.rho,'rho0_point':rho0,'material_label_centers':labels,
        'reference_rate':reference,'kernel_rate':rate,'rate_error':rate-reference,
        'L_data':L.data,'L_indices':L.indices,'L_indptr':L.indptr,'L_shape':np.array(L.shape),**a,**map_arrays,**geometry}
    for order in [8,16]:
        for prefix,control in [('gauss',gauss[order]),('face',face[order])]:
            for key,value in control.items():saved[f'{prefix}{order}_{key}']=value
    np.savez_compressed(out/'shear-frozen.npz',**saved)
    for name,value in [('results.json',result),('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
