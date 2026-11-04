"""One predeclared smooth-compression frozen point-kernel control."""
from pathlib import Path
import argparse,hashlib,json,platform,resource
import numpy as np
import scipy
from chain_study import verify
from positive_kernel_2d import build_periodic_exchange
from sparse_exchange_audit import audit_laplacian,sparse_laplacian,sparse_max_abs
from compression_probe import make_compression_samples,analytic_material_rate,compression_geometry,compression_gauss_reference


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-kernel2d-density-transient';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_compression_probe_predeclared']
    if (protocol['N_per_direction'],protocol['h_m'],protocol['D_m2_per_s'],protocol['alpha_frozen'])!=(128,.04,.02,.25) or protocol['times_s']!=[0.,.05]:raise RuntimeError('predeclared compression control changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg={'study':'one smooth compressible-map frozen control, same saved mass and c0','source':str(source),'protocol':protocol,
        'protocol_source_sha256':hashlib.sha256((source/'stage3-status.json').read_bytes()).hexdigest(),
        'candidate_backend_unchanged':True,'candidate_J_F_C_or_true_faces_inputs':False,
        'scalar_state':'saved c0 as prescribed frozen material means, not an evolved compression-flow scalar',
        'reference':'material face balance plus Gauss8/16 of full physical Laplacian and density drift, weighted by initial rho0',
        'special_identity':'rho=rho0/J is exactly uniform when alpha=epsilon; density is not overwritten or fitted',
        'reference_geometry_only':'unwrapped material faces and J not passed to candidate',
        'new_spatial_levels':0,'new_physical_cases':1,'parameter_scan':False,'new_transient_runs':0,'full_P_constructed':False,
        'clipping_or_state_weight_normalization':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    with np.load(source/'density-transient.npz') as z:
        m=z['mass'];c0=z['c0'];labels=z['material_label_centers'];ids=z['ids'];rho0=z['rho'][0]
    N=128;epsilon=.25;D=.02;h=.04;times=np.array(protocol['times_s'])
    data,map_arrays,map_audit=make_compression_samples(labels,ids,m,c0,rho0,times,.25,.05)
    frame=protocol['frozen_frame'];alpha=map_arrays['alpha'][frame];t=times[frame]
    reference,split=analytic_material_rate(N,epsilon,alpha,D,m,c0,labels)
    geometry=compression_geometry(N,labels,alpha,t)
    checks={order:compression_gauss_reference(N,epsilon,alpha,D,labels,m,order) for order in [8,16]}
    L,a,diag=build_periodic_exchange(data,frame,D,h,[1.,1.]);rate=-(L@c0)/m;audit=audit_laplacian(m,L)
    independent=sparse_laplacian(len(m),a['pairs'],a['conductances']);shared_difference=sparse_max_abs(independent-L);del independent
    i,j=a['pairs'].T;pair_rate=np.zeros(len(m));flux=a['conductances']*(c0[j]-c0[i]);np.add.at(pair_rate,i,flux);np.add.at(pair_rate,j,-flux);pair_rate/=m
    norm=np.sqrt(m@(reference*reference));error=lambda v:float(np.sqrt(m@(v*v))/norm)
    result={'map_audit':map_audit,'operator_diagnostics':diag,'sparse_graph_audit':audit,
        'relative_mass_weighted_instantaneous_rate_error':error(rate-reference),'maximum_absolute_rate_error_per_s':float(np.max(abs(rate-reference))),
        'analytic_face_vs_Gauss16_reference_relative_difference':error(reference-checks[16]['rate']),
        'analytic_face_vs_Gauss16_reference_max_difference_per_s':float(np.max(abs(reference-checks[16]['rate']))),
        'Gauss8_vs_16_reference_relative_difference':error(checks[8]['rate']-checks[16]['rate']),
        'saved_mass_vs_Gauss16_mass_max_relative_difference':float(np.max(abs(m-checks[16]['mass'])/m)),
        'saved_c0_vs_Gauss16_mass_average_max_difference':float(np.max(abs(c0-checks[16]['state']))),
        'frozen_point_density_range_kg_per_m3':[float(data.rho[frame].min()),float(data.rho[frame].max())],
        'frozen_Gauss16_density_drift_max_abs_per_s':float(np.max(abs(checks[16]['drift_rate']))),
        'true_area_face_vs_closed_form_max_relative_difference':float(np.max(abs(geometry['true_cell_area_from_unwrapped_faces']/geometry['closed_form_cell_area']-1))),
        'saved_mass_vs_true_current_area_max_relative_difference':float(np.max(abs(m/geometry['true_cell_area_from_unwrapped_faces']-1))),
        'raw_quadrature_area_vs_true_area_max_relative_difference':float(np.max(abs(a['quadrature_area']/geometry['true_cell_area_from_unwrapped_faces']-1))),
        'raw_quadrature_area_total_m2':float(a['quadrature_area'].sum()),'true_area_total_m2':float(geometry['true_cell_area_from_unwrapped_faces'].sum()),
        'maximum_node_geometric_center_offset_m':float(np.max(abs(geometry['node_minus_geometric_center_x']))),
        'minimum_physical_cell_width_m':float(geometry['true_cell_width'].min()),'maximum_physical_cell_width_m':float(geometry['true_cell_width'].max()),
        'raw_weighted_rate_mass_residual_kg_per_s':float(m@rate),'analytic_reference_weighted_rate_mass_residual_kg_per_s':float(m@reference),
        'pair_scatter_vs_sparse_rate_max_difference_per_s':float(np.max(abs(pair_rate-rate))),
        'shared_sparse_graph_vs_candidate_L_max_difference':shared_difference,
        'same_saved_mass_c0_ids':True,'reference_J_or_center_shift_used_to_change_candidate':False,
        'new_physical_cases':1,'new_spatial_levels':0,'new_transient_runs':0,'full_P_constructed':False,
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'limits':['special uniform-final-density state, not generic compressible closure','one N/h and smooth periodic nonuniform cloud','finite-width and nonuniform quadrature errors not separated','no new compression transient or ORC acceptance']}
    saved={'time':times,'positions':data.positions,'ids':ids,'mass':m,'c0':c0,'rho':data.rho,'rho0_point':rho0,'material_label_centers':labels,
        'reference_rate':reference,'kernel_rate':rate,'rate_error':rate-reference,
        'L_data':L.data,'L_indices':L.indices,'L_indptr':L.indptr,'L_shape':np.array(L.shape),**a,**map_arrays,**geometry,**split}
    for order,control in checks.items():
        for key,value in control.items():saved[f'gauss{order}_{key}']=value
    np.savez_compressed(out/'compression-kernel2d.npz',**saved)
    for name,value in [('results.json',result),('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
