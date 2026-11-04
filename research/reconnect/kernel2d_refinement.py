"""Only the N=64,h=.06 control predeclared in the immutable previous run."""
from pathlib import Path
import argparse,hashlib,json,platform,resource
import numpy as np
import scipy
from kernel2d_study import fixture,continuum_decay
from positive_kernel_2d import build_periodic_exchange
from sparse_exchange_audit import sparse_laplacian,sparse_max_abs,audit_laplacian,restrict_equal_nested_squares
from chain_study import verify


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-kernel2d-oracle';verify(root,source)
    prior=json.loads((source/'stage3-status.json').read_text());protocol=prior['next_control_predeclared']
    if protocol!={'N_per_direction':64,'h_m':.06,'D_m2_per_s':.02,'rho_kg_per_m3':1.,'same_periodic_mode':True,'sparse_audit_without_full_P':True}:
        raise RuntimeError('predeclared second-level protocol changed')
    oldcfg=json.loads((source/'config.json').read_text());old=json.loads((source/'results.json').read_text())
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg={'study':'one predeclared 2D oracle-density joint refinement, no scan','source':str(source),'second_level':protocol,
        'protocol_source_sha256':hashlib.sha256((source/'stage3-status.json').read_bytes()).hexdigest(),
        'periods_m':oldcfg['periods_m'],'thickness_m':oldcfg['thickness_m'],'time_samples_s':oldcfg['time_samples_s'],
        'translation_velocity_m_per_s':oldcfg['translation_velocity_m_per_s'],'frozen_frame':oldcfg['frozen_frame'],
        'state':oldcfg['state'],'point_kernel_backend_unchanged':True,'coarse_level_rerun':False,
        'candidate_true_faces_F_C_or_scalar_probe_inputs':False,'same_D_rho_and_continuum_mode':True,
        'analysis_only_projection':'fine mass-weighted 2x2 cell averages divided by saved parent masses, no renormalization',
        'sparse_only_no_N_by_N_dense_matrix':True,'new_fluid_transient_runs':0,'new_propagators_constructed':0,
        'new_parameter_or_mesh_search':False,'clipping_or_weight_fitting':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    with np.load(source/'periodic-kernel2d.npz') as z:
        base={key:z[key] for key in ['mass','c0','reference_rate','kernel_rate','continuum_finite_width_rate','finite_width_error','node_quadrature_error']}
    data,labels,factor=fixture(protocol['N_per_direction']);h=protocol['h_m'];D=protocol['D_m2_per_s'];k=2*np.pi*np.sqrt(2);local=k*k*D
    L,a,diag=build_periodic_exchange(data,1,D,h,cfg['periods_m']);rate=-(L@data.c0)/data.mass
    shared=sparse_laplacian(len(data.mass),a['pairs'],a['conductances']);audit=audit_laplacian(data.mass,L)
    reference=-local*(data.c0-.5);norm=float(np.sqrt(data.mass@(reference*reference)))
    lambdas={f'{form}{order}':continuum_decay(h,D,k,order,form) for form in ['B','W'] for order in [64,128]}
    finite=-lambdas['W128']*(data.c0-.5);width=finite-reference;quadrature=rate-finite
    error=lambda v:float(np.sqrt(data.mass@(v*v))/norm)
    i,j=a['pairs'].T;touch=(i==0)|(j==0);dx=a['minimum_image_displacement'][touch];gg=a['conductances'][touch]
    symbols=[float(np.sum(gg*(1-np.cos(2*np.pi*(dx[:,0]+sign*dx[:,1]))))/data.mass[0]) for sign in [1,-1]]
    symbol_residual=float(np.max(abs(rate+symbols[0]*(data.c0-.5))))
    pair_rate=np.zeros(len(data.mass));flux=a['conductances']*(data.c0[j]-data.c0[i]);np.add.at(pair_rate,i,flux);np.add.at(pair_rate,j,-flux);pair_rate/=data.mass
    spread=max(lambdas.values())-min(lambdas.values())
    restrict=lambda v:restrict_equal_nested_squares(v,data.mass,base['mass'])
    projected={key:restrict(v) for key,v in [('c0',data.c0),('reference_rate',reference),('kernel_rate',rate),('continuum_finite_width_rate',finite),('finite_width_error',width),('node_quadrature_error',quadrature)]}
    coarse_norm=float(np.sqrt(base['mass']@(base['reference_rate']**2)))
    coarse_error=lambda v:float(np.sqrt(base['mass']@(v*v))/coarse_norm)
    old_total=base['kernel_rate']-base['reference_rate'];new_total=projected['kernel_rate']-base['reference_rate']
    action_change=projected['kernel_rate']-base['kernel_rate'];continuous_change=projected['continuum_finite_width_rate']-base['continuum_finite_width_rate']
    quadrature_change=projected['node_quadrature_error']-base['node_quadrature_error']
    old_ref_spread=old['continuum_quadrature_and_form_spread_per_s'];numerical_indicator=(old_ref_spread+spread)/local
    numerical_indicator+=(old['raw_action_vs_discrete_Fourier_mode_max_difference_per_s']+symbol_residual)*np.sqrt(base['mass'].sum())/coarse_norm
    numerical_indicator+=coarse_error(projected['reference_rate']-base['reference_rate'])
    improvement=coarse_error(old_total)-coarse_error(new_total)
    comparison={'coarse_N_per_direction':oldcfg['N_per_direction'],'coarse_h_m':oldcfg['h_m'],'fine_N_per_direction':protocol['N_per_direction'],'fine_h_m':h,
        'Delta_over_h_before':1/(oldcfg['N_per_direction']*oldcfg['h_m']),'Delta_over_h_after':1/(protocol['N_per_direction']*h),
        'old_total_rate_error':old['relative_mass_weighted_instantaneous_rate_error'],'new_total_rate_error':error(rate-reference),
        'old_finite_width_error':old['finite_width_rate_bias_relative_to_local_PDE'],'new_finite_width_error':error(width),
        'old_node_quadrature_error':old['node_quadrature_rate_difference_relative_to_local_PDE'],'new_node_quadrature_error':error(quadrature),
        'new_rate_restricted_to_saved_parent_error':coarse_error(new_total),'raw_restricted_state_vs_saved_parent_max_difference':float(np.max(abs(projected['c0']-base['c0']))),
        'raw_restricted_reference_vs_saved_parent_max_difference_per_s':float(np.max(abs(projected['reference_rate']-base['reference_rate']))),
        'absolute_rate_error_reduction_on_same_parent_cells':improvement,
        'relative_error_reduction_fraction':improvement/coarse_error(old_total),'observed_operator_change_on_same_parent_cells':coarse_error(action_change),
        'independent_continuous_kernel_predicted_change':coarse_error(continuous_change),'node_quadrature_change':coarse_error(quadrature_change),
        'signed_change_decomposition_max_residual_per_s':float(np.max(abs(action_change-continuous_change-quadrature_change))),
        'empirical_reference_and_implementation_indicator':float(numerical_indicator),'improvement_over_empirical_indicator':improvement/numerical_indicator,
        'indicator_is_rigorous_bound':False,'second_level_improves_fixed_state':bool(improvement>numerical_indicator),
        'two_level_control_is_a_general_convergence_proof':False,'no_new_physical_acceptance_threshold':True}
    result={'operator_diagnostics':diag,'sparse_graph_layer_audit':audit,'relative_mass_weighted_instantaneous_rate_error':error(rate-reference),
        'finite_width_rate_bias_relative_to_local_PDE':error(width),'node_quadrature_rate_difference_relative_to_local_PDE':error(quadrature),
        'continuum_local_decay_per_s':local,'continuum_finite_width_decay_per_s':lambdas,'discrete_Fourier_symbols_plus_minus_per_s':symbols,
        'continuum_quadrature_and_form_spread_per_s':float(spread),'raw_action_vs_discrete_Fourier_mode_max_difference_per_s':symbol_residual,
        'pair_scatter_vs_sparse_action_max_difference_per_s':float(np.max(abs(rate-pair_rate))),
        'shared_sparse_graph_interface_vs_backend_L_max_difference':sparse_max_abs(shared-L),
        'raw_weighted_rate_mass_residual_kg_per_s':float(data.mass@rate),'exact_mass_average_attenuation_factor':float(factor),
        'comparison':comparison,'new_fluid_transient_runs':0,'new_propagators_constructed':0,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'sparse_only':True,'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'limits':['only two levels of one uniform periodic Fourier state','not variable density or shock/contact acceptance','no transient or ORC acceptance']}
    np.savez_compressed(out/'refined-kernel2d.npz',time=data.time,positions=data.positions,ids=data.ids,mass=data.mass,c0=data.c0,rho=data.rho,
        material_label_centers=labels,reference_rate=reference,kernel_rate=rate,continuum_finite_width_rate=finite,finite_width_error=width,node_quadrature_error=quadrature,
        L_data=L.data,L_indices=L.indices,L_indptr=L.indptr,L_shape=np.array(L.shape),**a)
    saved={'parent_'+key:v for key,v in base.items()};saved.update({'restricted_'+key:v for key,v in projected.items()})
    saved.update({'action_change':action_change,'continuous_change':continuous_change,'quadrature_change':quadrature_change})
    np.savez_compressed(out/'same-parent-comparison.npz',**saved)
    for name,value in [('results.json',result),('comparison.json',comparison),('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
