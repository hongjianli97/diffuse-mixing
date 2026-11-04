"""One predeclared N=128,h=.04 smooth-density control, unchanged backend."""
from pathlib import Path
import argparse,hashlib,json,platform,resource
import numpy as np
import scipy
from chain_study import verify
from smooth_density_probe import make_samples,gauss_reference
from positive_kernel_2d import build_periodic_exchange
from sparse_exchange_audit import audit_laplacian,sparse_laplacian,sparse_max_abs
from continuous_density_kernel import PolarRule,reference_rates,split_budget
from variable_parent_observation import parent_mass_audit,observe_parent


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-kernel2d-continuous-density';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_control_predeclared']
    if (protocol['N_per_direction'],protocol['h_m'],protocol['D_m2_per_s'],protocol['density_amplitude'])!=(128,.04,.02,.25):raise RuntimeError('predeclared refinement changed')
    if protocol['new_candidate_levels']!=1 or not protocol['sparse_only'] or protocol['renormalize_child_masses']:raise RuntimeError('fixed single-level/no-renormalization protocol changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg={'study':'one predeclared variable-density joint refinement and continuous-reference comparison','source':str(source),
        'protocol':protocol,'protocol_source_sha256':hashlib.sha256((source/'stage3-status.json').read_bytes()).hexdigest(),
        'coarse_case_rerun':False,'candidate_kernel_unchanged':True,'physical_D_rho_scalar_motion_and_periods_unchanged':True,
        'parent_observation':'raw child component sums divided by saved parent masses; audit but do not repair analytic mass rounding',
        'sparse_only':True,'new_candidate_levels':1,'parameter_or_mesh_scan':False,'fit_or_clipping':False,
        'new_fluid_transient_runs':0,'new_propagators_constructed':0,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    with np.load(source/'continuous-density-budget.npz') as z:
        parent={key:z[key] for key in ['mass','c0','reference_rate','kernel_rate','rule1_physical_rate','rule1_input_closure_rate','finite_width_error','closure_error','node_quadrature_error']}
    old=json.loads((source/'results.json').read_text());N=protocol['N_per_direction'];h=protocol['h_m'];D=protocol['D_m2_per_s'];eps=protocol['density_amplitude']
    data,ref=make_samples(N,eps,D)
    L,a,diagnostics=build_periodic_exchange(data,1,D,h,[1.,1.]);rate=-(L@data.c0)/data.mass
    audit=audit_laplacian(data.mass,L)
    independent=sparse_laplacian(len(data.mass),a['pairs'],a['conductances']);shared_difference=sparse_max_abs(independent-L);del independent
    i,j=a['pairs'].T;pair_rate=np.zeros(len(data.mass));flux=a['conductances']*(data.c0[j]-data.c0[i])
    np.add.at(pair_rate,i,flux);np.add.at(pair_rate,j,-flux);pair_rate/=data.mass
    scatter_difference=float(np.max(abs(pair_rate-rate)));del pair_rate,flux
    gaussian={order:gauss_reference(N,eps,D,ref['labels'],order) for order in [8,16]}
    continuous=[reference_rates(N,eps,D,h,PolarRule(**rule)) for rule in protocol['continuous_reference_rules']]
    components,budget=split_budget(data.mass,ref['rate'],rate,continuous[1]['physical_rate'],continuous[1]['input_closure_rate'])
    low_components,low_budget=split_budget(data.mass,ref['rate'],rate,continuous[0]['physical_rate'],continuous[0]['input_closure_rate'])
    norm=np.sqrt(data.mass@(ref['rate']**2));relative=lambda v:float(np.sqrt(data.mass@(v*v))/norm)
    reference_control={'physical_rule_difference_relative':relative(continuous[1]['physical_rate']-continuous[0]['physical_rate']),
        'input_rule_difference_relative':relative(continuous[1]['input_closure_rate']-continuous[0]['input_closure_rate']),
        'component_rule_differences_relative':[relative(v) for v in components-low_components],
        'analytic_vs_Gauss16_rate_relative':relative(ref['rate']-gaussian[16]['rate']),
        'Gauss8_vs_16_rate_relative':relative(gaussian[8]['rate']-gaussian[16]['rate']),
        'analytic_vs_Gauss16_mass_max_relative':float(np.max(abs(data.mass-gaussian[16]['mass'])/data.mass)),
        'analytic_vs_Gauss16_state_max_difference':float(np.max(abs(data.c0-gaussian[16]['state']))),
        'differences_are_empirical_not_rigorous_bounds':True}
    child_sum,mass_rounding,mass_audit=parent_mass_audit(data.mass,parent['mass'])
    R=lambda v:observe_parent(v,data.mass,parent['mass'])
    fine_values={'c0':data.c0,'reference_rate':ref['rate'],'kernel_rate':rate,'physical_rate':continuous[1]['physical_rate'],
        'input_rate':continuous[1]['input_closure_rate'],'finite_width_error':components[0],'closure_error':components[1],'node_quadrature_error':components[2]}
    observed={key:R(value) for key,value in fine_values.items()}
    parent_norm=np.sqrt(parent['mass']@(parent['reference_rate']**2));parent_relative=lambda v:float(np.sqrt(parent['mass']@(v*v))/parent_norm)
    actual_change=observed['kernel_rate']-parent['kernel_rate'];physical_change=observed['physical_rate']-parent['rule1_physical_rate']
    closure_change=(observed['input_rate']-observed['physical_rate'])-(parent['rule1_input_closure_rate']-parent['rule1_physical_rate'])
    quadrature_change=(observed['kernel_rate']-observed['input_rate'])-(parent['kernel_rate']-parent['rule1_input_closure_rate'])
    old_error=parent_relative(parent['kernel_rate']-parent['reference_rate']);new_error=parent_relative(observed['kernel_rate']-parent['reference_rate'])
    indicator=reference_control['physical_rule_difference_relative']+reference_control['input_rule_difference_relative']+reference_control['analytic_vs_Gauss16_rate_relative']
    indicator+=old['reference_audit']['physical_reference_rule_difference_relative']+old['reference_audit']['input_closure_rule_difference_relative']
    indicator+=parent_relative(observed['reference_rate']-parent['reference_rate'])+scatter_difference*np.sqrt(parent['mass'].sum())/parent_norm
    comparison={'old_total_error':old_error,'new_total_error_own_cells':budget['total_relative_error'],'new_total_error_saved_parent_cells':new_error,
        'absolute_error_reduction':old_error-new_error,'relative_error_reduction_fraction':1-new_error/old_error,
        'old_budget':old['budget'],'new_budget':budget,'Delta_over_h_before':1/(64*.06),'Delta_over_h_after':1/(N*h),
        'observed_operator_change_relative_norm':parent_relative(actual_change),'continuous_physical_prediction_change_relative_norm':parent_relative(physical_change),
        'closure_change_relative_norm':parent_relative(closure_change),'node_quadrature_change_relative_norm':parent_relative(quadrature_change),
        'predicted_new_error_changing_only_continuous_physical_part':parent_relative(parent['kernel_rate']+physical_change-parent['reference_rate']),
        'actual_vs_physical_only_prediction_difference':parent_relative(actual_change-physical_change),
        'signed_change_decomposition_max_residual_per_s':float(np.max(abs(actual_change-physical_change-closure_change-quadrature_change))),
        'parent_state_projection_max_difference':float(np.max(abs(observed['c0']-parent['c0']))),
        'parent_local_reference_projection_max_difference_per_s':float(np.max(abs(observed['reference_rate']-parent['reference_rate']))),
        'empirical_reference_implementation_and_projection_indicator':float(indicator),'improvement_over_indicator':(old_error-new_error)/indicator,
        'indicator_is_a_rigorous_bound':False,'general_convergence_proof':False,'physical_acceptance_threshold_changed':False}
    result={'operator_diagnostics':diagnostics,'sparse_graph_audit':audit,'budget':budget,'reference_control':reference_control,
        'parent_mass_roundoff_audit':mass_audit,'comparison':comparison,'shared_sparse_graph_difference':shared_difference,
        'pair_scatter_vs_sparse_rate_max_difference_per_s':scatter_difference,'raw_weighted_rate_mass_residual_kg_per_s':float(data.mass@rate),
        'new_candidate_levels':1,'new_fluid_transient_runs':0,'new_propagators_constructed':0,'sparse_only':True,
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'limits':['two smooth variable-density grid/width levels only','no shock/contact, nonuniform cloud, wall, ledger or DNS validation','no transient or ORC acceptance']}
    arrays={'time':data.time,'positions':data.positions,'ids':data.ids,'mass':data.mass,'c0':data.c0,'rho':data.rho,
        'material_label_centers':ref['labels'],'rho_cell_average':ref['rho_average'],'true_cell_area':ref['true_area'],
        'reference_rate':ref['rate'],'kernel_rate':rate,'finite_width_error':components[0],'closure_error':components[1],'node_quadrature_error':components[2],
        'L_data':L.data,'L_indices':L.indices,'L_indptr':L.indptr,'L_shape':np.array(L.shape),**a}
    for index,reference in enumerate(continuous):
        for key,value in reference.items():arrays[f'rule{index}_{key}']=value
    for order,control in gaussian.items():
        for key,value in control.items():arrays[f'gauss{order}_{key}']=value
    np.savez_compressed(out/'variable-density-refined.npz',**arrays)
    common={'parent_'+key:value for key,value in parent.items()};common.update({'restricted_'+key:value for key,value in observed.items()})
    common.update({'child_mass_sum':child_sum,'mass_rounding_relative':mass_rounding,'actual_change':actual_change,'physical_change':physical_change,
        'closure_change':closure_change,'quadrature_change':quadrature_change})
    np.savez_compressed(out/'same-parent-comparison.npz',**common)
    for name,value in [('results.json',result),('comparison.json',comparison),('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
