"""Only continuous references on unchanged saved inputs and graph action."""
from pathlib import Path
import argparse,hashlib,json,platform,resource
import numpy as np
import scipy
from chain_study import verify
from continuous_density_kernel import PolarRule,reference_rates,input_extension,split_budget


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-kernel2d-variable-density';verify(root,source)
    source_cfg=json.loads((source/'config.json').read_text());p=source_cfg['predeclared_protocol']
    protocol=json.loads((source/'stage3-status.json').read_text())['next_reference_predeclared']
    rules=[PolarRule(**v) for v in protocol['quadrature_levels']]
    if protocol['quadrature_levels']!=[{'radial':32,'angular':64,'cell_x':8},{'radial':64,'angular':128,'cell_x':16}]:raise RuntimeError('reference protocol changed')
    if not protocol['same_saved_N_h_D_rho_scalar_and_graph'] or protocol['new_graphs']!=0:raise RuntimeError('saved state/graph must remain fixed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg={'study':'same saved smooth-density state: continuous-kernel three-part error budget','source':str(source),
        'reference_protocol':protocol,'reference_protocol_source_sha256':hashlib.sha256((source/'stage3-status.json').read_bytes()).hexdigest(),
        'physical_protocol':p,'analytic_input_extension':'rho_bar_delta/rho_point; exact saved rho-weighted state amplitude, no data fit',
        'physical_comparator':'continuous harmonic kernel acting on true c, then true rho-weighted material-cell average',
        'input_comparator':'continuous integration of fixed-Delta analytic saved-state/weight extension, evaluated at saved centers',
        'components_order':['physical continuous kernel minus local PDE','saved-input continuous closure minus physical averaged kernel','saved discrete action minus saved-input continuous closure'],
        'candidate_modified':False,'new_graphs':0,'new_discrete_operator_applications':0,'new_fluid_transient_runs':0,'new_propagators_constructed':0,
        'parameter_or_mesh_scan':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    with np.load(source/'variable-density-kernel2d.npz') as z:
        saved={key:z[key] for key in ['mass','c0','reference_rate','kernel_rate','rho','material_label_centers','quadrature_area','true_cell_area']}
    N=p['N_per_direction'];eps=p['density_amplitude'];D=p['D_m2_per_s'];h=p['h_m']
    references=[reference_rates(N,eps,D,h,rule) for rule in rules]
    components,budget=split_budget(saved['mass'],saved['reference_rate'],saved['kernel_rate'],references[1]['physical_rate'],references[1]['input_closure_rate'])
    coarse_components,coarse_budget=split_budget(saved['mass'],saved['reference_rate'],saved['kernel_rate'],references[0]['physical_rate'],references[0]['input_closure_rate'])
    norm=float(np.sqrt(saved['mass']@(saved['reference_rate']**2)));relative=lambda v:float(np.sqrt(saved['mass']@(v*v))/norm)
    component_difference=components-coarse_components
    physical_difference=references[1]['physical_rate']-references[0]['physical_rate'];input_difference=references[1]['input_closure_rate']-references[0]['input_closure_rate']
    names=['finite_width','averaged_input_point_density_closure','node_quadrature']
    reference_audit={'physical_reference_rule_difference_relative':relative(physical_difference),
        'input_closure_rule_difference_relative':relative(input_difference),
        'component_rule_differences_relative':{key:relative(v) for key,v in zip(names,component_difference)},
        'component_rule_difference_to_component_norm':{key:relative(v)/budget[key+'_relative_norm'] if budget[key+'_relative_norm']>0 else None for key,v in zip(names,component_difference)},
        'maximum_physical_rule_difference_per_s':float(np.max(abs(physical_difference))),
        'maximum_input_closure_rule_difference_per_s':float(np.max(abs(input_difference))),
        'differences_are_empirical_not_rigorous_bounds':True}
    labels=saved['material_label_centers'];rho,mean,amplitude,ratio=input_extension(labels[:,0],N,eps)
    state_extension=.5+amplitude*np.cos(2*np.pi*labels[:,1])
    extension_checks={'state_extension_vs_saved_mass_average_max_difference':float(np.max(abs(state_extension-saved['c0']))),
        'rho_extension_vs_saved_point_rho_max_difference':float(np.max(abs(rho-saved['rho'][1]))),
        'weight_extension_vs_saved_area_ratio_max_difference':float(np.max(abs(ratio-saved['quadrature_area']/saved['true_cell_area']))),
        'mass_extension_vs_saved_mass_max_difference':float(np.max(abs(mean/(N*N)-saved['mass'])))}
    result={'budget':budget,'coarse_rule_budget':coarse_budget,'reference_audit':reference_audit,'input_extension_checks':extension_checks,
        'physical_reference_weighted_mass_rate_kg_per_s':float(saved['mass']@references[1]['physical_rate']),
        'input_reference_weighted_mass_rate_kg_per_s':float(saved['mass']@references[1]['input_closure_rate']),
        'new_graphs':0,'new_discrete_operator_applications':0,'new_fluid_transient_runs':0,'new_propagators_constructed':0,
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'limits':['fixed analytic extension, not a general reconstruction theorem','quadrature refinement differences are not certified bounds','single smooth periodic state and fixed h','no shock/contact, transient or ORC acceptance']}
    arrays=dict(saved);arrays.update({'finite_width_error':components[0],'closure_error':components[1],'node_quadrature_error':components[2],
        'total_error':saved['kernel_rate']-saved['reference_rate'],'input_extension_state':state_extension,'input_extension_weight_ratio':ratio})
    for index,reference in enumerate(references):
        for key,value in reference.items():arrays[f'rule{index}_{key}']=value
    np.savez_compressed(out/'continuous-density-budget.npz',**arrays)
    for name,value in [('results.json',result),('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
