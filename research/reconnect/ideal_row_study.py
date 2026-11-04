"""Two predeclared rules for one aggregate row; no graph or propagation."""
from pathlib import Path
import argparse,json,platform,resource
import numpy as np
import scipy
from audit_contact_constitutive import digest
from package_compression_midpoint import dump
from ideal_row_response import aggregate_row


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-mass-projection'
    entries=json.loads((source/'sha256.json').read_text())
    for rel,h in entries.items():assert digest(root/rel)==h,rel
    protocol=json.loads((source/'stage3-status.json').read_text())['next_single_row_response_predeclared']
    levels=[{'normal_order':32,'transverse_order':32},{'normal_order':64,'transverse_order':64}]
    assert protocol['fixed_quadrature_levels']==levels and (protocol['N_per_direction'],protocol['h_m'],protocol['D_m2_per_s'])==(128,.04,.02)
    assert protocol['no_new_edge_weights_or_full_graph'] and protocol['no_division_by_one_plus_eta_or_moment_renormalization']
    assert not protocol['parameter_or_mesh_matrix_scan'] and not protocol['fluid_orc_enabled']
    input_path=root/'runs/reconnect-contact-frozen/contact-observations.npz'
    keys=['mass','c0','ids','material_label_centers','rho0_point','reference_rate','kernel_rate']
    with np.load(input_path) as z:saved={key:z[key] for key in keys}
    N=128;ell=1/N;h=.04;D=.02;target=N//4-1;center=float(saved['material_label_centers'][target,0])
    mass=float(saved['mass'][target]);rho=float(saved['rho0_point'][target]);distance=min(center,.5-center)
    assert saved['material_label_centers'].shape==(N*N,2) and center==63/256 and rho==1. and mass==rho*ell**2
    assert distance-ell/2>2*h
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    dump(out/'config.json',{'scope':'one predeclared ideal aggregate quadratic row-response integral',
        'source':str(source),'source_manifest_entries_verified':len(entries),'protocol':protocol,'protocol_source_sha256':digest(source/'stage3-status.json'),
        'original_input_source':str(input_path.relative_to(root)),'original_input_sha256':digest(input_path),
        'target_selection':'row 0, column N/4-1; fixed geometry interior low phase, never selected by error',
        'target_index':target,'target_id':int(saved['ids'][target]),'target_center_a_m':center,'saved_original_mass_kg':mass,
        'unchanged_original_c0_mass_average':float(saved['c0'][target]),
        'probe':'auxiliary analytic c=a^2/2 and its exact cell means, not original saved cosine initial state',
        'direct_check':'independent two-point exact Gauss phase integration of projected neighbor cell means within the same row/rules',
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'fluid_orc_enabled':False})
    runs={}
    for name,level in zip(['coarse','fine'],levels):
        arrays,metrics=aggregate_row(ell,h,D,rho,mass,center,level['normal_order'],level['transverse_order'])
        for key,value in arrays.items():saved[name+'_'+key]=value
        runs[name]=metrics
    lower=6890625/3023656976384;upper=5625/131072
    for row in runs.values():assert lower<row['eta']<upper
    controls={key:abs(runs['coarse'][key]-runs['fine'][key]) for key in ['eta','effective_rate_exact_moment_prediction_per_s',
        'raw_variance_aggregate_rate_per_s','raw_direct_aggregate_rate_per_s','raw_second_moment']}
    saved['panel_eta_coarse_minus_fine']=saved['coarse_panel_eta']-saved['fine_panel_eta']
    result={'rules':runs,'coarse_fine_absolute_differences':controls,'fine_eta_percent':100*runs['fine']['eta'],
        'fine_predicted_rate_minus_PDE_per_s':runs['fine']['effective_rate_exact_moment_prediction_per_s']-D,
        'strict_analytic_eta_bounds':{'lower':lower,'upper':upper},'strict_bounds_satisfied_by_both_rules':True,
        'best_raw_rule':'fine','quadrature_difference_is_rigorous_bound':False,'auxiliary_PDE_rate_per_s':D,
        'reference_only_ideal_geometry_mean_closure':True,'original_c0_and_point_candidate_and_PDE_unchanged':True,
        'new_candidate_graphs':0,'edge_weights_created':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'aggregate_row_rule_evaluations':2,'finite_node_candidate_admitted':False,'contact_transient_admitted':False,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'limits':['same original ratio and one ideal pure-phase aggregate row only','not a new numerical verdict on the point candidate or saved cosine state',
            'no coefficient or moment adjustment, full graph, physical flow solve or propagation']}
    np.savez_compressed(out/'ideal-row.npz',**saved);dump(out/'results.json',result)
    dump(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
