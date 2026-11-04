"""Predeclared directional analysis of saved total rates, no new model."""
from pathlib import Path
import argparse,json,platform,resource
import numpy as np
import scipy
from chain_study import verify
from package_compression_midpoint import dump,digest
from oracle_directional import material_components,gram_budget
from compression_midpoint import weighted_norm


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-oracle-path';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_oracle_directional_budget_predeclared']
    levels=[{'cell_x_order':8,'normal_displacement_order':32,'transverse_order':32},
        {'cell_x_order':16,'normal_displacement_order':64,'transverse_order':64}]
    if protocol['fixed_quadrature_levels']!=levels or (protocol['N_per_direction'],protocol['h_m'],protocol['D_m2_per_s'])!=(128,.04,.02) or not protocol['reference_only'] or protocol['new_candidate_graphs'] or protocol['new_candidate_transient_runs'] or protocol['fluid_orc_enabled']:
        raise RuntimeError('predeclared same-operator directional analysis changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    data_source=root/protocol['source'];PDE_source=root/protocol['PDE_component_source']
    dump(out/'config.json',{'study':'normal-first raw directional residual budget of unchanged saved oracle operator',
        'source':str(source),'protocol':protocol,'protocol_source_sha256':digest(source/'stage3-status.json'),
        'input_source_sha256':digest(data_source),'PDE_component_source_sha256':digest(PDE_source),
        'raw_saved_total_and_candidate_unchanged':True,'decomposition_order_fixed':'normal-first',
        'components_computed_independently':True,'evaluation_residual_retained':'saved total - raw normal - raw transverse, not forced to zero',
        'normalization':'original global sqrt(sum mass*PDE_total_rate^2) for every component and region',
        'reference_only_extra_information':'full original analytic density and contact locations; still not identified from finite-node candidate inputs',
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'})
    keys=['mass','c0','ids','material_label_centers','rho0_point','reference_rate','kernel_rate','harmonic_fine_rate','near_contact','far_from_contact','distance_to_contact']
    records={};saved={}
    with np.load(data_source) as z:
        saved.update({key:z[key] for key in keys})
        for name in ['coarse','fine']:
            saved[name+'_saved_total_rate']=z[name+'_rate']
            records[name]={key:z[name+'_'+key] for key in ['outer_x','outer_weight','outer_column','y_cosine_integrals']}
            for key,value in records[name].items():saved[name+'_'+key]=value
    with np.load(PDE_source) as z:
        saved['PDE_rate_a']=z['face_rate_a'];saved['PDE_rate_b']=z['face_rate_b']
        np.testing.assert_array_equal(saved['reference_rate'],saved['PDE_rate_a']+saved['PDE_rate_b'])
        for axis in ['a','b']:
            saved['PDE_'+axis+'_face_volume16_difference']=z['face_rate_'+axis]-z['volume16_rate_'+axis]
            saved['PDE_'+axis+'_volume8_16_difference']=z['volume8_rate_'+axis]-z['volume16_rate_'+axis]
    m=saved['mass'];pde=saved['reference_rate'];near=saved['near_contact'];norm=weighted_norm(m,pde);runs={}
    for name,quad in zip(['coarse','fine'],levels):
        print(json.dumps({'directional_rule':name,'orders':quad}),flush=True)
        arrays=material_components(m,records[name],.04,.02,quad)
        for key,value in arrays.items():saved[name+'_'+key]=value
        ra=arrays['rate_a'];rb=arrays['rate_b'];total=saved[name+'_saved_total_rate']
        errors=np.array([ra-saved['PDE_rate_a'],rb-saved['PDE_rate_b'],total-ra-rb])
        saved[name+'_error_vectors']=errors;saved[name+'_total_error']=total-pde
        saved[name+'_error_identity_residual']=total-pde-errors.sum(axis=0)
        runs[name]={'quadrature':quad,'outer_quadrature_points':len(records[name]['outer_x']),
            'budget':gram_budget(m,errors,total-pde,pde,near),
            'raw_component_mass_rates_kg_per_s':[float(m@v) for v in [ra,rb,total]],
            'projected_chi_components_kg_per_s':[-float(m@(saved['c0']*v)) for v in [ra,rb,errors[2]]],
            'saved_total_projected_chi_kg_per_s':-float(m@(saved['c0']*total))}
    controls={}
    for key in ['rate_a','rate_b']:
        difference=saved['coarse_'+key]-saved['fine_'+key];saved[key+'_rule_difference']=difference
        controls[key]={region:weighted_norm(m[mask],difference[mask])/norm for region,mask in [('all',np.ones(len(m),dtype=bool)),('near_contact',near),('far_from_contact',~near)]}
    reference_controls={key:{region:weighted_norm(m[mask],value[mask])/norm for region,mask in [('all',np.ones(len(m),dtype=bool)),('near_contact',near),('far_from_contact',~near)]}
        for key,value in saved.items() if key.startswith('PDE_') and key.endswith('_difference')}
    result={'directional_runs':runs,'component_quadrature_differences':controls,'saved_PDE_component_controls':reference_controls,
        'best_raw_rule':'fine','component_order':['normal','transverse','evaluation_residual'],'reference_only':True,
        'candidate_and_total_operator_unchanged':True,'norms_are_additive_causal_fractions':False,
        'quadrature_differences_are_rigorous_bounds':False,'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'finite_node_candidate_admitted':False,'contact_transient_admitted':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'limits':['one prescribed normal-first decomposition, not unique causal attribution','same one h/contact state and full oracle density',
            'no new coefficient, graph, scalar propagation, fitting or threshold change']}
    np.savez_compressed(out/'oracle-directional.npz',**saved);dump(out/'results.json',result)
    dump(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
