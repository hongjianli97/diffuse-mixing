"""Predeclared oracle constitutive comparison; original candidate untouched."""
from pathlib import Path
import argparse,json,platform,resource
import numpy as np
import scipy
from chain_study import verify
from package_compression_midpoint import dump,digest
from compression_midpoint import weighted_norm
from oracle_path_reference import material_means
from contact_frozen_probe import AMPLITUDE,point_fields


def regional_norms(m,values,pde,near):
    norm=weighted_norm(m,pde)
    return {key:weighted_norm(m[mask],values[mask])/norm for key,mask in
        [('all',np.ones(len(m),dtype=bool)),('near_contact',near),('far_from_contact',~near)]}


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-contact-constitutive';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_oracle_path_control_predeclared']
    levels=[{'cell_x_order':8,'normal_displacement_order':32,'transverse_order':32},
        {'cell_x_order':16,'normal_displacement_order':64,'transverse_order':64}]
    if (protocol['N_per_direction'],protocol['h_m'],protocol['D_m2_per_s'])!=(128,.04,.02) or protocol['fixed_quadrature_levels']!=levels or not protocol['reference_only'] or protocol['new_candidate_graphs'] or protocol['new_candidate_transient_runs'] or protocol['fluid_orc_enabled']:
        raise RuntimeError('oracle-only single-case protocol changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);data_source=root/protocol['source']
    dump(out/'config.json',{'study':'same contact continuous oracle path-resistance versus saved harmonic operator',
        'source':str(source),'protocol':protocol,'protocol_source_sha256':digest(source/'stage3-status.json'),
        'input_source_sha256':digest(data_source),'candidate_backend_and_saved_graph_unchanged':True,
        'additional_information':'complete original analytic periodic rho(a) and contact locations; reference only, unavailable from finite node inputs without closure assumptions',
        'best_rule_selection':'second/fine raw rule predeclared; no extrapolation or reference-error selection',
        'observation':'unchanged saved mass denominator and original material regions, continuous field first then mass average',
        'indicator_fixed_before_run':protocol['empirical_indicator_fixed_before_run'],
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'finite_node_candidate_admitted':False,'full_P_constructed':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'})
    keys=['mass','c0','ids','material_label_centers','rho0_point','reference_rate','kernel_rate','near_contact','far_from_contact','distance_to_contact']
    with np.load(data_source) as z:
        saved={key:z[key] for key in keys};harmonic={name:z[name+'_rate'] for name in ['coarse','fine']}
    with np.load(root/'runs/reconnect-contact-frozen/contact-observations.npz') as z:
        np.testing.assert_array_equal(saved['reference_rate'],z['face_rate'])
        saved['PDE_face_volume16_difference']=z['face_rate']-z['volume16_rate']
        saved['PDE_volume8_16_difference']=z['volume8_rate']-z['volume16_rate']
    for name,rate in harmonic.items():saved['harmonic_'+name+'_rate']=rate
    m=saved['mass'];pde=saved['reference_rate'];near=saved['near_contact'];runs={};answers=[]
    for name,quad in zip(['coarse','fine'],levels):
        print(json.dumps({'oracle_rule':name,'orders':quad}),flush=True)
        arrays=material_means(128,m,saved['material_label_centers'],.04,.02,quad);rate=arrays['rate'];answers.append(rate)
        for key,value in arrays.items():saved[name+'_'+key]=value
        phi=point_fields(arrays['outer_x'],0.,.02)['phase']
        runs[name]={'quadrature':quad,'outer_quadrature_points':len(arrays['outer_x']),
            'errors':regional_norms(m,rate-pde,pde,near),'raw_mass_rate_kg_per_s':float(m@rate),
            'projected_chi_kg_per_s':-float(m@(saved['c0']*rate)),
            'physical_nonlocal_chi_kg_per_s':-AMPLITUDE**2*float(arrays['outer_weight']@(np.cos(phi)*arrays['outer_normal_response'])),
            'maximum_absolute_rate_per_s':float(np.max(abs(rate)))}
    best=answers[1];oracle_diff=answers[0]-answers[1];harmonic_diff=harmonic['coarse']-harmonic['fine']
    vectors={'oracle_rule_difference':oracle_diff,'harmonic_rule_difference':harmonic_diff,
        'PDE_face_volume16_difference':saved['PDE_face_volume16_difference'],'PDE_volume8_16_difference':saved['PDE_volume8_16_difference']}
    controls={key:regional_norms(m,value,pde,near) for key,value in vectors.items()}
    old_errors=regional_norms(m,harmonic['fine']-pde,pde,near);new_errors=regional_norms(m,best-pde,pde,near)
    operator_difference=regional_norms(m,best-harmonic['fine'],pde,near)
    comparison={}
    for region in ['all','near_contact','far_from_contact']:
        parts={key:value[region] for key,value in controls.items()};indicator=sum(parts.values());reduction=old_errors[region]-new_errors[region]
        comparison[region]={'saved_harmonic_error':old_errors[region],'oracle_path_error':new_errors[region],
            'absolute_error_reduction':reduction,'relative_error_reduction_fraction':reduction/old_errors[region],
            'oracle_vs_harmonic_difference':operator_difference[region],
            'empirical_indicator_components':parts,'combined_empirical_indicator':indicator,
            'reduction_over_indicator':reduction/indicator if indicator else None,'reduction_exceeds_indicator':bool(reduction>indicator)}
    saved.update(vectors);saved.update({'best_oracle_rate':best,'oracle_error':best-pde,'oracle_minus_harmonic':best-harmonic['fine']})
    result={'oracle_runs':runs,'comparison':comparison,'best_raw_rule':'fine','reference_only':True,
        'uses_additional_oracle_density_information':True,'quadrature_indicator_is_rigorous_bound':False,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'finite_node_candidate_admitted':False,'full_P_constructed':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'limits':['full original density field is extra oracle information, not identified from finite node inputs',
            'one frozen continuous field and one h, no spatial or transient accuracy proof',
            'original discrete scalar graph and ordinary FV baseline unchanged',
            'no state/weight fit, clipping, normalization, physical threshold change or ORC admission']}
    np.savez_compressed(out/'oracle-path.npz',**saved);dump(out/'results.json',result)
    dump(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
