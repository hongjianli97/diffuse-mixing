"""Two fixed quadrature controls for the saved contact; no candidate solves."""
from pathlib import Path
import argparse,json,platform,resource
import numpy as np
import scipy
from chain_study import verify
from package_compression_midpoint import dump,digest
from compression_midpoint import weighted_norm
from contact_continuous import material_means
from contact_frozen_probe import AMPLITUDE,point_fields


def budget(m,discrete,continuous,pde,near):
    norm=weighted_norm(m,pde);total=discrete-pde;constitutive=continuous-pde;remainder=discrete-continuous
    result={}
    for key,mask in [('all',np.ones(len(m),dtype=bool)),('near_contact',near),('far_from_contact',~near)]:
        n=lambda v:weighted_norm(m[mask],v[mask])/norm
        cross=2*float(m[mask]@(constitutive[mask]*remainder[mask]))/norm**2
        result[key]={'total_error':n(total),'continuous_fixed_width_error':n(constitutive),'discrete_mean_closure_remainder':n(remainder),
            'signed_twice_cross_term_over_reference_norm_squared':cross,
            'squared_norm_identity_residual':n(total)**2-n(constitutive)**2-n(remainder)**2-cross,
            'vector_identity_residual':n(total-constitutive-remainder),'cells':int(mask.sum())}
    return result


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-contact-frozen';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_contact_continuous_control_predeclared']
    levels=[{'cell_x_order':8,'normal_displacement_order':32,'transverse_order':32},
        {'cell_x_order':16,'normal_displacement_order':64,'transverse_order':64}]
    if protocol['fixed_quadrature_levels']!=levels or (protocol['N_per_direction'],protocol['h_m'],protocol['D_m2_per_s'])!=(128,.04,.02) or protocol['new_candidate_graphs'] or protocol['new_candidate_transient_runs'] or protocol['fluid_orc_enabled']:
        raise RuntimeError('predeclared fixed-h quadrature or no-new-candidate scope changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    dump(out/'config.json',{'study':'same fixed-h continuous nonlocal material-average reference and saved-error budget',
        'source':str(source),'protocol':protocol,'protocol_source_sha256':digest(source/'stage3-status.json'),
        'quadrature_implementation':'exact transverse parity: Gauss on [0,width] times 2, unchanged prescribed orders; split normal and cell intervals',
        'candidate_graph_and_state_unchanged':True,'best_reference_selection':'predeclared second/fine raw rule, no extrapolation or selection against PDE error',
        'raw_mass_observation':'integral rho*r_h divided by unchanged saved mass, no normalization',
        'error_normalization':'same all-cell sqrt(sum m*saved_PDE_rate**2) for all regions and components',
        'decomposition':'combined discrete/mean-state remainder plus continuous fixed-width deviation, with signed cross term',
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'full_P_constructed':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'})
    keys=['mass','c0','ids','material_label_centers','rho0_point','reference_rate','kernel_rate','rate_error','near_contact','far_from_contact','distance_to_contact']
    with np.load(source/'contact-observations.npz') as z:saved={key:z[key] for key in keys}
    m=saved['mass'];pde=saved['reference_rate'];discrete=saved['kernel_rate'];near=saved['near_contact'];norm=weighted_norm(m,pde)
    runs={};answers=[]
    for index,quadrature in enumerate(levels):
        name=['coarse','fine'][index]
        print(json.dumps({'continuous_rule':name,'orders':quadrature}),flush=True)
        arrays=material_means(128,m,saved['material_label_centers'],.04,.02,quadrature)
        response=arrays['rate'];answers.append(response)
        for key,value in arrays.items():saved[name+'_'+key]=value
        phi=point_fields(arrays['outer_x'],0.,.02)['phase']
        continuous_chi=-AMPLITUDE**2*float(arrays['outer_weight']@(np.cos(phi)*arrays['outer_normal_response']))
        runs[name]={'quadrature':quadrature,'outer_quadrature_points':len(arrays['outer_x']),
            'budget':budget(m,discrete,response,pde,near),'raw_mass_rate_kg_per_s':float(m@response),
            'projected_chi_kg_per_s':-float(m@(saved['c0']*response)),
            'physical_continuous_nonlocal_chi_kg_per_s':continuous_chi,
            'maximum_absolute_rate_per_s':float(np.max(abs(response)))}
    difference=answers[0]-answers[1];best=answers[1]
    controls={key:weighted_norm(m[mask],difference[mask])/norm for key,mask in [('all',np.ones(len(m),dtype=bool)),('near_contact',near),('far_from_contact',~near)]}
    saved.update({'quadrature_difference':difference,'best_continuous_rate':best,
        'fixed_width_deviation':best-pde,'discrete_mean_closure_remainder':discrete-best,
        'vector_identity_residual':discrete-pde-(best-pde)-(discrete-best)})
    result={'continuous_runs':runs,'best_raw_rule':'fine','quadrature_difference_controls':controls,
        'quadrature_difference_is_rigorous_bound':False,'norms_are_additive_error_fractions':False,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'full_P_constructed':False,'fluid_orc_enabled':False,
        'stage3_physical_acceptance':'incomplete','contact_transient_admitted':False,
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'limits':['one fixed h and one contact scalar','first budget remainder combines point quadrature and mean-state closure',
            'two quadrature levels are empirical controls, not rigorous bounds','strong fixed-h defect is not proof about h->0 weak limits',
            'no fitted or corrected state/weight, no transient or fluid ORC admission']}
    np.savez_compressed(out/'contact-continuous.npz',**saved)
    dump(out/'results.json',result)
    dump(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
