"""Read saved vectors and scope their signed residual budget; no operators."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import sys
import numpy as np

from audit_route_decision import ROOT,read,save,digest

SOURCE=ROOT/'runs/reconnect-reconstructed-global-frozen'
ORACLE=ROOT/'runs/reconnect-oracle-path/oracle-path.npz'
DIRECTIONAL=ROOT/'runs/reconnect-oracle-directional/oracle-directional.npz'
NORM=0.2479286450687567


def record(a):
    return {'shape':list(a.shape),'dtype':a.dtype.str,'C_order_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def inner(m,u,v):
    return math.fsum(float(a)*float(b)*float(c) for a,b,c in zip(m,u,v))/(NORM*NORM)


def norm(m,u):
    return math.sqrt(inner(m,u,u))


def evaluate(data,prior,oracle_results,directional_results):
    m=data['mass'];ref=data['reference_rate'];r=data['reconstructed_rate'];o=data['oracle_fine_rate']
    # The intermediate vector is fixed: the original fine/raw oracle, no
    # interpolation, extrapolation, fitted scalar or post-result rule selection.
    combined=r-o;continuous=o-ref;total=r-ref
    arithmetic=total-combined-continuous
    vectors=np.array([combined,continuous,arithmetic])
    coarse_combined=r-data['oracle_coarse_rate']
    oracle_rule=data['oracle_coarse_rate']-o
    regions={}
    for label,mask in [('all',np.ones(len(m),dtype=bool)),('near_contact',data['near_contact']),('far_from_contact',data['far_from_contact'])]:
        mm=m[mask];v=vectors[:,mask];tt=total[mask]
        gram=[[inner(mm,a,b) for b in v] for a in v]
        norms=[math.sqrt(gram[i][i]) for i in range(3)]
        total_norm=norm(mm,tt)
        corr=gram[0][1]/(norms[0]*norms[1]) if norms[0]*norms[1] else None
        local_mean=norm(mm,coarse_combined[mask]);local_oracle=norm(mm,oracle_rule[mask])
        cross=2*gram[0][1]
        identity=tt-v.sum(axis=0)
        regions[label]={'cells':int(np.count_nonzero(mask)),
            'combined_point_pair_and_mass_projection_remainder':norms[0],
            'continuous_oracle_minus_PDE':norms[1],
            'arithmetic_closure_remainder':norms[2],
            'reconstructed_minus_PDE':total_norm,'signed_Gram':gram,
            'twice_main_signed_cross_term':cross,'main_component_correlation':corr,
            'gram_sum_minus_total_squared':math.fsum(map(math.fsum,gram))-total_norm**2,
            'vector_identity_residual':norm(mm,identity),
            'saved_oracle_coarse_fine_vector_difference':local_oracle,
            'combined_remainder_using_saved_coarse_oracle':local_mean,
            'norm_gap_not_component_norm':total_norm-norms[1],
            'combined_remainder_over_oracle_rule_indicator':norms[0]/local_oracle if local_oracle else None,
            'combined_to_continuous_component_norm_ratio':norms[0]/norms[1] if norms[1] else None,
            'mass_fraction_of_negative_main_pointwise_product':float(math.fsum(map(float,mm[(v[0]*v[1])<0]))/math.fsum(map(float,mm))),
            'no_additive_causal_percentages':True}
    rows=[];lookup={int(v):i for i,v in enumerate(data['ids'])}
    # Already-saved global extrema and the original pair, not new scalar probes.
    for identity in [1,32,63,64]:
        i=lookup[identity]
        rows.append({'id':identity,'reconstructed_rate':float(r[i]),'oracle_fine_rate':float(o[i]),
            'PDE_rate':float(ref[i]),'combined_point_pair_and_projection_signed':float(combined[i]),
            'continuous_signed':float(continuous[i]),'total_signed':float(total[i]),
            'arithmetic_closure_signed':float(arithmetic[i])})
    projected=lambda rate:-math.fsum(float(a)*float(b)*float(c) for a,b,c in zip(m,data['c0'],rate))
    energy={'new_graph_projected':projected(r),'saved_old_kernel_projected':projected(data['kernel_rate']),
        'continuous_oracle_projected':projected(o),'PDE_projected':projected(ref),
        'same_original_cbar_test_not_a_finite_time_mixing_prediction':True}
    summary={'fixed_global_PDE_norm':NORM,'intermediate':'unchanged saved fine continuous oracle on same original material observation',
        'regions':regions,'selected_saved_extrema_and_original_pair':rows,'frozen_projected_energy':energy,
        'saved_primary_global_error':prior['reconstructed_global_error'],
        'saved_continuous_oracle_error':oracle_results['oracle_runs']['fine']['errors']['all'],
        'oracle_quadrature_difference_is_empirical_not_rigorous':True,
        'only_two_model_components_plus_arithmetic_closure_identified':True,
        'true_ideal_cell_pair_c0_action_not_available_in_saved_evidence':True,
        'combined_remainder_not_named_pure_projection_error':True,
        'ideal_quadratic_eta_not_a_quantitative_explanation_of_original_c0':True,
        'old_directional_budget_is_a_different_split_of_the_continuous_component_only':True,
        'full_density_continuous_oracle_not_finite_node_backend':True,
        'new_weights_or_operator_evaluations':0,'new_numerical_or_flow_cases':0,'new_software_checks':0,
        'recorded_checks_total':417,'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    arrays={'combined_point_pair_and_projection':combined,'continuous_oracle_minus_PDE':continuous,
        'reconstructed_minus_PDE':total,'arithmetic_closure':arithmetic,
        'combined_using_saved_coarse_oracle':coarse_combined,'saved_oracle_rule_difference':oracle_rule}
    return summary,arrays


def main(output):
    manifest=read(SOURCE/'sha256.json')
    for rel,h in manifest.items():assert digest(ROOT/rel)==h,rel
    status=read(SOURCE/'stage3-status.json')
    assert status['next_work'].startswith('read-only residual scope review') and status['recorded_checks_total']==417
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',{'scope':'saved-vector residual scope review; no new operator, graph or quadrature',
        'parent_status_sha256':digest(SOURCE/'stage3-status.json'),'parent_manifest_entries_verified':len(manifest),
        'split_fixed_before_evaluation':'(reconstructed-saved fine continuous oracle) + (saved fine continuous oracle-PDE) + uncorrected arithmetic closure',
        'reference_controls':'reuse saved oracle coarse/fine vector difference and prior arithmetic/PDE indicators, no new quadrature',
        'regions':'original byte-identical near_contact/far_from_contact; locked original all-cell denominator',
        'no_new_absolute_acceptance_threshold':True})
    shutil.copyfile(SOURCE/'stage3-status.json',out/'source-stage3-status.json')
    with np.load(SOURCE/'original-inputs.npz',allow_pickle=False) as z:
        data={k:z[k].copy() for k in ['ids','mass','c0','positions','rho','reference_rate','kernel_rate','near_contact','far_from_contact']}
    with np.load(SOURCE/'global-action.npz',allow_pickle=False) as z:data['reconstructed_rate']=z['rate'].copy()
    matches=[]
    with np.load(ORACLE,allow_pickle=False) as z:
        common=['ids','mass','c0','reference_rate','kernel_rate','near_contact','far_from_contact']
        for k in common+['positions','rho']:
            oldkey={'positions':'material_label_centers','rho':'rho0_point'}.get(k,k)
            assert record(z[oldkey])==record(data[k]),k
            matches.append({'current_member':k,'oracle_member':oldkey,'identical_bytes':record(data[k])})
        assert np.array_equal(z['fine_rate'],z['best_oracle_rate'])
        data['oracle_fine_rate']=z['fine_rate'].copy();data['oracle_coarse_rate']=z['coarse_rate'].copy()
        assert np.array_equal(z['oracle_rule_difference'],data['oracle_coarse_rate']-data['oracle_fine_rate'])
    with np.load(DIRECTIONAL,allow_pickle=False) as z:
        for k in common:assert record(z[k])==record(data[k]),k
        assert record(z['fine_saved_total_rate'])==record(data['oracle_fine_rate'])
        assert np.array_equal(z['PDE_rate_a']+z['PDE_rate_b'],data['reference_rate'])
    np.savez_compressed(out/'saved-vector-inputs.npz',**data)
    prior=read(SOURCE/'results.json');oracle_results=read(ROOT/'runs/reconnect-oracle-path/results.json')
    directional_results=read(ROOT/'runs/reconnect-oracle-directional/results.json')
    summary,arrays=evaluate(data,prior,oracle_results,directional_results)
    save(out/'source-new-global-results.json',prior);save(out/'source-oracle-results.json',oracle_results)
    save(out/'source-directional-results.json',directional_results)
    save(out/'residual-budget.json',summary);np.savez_compressed(out/'residual-vectors.npz',**arrays)
    save(out/'input-provenance.json',{'parent_input_sha256':digest(SOURCE/'original-inputs.npz'),
        'parent_action_sha256':digest(SOURCE/'global-action.npz'),'oracle_snapshot_sha256':digest(ORACLE),
        'directional_snapshot_sha256':digest(DIRECTIONAL),'original_common_member_matches':matches,
        'directional_same_observation_and_total_rate_verified':True,
        'saved_input_artifact_sha256':digest(out/'saved-vector-inputs.npz'),
        'source_array_records':{k:record(v) for k,v in data.items()},
        'original_arrays_unchanged':True,'no_interpolation_reordering_or_normalization':True})
    for name in ['scipy','cached_global_exchange','reconstructed_global_study','periodic_density_path','positive_kernel_2d',
        'oracle_path_reference','mass_projection_certificates','contact_spectral_reference']:
        assert name not in sys.modules,name
    print(json.dumps(summary,indent=2),flush=True)


def replay(directory):
    out=Path(directory);provenance=read(out/'input-provenance.json')
    assert digest(out/'saved-vector-inputs.npz')==provenance['saved_input_artifact_sha256']
    with np.load(out/'saved-vector-inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    result,arrays=evaluate(data,read(out/'source-new-global-results.json'),read(out/'source-oracle-results.json'),read(out/'source-directional-results.json'))
    assert result==read(out/'residual-budget.json')
    with np.load(out/'residual-vectors.npz',allow_pickle=False) as z:
        assert set(z.files)==set(arrays)
        for k,v in arrays.items():assert np.array_equal(v,z[k]),k
    print('Saved-vector signed residual budget replays exactly; no kernel, path, graph, quadrature or propagation executed.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output');parser.add_argument('--replay');args=parser.parse_args()
    if args.replay:replay(args.replay)
    elif args.output:main(args.output)
    else:parser.error('--output or --replay required')
