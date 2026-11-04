"""One saved-state global ordinary FV action and replayable error comparison.

NumPy is used only for immutable NPZ extraction and old CSR row-norm indicators.
The FV action, diagnostics and standalone replay use the standard library.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform

from audit_route_decision import ROOT, digest, read, save
from cartesian_fv_stream import MATERIAL_PRIOR, stream_action


SOURCE = ROOT/'runs/reconnect-four-mean-interface'
STENCIL = ROOT/'runs/reconnect-FV-interface-expansion/source-stencil.json'
COMPARISON = ROOT/'runs/reconnect-FV-face-budget/source-main-c0-comparison.json'
LOCAL_FLUXES = ROOT/'runs/reconnect-FV-face-budget/source-main-c0-local-fluxes.json'
CONTACT_RESULTS = ROOT/'runs/reconnect-contact-frozen/results.json'


def record(array):
    return {'shape':list(array.shape),'dtype':array.dtype.str,
            'C_order_sha256':hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def norm(mass, values, mask=None):
    return math.sqrt(math.fsum(m*v*v for i,(m,v) in enumerate(zip(mass,values)) if mask is None or mask[i]))


def float_gather(packet):
    data,model=packet['saved_arrays'],packet['material_model']
    points=data['positions'];rho=data['rho'];means=data['c0'];mass=data['mass']
    axes=[sorted({p[k] for p in points}) for k in range(2)]
    shape=list(map(len,axes));step=[model['periods'][k]/shape[k] for k in range(2)]
    bypoint={tuple(p):i for i,p in enumerate(points)}
    out=[0.]*len(points)
    for j,y in enumerate(axes[1]):
        for i,x in enumerate(axes[0]):
            left=bypoint[(x,y)];terms=[]
            for axis,sign in [(0,-1),(0,1),(1,-1),(1,1)]:
                other=[i,j];other[axis]=(other[axis]+sign)%shape[axis]
                right=bypoint[(axes[0][other[0]],axes[1][other[1]])]
                g=2*model['thickness']*step[1-axis]*model['D']*rho[left]*rho[right]/(step[axis]*(rho[left]+rho[right]))
                terms.append(g*(means[right]-means[left]))
            out[left]=math.fsum(terms)/mass[left]
    return out


def evaluate(packet):
    data,model=packet['saved_arrays'],packet['material_model']
    controls=packet['source_controls'];stencil=controls['stencil']
    selected=[(*edge['ids'],edge['axis']) for edge in stencil['edges']]
    action=stream_action(data['positions'],data['ids'],data['mass'],data['rho'],model['periods'],model['D'],data['c0'],
        model_prior=model['model_prior'],thickness=model['thickness'],selected_edges=selected)
    assert action.shape == (128,128) and action.face_count == 32768
    edge_lookup={(*sorted((a,b)),axis):(g,q,a,b) for a,b,axis,g,q in action.selected_faces}
    for edge in stencil['edges']:
        assert edge_lookup[(*edge['ids'],edge['axis'])][0] == F(edge['conductance_exact'])
    byid={identity:i for i,identity in enumerate(action.ids)}
    previous=controls['two_row_comparison']
    for row in previous['rows']:
        assert action.rates[byid[row['target_id']]] == F(row['FV_exact_binary_flux_rate_exact'])
    for row in controls['local_fluxes']:
        for item in row['outward_contributions']:
            target,other=row['target_id'],item['neighbor_id']
            axis=0 if item['direction'].startswith('x') else 1
            g,q,left,right=edge_lookup[(*sorted((target,other)),axis)]
            assert (q if left == target else -q) == F(item['mass_rate_exact_kg_per_s'])
    mass=data['mass'];reference=data['reference_rate'];kernel=data['kernel_rate']
    fv=list(map(float,action.rates));floating=float_gather(packet)
    denominator=previous['original_all_cell_mass_weighted_PDE_rate_norm']
    assert denominator>0
    relative=lambda values,mask=None:norm(mass,values,mask)/denominator
    error_fv=[a-b for a,b in zip(fv,reference)]
    error_kernel=[a-b for a,b in zip(kernel,reference)]
    assert error_kernel == data['rate_error']
    fv_error,kernel_error=relative(error_fv),relative(error_kernel)
    exact_error=[a-F(b) for a,b in zip(action.rates,reference)]
    exact_error_square=sum((F(m)*e*e for m,e in zip(mass,exact_error)),F(0))
    exact_error_norm=math.sqrt(float(exact_error_square))/denominator
    regional={}
    for label,mask in [('all',[True]*len(mass)),('near_contact',data['near_contact']),('far_from_contact',data['far_from_contact'])]:
        assert label=='all' or all(isinstance(x,bool) for x in mask)
        regional[label]={'cells':sum(mask),'mass_kg':math.fsum(m for m,keep in zip(mass,mask) if keep)}
        for method,error in [('ordinary_FV',error_fv),('saved_kernel',error_kernel)]:
            value=relative(error,mask)
            regional[label][method]={'error_over_original_global_PDE_norm':value,
                'error_squared_fraction_of_this_method_global':(value/(fv_error if method=='ordinary_FV' else kernel_error))**2,
                'maximum_absolute_rate_error_per_s':max(abs(e) for e,keep in zip(error,mask) if keep)}
    mean_difference=max(abs(F(c)-F(v)) for c,v in zip(data['c0'],data['volume16_state']))
    mean_float=float(mean_difference)
    uncertainty={
        'saved_face_vs_volume16_over_original_norm':relative([a-b for a,b in zip(reference,data['volume16_rate'])]),
        'saved_volume8_vs_volume16_over_original_norm':relative([a-b for a,b in zip(data['volume8_rate'],data['volume16_rate'])]),
        'FV_exact_binary_vs_float_cell_gather_over_original_norm':relative([a-b for a,b in zip(fv,floating)]),
        'FV_exact_error_subtraction_vs_display_norm_difference':abs(exact_error_norm-fv_error),
        'saved_kernel_CSR_vs_saved_pair_scatter_over_original_norm':relative([a-b for a,b in zip(kernel,data['pair_scatter_rate'])]),
        'FV_row_norm_times_saved_mean_difference_indicator':relative([float(a)*mean_float for a in action.row_absolute_sum_bounds]),
        'saved_kernel_row_norm_times_saved_mean_difference_indicator':relative([a*mean_float for a in packet['derived_controls']['kernel_absolute_row_sum_per_mass']]),
    }
    indicator=math.fsum(uncertainty.values());gain=kernel_error-fv_error
    results={
        'scope':'one full-domain frozen comparison on original saved contact material means',
        'nodes':len(mass),'source_state':'original saved c0, unchanged',
        'original_global_PDE_rate_norm':denominator,
        'recomputed_norm_minus_original_scalar':norm(mass,reference)-denominator,
        'ordinary_FV_global_relative_error':fv_error,'saved_kernel_global_relative_error':kernel_error,
        'saved_kernel_global_error_minus_original_report':kernel_error-controls['contact_results']['regional_comparison']['all']['error_over_global_reference_norm'],
        'relative_error_reduction':gain,'reduction_percent_of_saved_kernel_error':100*gain/kernel_error,
        'saved_kernel_error_divided_by_FV_error':kernel_error/fv_error,
        'FV_exact_binary_error_norm_control':exact_error_norm,
        'FV_two_row_control_over_original_global_norm':relative(error_fv,[identity in [63,64] for identity in action.ids]),
        'all_seven_saved_g_two_saved_rates_and_eight_local_flux_contributions_equal':True,
        'regional_comparison_with_original_masks':regional,
        'saved_mean_difference_max_absolute_exact':str(mean_difference),
        'empirical_uncertainty_components':uncertainty,'sum_of_empirical_uncertainty_indicators':indicator,
        'gain_divided_by_empirical_indicator':gain/indicator if indicator else None,
        'gain_exceeds_saved_reference_and_arithmetic_indicators':gain>indicator,
        'indicator_is_not_a_certified_total_physical_error_bound_or_acceptance_threshold':True,
        'time_discretization_present':False,
        'exact_FV_mass_rate_sum_kg_per_s':str(action.exact_mass_balance),
        'exact_FV_energy_residual_kg_per_s':str(action.exact_energy_residual),
        'exact_FV_projected_dissipation_kg_per_s':str(action.dissipation),
        'FV_projected_dissipation_display_kg_per_s':float(action.dissipation),
        'nonnegative_shared_conductances':action.minimum_g>=0,
        'minimum_shared_g_exact':str(action.minimum_g),'maximum_shared_g_exact':str(action.maximum_g),
        'main_FV_face_law_evaluations':action.face_count,'face_counts_by_axis':list(action.face_counts_by_axis),
        'previously_saved_control_faces':7,'additional_face_locations_beyond_saved_local_stencil':action.face_count-7,
        'independent_float_cell_gather_face_law_evaluations':4*len(mass),
        'new_FV_global_frozen_actions':1,'new_point_kernel_builds':0,'new_point_candidate_graphs':0,
        'new_global_matrices':0,'new_reference_integrations':0,'LP_solves':0,
        'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'new_exact_algebra_certificate_groups':0,'new_software_checks':4,'previous_checks_rerun':0,
        'recorded_checks_total':392,'four_mean_correction_installed':False,
        'geometry_is_extra_information_reference_baseline_not_generic_point_input':True,
        'finite_node_candidate_admitted':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    exact={'ids':list(action.ids),'mass_rates_kg_per_s':list(map(str,action.mass_rates)),
        'rates_per_s':list(map(str,action.rates)),
        'row_absolute_sum_bounds_per_s':list(map(str,action.row_absolute_sum_bounds)),
        'selected_positive_axis_faces':[{'from_id':a,'to_id':b,'axis':axis,'g_exact':str(g),'q_exact':str(q)}
                                       for a,b,axis,g,q in action.selected_faces],
        'global_error_square_from_exact_binary_FV_minus_saved_reference':str(exact_error_square)}
    return results,exact


def replay(output):
    out=Path(output).resolve()
    results,exact=evaluate(read(out/'frozen-inputs.json'))
    assert results==read(out/'results.json')
    assert exact==read(out/'exact-action.json')
    for name,expected in read(out/'input-provenance.json')['receipt_SHA256'].items():
        assert digest(out/name)==expected,name
    print(json.dumps({'standard_library_replay_identical':True,'global_FV_error':results['ordinary_FV_global_relative_error'],
        'saved_kernel_error':results['saved_kernel_global_relative_error'],'recorded_checks_total':392,
        'fluid_orc_enabled':False}),flush=True)


def main(output,tests,test_log):
    import numpy as np
    source_manifest=read(SOURCE/'sha256.json')
    for rel,expected in source_manifest.items():assert digest(ROOT/rel)==expected,rel
    protocol_path=SOURCE/'stage3-status.json'
    protocol=read(protocol_path)['next_ordinary_FV_global_frozen_predeclared']
    assert not protocol['already_executed'] and protocol['N']==128
    snapshot=ROOT/protocol['source_snapshot']
    assert digest(snapshot)==protocol['source_snapshot_sha256']
    checks=read(tests);assert checks['successful'] and checks['checks_run']==4
    assert checks['failures']==checks['errors']==checks['skipped']==0
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    keys=['ids','mass','c0','reference_rate','kernel_rate','rate_error','pair_scatter_rate',
          'volume8_rate','volume16_rate','volume16_state','near_contact','far_from_contact']
    with np.load(snapshot,allow_pickle=False) as z:
        raw={key:z[key].copy() for key in keys}
        raw['positions']=z['positions'][0].copy();raw['rho']=z['rho'][0].copy()
        data=z['L_data'];ptr=z['L_indptr']
        assert len(ptr)==16385 and ptr[0]==0 and ptr[-1]==len(data)
        old_CSR_records={'L_data':record(data),'L_indptr':record(ptr)}
        kernel_row_abs=np.add.reduceat(np.abs(data),ptr[:-1])/raw['mass']
    np.savez_compressed(out/'selected-source-observations.npz',**raw)
    input_arrays={key:value.tolist() for key,value in raw.items()}
    assert all(a!=b for a,b in zip(input_arrays['near_contact'],input_arrays['far_from_contact']))
    model={'periods':[1.,1.],'D':float(F(protocol['D_binary_input_exact'])),'thickness':1.,'model_prior':MATERIAL_PRIOR}
    packet={'saved_arrays':input_arrays,'material_model':model,
        'derived_controls':{'kernel_absolute_row_sum_per_mass':kernel_row_abs.tolist()},
        'source_controls':{'stencil':read(STENCIL),'two_row_comparison':read(COMPARISON),
                           'local_fluxes':read(LOCAL_FLUXES),'contact_results':read(CONTACT_RESULTS)}}
    save(out/'frozen-inputs.json',packet)
    receipts={'source-stencil.json':STENCIL,'source-two-row-comparison.json':COMPARISON,
              'source-local-fluxes.json':LOCAL_FLUXES,'source-contact-results.json':CONTACT_RESULTS,
              'tests.json':Path(tests),'tests.log':Path(test_log)}
    for name,path in receipts.items():
        with (out/name).open('xb') as f:f.write(path.read_bytes())
    save(out/'config.json',{'study':'one predeclared full-domain ordinary FV frozen action',
        'protocol':protocol,'protocol_source_sha256':digest(protocol_path),
        'source_manifest_entries_verified':len(source_manifest),
        'main_arithmetic':'Fraction streaming shared faces on unchanged binary source inputs',
        'norm':'original mass-weighted rate error divided by unchanged original global PDE norm',
        'uncertainty_combination_fixed_before_evaluation':'sum of two saved reference differences, FV floating/error-subtraction controls, saved kernel pair scatter difference, and two saved-mean row-norm indicators',
        'old_CSR_arrays_read_for_row_norm_indicator_only':True,
        'no_transport_matrix_constructed_and_no_old_kernel_applied_to_new_state':True,
        'software_checks':checks,'no_new_physical_acceptance_threshold':True})
    save(out/'input-provenance.json',{'source_snapshot':protocol['source_snapshot'],
        'source_snapshot_sha256':digest(snapshot),'selected_source_member_records':{key:record(value) for key,value in raw.items()},
        'selected_NPZ_sha256':digest(out/'selected-source-observations.npz'),
        'old_CSR_member_records_used_only_for_row_norm':old_CSR_records,
        'receipt_SHA256':{name:digest(path) for name,path in receipts.items()},
        'input_JSON_preserves_original_source_binary_values':True})
    results,exact=evaluate(packet)
    save(out/'results.json',results);save(out/'exact-action.json',exact)
    save(out/'environment.json',{'python':platform.python_version(),'numpy_extraction_only':np.__version__,
        'FV_and_independent_replay_dependencies':'standard library only'})
    print(json.dumps({key:value for key,value in results.items() if key not in ['exact_FV_projected_dissipation_kg_per_s']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--output');group.add_argument('--replay')
    parser.add_argument('--tests');parser.add_argument('--test-log');args=parser.parse_args()
    if args.replay:replay(args.replay)
    else:
        if not args.tests or not args.test_log:parser.error('--tests and --test-log are required for source extraction')
        main(args.output,args.tests,args.test_log)
