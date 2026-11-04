"""One sealed-protocol full-domain frozen action; no matrix or evolution."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import math
import platform
import resource
import shutil
import signal
import time
import numpy as np
import scipy

from audit_route_decision import ROOT,read,save,digest
from global_frozen_gate_plan import array_record
from cached_global_exchange import build_original_global_cache,coefficient_for_indices,apply_cached_exchange,floating_action
from reconstructed_pair_study import high_B,pi_decimal

SOURCE=ROOT/'runs/reconnect-global-frozen-plan'
NORM=0.2479286450687567


def relative(mass,values,mask=None):
    m=mass if mask is None else mass[mask]
    v=np.asarray(values,dtype=float) if mask is None else np.asarray(values,dtype=float)[mask]
    return float(np.sqrt(m@(v*v)))/NORM


def encode_cache(cache):
    return {'N':cache.N,'D_exact':str(cache.D),'h_exact':str(cache.h),'q_exact':str(cache.q),
        'offsets_positive_half':[list(v) for v in cache.offsets],
        'signatures':[{'normal_offset':dx,'given_density_sequence':list(map(str,p))} for dx,p in cache.signatures],
        'path_map':[list(v) for v in cache.path_map],'coefficient_map':[list(v) for v in cache.coefficient_map],
        'coefficient_keys':[list(v) for v in cache.coefficient_keys],
        'radius_squared_numerators':list(cache.radius_squared_numerators),'radius_squared_denominator':cache.N**2,
        'B_h_original_values':list(cache.kernel_values),'mean_inverse_density_exact':list(map(str,cache.path_means)),
        'conductances_exact':list(map(str,cache.conductances)),
        'raw_ownership_receipts_apply_only_to_representatives':True}


def pair_control(cache,c0,edges,pair):
    lookup={v:i for i,v in enumerate(cache.ids)};checked=0
    for e in edges:
        i,j=[lookup[v] for v in e['ids']]
        actual=coefficient_for_indices(cache,i,j)
        assert actual==F(e['conductance_exact_given_binary_B_h']),e['ids']
        checked+=1
    state=tuple(F(float(v)) for v in c0);records=[]
    for row in pair['rows']:
        i=lookup[row['target_id']]
        value=sum((coefficient_for_indices(cache,i,lookup[v['id']])*(state[lookup[v['id']]]-state[i])
                   for v in row['neighbors']),F(0))/cache.mass[i]
        assert value==F(row['reconstructed_rate_exact_given_binary_B_h'])
        records.append({'target_id':row['target_id'],'exact_rate':str(value),'identical_to_saved_two_row_result':True})
    assert checked==663
    return {'all_saved_conductances_compared':checked,'all_conductances_identical_exactly':True,
        'rows':records,'completed_before_global_action':True}


def high_coefficients(cache,precision):
    pi=pi_decimal(precision)
    values=tuple(high_B(F(v,cache.N**2),cache.h,precision,pi) for v in cache.radius_squared_numerators)
    return tuple(2*cache.q*cache.q*cache.D/cache.path_means[sid]*values[rid]
                 for sid,rid in cache.coefficient_keys)


def evaluate(cache,data,controls,old_results,pair_receipt,progress=None):
    mass=data['mass'];state={v:F(float(c)) for v,c in zip(cache.ids,data['c0'])}
    primary=apply_cached_exchange(cache,state,progress=progress)
    assert primary.edge_count==2719744
    for row in pair_receipt['rows']:
        assert primary.rates[cache.ids.index(row['target_id'])]==F(row['exact_rate'])
    rates=np.array(list(map(float,primary.rates)))
    float_rates=floating_action(cache,data['c0'])
    actions={}
    high_tables={}
    for p in (60,100):
        high_tables[p]=high_coefficients(cache,p)
        if progress is not None:progress(f'kernel_control_{p}',0,166)
        actions[p]=apply_cached_exchange(cache,state,conductances=high_tables[p],progress=progress)
    reference=data['reference_rate'];error=rates-reference
    exact_error_display=np.array([float(r-F(float(ref))) for r,ref in zip(primary.rates,reference)])
    row_L1=np.array(list(map(float,primary.row_L1)))
    exact_delta100=np.array([float(a-b) for a,b in zip(primary.rates,actions[100].rates)])
    exact_precision_delta=np.array([float(a-b) for a,b in zip(actions[60].rates,actions[100].rates)])
    norm_numpy=float(np.sqrt(mass@(reference*reference)))
    norm_fsum=math.sqrt(math.fsum(float(m)*float(r)*float(r) for m,r in zip(mass,reference)))
    old_error=old_results['regional_comparison']['all']['error_over_global_reference_norm']
    new_error=relative(mass,error)
    uncertainty={
        'saved_face_vs_volume16':relative(mass,reference-data['volume16_rate']),
        'saved_volume8_vs_volume16':relative(mass,data['volume8_rate']-data['volume16_rate']),
        'saved_kernel_CSR_vs_saved_pair_scatter':relative(mass,data['kernel_rate']-data['pair_scatter_rate']),
        'saved_kernel_error_reduction_recomputed_vs_saved':abs(relative(mass,data['kernel_rate']-reference)-old_error),
        'new_exact_vs_float_flux_reduction':relative(mass,rates-float_rates),
        'new_binary_B_h_vs_100_digit_kernel_action':relative(mass,exact_delta100),
        'new_60_vs_100_digit_action_before_float_subtraction':relative(mass,exact_precision_delta),
        'new_display_subtraction_vs_exact_error':relative(mass,error-exact_error_display),
        'new_row_norm_times_saved_mean_difference':relative(mass,row_L1*controls['exact_c0_vs_Gauss16_max_absolute']),
        'old_row_norm_times_saved_mean_difference':relative(mass,data['old_kernel_row_L1']*controls['exact_c0_vs_Gauss16_max_absolute']),
    }
    indicator=sum(uncertainty.values());benefit=old_error-new_error
    decision='resolvable_global_improvement' if benefit>indicator else ('resolvable_global_deterioration' if benefit < -indicator else 'no_resolvable_global_benefit')
    regions={}
    for label,mask in [('all',np.ones(len(mass),dtype=bool)),('near_contact',data['near_contact']),('far_from_contact',data['far_from_contact'])]:
        indices=np.flatnonzero(mask);peak=int(indices[np.argmax(np.abs(error[mask]))])
        contribution=relative(mass,error,mask)
        regions[label]={'cells':int(np.count_nonzero(mask)),'mass':float(math.fsum(map(float,mass[mask]))),
            'candidate_error_over_locked_global_norm':contribution,
            'saved_kernel_error_over_locked_global_norm':old_results['regional_comparison'][label]['error_over_global_reference_norm'],
            'error_squared_fraction_of_candidate_global':(contribution/new_error)**2,
            'maximum_absolute_rate_error':float(abs(error[peak])),'maximum_error_id':cache.ids[peak],
            'signed_error_at_maximum':float(error[peak])}
    energy_exact=primary.energy_row_exact+primary.energy_pair_exact
    inventory_exact=sum((m*r for m,r in zip(cache.mass,primary.rates)),F(0))
    energy_float=float(np.dot(mass*data['c0'],rates))
    energy_pair_float=float(primary.energy_pair_exact)
    summary={'locked_original_global_PDE_norm':NORM,'recomputed_norm_numpy_control':norm_numpy,
        'recomputed_norm_fsum_control':norm_fsum,'numpy_norm_minus_locked':norm_numpy-NORM,
        'saved_kernel_global_error':old_error,'reconstructed_global_error':new_error,
        'saved_ordinary_FV_global_error_context_only':0.032630071085544846,
        'continuous_oracle_path_value_not_finite_node_result':0.031229167742325044,
        'benefit_saved_kernel_minus_reconstructed':benefit,'error_reduction_percent_of_saved_kernel':100*benefit/old_error,
        'empirical_difference_indicators':uncertainty,'sum_of_empirical_difference_indicators':indicator,
        'benefit_over_empirical_indicator':benefit/indicator,
        'indicator_not_certified_physical_error_bound_or_new_acceptance_threshold':True,
        'decision':decision,'regions':regions,'all_old_pair_control_passed':True,
        'all_undirected_edges_covered':primary.edge_count,'path_representatives':len(cache.path_means),
        'kernel_radii':len(cache.kernel_values),'coefficient_cache_slots':len(cache.conductances),
        'minimum_raw_conductance':float(min(cache.conductances)),
        'negative_raw_conductance_slots':sum(v<0 for v in cache.conductances),
        'primary_coefficient_denominator_bits':primary.Q.bit_length(),'original_state_denominator_bits':primary.S.bit_length(),
        'primary_flux_denominator_bits':(primary.Q*primary.S).bit_length(),
        'exact_mass_inventory_rate':str(inventory_exact),'exact_energy_identity_residual':str(energy_exact),
        'exact_graph_energy_dissipation':str(primary.energy_pair_exact),
        'graph_energy_dissipation_display':float(primary.energy_pair_exact),
        'float_mass_inventory_rate_numpy_unrepaired':float(mass@rates),
        'float_mass_inventory_rate_fsum_unrepaired':math.fsum(float(m)*float(r) for m,r in zip(mass,rates)),
        'float_energy_identity_residual_unrepaired':energy_float+energy_pair_float,
        'exact_constant_state_rate_from_shared_differences':0,
        'new_sparse_matrices':0,'new_PDE_reference_solves':0,'new_candidate_transient_runs':0,
        'no_clipping_renormalization_fitting_or_threshold_change':True,
        'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    assert inventory_exact==energy_exact==0
    arrays={'rate':rates,'signed_error':error,'exact_error_display_control':exact_error_display,
        'float_flux_control_rate':float_rates,'row_L1':row_L1,'binary_B_h_minus_100_digit_action':exact_delta100,
        'kernel_60_minus_100_digit_action':exact_precision_delta,
        'kernel_60_digit_rate':np.array(list(map(float,actions[60].rates))),
        'kernel_100_digit_rate':np.array(list(map(float,actions[100].rates)))}
    integer_receipt={'Q':str(primary.Q),'S':str(primary.S),'flux_numerators':list(map(str,primary.flux_numerators)),
        'energy_pair_exact':str(primary.energy_pair_exact),'energy_row_exact':str(primary.energy_row_exact),
        'edge_count':primary.edge_count,'mass_inventory_numerator_exact':str(sum(primary.flux_numerators))}
    arithmetic={'high_precision_conductances':{str(p):list(map(str,high_tables[p])) for p in (60,100)},
        'high_precision_Q_bits':{str(p):actions[p].Q.bit_length() for p in (60,100)},
        'all_controls_use_same_support_and_original_c0':True}
    return summary,arrays,integer_receipt,arithmetic


def main(output,tests,test_log):
    entries=read(SOURCE/'sha256.json')
    for rel,h in entries.items():assert digest(ROOT/rel)==h,rel
    protocol=read(SOURCE/'gate-spec.json');assert not protocol['already_executed']
    snapshot=ROOT/protocol['source_snapshot'];assert digest(snapshot)==protocol['source_snapshot_sha256']
    checks=read(tests);assert checks=={'successful':True,'checks_run':7,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',{'protocol':protocol,'source_protocol_sha256':digest(SOURCE/'gate-spec.json'),
        'source_manifest_entries_verified':len(entries),'primary_error_definition':'sqrt(sum m*(float(exact rate)-saved reference)**2)/locked original global norm',
        'control_combination_fixed_before_evaluation':'sum saved reference/kernel arithmetic, new exact/float/60/100-digit/error-display and old/new row-norm indicators',
        'no_new_absolute_physical_acceptance_threshold':True})
    for name,path in [('source-gate-spec.json',SOURCE/'gate-spec.json'),('tests.json',Path(tests)),('tests.log',Path(test_log)),
        ('source-pair-edges.json',ROOT/'runs/reconnect-reconstructed-pair/edges.json'),
        ('source-pair-comparison.json',ROOT/'runs/reconnect-reconstructed-pair/main-c0-comparison.json')]:shutil.copyfile(path,out/name)
    started=time.monotonic();last_stage=['start']
    def guard(stage='checkpoint',n=0,total=0,record=None):
        last_stage[0]=stage
        if record is not None:
            with (out/'representative-progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        elapsed=time.monotonic()-started;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>600 or rss>512:raise RuntimeError(f'predeclared execution resource cap exceeded: seconds={elapsed}, RSS_MiB={rss}')
        if n%20==0 or n==total:print(f'{stage}: {n}/{total}; elapsed {elapsed:.3f}s; peak RSS {rss:.2f} MiB',flush=True)
    def timeout_handler(signum,frame):raise TimeoutError('predeclared 600 second resource cap reached')
    signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(600)
    try:
        with np.load(snapshot,allow_pickle=False) as z:
            data={k:z[k].copy() for k in ['ids','mass']};data['positions']=z['positions'][0].copy();data['rho']=z['rho'][0].copy()
        point_records={k:array_record(v) for k,v in data.items()}
        cache=build_original_global_cache(**data,D=float(F(protocol['D_exact'])),h=float(F(protocol['h_exact'])),periods=protocol['periods'],progress=guard)
        cost=read(SOURCE/'cost-and-reuse.json')
        assert encode_cache(cache)['coefficient_keys']==cost['coefficient_cache_keys']
        with np.load(SOURCE/'request-class-map.npz',allow_pickle=False) as z:
            assert np.array_equal(cache.path_map,z['path_class']) and np.array_equal(cache.coefficient_map,z['coefficient_class'])
        save(out/'coefficient-cache.json',encode_cache(cache));save(out/'representative-paths.json',cache.representative_records)
        # Only after the coefficient cache is complete are c0, masks and rates read.
        with np.load(snapshot,allow_pickle=False) as z:
            for key in ['c0','reference_rate','kernel_rate','volume8_rate','volume16_rate','pair_scatter_rate','near_contact','far_from_contact']:
                data[key]=z[key].copy()
            old_L=z['L_data'];old_indptr=z['L_indptr']
            old_L_hashes={key:array_record(value) for key,value in [('L_data',old_L),('L_indptr',old_indptr)]}
            data['old_kernel_row_L1']=np.add.reduceat(np.abs(old_L),old_indptr[:-1])/data['mass']
            del old_L,old_indptr
        originals={k:array_record(v) for k,v in data.items() if k!='old_kernel_row_L1'}
        np.savez_compressed(out/'original-inputs.npz',**data)
        pair=pair_control(cache,data['c0'],read(out/'source-pair-edges.json'),read(out/'source-pair-comparison.json'))
        save(out/'old-pair-control.json',pair);print('All 663 old exact conductances and both exact rows match before global action.',flush=True)
        old_results=read(ROOT/'runs/reconnect-contact-frozen/results.json');controls=old_results['reference_controls']
        save(out/'source-original-kernel-results.json',old_results);save(out/'saved-reference-controls.json',controls)
        summary,arrays,integer_receipt,arithmetic=evaluate(cache,data,controls,old_results,pair,guard)
        guard('completed',1,1);signal.alarm(0)
        assert originals=={k:array_record(data[k]) for k in originals}
        assert point_records=={k:array_record(data[k]) for k in point_records}
        assert digest(snapshot)==protocol['source_snapshot_sha256']
        save(out/'results.json',summary);np.savez_compressed(out/'global-action.npz',**arrays)
        save(out/'integer-action.json',integer_receipt);save(out/'arithmetic-controls.json',arithmetic)
        save(out/'input-provenance.json',{'source_snapshot':protocol['source_snapshot'],'source_snapshot_sha256':digest(snapshot),
            'original_members':originals,'source_old_kernel_L_hashes_for_row_norm_control':old_L_hashes,
            'input_artifact_sha256':digest(out/'original-inputs.npz'),'coefficient_cache_fixed_before_c0_or_reference_read':True})
        status={'updated':'2026-11-01','timezone':'Etc/UTC','gate_executed':True,
            'new_software_checks':7,'previous_recorded_checks':410,'recorded_checks_total':417,'previous_checks_rerun':0,
            'new_original_snapshot_full_domain_frozen_diagnostics':1,'new_candidate_exchange_graphs_represented_implicitly':1,
            'new_edge_weights_logical':2719744,'new_coefficient_cache_slots':964,'new_representative_path_integrals':132,
            'new_sparse_matrices':0,'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,'new_flow_solves':0,
            'LP_solves':0,'new_exact_algebra_certificate_groups':0,'decision':summary['decision'],
            'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
            'ordinary_central_resistance_FV_baseline_retained':True,'mass_average_resistance_remains_experimental':True,
            'full_domain_improvement_does_not_authorize_propagation':True,
            'next_work':'read-only residual scope review using current global arrays and existing conditional interface/projection evidence; no new h/grid/state/time scan or automatic propagation',
            'pdf_generated':False,'software_release_created':False,'published':False}
        save(out/'stage3-status.json',status)
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'first_attempt_passed':True,
            'source_manifest_entries_verified':len(entries),'new_software_checks':7,'recorded_checks_total':417,'previous_checks_rerun':0,
            'candidate_elapsed_seconds_including_arithmetic_controls':time.monotonic()-started,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'predeclared_resource_caps_passed':True,'wall_clock_cap_seconds':600,'peak_RSS_MiB_cap':512,
            'all_controls_same_physical_case_not_extra_parameter_cases':True,'python':platform.python_version(),
            'numpy':np.__version__,'scipy':scipy.__version__,'network_used':False,'environment_configuration_changed':False,
            'committed_or_pushed':False,'published':False})
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as error:
        signal.alarm(0)
        save(out/'failure.json',{'type':type(error).__name__,'reason':str(error),'last_stage':last_stage[0],
            'elapsed_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'partial_evidence_preserved':True,'no_input_formula_or_threshold_retuning':True,'fluid_orc_enabled':False})
        raise


def replay(directory):
    """Reconstruct the same coefficients/action, never a new physical case."""
    out=Path(directory);p=read(out/'input-provenance.json')
    assert digest(out/'original-inputs.npz')==p['input_artifact_sha256']
    with np.load(out/'original-inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    spec=read(out/'source-gate-spec.json')
    cache=build_original_global_cache(**{k:data[k] for k in ['positions','ids','mass','rho']},
        D=float(F(spec['D_exact'])),h=float(F(spec['h_exact'])),periods=spec['periods'])
    assert encode_cache(cache)==read(out/'coefficient-cache.json')
    assert list(cache.representative_records)==read(out/'representative-paths.json')
    pair=pair_control(cache,data['c0'],read(out/'source-pair-edges.json'),read(out/'source-pair-comparison.json'))
    assert pair==read(out/'old-pair-control.json')
    result,arrays,integer,arithmetic=evaluate(cache,data,read(out/'saved-reference-controls.json'),
        read(out/'source-original-kernel-results.json'),pair)
    assert result==read(out/'results.json') and integer==read(out/'integer-action.json')
    assert arithmetic==read(out/'arithmetic-controls.json')
    with np.load(out/'global-action.npz',allow_pickle=False) as z:
        assert set(z.files)==set(arrays)
        for key,a in arrays.items():assert np.array_equal(z[key],a),key
    print('Same original N128 full-domain cache, all 663 old edges, exact shared action, controls and comparison replay exactly; no transient or physical admission.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output');parser.add_argument('--tests');parser.add_argument('--test-log');parser.add_argument('--replay')
    args=parser.parse_args()
    if args.replay:replay(args.replay)
    else:
        if not all([args.output,args.tests,args.test_log]):parser.error('--output, --tests and --test-log required')
        main(args.output,args.tests,args.test_log)
