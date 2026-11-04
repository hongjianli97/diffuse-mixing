"""One predeclared original-c0 two-row ideal material-pair reference."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
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
from ideal_material_pair_reference import integrate_ideal_material_rows,cell_support_offsets

SOURCE=ROOT/'runs/reconnect-global-residual-review'
NORM=0.2479286450687567
ARRAY_FIELDS={'neighbor_indices','neighbor_ids','neighbor_offsets','neighbor_lifted_centers','target_nodes',
    'target_weights','target_node_inner_integrals','neighbor_signed_mass_rate_contributions'}


def record(a):
    return {'shape':list(a.shape),'dtype':a.dtype.str,'C_order_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def encode_rule(rule,rows):
    arrays={k:v for k,v in rule.items() if isinstance(v,np.ndarray)}
    descriptions=[]
    for row in rows:
        descriptions.append({k:v for k,v in row.items() if k not in ARRAY_FIELDS})
        prefix=f'row{row["target_id"]}_'
        for k in ARRAY_FIELDS:arrays[prefix+k]=row[k]
    return {'order':rule['order'],'rows':descriptions,'raw_weights_not_renormalized':True},arrays


def comparison(data,levels,prior_pair):
    lookup={int(v):i for i,v in enumerate(data['ids'])};indices=np.array([lookup[63],lookup[64]])
    mass=data['mass'][indices];point=data['reconstructed_rate'][indices]
    oracle=data['oracle_fine_rate'][indices];oracle_coarse=data['oracle_coarse_rate'][indices]
    pde=data['reference_rate'][indices]
    rates={order:np.array([r['rate'] for r in levels[order]]) for order in [8,16]}
    fine=rates[16];coarse=rates[8]
    first=point-fine;second=fine-oracle;third=oracle-pde;total=point-pde
    closure=total-first-second-third;vectors=np.array([first,second,third,closure])
    inner=lambda u,v:math.fsum(float(m)*float(a)*float(b) for m,a,b in zip(mass,u,v))/(NORM*NORM)
    norm=lambda v:math.sqrt(inner(v,v))
    gram=[[inner(u,v) for v in vectors] for u in vectors]
    norms=[math.sqrt(gram[i][i]) for i in range(4)]
    rule_delta=coarse-fine;oracle_delta=oracle_coarse-oracle
    reduction=np.array([r['mass_rate_reduction_difference']/r['original_mass'] for r in levels[16]])
    weight_area=np.array([abs(r['quadrature_area']*128**2-1) for r in levels[16]])
    indicator_parts={'ideal_Gauss8_vs_16_vector':norm(rule_delta),'saved_continuous_oracle_rule_vector':norm(oracle_delta),
        'new_point_vs_cell_reduction_arithmetic':norm(reduction),
        'raw_weight_area_multiplicative_indicator':norm(2*weight_area*abs(fine)),
        'saved_point_reference_and_arithmetic_indicators':prior_pair['sum_of_empirical_difference_indicators']}
    indicator=sum(indicator_parts.values())
    resolved_first=norms[0]>indicator;resolved_second=norms[1]>indicator
    decision='two_row_signed_split_resolved_relative_to_saved_controls' if resolved_first and resolved_second else 'two_row_split_inconclusive_at_fixed_rules'
    rows=[]
    for k,identity in enumerate([63,64]):
        row_indicator=abs(rule_delta[k])+abs(oracle_delta[k])+abs(reduction[k])+abs(2*weight_area[k]*fine[k])+prior_pair['sum_of_empirical_difference_indicators']*NORM/math.sqrt(mass[k])
        rows.append({'target_id':identity,'original_mass':float(mass[k]),'original_cbar':float(data['c0'][indices[k]]),
            'saved_point_rate':float(point[k]),'ideal_Gauss8_rate':float(coarse[k]),'ideal_primary_Gauss16_rate':float(fine[k]),
            'saved_continuous_fine_rate':float(oracle[k]),'saved_continuous_coarse_rate':float(oracle_coarse[k]),'saved_PDE_rate':float(pde[k]),
            'point_minus_ideal':float(first[k]),'ideal_minus_continuous':float(second[k]),'continuous_minus_PDE':float(third[k]),
            'total_point_minus_PDE':float(total[k]),'arithmetic_closure':float(closure[k]),
            'ideal_coarse_minus_fine':float(rule_delta[k]),'continuous_coarse_minus_fine':float(oracle_delta[k]),
            'row_empirical_difference_sum':float(row_indicator),
            'point_minus_ideal_sign_resolved':bool(abs(first[k])>row_indicator),
            'ideal_minus_continuous_sign_resolved':bool(abs(second[k])>row_indicator)})
    shared=[]
    for row,other in zip(levels[16],[64,63]):
        local=int(np.flatnonzero(row['neighbor_ids']==other)[0])
        shared.append(float(row['neighbor_signed_mass_rate_contributions'][local]))
    result={'main_state':'unchanged original c0 material mass means, not an auxiliary polynomial',
        'fixed_global_PDE_norm':NORM,'primary_rule_predeclared':16,'rows':rows,
        'component_names':['point_minus_ideal','ideal_minus_continuous','continuous_minus_PDE','arithmetic_closure'],
        'two_row_component_norms_over_original_global_norm':norms,'signed_Gram':gram,
        'twice_point_quadrature_projection_cross_term':2*gram[0][1],
        'point_quadrature_projection_correlation':gram[0][1]/(norms[0]*norms[1]) if norms[0]*norms[1] else None,
        'same_pair_point_total_error':norm(total),'same_pair_ideal_total_error_context_only':norm(fine-pde),
        'same_pair_continuous_total_error':norm(third),
        'same_pair_combined_point_minus_continuous':norm(point-oracle),
        'Gram_sum_minus_total_squared':math.fsum(map(math.fsum,gram))-norm(total)**2,
        'vector_identity_remainder':norm(total-vectors.sum(axis=0)),
        'empirical_difference_indicators':indicator_parts,'sum_of_empirical_difference_indicators':indicator,
        'point_minus_ideal_pair_resolvable':bool(resolved_first),'ideal_minus_continuous_pair_resolvable':bool(resolved_second),
        'decision':decision,'no_rigorous_error_bound_or_physical_threshold_claim':True,
        'open_pair_ideal_mass_rate':math.fsum(r['raw_mass_rate_numerator'] for r in levels[16]),
        'shared_target_mass_contributions':shared,'shared_target_antisymmetry_residual_unrepaired':sum(shared),
        'two_rows_not_global_causal_percentages_or_candidate_improvement':True,
        'candidate_weights_or_saved_results_changed':False,'new_candidate_graphs':0,'new_candidate_transient_runs':0,
        'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    arrays={'order8_rates':coarse,'order16_rates':fine,'component_vectors':vectors,'point_total_error':total,
        'ideal_rule_difference':rule_delta,'saved_oracle_rule_difference':oracle_delta,'target_mass':mass,'target_ids':np.array([63,64],dtype=np.int64)}
    return result,arrays


def main(output,tests,test_log):
    manifest=read(SOURCE/'sha256.json')
    for rel,h in manifest.items():assert digest(ROOT/rel)==h,rel
    protocol=read(SOURCE/'next-gate-spec.json')
    assert not protocol['already_executed'] and protocol['two_predeclared_tensor_Gauss_orders']==[8,16]
    assert protocol['target_ids']==[63,64] and protocol['fine_raw_rule_is_primary']==16
    snapshot=ROOT/protocol['source_snapshot'];assert digest(snapshot)==protocol['source_snapshot_sha256']
    checks=read(tests);assert checks=={'successful':True,'checks_run':6,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',{'protocol':protocol,'parent_gate_sha256':digest(SOURCE/'next-gate-spec.json'),
        'parent_manifest_entries_verified':len(manifest),'primary_arithmetic_fixed_before_run':'raw Gauss16 target-weight dot of full neighbor integral, then factor2 and original mass',
        'controls_fixed_before_run':'sum ideal/old oracle rule vector differences, new raw reduction/area indicators and saved old two-row arithmetic/reference indicators',
        'tensor_target_block':8,'fine_raw_rule_not_selected_by_PDE_closeness':True})
    for name,path in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('tests.json',Path(tests)),('tests.log',Path(test_log)),
        ('source-point-pair.json',ROOT/'runs/reconnect-reconstructed-pair/main-c0-comparison.json')]:shutil.copyfile(path,out/name)
    with np.load(SOURCE/'saved-vector-inputs.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    original_records={k:record(v) for k,v in data.items()}
    with np.load(snapshot,allow_pickle=False) as z:
        for k in ['ids','mass','c0','reference_rate']:
            assert record(z[k])==original_records[k],k
        assert record(z['positions'][0])==original_records['positions']
        assert record(z['rho'][0])==original_records['rho']
    np.savez_compressed(out/'original-inputs.npz',**data)
    N=protocol['N'];D=float(F(protocol['D_exact']));h=float(F(protocol['h_exact']))
    assert [list(v) for v in cell_support_offsets(N,h)]==protocol['neighbor_cell_offsets']
    started=time.monotonic();last=['start']
    def guard(identity,order,done,total):
        last[0]=f'ID{identity}-Gauss{order}-{done}/{total}'
        elapsed=time.monotonic()-started;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>600 or rss>512:raise RuntimeError(f'predeclared resource limit exceeded: seconds={elapsed}, peak_RSS_MiB={rss}')
        if done%64==0 or done==total:print(f'{last[0]}; {elapsed:.3f}s; peak {rss:.2f} MiB',flush=True)
    def timeout_handler(signum,frame):raise TimeoutError('predeclared 600 second limit reached')
    signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(600)
    try:
        levels={};descriptions={};evaluations=0
        for order in [8,16]:
            rule,rows=integrate_ideal_material_rows(N,data['ids'],data['mass'],data['c0'],data['positions'],D,h,[63,64],order,target_block=8,progress=guard)
            description,arrays=encode_rule(rule,rows);levels[order]=rows;descriptions[order]=description
            save(out/f'order{order}-records.json',description);np.savez_compressed(out/f'order{order}-integrals.npz',**arrays)
            evaluations+=sum(row['kernel_node_pairs_evaluated'] for row in rows)
        assert evaluations==protocol['planned_max_kernel_node_pairs_two_targets_two_rules']==58490880
        result,arrays=comparison(data,levels,read(out/'source-point-pair.json'))
        guard(64,16,256,256);signal.alarm(0)
        assert original_records=={k:record(v) for k,v in data.items()}
        assert digest(snapshot)==protocol['source_snapshot_sha256']
        save(out/'results.json',result);np.savez_compressed(out/'signed-budget.npz',**arrays)
        save(out/'input-provenance.json',{'source_snapshot':protocol['source_snapshot'],'source_snapshot_sha256':digest(snapshot),
            'source_saved_vector_inputs_sha256':digest(SOURCE/'saved-vector-inputs.npz'),
            'copied_original_array_records':original_records,'input_artifact_sha256':digest(out/'original-inputs.npz'),
            'state_and_mass_unchanged':True,'extra_geometry_and_full_density_reference_only':True})
        status={'updated':'2026-11-01','timezone':'Etc/UTC','gate_executed':True,'primary_rule':16,'orders_executed':[8,16],
            'new_software_checks':6,'previous_recorded_checks':417,'recorded_checks_total':423,'previous_checks_rerun':0,
            'new_reference_only_fixed_configurations':1,'new_ideal_original_c0_rows':2,
            'new_reference_kernel_node_pairs':evaluations,'new_candidate_weights':0,'new_candidate_graphs':0,
            'new_global_matrices':0,'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,'new_exact_algebra_certificate_groups':0,
            'decision':result['decision'],'global_point_pair_vs_projection_split':'still_not_identified_by_two_rows',
            'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
            'ordinary_central_resistance_FV_baseline_retained':True,'mass_average_resistance_remains_experimental':True,
            'next_work':'read-only interpretation of this fixed original-c0 pair and existing global/conditional evidence; stop quadrature growth, no coefficient repair or propagation',
            'pdf_generated':False,'software_release_created':False,'published':False}
        save(out/'stage3-status.json',status)
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'first_attempt_passed':True,
            'source_manifest_entries_verified':len(manifest),'new_software_checks':6,'recorded_checks_total':423,'previous_checks_rerun':0,
            'candidate_oracle_reference_elapsed_seconds':time.monotonic()-started,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'predeclared_resource_caps_passed':True,'wall_clock_cap_seconds':600,'peak_RSS_MiB_cap':512,
            'old_candidate_or_reference_operators_not_rerun':True,'old_results_and_thresholds_unchanged':True,
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as error:
        signal.alarm(0)
        save(out/'failure.json',{'type':type(error).__name__,'reason':str(error),'last_stage':last[0],
            'elapsed_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'partial_records_preserved':True,'inputs_formula_rules_and_thresholds_not_changed':True,'fluid_orc_enabled':False})
        raise


def replay(directory):
    out=Path(directory);p=read(out/'input-provenance.json')
    assert digest(out/'original-inputs.npz')==p['input_artifact_sha256']
    with np.load(out/'original-inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    spec=read(out/'source-gate-spec.json');levels={}
    for order in [8,16]:
        rule,rows=integrate_ideal_material_rows(128,data['ids'],data['mass'],data['c0'],data['positions'],
            float(F(spec['D_exact'])),float(F(spec['h_exact'])),[63,64],order,target_block=8)
        description,arrays=encode_rule(rule,rows);assert description==read(out/f'order{order}-records.json')
        with np.load(out/f'order{order}-integrals.npz',allow_pickle=False) as z:
            assert set(z.files)==set(arrays)
            for k,v in arrays.items():assert np.array_equal(v,z[k]),k
        levels[order]=rows
    result,arrays=comparison(data,levels,read(out/'source-point-pair.json'))
    assert result==read(out/'results.json')
    with np.load(out/'signed-budget.npz',allow_pickle=False) as z:
        assert set(z.files)==set(arrays)
        for k,v in arrays.items():assert np.array_equal(v,z[k]),k
    print('Same original-c0 two-row Gauss8/16 ideal material-pair reference and signed budget replay exactly; candidate unchanged, no propagation or physical admission.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output');parser.add_argument('--tests');parser.add_argument('--test-log');parser.add_argument('--replay');args=parser.parse_args()
    if args.replay:replay(args.replay)
    else:
        if not all([args.output,args.tests,args.test_log]):parser.error('--output, --tests and --test-log required')
        main(args.output,args.tests,args.test_log)
