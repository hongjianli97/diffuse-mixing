"""Independent periodic integer reassembly and immutable-source audit."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import math
import numpy as np

from audit_route_decision import ROOT,read,save,digest,preservation
from reconstructed_global_study import SOURCE,NORM,relative
from global_frozen_gate_plan import array_record


def independent_action(cache,g,c0,mass):
    """Array shifts/object integers, independent of the scalar edge streamer."""
    N=cache['N'];S=math.lcm(*(F(float(v)).denominator for v in c0))
    C=np.array([int(F(float(v))*S) for v in c0],dtype=object).reshape(N,N)
    Q=math.lcm(*(v.denominator for v in g));G=[int(v*Q) for v in g]
    accum=np.zeros((N,N),dtype=object);row_sum=np.zeros((N,N),dtype=object);energy=0
    for k,(dx,dy) in enumerate(cache['offsets_positive_half']):
        columns=np.array([G[cache['coefficient_map'][c][k]] for c in range(N)],dtype=object)
        delta=np.roll(C,(-dy,-dx),(0,1))-C
        flux=columns[None,:]*delta
        accum+=flux;accum-=np.roll(flux,(dy,dx),(0,1))
        row_sum+=columns[None,:];row_sum+=np.roll(columns,dx)[None,:]
        energy+=sum((columns[None,:]*delta*delta).ravel())
    rates=[F(int(n),Q*S)/F(float(m)) for n,m in zip(accum.ravel(),mass)]
    row_L1=[F(2*int(n),Q)/F(float(m)) for n,m in zip(row_sum.ravel(),mass)]
    row_energy=sum(int(c)*int(n) for c,n in zip(C.ravel(),accum.ravel()))
    assert row_energy+energy==0 and sum(accum.ravel())==0
    return rates,row_L1,list(map(int,accum.ravel())),Q,S,F(int(energy),Q*S*S)


def main(output):
    out=Path(output).resolve();assert not (out/'sha256.json').exists()
    config=read(out/'config.json');spec=read(SOURCE/'gate-spec.json')
    assert config['protocol']==spec and not spec['already_executed']
    assert config['source_protocol_sha256']==digest(SOURCE/'gate-spec.json')
    assert (out/'source-gate-spec.json').read_bytes()==(SOURCE/'gate-spec.json').read_bytes()
    assert read(out/'tests.json')=={'successful':True,'checks_run':7,'failures':0,'errors':0,'skipped':0}
    status=read(out/'stage3-status.json');summary=read(out/'results.json');execution=read(out/'execution.json')
    assert status['recorded_checks_total']==410+7==417 and status['previous_checks_rerun']==0
    assert execution['study_exit_code']==execution['tests_exit_code']==0
    assert execution['candidate_elapsed_seconds_including_arithmetic_controls']<600 and execution['peak_RSS_MiB']<512
    provenance=read(out/'input-provenance.json');snapshot=ROOT/spec['source_snapshot']
    assert digest(snapshot)==spec['source_snapshot_sha256']==provenance['source_snapshot_sha256']
    assert digest(out/'original-inputs.npz')==provenance['input_artifact_sha256']
    with np.load(out/'original-inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(snapshot,allow_pickle=False) as z:
        for key,record in provenance['original_members'].items():
            original=z[key][0] if key in ['positions','rho'] else z[key]
            assert array_record(original)==record==array_record(data[key]),key
        for key,record in provenance['source_old_kernel_L_hashes_for_row_norm_control'].items():
            assert array_record(z[key])==record,key
        L=z['L_data'];ptr=z['L_indptr']
        assert np.array_equal(np.add.reduceat(np.abs(L),ptr[:-1])/data['mass'],data['old_kernel_row_L1'])
    assert data['near_contact'].dtype==data['far_from_contact'].dtype==np.dtype(bool)
    assert np.array_equal(data['near_contact'],~data['far_from_contact'])
    assert np.count_nonzero(data['near_contact'])==5120 and np.count_nonzero(data['far_from_contact'])==11264
    cache=read(out/'coefficient-cache.json');paths=read(out/'representative-paths.json')
    cost=read(SOURCE/'cost-and-reuse.json');N=cache['N'];assert N==128
    assert cache['offsets_positive_half']==cost['offsets_positive_half']
    assert cache['coefficient_keys']==cost['coefficient_cache_keys']
    assert cache['radius_squared_numerators']==cost['radius_squared_lattice_numerators']
    with np.load(SOURCE/'request-class-map.npz',allow_pickle=False) as z:
        assert np.array_equal(cache['path_map'],z['path_class']) and np.array_equal(cache['coefficient_map'],z['coefficient_class'])
    assert len(paths)==132 and len(cache['B_h_original_values'])==45 and len(cache['conductances_exact'])==964
    assert len(cache['offsets_positive_half'])*N*N==2719744
    progress_records=[json.loads(s) for s in (out/'representative-progress.jsonl').read_text().splitlines()]
    assert progress_records==paths
    for sid,record in enumerate(paths):
        assert record['class_id']==sid and record['ownership_applies_only_to_representative']
        dx=record['normal_offset'];given=list(map(F,record['given_density_sequence']))
        # Independent consequence of the verified given product point field,
        # not a continuous oracle field and not an input to coefficient creation.
        expected=1/given[0] if dx==0 else (F(1,2)/given[0]+sum((1/v for v in given[1:-1]),F(0))+F(1,2)/given[-1])/dx
        mean=F(cache['mean_inverse_density_exact'][sid])
        assert mean==expected==F(record['path']['mean_inverse_density_exact'])
        parts=record['path']['intervals']
        assert sum(((F(p['stop'])-F(p['start']))/F(p['density']) for p in parts),F(0))==mean
        assert parts[0]['start']=='0' and parts[-1]['stop']=='1'
        i,j=record['representative_input_indices']
        assert record['path']['start_id']==int(data['ids'][i]) and record['path']['end_id']==int(data['ids'][j])
        assert cache['signatures'][sid]['given_density_sequence']==record['given_density_sequence']
    g=list(map(F,cache['conductances_exact']));q=F(cache['q_exact']);D=F(cache['D_exact'])
    assert q==F(1,N*N) and D==F(spec['D_exact']) and F(cache['h_exact'])==F(spec['h_exact'])
    for slot,(sid,rid) in enumerate(cache['coefficient_keys']):
        assert g[slot]==2*q*q*D/F(cache['mean_inverse_density_exact'][sid])*F(cache['B_h_original_values'][rid])>=0
    result=read(out/'integer-action.json')
    rates,row_L1,acc,Q,S,energy=independent_action(cache,g,data['c0'],data['mass'])
    assert list(map(str,acc))==result['flux_numerators'] and str(Q)==result['Q'] and str(S)==result['S']
    assert energy==F(result['energy_pair_exact'])==-F(result['energy_row_exact'])
    with np.load(out/'global-action.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    assert np.array_equal(np.array(list(map(float,rates))),arrays['rate'])
    assert np.array_equal(np.array(list(map(float,row_L1))),arrays['row_L1'])
    assert np.array_equal(arrays['signed_error'],arrays['rate']-data['reference_rate'])
    arithmetic=read(out/'arithmetic-controls.json')
    precision_actions={}
    for p in (60,100):
        values=list(map(F,arithmetic['high_precision_conductances'][str(p)]))
        precise,_,_,_,_,_=independent_action(cache,values,data['c0'],data['mass'])
        precision_actions[p]=precise
        assert np.array_equal(np.array(list(map(float,precise))),arrays[f'kernel_{p}_digit_rate'])
    assert np.array_equal(np.array([float(a-b) for a,b in zip(rates,precision_actions[100])]),arrays['binary_B_h_minus_100_digit_action'])
    assert np.array_equal(np.array([float(a-b) for a,b in zip(precision_actions[60],precision_actions[100])]),arrays['kernel_60_minus_100_digit_action'])
    pair=read(out/'old-pair-control.json');assert pair['all_saved_conductances_compared']==663
    assert pair['all_conductances_identical_exactly'] and pair['completed_before_global_action']
    assert (out/'source-pair-edges.json').read_bytes()==(ROOT/'runs/reconnect-reconstructed-pair/edges.json').read_bytes()
    assert (out/'source-pair-comparison.json').read_bytes()==(ROOT/'runs/reconnect-reconstructed-pair/main-c0-comparison.json').read_bytes()
    lookup={int(v):i for i,v in enumerate(data['ids'])}
    for row in pair['rows']:assert rates[lookup[row['target_id']]]==F(row['exact_rate'])
    assert summary['locked_original_global_PDE_norm']==NORM
    assert relative(data['mass'],arrays['signed_error'])==summary['reconstructed_global_error']
    assert sum(summary['empirical_difference_indicators'].values())==summary['sum_of_empirical_difference_indicators']
    benefit=summary['saved_kernel_global_error']-summary['reconstructed_global_error']
    assert benefit==summary['benefit_saved_kernel_minus_reconstructed']
    assert benefit>summary['sum_of_empirical_difference_indicators'] and summary['decision']=='resolvable_global_improvement'
    for label,mask in [('near_contact',data['near_contact']),('far_from_contact',data['far_from_contact'])]:
        assert relative(data['mass'],arrays['signed_error'],mask)==summary['regions'][label]['candidate_error_over_locked_global_norm']
    assert summary['minimum_raw_conductance']==float(min(g))>0 and summary['negative_raw_conductance_slots']==0
    assert summary['exact_mass_inventory_rate']==summary['exact_energy_identity_residual']=='0'
    assert summary['float_mass_inventory_rate_numpy_unrepaired']==float(data['mass']@arrays['rate'])
    for obj in [summary,status]:
        assert not obj['finite_node_candidate_admitted'] and not obj['fluid_orc_enabled']
        assert obj['stage3_physical_acceptance']=='incomplete'
    assert status['new_sparse_matrices']==status['new_candidate_transient_runs']==status['new_PDE_reference_solves']==0
    preserved=preservation()
    known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']
    prior=prior+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
        'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in prior:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256']
        entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==69
    save(out/'preservation.json',preserved)
    audit={'original_input_reference_region_and_old_matrix_member_hashes_verified':True,
        'all_representative_integrals_match_given_product_field_sequence_control':True,
        'integral_only_reuse_preserves_representative_ID_receipt_scope':True,
        'all_coefficient_factors_and_plan_maps_verified':True,
        'independent_object_integer_periodic_shift_assembly_matches_every_primary_flux_numerator':True,
        'independent_60_and_100_digit_control_actions_match_every_saved_rate':True,
        'exact_shared_inventory_and_energy_identity_verified':True,
        'old_663_conductances_and_both_exact_rows_match_before_global_evaluation':True,
        'locked_norm_and_unchanged_region_metrics_verified':True,
        'all_2719744_edges_covered_without_sparse_matrix':True,
        'predeclared_execution_caps_passed':True,'new_software_checks':7,'recorded_checks_total':417,'previous_checks_rerun':0,
        'one_fixed_physical_configuration_arithmetic_reassembly_not_new_cases':True,
        'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,
        'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':69,
        'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'receipt-audit.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
