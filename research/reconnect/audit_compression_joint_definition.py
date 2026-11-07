"""Independent saved-reference, exact observation, clock and contract audit."""
from pathlib import Path
from fractions import Fraction as F
import argparse,ast,resource,signal,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest


def audit(out):
    p=read(out/'protocol.json');proposal=read(out/'proposed-experiment.json');inv=read(out/'inventory.json')
    for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
    for name,h in read(out/'implementation-lock.json')['files'].items():assert digest(ROOT/name)==h,name
    assert digest(ROOT/p['contract'])==p['contract_sha256']
    ex=read(out/'execution.json');assert ex['completed'] and ex['elapsed_seconds']<120 and ex['peak_RSS_MiB']<1024
    for name in ('compression_joint_definition.py','audit_compression_joint_definition.py'):
        tree=ast.parse((ROOT/'research/reconnect'/name).read_text())
        assert {n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}<={'pathlib','fractions','audit_route_decision'}
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):assert {a.name for a in n.names}<={'argparse','ast','resource','signal','time','traceback','numpy'}
    source=ROOT/'runs/reconnect-compression-union64';ref=ROOT/'runs/reconnect-compression-reference';old=ROOT/'runs/reconnect-compression-density-calibration'
    assert digest(ref/'compression-reference.npz')==digest(source/'original-parent-reference.npz')
    with np.load(ref/'compression-reference.npz',allow_pickle=False) as z:
        cases=sorted(k.removesuffix('_B_snapshots') for k in z.files if k.endswith('_B_snapshots'))
        assert cases==['n1024-k1024','n1024-k256','n1024-k512','n512-k256']
        parent_m,parent_c=z['mass'],z['c0'];parent_ref=z['n1024-k1024_projected_states']
        for row in inv['reference_cases']:
            name=row['name'];assert z[name+'_B_snapshots'].shape==tuple(row['B_snapshot_shape'])
            assert z[name+'_projected_states'].shape==tuple(row['projected_shape'])
            assert (name+'_B_history' in z.files)==row['complete_B_history_saved']
    with np.load(source/'union-definition.npz',allow_pickle=False) as z:m,c0,R=z['mass'],z['c0'],z['reference'];assert np.array_equal(m,z['n128_mass_sum'])
    # Verify the existing hierarchy scalar by scalar, never construct a new reference array.
    largest=0.;mass_largest=0.
    for i in range(4096):
        r,c=divmod(i,64);children=[(2*r+dr)*128+2*c+dc for dr in (0,1) for dc in (0,1)]
        exact_m=sum((F(float(parent_m[j])) for j in children),F(0));mass_largest=max(mass_largest,abs(float(exact_m)-m[i]))
        assert abs(exact_m-F(float(m[i])))<64*np.finfo(float).eps*F(float(m[i]))
        for values,stored in ((parent_c,c0),*((parent_ref[k],R[k]) for k in range(3))):
            expected=sum((F(float(parent_m[j]))*F(float(values[j])) for j in children),F(0))/F(float(m[i]))
            difference=abs(float(expected)-float(stored[i]));largest=max(largest,difference)
            assert difference<=64*np.finfo(float).eps
    with np.load(source/'builder-inputs.npz',allow_pickle=False) as z:
        for key in ('time','positions','rho'):np.testing.assert_array_equal(z['k16_'+key],z['k32_'+key][::2])
        assert z['k32_positions'].shape==(65,4096,2) and not any(k.startswith('k64_') for k in z.files)
    for row in inv['old_paths']:
        folder=(source if row['source']=='oracle' else old/'ledger')/f"k{row['K']}"
        assert digest(folder/'trajectory-states.npz')==row['sha256']
        with np.load(folder/'trajectory-states.npz',allow_pickle=False) as z:
            for key in ('states','independent_states'):assert z[key].shape==(row['K']+1,4096) and np.array_equal(z[key][0],c0)
    old_clock=[F(i,1280) for i in range(65)];fine_clock=[F(i,2560) for i in range(129)]
    assert fine_clock[::2]==old_clock and len(set(fine_clock)-set(old_clock))==64
    old_midpoints={F(2*i+1,40*K) for K in (16,32) for i in range(K)}
    assert not (set(fine_clock[1::2])&old_midpoints)
    required=set(map(tuple,proposal['reference_space_pair']+proposal['reference_time_pair']))
    available={(512,256),(1024,256),(1024,512),(1024,1024)};missing=set(map(tuple,proposal['new_reference_cases']))
    assert required-available==missing=={(2048,1024),(2048,2048)}
    assert all(not required <= (available|(missing-{item})) for item in missing)
    assert proposal['reference_space_pair']==[[1024,1024],[2048,1024]] and proposal['reference_time_pair']==[[2048,1024],[2048,2048]]
    assert proposal['reference_observation_N_sequence']==[128,64] and proposal['new_best_reference']==[2048,2048]
    counts={'new_position_frames':64,'total_position_frames':129,'shared_saved_position_frames':65,'new_geometry_graphs':64,
        'faces_per_geometry':8192,'new_density_fields':128,'new_coefficient_packets':128,'new_actual_paths':2,'new_independent_paths':2,
        'new_action_intervals':128,'old_actual_paths_reused':4,'old_independent_paths_reused':4,'old_paths_rerun':0,
        'new_reference_PDE_solves':2,'new_reference_CN_intervals':3072,'reference_B_history_raw_bytes':sum((1025*2048*8,2049*2048*8))}
    for key,value in counts.items():assert proposal[key]==value,key
    assert proposal['reference_B_history_shapes']==[[1025,2048],[2049,2048]]
    C=proposal['new_C_budget_positive_weights'];H=proposal['new_H_budget_positive_weights']
    assert C=={'ledger_time32_64':1,'oracle_time32_64':1,'reference_space_same1024steps':4,'reference_time_same2048columns':4,
        'initial_bridge_best_reference':2,'ledger_e32':1,'ledger_e64':1,'oracle_e32':1,'oracle_e64':1,'same_N64_identity':0}
    assert H=={'ledger_H32_64':1,'oracle_H32_64':1,'reference_H_space_same1024steps':2,'reference_H_time_same2048columns':2,
        'initial_H_bridge_best_reference':2,'ledger_sequence_H32':1,'ledger_sequence_H64':1,'oracle_sequence_H32':1,'oracle_sequence_H64':1,
        'hm_of_all_four_machine_tail_terms':1,'same_N64_identity':0}
    incidence={16:0,32:0,64:0}
    for edge in ((16,32),(32,64)):
        for K in edge:incidence[K]+=1
    assert proposal['time_decrease_machine_weights']=={f'K{k}':v for k,v in incidence.items()}=={'K16':1,'K32':2,'K64':1}
    assert proposal['old_K16_K32_budgets_and_classifications_immutable'] and proposal['new_K64_rows_cannot_overwrite_old_rows']
    assert not proposal['new_physical_acceptance_threshold'] and not proposal['proxy_must_beat_oracle_to_pass_Stage3'] and not proposal['time_order_claim']
    assert proposal['new_coefficients_and_pure_geometry_audits_sealed_before_scalar_or_reference_loading']
    assert proposal['reference_no_candidate_density_weights_or_errors'] and proposal['reference_wrapper_must_match_original_small_case_bitwise']
    assert proposal['algebra_guard']==1e-10 and F(proposal['exact_tail_per_interval'])==F(1,10**16) and proposal['seed']==20261101
    assert proposal['reference_initial_N128_atol']==3e-16 and proposal['reference_CN_energy_balance_atol_exclusive']==1e-15
    assert proposal['reference_energy_strictly_decreases'] and proposal['CN_reference_not_positive_candidate_backend']
    old_seconds=sum(read(q)[k] for q,k in [(source/'execution.json','elapsed_main_seconds'),(source/'geometry-audit-execution.json','elapsed_including_wait_seconds'),(old/'execution.json','elapsed_seconds')])
    assert proposal['resource_basis_old_combined_seconds']==old_seconds and proposal['resource_basis_scaled64_over48_seconds']==old_seconds*64/48
    assert proposal['main_wall_seconds_cap']==1600>proposal['resource_basis_scaled64_over48_seconds']
    assert proposal['software_wall_seconds_cap']==120 and proposal['independent_saved_audit_wall_seconds_cap']==400 and proposal['peak_RSS_MiB_cap']==1024
    assert proposal['software_groups_required']==2 and proposal['current_recorded_checks_total']==506 and proposal['future_checks_if_two_groups_pass']==508
    assert not proposal['already_executed'] and not proposal['kernel_rerun'] and not proposal['automatic_K128_reference4096_kernel_or_model_extension']
    return {'passed':True,'canonical_reference_cases_verified':4,'old_actual_and_independent_pairs_verified':4,
        'existing_raw_N128_to_N64_observation_verified':True,'maximum_exact_observation_vs_saved_difference':largest,
        'maximum_exact_parent_mass_vs_saved_difference':mass_largest,'nested_clock_and_new_midpoint_counts_verified':True,
        'two_missing_reference_cases_minimal_for_fixed_matched_bases':True,'positive_budget_weights_and_middle_time_double_count_verified':True,
        'proposed_future_counts':counts,'resource_and_stop_contract_verified':True,
        'new_physical_fields_or_solves':0,'new_software_checks':0,'recorded_checks_total':506,'stage3':'incomplete','fluid_orc_enabled':False}


def main(directory,write=False):
    out=Path(directory).resolve();start=time.monotonic()
    def timeout(*_):raise TimeoutError('read-only definition audit120second cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        findings=audit(out);elapsed=time.monotonic()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed<120 and peak<1024
        if write:save(out/'audit.json',{'passed':True,'findings':findings,'elapsed_seconds':elapsed,'peak_RSS_MiB':peak})
        else:assert findings==read(out/'audit.json')['findings']
        print('Independent source/observation,matched-reference minimality,clock,positive budgets andresource audit passed;future case unexecuted.',flush=True)
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();main(a.directory,a.write)
