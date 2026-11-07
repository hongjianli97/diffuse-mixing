"""Read-only inventory and finite joint-control contract; no sampling or solving."""
from pathlib import Path
from fractions import Fraction as F
import argparse,resource,signal,time,traceback
import numpy as np
from audit_route_decision import ROOT,read,save,digest

PARENT=ROOT/'runs/reconnect-compression-density-diagnostics'
CALIBRATION=ROOT/'runs/reconnect-compression-density-calibration'
ORIGINAL=ROOT/'runs/reconnect-compression-union64'
REFERENCE=ROOT/'runs/reconnect-compression-reference'
PROTOCOL=ROOT/'research/reconnect/compression_joint_definition_protocol.json'


def inventory():
    config=read(REFERENCE/'config.json');cases=[(c['columns'],c['time_steps']) for c in config['reference_cases']]
    assert cases==[(512,256),(1024,256),(1024,512),(1024,1024)]
    assert digest(REFERENCE/'compression-reference.npz')==digest(ORIGINAL/'original-parent-reference.npz')
    with np.load(REFERENCE/'compression-reference.npz',allow_pickle=False) as z:
        assert z['mass'].shape==z['c0'].shape==(16384,)
        refs=[]
        for M,K in cases:
            name=f'n{M}-k{K}';assert z[name+'_B_snapshots'].shape==(3,M) and z[name+'_projected_states'].shape==(3,16384)
            np.testing.assert_array_equal(z[name+'_snapshot_time'],[0.,.025,.05])
            assert z[name+'_step_time'].shape==(K+1,)
            refs.append({'name':name,'columns':M,'steps':K,'B_snapshot_shape':[3,M],'projected_shape':[3,16384],
                         'complete_B_history_saved':name+'_B_history' in z.files})
        np.testing.assert_array_equal(z['best_projected_states'],z['n1024-k1024_projected_states'])
        parent_mass=z['mass'];parent_ids=z['ids']
    with np.load(ORIGINAL/'union-definition.npz',allow_pickle=False) as z:
        m,c0=z['mass'],z['c0'];np.testing.assert_array_equal(m,z['n128_mass_sum'])
        assert m.shape==c0.shape==(4096,) and np.all(m>0)
        for key in ('reference','space','time','reference_space_coarse','reference_space_fine','reference_time_check'):assert z[key].shape==(3,4096)
    with np.load(ORIGINAL/'trajectory-source.npz',allow_pickle=False) as z:
        ids=z['ids'];np.testing.assert_array_equal(z['n128_ids'],parent_ids)
        membership=z['n128_to_n64'];assert membership.shape==(16384,) and np.bincount(membership).tolist()==[4]*4096
    # Capacity equality from the already saved union, not newly generated mass.
    assert float(parent_mass.sum())>0 and len(ids)==4096
    with np.load(ORIGINAL/'builder-inputs.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],ids)
        for key in ('time','positions','rho'):np.testing.assert_array_equal(z['k16_'+key],z['k32_'+key][::2])
        assert z['k32_positions'].shape==(65,4096,2) and z['k16_positions'].shape==(33,4096,2)
        assert not any(key.startswith('k64_') for key in z.files)
    paths=[]
    for s in ('oracle','ledger'):
        for K in (16,32):
            folder=(ORIGINAL if s=='oracle' else CALIBRATION/s)/f'k{K}';run=read(folder/'native-results.json')
            with np.load(folder/'trajectory-states.npz',allow_pickle=False) as z:
                for key in ('states','independent_states'):
                    assert z[key].shape==(K+1,4096);np.testing.assert_array_equal(z[key][0],c0)
            assert len(run['steps'])==K
            paths.append({'source':s,'K':K,'file':str((folder/'trajectory-states.npz').relative_to(ROOT)),
                          'sha256':digest(folder/'trajectory-states.npz'),'actual_and_independent_reused':True})
    index=read(ORIGINAL/'graph-index.json');assert len(index)==48
    p=read(ORIGINAL/index['k32-step00']['file']);assert len(p['edges'])==8192 and p['ids']==ids.tolist() and F(p['D_exact'])==F(.02)
    return {'reference_cases':refs,'canonical_reference_byte_identity_with_saved_parent':True,
        'reference_origin_N':128,'material_union_N':64,'parent_cells_per_union':4,'saved_N64_mass_identity_verified':True,
        'same_observation_hierarchy_required':'fine material columns -> original N128 mass means -> original N64 four-cell unions',
        'saved_K32_frames':65,'saved_K16_frames':33,'shared_frames_bitwise_equal':True,'new_target_frames_generated':0,
        'K64_absent_from_canonical_builder':True,'n2048_absent_from_canonical_reference':True,
        'absence_scope':'canonical frozen inputs,not unavailablev080 or all external material',
        'old_paths':paths,'original_geometries':48,'faces_per_geometry':8192,'new_geometry_or_PDE_calls':0}


def plan():
    old32={F(i,1280) for i in range(65)};fine64={F(i,2560) for i in range(129)};newmid={F(2*j+1,2560) for j in range(64)}
    oldmid={F(2*j+1,40*K) for K in (16,32) for j in range(K)}
    assert old32<fine64 and fine64-old32==newmid and newmid.isdisjoint(oldmid)
    existing={(512,256),(1024,256),(1024,512),(1024,1024)}
    required={(1024,1024),(2048,1024),(2048,2048)};new=required-existing
    assert new=={(2048,1024),(2048,2048)}
    elapsed=sum(read(q)[key] for q,key in [(ORIGINAL/'execution.json','elapsed_main_seconds'),
        (ORIGINAL/'geometry-audit-execution.json','elapsed_including_wait_seconds'),(CALIBRATION/'execution.json','elapsed_seconds')])
    concentration={'ledger_time32_64':1,'oracle_time32_64':1,'reference_space_same1024steps':4,'reference_time_same2048columns':4,
                   'initial_bridge_best_reference':2,'ledger_e32':1,'ledger_e64':1,'oracle_e32':1,'oracle_e64':1,'same_N64_identity':0}
    H={'ledger_H32_64':1,'oracle_H32_64':1,'reference_H_space_same1024steps':2,'reference_H_time_same2048columns':2,
       'initial_H_bridge_best_reference':2,'ledger_sequence_H32':1,'ledger_sequence_H64':1,'oracle_sequence_H32':1,'oracle_sequence_H64':1,
       'hm_of_all_four_machine_tail_terms':1,'same_N64_identity':0}
    return {'type':'single_N64_ledger_oracle_joint_time_reference_control','already_executed':False,'N':64,'nodes':4096,
        'D':.02,'T_exact':'1/20','observation_times_exact':['0','1/40','1/20'],'sources':['oracle','ledger'],
        'old_candidate_K':[16,32],'new_candidate_K':[64],'comparison_time_pair':[32,64],
        'new_position_frames':64,'total_position_frames':129,'shared_saved_position_frames':65,
        'new_geometry_graphs':64,'faces_per_geometry':8192,'new_density_fields':128,'new_coefficient_packets':128,
        'new_actual_paths':2,'new_independent_paths':2,'new_action_intervals':128,
        'old_actual_paths_reused':4,'old_independent_paths_reused':4,'old_paths_rerun':0,'kernel_rerun':False,
        'reference_cases_reused':[[1024,1024]],'new_reference_cases':[[2048,1024],[2048,2048]],
        'reference_space_pair':[[1024,1024],[2048,1024]],'reference_time_pair':[[2048,1024],[2048,2048]],
        'new_best_reference':[2048,2048],'new_reference_PDE_solves':2,'new_reference_CN_intervals':3072,
        'reference_B_history_shapes':[[1025,2048],[2049,2048]],'reference_B_history_raw_bytes':50364416,
        'reference_observation_N_sequence':[128,64],'original_reference_source_unchanged':True,
        'reference_wrapper_must_match_original_small_case_bitwise':True,
        'minimality_scope':'two missing results for fixed next adjacent levels and matched space/time bases;not global optimum',
        'new_midpoints_disjoint_from_old16_32_coefficients':True,'old_N64_mass_c0_ids_and_density_definitions_unchanged':True,
        'new_coefficients_and_pure_geometry_audits_sealed_before_scalar_or_reference_loading':True,
        'reference_no_candidate_density_weights_or_errors':True,'new_C_budget_positive_weights':concentration,'new_H_budget_positive_weights':H,
        'time_decrease_machine_weights':{'K16':1,'K32':2,'K64':1},'time_order_claim':False,
        'old_K16_K32_budgets_and_classifications_immutable':True,'new_K64_rows_cannot_overwrite_old_rows':True,
        'new_physical_acceptance_threshold':False,'proxy_must_beat_oracle_to_pass_Stage3':False,
        'software_groups_required':2,'software_max_nodes_per_group':16,'software_max_frames_per_group':8,
        'software_wall_seconds_cap':120,'main_wall_seconds_cap':1600,'independent_saved_audit_wall_seconds_cap':400,'peak_RSS_MiB_cap':1024,
        'resource_basis_old_combined_seconds':elapsed,'resource_basis_scaled64_over48_seconds':elapsed*64/48,
        'resource_estimate_is_not_accuracy_prediction':True,'current_recorded_checks_total':506,'future_checks_if_two_groups_pass':508,
        'stop_on_input_test_raw_guard_or_cap_failure':True,'stop_expansion_after_complete_case_if_time_or_source_control_unresolved_or_adverse':True,
        'automatic_K128_reference4096_kernel_or_model_extension':False,'seed':20261101,'algebra_guard':1e-10,'exact_tail_per_interval':'1/10000000000000000',
        'reference_initial_N128_atol':3e-16,'reference_CN_energy_balance_atol_exclusive':1e-15,
        'reference_energy_strictly_decreases':True,'CN_reference_not_positive_candidate_backend':True,
        'stage3':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}


def preserve():
    old=read(PARENT/'preservation.json');formal=old['formal_manifests_verified']+[{'manifest':str((PARENT/'sha256.json').relative_to(ROOT)),
        'manifest_sha256':digest(PARENT/'sha256.json'),'entries_verified':len(read(PARENT/'sha256.json'))}]
    assert len(formal)==120
    for item in formal:
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for name,h in entries.items():assert digest(ROOT/name)==h,name
    originals=read('/workspace/original-handoff-sha256.json');attachments=read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals)==782 and len(attachments)==15
    for name,h in originals.items():assert digest(ROOT/name)==h,name
    for item in attachments:
        p=Path('/workspace')/item['path'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
    for item in old['historical_diagnostic_differences_retained']:
        entries=read(ROOT/item['manifest']);assert digest(ROOT/item['manifest'])==item['manifest_sha256']
        diff=[{'path':k,'saved':v,'current':digest(ROOT/k)} for k,v in entries.items() if digest(ROOT/k)!=v]
        assert sorted(diff,key=lambda x:x['path'])==sorted(item['historical_paths_differing_from_later_formal_versions'],key=lambda x:x['path'])
    return {'formal_manifests_verified':formal,'originals_verified':782,'attachments_verified':15,
        'historical_diagnostic_differences_retained':old['historical_diagnostic_differences_retained'],'all_prior_scientific_files_unchanged':True}


def main(directory):
    out=Path(directory).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def timeout(*_):raise TimeoutError('read-only definition120second cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        p=read(PROTOCOL);gate=read(PARENT/'next-gate-spec.json')
        assert gate['type']=='read_only_ledger_oracle_joint_time_reference_control_contract' and not gate['already_executed']
        assert digest(PARENT/'next-gate-spec.json')==p['source_gate_sha256']
        for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
        save(out/'protocol.json',p);save(out/'source-gate-spec.json',gate)
        save(out/'implementation-lock.json',{'files':{name:digest(ROOT/name) for name in p['implementation_files']},'read_only_definition':True})
        save(out/'inventory.json',inventory());save(out/'proposed-experiment.json',plan())
        save(out/'preservation.json',preserve());elapsed=time.monotonic()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed<120 and peak<1024
        save(out/'execution.json',{'completed':True,'elapsed_seconds':elapsed,'peak_RSS_MiB':peak,'new_physical_fields_or_solves':0,
            'new_software_checks':0,'recorded_checks_total':506,'source_files_verified':len(p['sources']),'published_or_pushed':False})
        print('Canonical4references and4old source paths verified;one K64+two-reference joint contract defined without execution.',flush=True)
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False});raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);main(p.parse_args().directory)
