"""Read saved compression data and freeze a density-source experiment; no solvers."""
from pathlib import Path
from fractions import Fraction as F
import argparse,math,resource,shutil,signal,time,traceback
import numpy as np
from audit_route_decision import ROOT,read,save,digest

PARENT=ROOT/'runs/reconnect-t1-density-calibration'
SOURCE=ROOT/'runs/reconnect-compression-union64'
PROTOCOL=ROOT/'research/reconnect/compression_density_definition_protocol.json'


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def inventory():
    b=arrays(SOURCE/'builder-inputs.npz');s=arrays(SOURCE/'trajectory-source.npz');u=arrays(SOURCE/'union-definition.npz')
    index=read(SOURCE/'graph-index.json');seal=read(SOURCE/'coefficient-seal.json');old=read(SOURCE/'results.json')
    assert seal['graph_index_sha256']==digest(SOURCE/'graph-index.json') and seal['builder_packet_sha256']==digest(SOURCE/'builder-inputs.npz')
    assert read(SOURCE/'geometry-audit-execution.json')['all48_pure_geometry_audits_passed']
    assert read(SOURCE/'numerical-audit.json')['all48_geometries_exact_rates_and_complete_raw_sequences_verified']
    assert old['comparison_decision']=='compression_three_level_improvement_resolved_with_native_control'
    ids=s['ids'];labels=s['labels'];m,c=u['mass'],u['c0']
    assert ids.shape==m.shape==c.shape==s['rho0_point'].shape==(4096,) and labels.shape==(4096,2)
    assert all(int(v)==i for i,v in enumerate(ids)) and np.array_equal(b['ids'],ids)
    for i,(x,y) in enumerate(labels):
        assert F(float(x))==F(2*(i%64)+1,128) and F(float(y))==F(2*(i//64)+1,128)
    assert np.all(m>0) and np.isfinite(m).all() and c.min()>=0 and c.max()<=1
    assert all(u[k].shape==(3,4096) and np.isfinite(u[k]).all() for k in ('reference','space','time','reference_space_coarse','reference_space_fine','reference_time_check'))
    for name in ('time','positions','rho'):np.testing.assert_array_equal(b['k16_'+name],b['k32_'+name][::2])
    np.testing.assert_array_equal(b['k32_positions'][0],labels);np.testing.assert_array_equal(b['k32_rho'][0],s['rho0_point'])
    assert np.isfinite(b['k32_positions']).all() and b['k32_positions'].min()>=0 and b['k32_positions'].max()<1
    assert b['k32_rho'].min()>0 and np.isfinite(b['k32_rho']).all()
    mid16=set(map(float,b['k16_midpoint_time']));mid32=set(map(float,b['k32_midpoint_time']))
    assert not mid16&mid32 and len(mid16|mid32)==48
    crossings=[]
    for K in (16,32):
        times=b[f'k{K}_time'];position=b[f'k{K}_positions'];assert len(times)==2*K+1 and np.all(np.diff(times)>0)
        assert times[0]==0 and times[-1]==.05
        delta=np.diff(position,axis=0);short=delta-np.rint(delta)
        assert not np.any(abs(delta)==.5) and np.max(abs(short))<.5
        crossings.append({'K':K,'frames':len(times),'positive_wrapped_jumps':np.sum(delta>.5,axis=(0,1)).tolist(),
            'negative_wrapped_jumps':np.sum(delta<-.5,axis=(0,1)).tolist(),
            'maximum_short_displacement_per_axis':np.max(abs(short),axis=(0,1)).tolist(),
            'maximum_saved_dt':float(np.max(np.diff(times))),'new_lift_array_or_winding_field_generated':False})
    expected=set()
    for i in range(4096):
        row,col=divmod(i,64)
        for j in (row*64+(col+1)%64,((row+1)%64)*64+col):expected.add(tuple(sorted((i,j))))
    graphs=[];initial_shifts=None;paths={}
    for K in (16,32):
        run=read(SOURCE/f'k{K}/native-results.json');path=arrays(SOURCE/f'k{K}/trajectory-states.npz')
        assert len(run['steps'])==K and path['states'].shape==path['independent_states'].shape==(K+1,4096)
        np.testing.assert_array_equal(path['states'][0],c);np.testing.assert_array_equal(path['independent_states'][0],c)
        assert all(F(v['dt_exact'])==F(1,20*K) for v in run['steps'])
        paths[str(K)]={'saved_actual_and_independent_steps':K,'path_sha256':digest(SOURCE/f'k{K}/trajectory-states.npz'),
            'run_sha256':digest(SOURCE/f'k{K}/native-results.json'),'rerun':False}
        for step in range(K):
            item=index[f'k{K}-step{step:02d}'];g=read(SOURCE/item['file']);receipt=read(SOURCE/f'k{K}/geometry-only-audit-step{step:02d}.json')
            h=digest(SOURCE/item['file']);assert h==item['sha256']==receipt['cache_sha256']==run['steps'][step]['coefficient_cache_sha256']
            assert receipt['builder_packet_sha256']==seal['builder_packet_sha256'] and receipt['pure_geometry_only_no_mass_c0_reference_loaded']
            assert g['ids']==ids.tolist() and set(map(tuple,g['edges']))==expected and len(g['edges'])==8192
            assert len(g['cell_areas_exact'])==4096 and F(g['D_exact'])==F(.02) and g['periods_exact']==['1','1'] and g['thickness_exact']=='1'
            assert item['frame']==2*step+1 and item['midpoint_time_saved']==float(b[f'k{K}_midpoint_time'][step])
            shifts=dict(zip(map(tuple,g['edges']),map(tuple,g['image_shifts'])))
            if initial_shifts is None:initial_shifts=shifts
            graphs.append({'K':K,'step':step,'file':item['file'],'sha256':h,'faces':8192,
                'image_labels_different_from_first_saved_midpoint':sum(v!=initial_shifts[e] for e,v in shifts.items())})
    return {'N':64,'nodes':4096,'saved_geometries_verified':48,'faces_per_geometry':8192,'all_saved_adjacencies_match_same_cardinal_labels':True,
        'initial_geometry_graph_present':False,'initial_ring_defined_from_label_geometry_and_verified_existing_adjacency':True,
        'coarse_frames_bitwise_subset_of_fine_frames':True,'distinct_midpoint_times':48,'saved_motion_diagnostics':crossings,
        'graphs':graphs,'oracle_paths':paths,'common_initial_fluctuation_norm':old['common_initial_fluctuation_norm'],
        'saved_material_mass_sum':float(m.sum()),'minimum_saved_material_mass':float(m.min()),
        'given_initial_point_density_range':[float(s['rho0_point'].min()),float(s['rho0_point'].max())],
        'original_state_and_reference_rebuilt':False,'new_density_geometry_lift_or_state_arrays':0}


def fixed_budget_floor():
    u=arrays(SOURCE/'union-definition.npz');m,c=u['mass'],u['c0'];sigma=read(SOURCE/'results.json')['common_initial_fluctuation_norm']
    norm=lambda a:float(np.sqrt(m@(a*a)))
    mu=float(m@c/m.sum());V0=float(m@((c-mu)**2));assert math.sqrt(V0)==sigma
    H=lambda a:1-float(m@((a-float(m@a/m.sum()))**2))/V0
    hm=lambda e:(2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c))))*e+e*e)/V0
    paths={K:arrays(SOURCE/f'k{K}/trajectory-states.npz') for K in (16,32)}
    runs={K:read(SOURCE/f'k{K}/native-results.json') for K in (16,32)};rows=[]
    for k,t in ((1,.025),(2,.05)):
        cp={'oracle_16_32':norm(paths[16]['states'][8*k]-paths[32]['states'][16*k]),
            'four_reference_space':4*norm(u['space'][k]),'four_reference_time':4*norm(u['time'][k]),
            'two_initial_bridge':2*norm(u['reference'][0]-c),'same_N64_projection_identity':0.}
        hp={'oracle_16_32':abs(H(paths[16]['states'][8*k])-H(paths[32]['states'][16*k])),
            'two_reference_space_same_basis':2*abs(H(u['reference_space_coarse'][k])-H(u['reference_space_fine'][k])),
            'two_reference_time_same_basis':2*abs(H(u['reference_time_check'][k])-H(u['reference'][k])),
            'two_initial_bridge':2*abs(H(u['reference'][0])-H(c)),'same_N64_projection_identity':0.}
        numerical=0.
        for K in (16,32):
            step=runs[K]['steps'][k*K//2-1];a,b=paths[K]['states'][k*K//2],paths[K]['independent_states'][k*K//2]
            machine_tail=step['cumulative_machine_indicator_absolute']+step['cumulative_tail_indicator_absolute'];numerical+=machine_tail
            cp[f'oracle_e{K}']=machine_tail+norm(a-b);hp[f'oracle_complete_sequence_H{K}']=abs(H(a)-H(b))
        hp['oracle_machine_tail_to_H']=hm(numerical)
        rows.append({'time':t,'concentration_fixed_parts_absolute':cp,'concentration_floor_relative':sum(cp.values())/sigma,
            'H_fixed_parts':hp,'H_floor':sum(hp.values()),'future_source_positive_parts_still_required':True,
            'source_error_gain_predicted_or_computed':False})
    return {'common_sigma64':sigma,'rows':rows,'floors_not_acceptance_threshold_changes':True,
        'cannot_subtract_shared_reference_or_fit_correlation':True,'new_source_results_used':False}


def preserve():
    old=read(PARENT/'preservation.json');formal=old['formal_manifests_verified']+[{'manifest':str((PARENT/'sha256.json').relative_to(ROOT)),
        'manifest_sha256':digest(PARENT/'sha256.json'),'entries_verified':len(read(PARENT/'sha256.json'))}]
    assert len(formal)==117
    for item in formal:
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for name,h in entries.items():assert digest(ROOT/name)==h,name
    originals=read('/workspace/original-handoff-sha256.json');attachments=read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals)==782 and len(attachments)==15
    for name,h in originals.items():assert digest(ROOT/name)==h,name
    for item in attachments:
        p=Path('/workspace')/item['path'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
    for item in old['historical_diagnostic_differences_retained']:
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p)
        differences=[{'path':k,'saved':v,'current':digest(ROOT/k)} for k,v in entries.items() if digest(ROOT/k)!=v]
        assert sorted(differences,key=lambda a:a['path'])==sorted(item['historical_paths_differing_from_later_formal_versions'],key=lambda a:a['path'])
    return {'formal_manifests_verified':formal,'originals_verified':782,'attachments_verified':15,
        'historical_diagnostic_differences_retained':old['historical_diagnostic_differences_retained'],'all_prior_scientific_files_unchanged':True}


def main(directory):
    out=Path(directory);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def timeout(*_):raise TimeoutError('frozen120second read-only definition cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        p=read(PROTOCOL);gate=read(PARENT/'next-gate-spec.json')
        assert gate['type']=='read_only_smooth_compression_N64_density_extension_contract' and not gate['already_executed']
        assert digest(PARENT/'next-gate-spec.json')==p['source_gate_sha256']
        for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
        shutil.copyfile(PROTOCOL,out/'protocol.json');shutil.copyfile(PARENT/'next-gate-spec.json',out/'source-gate-spec.json')
        save(out/'implementation-lock.json',{'files':{name:digest(ROOT/name) for name in p['implementation_files']},'read_only_definition':True})
        save(out/'input-inventory.json',inventory());save(out/'fixed-budget-floor.json',fixed_budget_floor())
        save(out/'definition.json',{'date':'2026-11-07','timezone':'Asia/Shanghai','contract':p['contract'],
            'contract_sha256':digest(ROOT/p['contract']),'proposed_case':p['proposed_case'],
            'ledger_secant_density_not_point_oracle':True,'anchor_proxy_mass_ratio_not_assumed_two':True,
            'explicit_integer_lift_required_no_saved_coordinate_repair':True,'moving_ledger_oracle_path_reuse_forbidden':True,
            'static_semigroup_rule_not_applied':True,'new_density_coefficient_lift_graph_or_state_arrays':0,'new_software_checks':0,
            'recorded_checks_total':504,'future_software_checks_if_both_pass':506,'future_execution_performed':False,
            'all_Becker_unresolved_gates_preserved':True,'stage3':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
        save(out/'preservation.json',preserve());elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed<120 and peak<1024
        save(out/'execution.json',{'completed':True,'elapsed_seconds':elapsed,'peak_RSS_MiB':peak,
            'new_flow_density_geometry_or_solver_calls':False,'old_paths_rerun':False,'published_or_pushed':False})
        print('Read-only compression density contract:48 saved geometries and existing uncertainty inputs verified; no new physical arrays.',flush=True)
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});raise
    finally:signal.alarm(0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);main(p.parse_args().directory)
