"""One registered existingN256 compression control, ordinary unchanged Voronoi FV."""
from pathlib import Path
import argparse,gc,json,math,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save,digest
from shear_union64_study import arrays,native_run
from shear_voronoi_fv_study import record
from compression_voronoi_core import sample_points
from compression_spatial_fv_core import compare
from spatial_geometry_schedule import construct
from voronoi_spatial_projection import validate_record

SOURCE=ROOT/'runs/reconnect-compression-time32'
FINE=ROOT/'runs/reconnect-compression-spatial/compression-spatial.npz'
PROTOCOL=ROOT/'research/reconnect/compression_spatial_fv_protocol.json'
CONTROL=ROOT/'research/reconnect/COMPRESSION_SPATIAL_FV_CONTROL.md'
COPIES={'original-fine.npz':FINE,'original-parent-reference.npz':SOURCE/'original-reference.npz',
 'original-parent-reference-results.json':SOURCE/'original-reference-results.json','source-time32-results.json':SOURCE/'results.json',
 'coarse16-states.npz':SOURCE/'old-k16-states.npz','coarse16-results.json':SOURCE/'old-k16-results.json',
 'coarse32-states.npz':SOURCE/'k32/trajectory-states.npz','coarse32-results.json':SOURCE/'k32/native-results.json'}


def inputs(out):
    d=arrays(out/'original-fine.npz');ref=arrays(out/'original-parent-reference.npz');validate_record(d)
    for name in ('mass','c0','ids','material_label_centers'):assert record(d['parent_'+name])==record(ref[name]),name
    assert record(d['parent_reference_states'])==record(ref['best_projected_states'])
    assert record(d['parent_reference_times'])==record(ref['snapshot_time'])
    runs={(128,K):read(out/f'coarse{K}-results.json') for K in (16,32)}
    paths={(128,K):arrays(out/f'coarse{K}-states.npz') for K in (16,32)}
    return d,ref,runs,paths


def main(output,tests,test_log):
    prior=read(SOURCE/'sha256.json');assert len(prior)==2689
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    spec=read(SOURCE/'next-gate-spec.json');proto=read(PROTOCOL)
    assert not spec['already_executed'] and spec['new_ordered_interval_counts']==[16,32] and spec['new_midpoint_graphs']==48
    assert spec['N']==256 and spec['main_wall_seconds_cap']==8400 and spec['main_peak_RSS_MiB_cap']==2048
    assert digest(FINE)==spec['source_sha256'] and digest(ROOT/spec['coarse_FV32_source'])==spec['coarse_FV32_sha256']
    assert read(tests)=={'successful':True,'checks_run':3,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,p in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),
      ('control-at-lock.md',CONTROL),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(p,out/name)
    locked=['periodic_voronoi_fv.py','voronoi_time_action.py','voronoi_time16_core.py','shear_union64_study.py','compression_voronoi_core.py',
      'variable_parent_observation.py','voronoi_spatial_projection.py','compression_spatial_fv_core.py','test_compression_spatial_fv.py',
      'spatial_geometry_worker.py','spatial_geometry_schedule.py','compression_spatial_fv_study.py']
    save(out/'implementation-lock.json',{'files':{name:digest(ROOT/'research/reconnect'/name) for name in locked},
      'protocol_sha256':digest(PROTOCOL),'control_note_sha256':digest(CONTROL),'fixed_before_new_point_validation_and_geometry':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':2689,
      'parent_manifest_sha256':digest(SOURCE/'sha256.json'),'previous_recorded_checks':463,'new_software_checks':3,
      'recorded_checks_total':466,'previous_checks_rerun':0,'new_reference_PDE_solves':0,'old_candidate_reruns':0})
    started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>8400 or peak>2048:raise RuntimeError(f'registered main resource cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k%16384==0 or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;parent_peak{peak:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('registered8400second main cap reached')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(8400)
    try:
        with np.load(FINE,allow_pickle=False) as z:
            labels=z['material_label_centers'].copy();ids=z['ids'].copy();rho0=z['rho0_point'].copy()
        axis=(np.arange(256)+.5)/256;assert record(labels)==record(np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2))
        assert record(rho0)==record(1+.25*np.cos(2*np.pi*labels[:,0]))
        fine=sample_points(labels,ids,rho0,np.linspace(0.,.05,65));b={'ids':ids};checks=[]
        with np.load(FINE,allow_pickle=False) as z:
            for K in (4,8):
                generated=sample_points(labels,ids,rho0,z[f'k{K}_time'])
                for key in ('time','positions','rho'):assert record(generated[key])==record(z[f'k{K}_{key}'])==record(fine[key][::(32//K)])
                checks.append({'K':K,'all_original_time_positions_rho_and_nested32_records_match_bitwise':True})
        for K in (16,32):
            stride=32//K
            for key in ('time','positions','rho'):b[f'k{K}_{key}']=fine[key][::stride]
            b[f'k{K}_boundary_time']=b[f'k{K}_time'][::2];b[f'k{K}_midpoint_time']=b[f'k{K}_time'][1::2]
            (out/f'k{K}').mkdir()
        np.savez_compressed(out/'trajectory-source.npz',labels=labels,ids=ids,rho0_point=rho0)
        np.savez_compressed(out/'builder-inputs.npz',**b);jobs={}
        for K in (16,32):
            for step in range(K):
                jobs[K,step]=float(b[f'k{K}_midpoint_time'][step])
                np.savez_compressed(out/f'k{K}/frame-input-step{step:02d}.npz',ids=ids,
                  positions=b[f'k{K}_positions'][2*step+1],rho=b[f'k{K}_rho'][2*step+1])
        save(out/'input-matching.json',{'source':str(FINE.relative_to(ROOT)),'source_sha256':digest(FINE),'checks':checks,
          'builder_packet_sha256':digest(out/'builder-inputs.npz'),'actual_mass_c0_reference_or_old_states_loaded':False,
          'pure_frame_inputs_only_ids_positions_rho':True,'J_F_C_true_faces_or_gradients_in_builder':False})
        del b,fine,generated;gc.collect();guard('all-pure-inputs-and-frame-jobs-sealed')
        index,geometry_execution=construct(out,jobs,guard,2048)
        save(out/'graph-index.json',index);save(out/'geometry-execution.json',geometry_execution)
        save(out/'coefficient-seal.json',{'graphs':48,'graph_index_sha256':digest(out/'graph-index.json'),
          'builder_packet_sha256':digest(out/'builder-inputs.npz'),'all48_coefficients_sealed_before_mass_c0_reference_or_old_state_loading':True,
          'builder_received_mass_c0_reference_true_faces_or_F_C':False})
        guard('all48-coefficients-sealed-before-state');copied=[]
        for name,p in COPIES.items():
            rel=str(p.relative_to(ROOT));assert digest(p)==prior[rel];shutil.copyfile(p,out/name);copied.append({'copy':name,'source':rel,'sha256':digest(p)})
        save(out/'source-copies.json',copied);d,ref,runs,paths=inputs(out)
        assert record(d['ids'])==record(ids) and record(d['material_label_centers'])==record(labels)
        for K in (16,32):
            run,path=native_run(out,K,d,index,guard);runs[256,K]=run;paths[256,K]=path
            save(out/f'k{K}/native-results.json',run);np.savez_compressed(out/f'k{K}/trajectory-states.npz',**path)
        result,common=compare(d,ref,runs,paths)
        source_result=read(out/'source-time32-results.json')
        for k,row in enumerate(result['per_grid_observations']['128']):
            assert row['native_error_native_denominator']==source_result['observations_by_intervals']['32'][k]['relative_concentration_error']
        save(out/'results.json',result);np.savez_compressed(out/'common-observations.npz',**common);guard('two-grid-comparison-saved')
        parent_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        aggregate_peak=max(parent_peak,geometry_execution['sampled_main_and_active_workers_peak_RSS_MiB'],geometry_execution['conservative_main_and_workers_peak_RSS_MiB'])
        if aggregate_peak>2048:raise RuntimeError('phase-separated aggregate resource envelope exceeded cap')
        save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','previous_recorded_checks':463,'new_software_checks':3,
          'recorded_checks_total':466,'previous_checks_rerun':0,'new_candidate_graphs':48,'new_candidate_transient_runs':2,
          'new_independent_uniformization_sequences':2,'new_reference_PDE_solves':0,'old_candidate_reruns':0,'new_spatial_candidate_level':256,
          'new_material_definitions':0,'registered_N256_compression_control_executed':True,'comparison_decision':result['comparison_decision'],
          'native_error_required':True,'no_automatic_extra_grid_or_K64':True,'shock_extension_deferred':True,
          'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
          'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':True,
          'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':aggregate_peak,'parent_peak_RSS_MiB':parent_peak,
          'resource_measurement':'maximum of conservative concurrent-main/geometry-worker envelope and sequential state-phase parent peak; geometry workers gone before scalar phase',
          'geometry_workers_selected':geometry_execution['workers_selected'],'resource_caps_passed':True,
          'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads_per_process':4,
          'numerical_norm_estimator_seed':20261101,'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
          'parent_peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();d,ref,runs,paths=inputs(out);index=read(out/'graph-index.json')
    for K in (16,32):
        run,path=native_run(out,K,d,index,replay=True);assert run==read(out/f'k{K}/native-results.json')
        saved=arrays(out/f'k{K}/trajectory-states.npz')
        for key,v in path.items():np.testing.assert_array_equal(v,saved[key])
        runs[256,K]=run;paths[256,K]=path
    r,c=compare(d,ref,runs,paths);assert r==read(out/'results.json');saved=arrays(out/'common-observations.npz')
    for key,v in c.items():np.testing.assert_array_equal(v,saved[key])
    print('Same48 sealed ordinary Voronoi graphs: actualN25616/32 and complete independent paths replayed bitwise; originalN12816/32 reused; all native/common/H controls reproduced. No geometry,old candidate,PDE,normalization or ORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
