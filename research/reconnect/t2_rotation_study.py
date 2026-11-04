"""One preregistered positive-D continuous rotation on the corresponding rotating torus."""
from pathlib import Path
from fractions import Fraction as F
import argparse,gc,json,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save,digest
from oriented_periodic_boundary import OrientedPeriodicRectangle,build_oriented_periodic_faces
from rotation_world_trajectory import prescribed_world_trajectory
from shear_union64_study import arrays,native_run
from t2_rotation_core import compare

SOURCE=ROOT/'runs/reconnect-t2-rotation-domain'
PROTOCOL=ROOT/'research/reconnect/t2_rotation_protocol.json'
CONTROL=ROOT/'research/reconnect/T2_ROTATION_CONTROL.md'


def inputs(out):
    d=arrays(out/'original-material-reference.npz');d.update(read(out/'original-analytic-parameters.json'))
    return d,read(out/'static16-results.json'),arrays(out/'static16-states.npz'),read(out/'static-geometry.json')


def evidence(out,index):
    packets=[read(out/index[f'k16-step{i:02d}']['file']) for i in range(16)]
    actions=[arrays(out/f'k16/action-step{i:02d}.npz') for i in range(16)]
    static=[arrays(out/f'static-actions/action-step{i:02d}.npz') for i in range(16)]
    return packets,actions,static


def main(output,tests,test_log):
    prior=read(SOURCE/'sha256.json');assert len(prior)==3849
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json');proto=read(PROTOCOL)
    assert not gate['already_executed'] and gate['N']==32 and gate['K']==16 and gate['new_candidate_graphs']==16
    assert gate['planned_new_software_checks']==3 and gate['main_wall_seconds_cap']==300 and gate['main_peak_RSS_MiB_cap']==1024
    assert read(tests)=={'successful':True,'checks_run':3,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,path in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),('control-at-lock.md',CONTROL),
      ('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(path,out/name)
    locked=['oriented_periodic_boundary.py','audit_oriented_world_geometry.py','rotation_world_trajectory.py','test_oriented_periodic_boundary.py',
      't2_rotation_core.py','t2_rotation_study.py','periodic_voronoi_fv.py','voronoi_time_action.py','voronoi_time16_core.py','shear_union64_study.py']
    save(out/'implementation-lock.json',{'files':{n:digest(ROOT/'research/reconnect'/n) for n in locked},'protocol_sha256':digest(PROTOCOL),
      'control_note_sha256':digest(CONTROL),'fixed_before_new_scientific_points':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':3849,
      'parent_manifest_sha256':digest(SOURCE/'sha256.json'),'new_software_checks':3,'recorded_checks_total':477,'previous_checks_rerun':0,
      'N':32,'K':16,'D':.02,'T_exact':'1/20','new_reference_PDE_solves':0,'old_candidate_reruns':0,'new_material_definitions':0})
    started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>300 or peak>1024:raise RuntimeError(f'registered cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('registered300second main cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(300)
    try:
        source=gate['sources']['pure_points'];assert digest(ROOT/source['path'])==source['sha256'];shutil.copyfile(ROOT/source['path'],out/'original-pure-inputs.npz')
        z=arrays(out/'original-pure-inputs.npz');a=z['n32_positions'];ids=z['n32_ids'];rho=z['n32_rho'];assert np.array_equal(rho,np.ones(1024))
        world=prescribed_world_trajectory(a,ids,rho);save(out/'world-trajectory.json',world)
        save(out/'input-matching.json',{'source_pure_sha256':source['sha256'],'world_trajectory_sha256':digest(out/'world-trajectory.json'),
          'only_pure_source_points_IDs_oracle_density_and_given_boundary_used':True,'mass_c0_reference_static_geometry_or_old_states_loaded':False})
        (out/'k16').mkdir();index={};del z;gc.collect()
        for step in range(16):
            frame=world['frames'][2*step+1];boundary=OrientedPeriodicRectangle.from_packet(frame['boundary'])
            local,wp,cert=build_oriented_periodic_faces(ids,frame['world_positions_exact'],frame['rho'],.02,boundary,guard)
            local_name=f'k16/geometry-step{step:02d}.json';world_name=f'k16/world-geometry-step{step:02d}.json'
            save(out/local_name,local);save(out/world_name,wp);save(out/f'k16/isometry-step{step:02d}.json',cert)
            index[f'k16-step{step:02d}']={'N':32,'K':16,'step':step,'frame':2*step+1,'file':local_name,'sha256':digest(out/local_name),
              'world_file':world_name,'world_sha256':digest(out/world_name),'isometry_file':f'k16/isometry-step{step:02d}.json',
              'midpoint_time_saved':float(F(frame['time_exact'])),'geometry_diagnostics':local['geometry_diagnostics']}
            del local,wp,cert;gc.collect();guard(f'sealed-world-geometry-step{step:02d}')
        save(out/'graph-index.json',index);save(out/'coefficient-seal.json',{'graphs':16,'graph_index_sha256':digest(out/'graph-index.json'),
          'world_trajectory_sha256':digest(out/'world-trajectory.json'),'all16_fresh_local_and_world_graphs_sealed_before_material_or_static_data':True,
          'adapter_received_source_local_labels_mass_c0_reference_true_internal_faces_or_F_C':False})
        del world,a;gc.collect();guard('all16-local-and-world-graphs-sealed-before-state')
        mapping={'original-material-reference.npz':'material_mass_c0_reference','original-analytic-parameters.json':'static_analytic_parameters',
          'static-geometry.json':'static_geometry','static16-states.npz':'static16_states','static16-results.json':'static16_results'}
        copies=[]
        for name,key in mapping.items():
            item=gate['sources'][key];assert digest(ROOT/item['path'])==item['sha256'];shutil.copyfile(ROOT/item['path'],out/name)
            copies.append({'copy':name,'source':item['path'],'sha256':item['sha256']})
        (out/'static-actions').mkdir()
        for i in range(16):
            item=gate['static_action_sources'][f'step{i:02d}'];assert digest(ROOT/item['path'])==item['sha256']
            name=f'static-actions/action-step{i:02d}.npz';shutil.copyfile(ROOT/item['path'],out/name);copies.append({'copy':name,'source':item['path'],'sha256':item['sha256']})
        save(out/'source-copies.json',copies);d,old_run,old_path,old_packet=inputs(out)
        run,path=native_run(out,16,d,index,guard);save(out/'k16/native-results.json',run);np.savez_compressed(out/'k16/trajectory-states.npz',**path)
        packets,actions,static=evidence(out,index);result=compare(d,old_run,old_path,run,path,packets,old_packet,actions,static)
        save(out/'results.json',result);guard('rotation-covariance-comparison-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','previous_recorded_checks':474,'new_software_checks':3,
          'recorded_checks_total':477,'previous_checks_rerun':0,'registered_rotation_case_executed':True,'exact_boundary_adapter_implemented':True,
          'original_geometry_resistance_time_backends_changed':False,'T2_registered_rotation_case_passed':result['T2_registered_rotation_case_passed'],
          'comparison_decision':result['comparison_decision'],'new_candidate_graphs':16,'new_candidate_transient_runs':1,'new_independent_uniformization_sequences':1,
          'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_material_definitions':0,'arbitrary_float_world_inputs_or_nonorthogonal_lattices_admitted':False,
          'general_T2_admitted':False,'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete',
          'fluid_orc_enabled':False,'kappa_computed':False,'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':True,
          'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'resource_caps_passed':True,
          'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4,'numerical_norm_estimator_seed':20261101,
          'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps({'decision':result['comparison_decision'],'path_arrays_bitwise_static':result['path_arrays_bitwise_static'],'final':result['observations'][-1]},indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
          'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();d,old_run,old_path,old_packet=inputs(out);index=read(out/'graph-index.json')
    run,path=native_run(out,16,d,index,replay=True);assert run==read(out/'k16/native-results.json');saved=arrays(out/'k16/trajectory-states.npz')
    from t2_d0_core import bitwise_same
    for k,v in path.items():assert bitwise_same(v,saved[k]),k
    packets,actions,static=evidence(out,index);result=compare(d,old_run,old_path,run,path,packets,old_packet,actions,static);assert result==read(out/'results.json')
    print('Sixteen sealed exact-boundary rotation graphs: actual/complete independent paths andallhalfstates replayed bitwise; originalstaticpaths reused; exactIDg,rawCSR/weights andmaterialmixing comparisons reproduced. No geometry,PDE,oldcandidate,normalization orORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
