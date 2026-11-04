"""Registered T2 rigid translation; original material data and static path reused."""
from pathlib import Path
from fractions import Fraction as F
import argparse,gc,json,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save,digest
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from shear_voronoi_fv_study import verify_probe_cells
from shear_union64_study import arrays,native_run
from t2_translation_core import coefficient_perturbation,compare

SOURCE=ROOT/'runs/reconnect-t1-uniform'
PROTOCOL=ROOT/'research/reconnect/t2_translation_protocol.json'
CONTROL=ROOT/'research/reconnect/T2_TRANSLATION_CONTROL.md'
COPIES={'original-material-reference.npz':SOURCE/'n32/material-reference.npz',
 'original-analytic-parameters.json':SOURCE/'n32/analytic-parameters.json','static-geometry.json':SOURCE/'n32/geometry.json',
 'static16-states.npz':SOURCE/'n32/k16/trajectory-states.npz','static16-results.json':SOURCE/'n32/k16/native-results.json',
 'original-T1-results.json':SOURCE/'results.json'}


def inputs(out):
    d=arrays(out/'original-material-reference.npz');d.update(read(out/'original-analytic-parameters.json'))
    return d,read(out/'static16-results.json'),arrays(out/'static16-states.npz')


def perturbations(out,data,index):
    old=read(out/'static-geometry.json');result=[]
    for step in range(16):
        p=read(out/index[f'k16-step{step:02d}']['file'])
        result.append(coefficient_perturbation(data['mass'],old,p));del p
    return result


def main(output,tests,test_log):
    prior=read(SOURCE/'sha256.json');assert len(prior)==3592
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json');proto=read(PROTOCOL)
    assert not gate['already_executed'] and gate['N']==32 and gate['K']==16 and gate['new_midpoint_graphs']==16
    assert gate['velocity']==[.2,-.1] and gate['main_wall_seconds_cap']==300 and gate['main_peak_RSS_MiB_cap']==1024
    assert read(tests)=={'successful':True,'checks_run':2,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for n,p in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),('control-at-lock.md',CONTROL),
      ('tests.json',Path(tests)),('tests.log',Path(test_log)),('tests-first-attempt.json',Path('/tmp/ltn-t2-translation-tests.json')),
      ('tests-first-attempt.log',Path('/tmp/ltn-t2-translation-tests.log'))]:shutil.copyfile(p,out/n)
    names=['t2_translation_core.py','t2_translation_study.py','test_t2_translation.py','periodic_voronoi_fv.py','voronoi_time_action.py',
      'voronoi_time16_core.py','shear_union64_study.py','t1_uniform_core.py']
    save(out/'implementation-lock.json',{'files':{n:digest(ROOT/'research/reconnect'/n) for n in names},
      'protocol_sha256':digest(PROTOCOL),'control_note_sha256':digest(CONTROL),'fixed_before_new_scientific_points':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':3592,
      'parent_manifest_sha256':digest(SOURCE/'sha256.json'),'new_software_checks':2,'recorded_checks_total':472,'previous_checks_rerun':0,
      'N':32,'K':16,'D':.02,'T_exact':'1/20','velocity':[.2,-.1],'new_reference_PDE_solves':0,'old_candidate_reruns':0})
    started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>300 or peak>1024:raise RuntimeError(f'registered cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('registered300second main cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(300)
    try:
        pure=SOURCE/'builder-inputs.npz';assert digest(pure)==gate['pure_source_sha256'];shutil.copyfile(pure,out/'original-pure-inputs.npz')
        z=arrays(out/'original-pure-inputs.npz');a=z['n32_positions'];ids=z['n32_ids'];rho=z['n32_rho']
        assert a.shape==(1024,2) and np.array_equal(ids,np.arange(1024)) and np.array_equal(rho,np.ones(1024))
        times=np.array([float(F(j,640)) for j in range(33)]);v=np.array([.2,-.1])
        b={'ids':ids,'labels':a,'time':times,'positions':np.mod(a[None,:,:]+times[:,None,None]*v[None,None,:],1.),
          'rho':np.broadcast_to(rho,(33,1024)).copy(),'midpoint_time':times[1::2],'velocity':v}
        np.savez_compressed(out/'builder-inputs.npz',**b)
        save(out/'input-matching.json',{'original_pure_source_sha256':digest(pure),'builder_packet_sha256':digest(out/'builder-inputs.npz'),
          'only_positions_ids_given_point_density_used':True,'mass_c0_reference_static_geometry_or_state_loaded':False})
        (out/'k16').mkdir();index={};del z;gc.collect()
        for step in range(16):
            frame=2*step+1
            graph=build_periodic_faces(ids,b['positions'][frame],b['rho'][frame],.02,(1,1),1,guard)
            probes=verify_probe_cells(graph,proto['fixed_geometry_probe_ids']);name=f'k16/geometry-step{step:02d}.json'
            save(out/name,coefficient_packet(graph));index[f'k16-step{step:02d}']={'N':32,'K':16,'step':step,'frame':frame,
              'file':name,'sha256':digest(out/name),'midpoint_time_saved':float(times[frame]),'geometry_diagnostics':graph.diagnostics,'independent_fixed_probes':probes}
            del graph;gc.collect();guard(f'sealed-geometry-k16-step{step:02d}')
        save(out/'graph-index.json',index)
        save(out/'coefficient-seal.json',{'graphs':16,'graph_index_sha256':digest(out/'graph-index.json'),'builder_packet_sha256':digest(out/'builder-inputs.npz'),
          'all16_coefficients_sealed_before_mass_c0_reference_static_geometry_or_state':True,'builder_received_true_material_faces_F_C_or_scalar':False})
        del b;gc.collect();guard('all16-coefficients-sealed-before-state')
        sources=[]
        for name,p in COPIES.items():
            rel=str(p.relative_to(ROOT));assert digest(p)==prior[rel];shutil.copyfile(p,out/name);sources.append({'copy':name,'source':rel,'sha256':digest(p)})
        save(out/'source-copies.json',sources);data,old_run,old_path=inputs(out)
        p=perturbations(out,data,index);save(out/'coefficient-perturbations.json',p)
        run,path=native_run(out,16,data,index,guard);save(out/'k16/native-results.json',run);np.savez_compressed(out/'k16/trajectory-states.npz',**path)
        result=compare(data,old_run,old_path,run,path,p);save(out/'results.json',result);guard('T2-translation-comparison-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','previous_recorded_checks':470,
          'new_software_checks':2,'recorded_checks_total':472,'previous_checks_rerun':0,'registered_T2_translation_executed':True,
          'T2_registered_translation_case_passed':result['T2_registered_translation_case_passed'],'comparison_decision':result['comparison_decision'],
          'new_candidate_graphs':16,'new_candidate_transient_runs':1,'new_independent_uniformization_sequences':1,'old_candidate_reruns':0,'new_reference_PDE_solves':0,
          'rigid_rotation_tested':False,'zero_diffusion_arbitrary_motion_tested':False,'kappa_computed':False,'general_T2_admitted':False,
          'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
          'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':False,'tests_failed_attempts_preserved':1,'main_first_attempt_passed':True,
          'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'resource_caps_passed':True,
          'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4,'numerical_norm_estimator_seed':20261101,
          'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps({'decision':result['comparison_decision'],'observations':result['observations']},indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
          'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();data,old_run,old_path=inputs(out);index=read(out/'graph-index.json')
    p=perturbations(out,data,index);assert p==read(out/'coefficient-perturbations.json')
    r,path=native_run(out,16,data,index,replay=True);assert r==read(out/'k16/native-results.json')
    saved=arrays(out/'k16/trajectory-states.npz')
    for key,v in path.items():np.testing.assert_array_equal(v,saved[key])
    result=compare(data,old_run,old_path,r,path,p);assert result==read(out/'results.json')
    print('Sixteen sealed moving graphs: actual and complete independent paths replayed bitwise; staticT1 path reused; exact coefficient/Duhamel and state/H comparisons reproduced. No geometry,PDE,old candidate,normalization or ORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
