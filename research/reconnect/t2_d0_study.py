"""One registered D=0 given nonrigid material-transport control."""
from pathlib import Path
from fractions import Fraction as F
import argparse,gc,json,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save,digest
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from shear_voronoi_fv_study import verify_probe_cells
from shear_union64_study import arrays,native_run
from compression_voronoi_core import sample_points
from t2_d0_core import evaluate,bitwise_same

SOURCE=ROOT/'runs/reconnect-t2-translation'
PROTOCOL=ROOT/'research/reconnect/t2_d0_protocol.json'
CONTROL=ROOT/'research/reconnect/T2_D0_CONTROL.md'


def inputs(out):
    d=arrays(out/'material-inputs.npz')
    with np.load(out/'original-material-reference.npz',allow_pickle=False) as z:
        for key in ('mass','c0'):assert bitwise_same(d[key],z[key]),key
    return d


def cached_evidence(out):
    index=read(out/'graph-index.json');packets=[];actions=[]
    for step in range(16):
        item=index[f'k16-step{step:02d}'];assert digest(out/item['file'])==item['sha256']
        packets.append(read(out/item['file']));actions.append(arrays(out/f'k16/action-step{step:02d}.npz'))
    return packets,actions


def main(output,tests,test_log):
    prior=read(SOURCE/'sha256.json');assert len(prior)==3711
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json');proto=read(PROTOCOL)
    assert not gate['already_executed'] and gate['N']==32 and gate['K']==16 and gate['D']==0 and gate['new_candidate_graphs']==16
    assert gate['main_wall_seconds_cap']==300 and gate['main_peak_RSS_MiB_cap']==1024
    assert read(tests)=={'successful':True,'checks_run':2,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for n,p in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),('control-at-lock.md',CONTROL),
      ('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(p,out/n)
    names=['t2_d0_core.py','t2_d0_study.py','test_t2_d0.py','audit_d0_geometry_core.py','periodic_voronoi_fv.py','voronoi_time_action.py',
      'voronoi_time16_core.py','shear_union64_study.py','compression_voronoi_core.py']
    save(out/'implementation-lock.json',{'files':{n:digest(ROOT/'research/reconnect'/n) for n in names},'protocol_sha256':digest(PROTOCOL),
      'control_note_sha256':digest(CONTROL),'fixed_before_new_scientific_points':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':3711,
      'parent_manifest_sha256':digest(SOURCE/'sha256.json'),'new_software_checks':2,'recorded_checks_total':474,'previous_checks_rerun':0,
      'N':32,'K':16,'D':0.,'T_exact':'1/20','new_reference_PDE_solves':0,'old_candidate_reruns':0,'new_material_definitions':0})
    started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>300 or peak>1024:raise RuntimeError(f'registered cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('registered300second main cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(300)
    try:
        pure=SOURCE/'original-pure-inputs.npz';assert digest(pure)==gate['pure_source_sha256'];shutil.copyfile(pure,out/'original-pure-inputs.npz')
        z=arrays(out/'original-pure-inputs.npz');a=z['n32_positions'];ids=z['n32_ids'];rho0=z['n32_rho']
        assert a.shape==(1024,2) and np.array_equal(ids,np.arange(1024)) and np.array_equal(rho0,np.ones(1024))
        t=np.array([float(F(j,640)) for j in range(33)]);b=sample_points(a,ids,rho0,t)
        b['labels']=a;b['rho0']=rho0;b['midpoint_time']=t[1::2]
        np.savez_compressed(out/'builder-inputs.npz',**b)
        save(out/'input-matching.json',{'original_pure_source_sha256':digest(pure),'builder_packet_sha256':digest(out/'builder-inputs.npz'),
          'same_original_sampler_unchanged':True,'mass_c0_positiveD_reference_or_old_states_loaded':False,
          'given_point_density_minimum':float(b['rho'].min()),'given_point_density_maximum':float(b['rho'].max())})
        (out/'k16').mkdir();index={};del z;gc.collect()
        for step in range(16):
            frame=2*step+1;graph=build_periodic_faces(ids,b['positions'][frame],b['rho'][frame],0.,(1,1),1,guard)
            probes=verify_probe_cells(graph,proto['fixed_geometry_probe_ids']);name=f'k16/geometry-step{step:02d}.json'
            save(out/name,coefficient_packet(graph));index[f'k16-step{step:02d}']={'N':32,'K':16,'step':step,'frame':frame,
              'file':name,'sha256':digest(out/name),'midpoint_time_saved':float(t[frame]),'geometry_diagnostics':graph.diagnostics,'independent_fixed_probes':probes}
            del graph;gc.collect();guard(f'sealed-geometry-k16-step{step:02d}')
        save(out/'graph-index.json',index)
        save(out/'coefficient-seal.json',{'graphs':16,'graph_index_sha256':digest(out/'graph-index.json'),'builder_packet_sha256':digest(out/'builder-inputs.npz'),
          'all16_coefficients_sealed_before_mass_c0_or_old_states':True,'builder_received_true_material_faces_F_C_or_scalar':False})
        del b;gc.collect();guard('all16-coefficients-sealed-before-state')
        material=SOURCE/'original-material-reference.npz';assert digest(material)==gate['material_source_sha256']
        shutil.copyfile(material,out/'original-material-reference.npz')
        with np.load(out/'original-material-reference.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in ('mass','c0')}
        d['ids']=ids;np.savez_compressed(out/'material-inputs.npz',**d)
        np.savez_compressed(out/'identity-reference.npz',time=np.array([float(F(j,320)) for j in range(17)]),states=np.broadcast_to(d['c0'],(17,1024)).copy())
        save(out/'source-copies.json',[{'copy':'original-pure-inputs.npz','source':str(pure.relative_to(ROOT)),'sha256':digest(pure),'phase':'before-coefficients,pure-only'},
          {'copy':'original-material-reference.npz','source':str(material.relative_to(ROOT)),'sha256':digest(material),'phase':'after-all16-sealed,mass-c0-only-parsed'}])
        run,path=native_run(out,16,d,index,guard);save(out/'k16/native-results.json',run);np.savez_compressed(out/'k16/trajectory-states.npz',**path)
        packets,actions=cached_evidence(out);result=evaluate(d,run,path,packets,actions);save(out/'results.json',result);guard('D0-identity-comparison-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','previous_recorded_checks':472,'new_software_checks':2,
          'recorded_checks_total':474,'previous_checks_rerun':0,'registered_T2_D0_executed':True,'T2_registered_D0_case_passed':result['T2_registered_D0_case_passed'],
          'comparison_decision':result['comparison_decision'],'new_candidate_graphs':16,'new_candidate_transient_runs':1,
          'new_independent_uniformization_sequences':1,'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_material_definitions':0,
          'rigid_rotation_tested':False,'kappa_computed':False,'general_T2_admitted':False,'general_smooth_flow_admitted':False,
          'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':True,
          'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'resource_caps_passed':True,
          'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4,'numerical_norm_estimator_seed':20261101,
          'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps({'decision':result['comparison_decision'],'bitwise_state_checks':result['state_arrays_bitwise_initial'],'final':result['observations'][-1]},indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
          'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();d=inputs(out);index=read(out/'graph-index.json')
    run,path=native_run(out,16,d,index,replay=True);assert run==read(out/'k16/native-results.json')
    saved=arrays(out/'k16/trajectory-states.npz')
    for key,v in path.items():assert bitwise_same(v,saved[key]),key
    packets,actions=cached_evidence(out);result=evaluate(d,run,path,packets,actions);assert result==read(out/'results.json')
    print('Sixteen sealed D0 nonrigid graphs: original zero-action branch and complete independent path replayed bitwise; all endpoints/halfstates equal originalc0,all zero-generator/identity-kernel andrawmass/variance/H checks reproduced. No geometry,PDE,old candidate,normalization or ORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
