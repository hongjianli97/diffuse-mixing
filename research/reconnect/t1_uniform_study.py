"""One registered same-backend T1 static material-mean sine control."""
from pathlib import Path
from fractions import Fraction as F
import argparse,gc,json,math,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save,digest
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from shear_voronoi_fv_study import record,verify_probe_cells
from shear_union64_study import arrays,native_run
from voronoi_time_action import operators
from t1_uniform_core import pure_labels,analytic_data,compare,TIMES,D

SOURCE=ROOT/'runs/reconnect-compression-union64'
PROTOCOL=ROOT/'research/reconnect/t1_uniform_protocol.json'
CONTROL=ROOT/'research/reconnect/T1_UNIFORM_CONTROL.md'


def state_data(out,N):
    d=arrays(out/f'n{N}/material-reference.npz');d.update(read(out/f'n{N}/analytic-parameters.json'))
    return d


def main(output,tests,test_log):
    prior=read(SOURCE/'sha256.json');assert len(prior)==3342
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json');proto=read(PROTOCOL)
    assert not gate['already_executed'] and gate['N_values']==[16,32,64] and gate['ordered_interval_counts']==[4,8,16]
    assert gate['new_candidate_graphs']==3 and gate['main_wall_seconds_cap']==600 and gate['main_peak_RSS_MiB_cap']==1024
    assert read(tests)=={'successful':True,'checks_run':2,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for n,p in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),
      ('control-at-lock.md',CONTROL),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(p,out/n)
    names=['t1_uniform_core.py','t1_uniform_study.py','test_t1_uniform.py','periodic_voronoi_fv.py',
      'voronoi_time_action.py','voronoi_time16_core.py','material_union.py','shear_union64_study.py']
    save(out/'implementation-lock.json',{'files':{n:digest(ROOT/'research/reconnect'/n) for n in names},
      'protocol_sha256':digest(PROTOCOL),'control_note_sha256':digest(CONTROL),'fixed_before_new_scientific_points':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':3342,
      'parent_manifest_sha256':digest(SOURCE/'sha256.json'),'N_values':[16,32,64],'K_values':[4,8,16],
      'new_software_checks':2,'recorded_checks_total':470,'previous_checks_rerun':0,'new_reference_PDE_solves':0,'old_candidate_reruns':0})
    started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>600 or peak>1024:raise RuntimeError(f'registered cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('registered600second main cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    try:
        b={};index={}
        for N in (16,32,64):
            a=pure_labels(N);ids=np.arange(N*N,dtype=np.int64);rho=np.ones(N*N)
            for key,value in [('positions',a),('ids',ids),('rho',rho)]:b[f'n{N}_{key}']=value
            (out/f'n{N}').mkdir()
        np.savez_compressed(out/'builder-inputs.npz',**b)
        save(out/'input-matching.json',{'builder_packet_sha256':digest(out/'builder-inputs.npz'),
          'only_positions_ids_given_point_rho':True,'mass_c0_reference_or_symbol_loaded':False})
        for N in (16,32,64):
            graph=build_periodic_faces(b[f'n{N}_ids'],b[f'n{N}_positions'],b[f'n{N}_rho'],D,(1,1),1,guard)
            probes=verify_probe_cells(graph,[0,N//2,N-1,N*N//2,N*N-1])
            name=f'n{N}/geometry.json';save(out/name,coefficient_packet(graph))
            item={'N':N,'file':name,'sha256':digest(out/name),'geometry_diagnostics':graph.diagnostics,'independent_fixed_probes':probes}
            index[str(N)]=item;del graph;gc.collect();guard(f'sealed-geometry-n{N}')
            per={}
            for K in (4,8,16):
                (out/f'n{N}/k{K}').mkdir()
                for step in range(K):per[f'k{K}-step{step:02d}']={**item,'file':'geometry.json','K':K,'step':step,'midpoint_time_saved':float((2*step+1)*.05/(2*K))}
            save(out/f'n{N}/graph-index.json',per)
        save(out/'graph-index.json',index)
        save(out/'coefficient-seal.json',{'graphs':3,'graph_index_sha256':digest(out/'graph-index.json'),
          'builder_packet_sha256':digest(out/'builder-inputs.npz'),'all3_coefficients_sealed_before_mass_c0_reference_symbol':True,
          'builder_received_mass_c0_reference_true_faces_symbol_or_F_C':False})
        del b;gc.collect();guard('all3-coefficients-sealed-before-state')
        data={};runs={};paths={}
        for N in (16,32,64):
            folder=out/f'n{N}';d=analytic_data(N);packet=read(folder/'geometry.json')
            op=operators(d['mass'],np.array(packet['edges'],dtype=np.int64),map(F,packet['conductances_exact']));del packet
            root=np.sqrt(d['mass']);phi=d['c0']-.5
            norm=lambda v:float(np.sqrt(d['mass']@(v*v)))
            residue=norm(op.symmetric_generator@(root*phi)/root+d['discrete_lambda']*phi)
            constant=norm(.5*(op.symmetric_generator@root)/root)
            assert residue+constant<1e-10*max(1.,d['discrete_lambda']*norm(phi))
            d.update(eigenmode_residue_mass_norm=residue,constant_residue_mass_norm=constant)
            array={k:v for k,v in d.items() if isinstance(v,np.ndarray)};parameters={k:v for k,v in d.items() if not isinstance(v,np.ndarray)}
            np.savez_compressed(folder/'material-reference.npz',**array);save(folder/'analytic-parameters.json',parameters)
            data[N]=d;del op;gc.collect();per=read(folder/'graph-index.json')
            for K in (4,8,16):
                r,p=native_run(folder,K,d,per,guard);runs[N,K]=r;paths[N,K]=p
                save(folder/f'k{K}/native-results.json',r);np.savez_compressed(folder/f'k{K}/trajectory-states.npz',**p)
        result,common=compare(data,runs,paths);save(out/'results.json',result);np.savez_compressed(out/'common-observations.npz',**common)
        guard('T1-comparison-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','previous_recorded_checks':468,
          'new_software_checks':2,'recorded_checks_total':470,'previous_checks_rerun':0,'new_candidate_graphs':3,
          'new_candidate_transient_runs':9,'new_independent_uniformization_sequences':9,'new_reference_PDE_solves':0,'old_candidate_reruns':0,
          'registered_T1_control_executed':True,'T1_registered_case_passed':result['T1_registered_case_passed'],
          'comparison_decision':result['comparison_decision'],'general_temporal_order_claimed':False,
          'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
          'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':True,
          'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'resource_caps_passed':True,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
          'threads':4,'numerical_norm_estimator_seed':20261101,'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps({'decision':result['comparison_decision'],'final_native_errors':{N:v[-1]['native_error_native_denominator'] for N,v in result['per_grid_observations'].items()}},indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
          'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();data={};runs={};paths={}
    for N in (16,32,64):
        data[N]=state_data(out,N);folder=out/f'n{N}';index=read(folder/'graph-index.json')
        for K in (4,8,16):
            r,p=native_run(folder,K,data[N],index,replay=True);assert r==read(folder/f'k{K}/native-results.json')
            saved=arrays(folder/f'k{K}/trajectory-states.npz')
            for key,v in p.items():np.testing.assert_array_equal(v,saved[key])
            runs[N,K]=r;paths[N,K]=p
    r,c=compare(data,runs,paths);assert r==read(out/'results.json');saved=arrays(out/'common-observations.npz')
    for key,v in c.items():np.testing.assert_array_equal(v,saved[key])
    print('Three sealed static graphs: nine actual and nine complete independent paths replayed bitwise; all native/common/H,static time and independent symbol comparisons reproduced. No geometry,PDE,old candidate,normalization or ORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
