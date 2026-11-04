"""Registered compression N64 physical unions; reuse saved N128/N25616/32 paths."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,math,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save,digest
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from shear_voronoi_fv_study import record,verify_probe_cells
from voronoi_time_action import operators,metrics
from voronoi_time16_core import ordered_path
from compression_voronoi_core import sample_points
from compression_union64_core import native_data,compare
from material_union import group_sum,block_index,union_labels,oracle_density,define_union,observe_path

SOURCE=ROOT/'runs/reconnect-compression-spatial-fv'
PROTOCOL=ROOT/'research/reconnect/compression_union64_protocol.json'
CONTROL=ROOT/'research/reconnect/COMPRESSION_UNION64_CONTROL.md'
SEED=20261101
COPIES={'original-fine.npz':SOURCE/'original-fine.npz',
 'original-parent-reference.npz':SOURCE/'original-parent-reference.npz',
 'old-results.json':SOURCE/'results.json'}
for N in (128,256):
    for K in (16,32):
        for kind,suffix in [('states','npz'),('results','json')]:
            origin=SOURCE/f'coarse{K}-{kind}.{suffix}' if N==128 else SOURCE/f'k{K}'/('trajectory-states.npz' if kind=='states' else 'native-results.json')
            COPIES[f'old-n{N}-k{K}-{kind}.{suffix}']=origin



def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def load_old(out):
    runs={};paths={}
    for N in (128,256):
        for K in (16,32):
            paths[N,K]=arrays(out/f'old-n{N}-k{K}-states.npz')
            runs[N,K]=read(out/f'old-n{N}-k{K}-results.json')
    return runs,paths


from shear_union64_study import native_run


def main(output,tests,test_log):
    prior=read(SOURCE/'sha256.json');assert len(prior)==3090
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json');protocol=read(PROTOCOL)
    assert not gate['already_executed'] and gate['new_midpoint_graphs']==48 and gate['new_ordered_interval_counts']==[16,32]
    assert gate['main_wall_seconds_cap']==700 and gate['main_peak_RSS_MiB_cap']==1024
    assert read(tests)=={'successful':True,'checks_run':2,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for n,p in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),
       ('control-at-lock.md',CONTROL),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(p,out/n)
    locked=['periodic_voronoi_fv.py','voronoi_time_action.py','voronoi_time16_core.py','shear_probe.py',
      'material_union.py','compression_voronoi_core.py','compression_union64_core.py','test_compression_union64.py','compression_union64_study.py','shear_union64_study.py']
    save(out/'implementation-lock.json',{'files':{n:digest(ROOT/'research/reconnect'/n) for n in locked},
      'protocol_sha256':digest(PROTOCOL),'control_note_sha256':digest(CONTROL),'fixed_before_any_new_input_or_geometry':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':3090,
      'parent_manifest_sha256':digest(SOURCE/'sha256.json'),'new_software_checks':2,'recorded_checks_total':468,
      'previous_checks_rerun':0,'new_reference_PDE_solves':0,'new_spatial_levels':[64]})
    started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>700 or peak>1024:raise RuntimeError(f'registered resource cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k%4096==0 or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('registered700second main cap reached')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(700)
    try:
        source={};checks={}
        for N,name in [(128,'coarse-trajectory-source.npz'),(256,'trajectory-source.npz')]:
            z=arrays(SOURCE/name)
            for key in ('labels','ids','rho0_point'):source[f'n{N}_{key}']=z[key]
        del z
        for N in (128,256):
            r=oracle_density(source[f'n{N}_labels']);assert np.array_equal(r,source[f'n{N}_rho0_point'])
            checks[str(N)]={'same_given_oracle_density_bitwise_verified':True,'rho_record':record(r)}
        a=union_labels(source['n128_labels']);ids=np.arange(64**2,dtype=np.int64);rho=oracle_density(a)
        membership={'labels':a,'ids':ids,'rho0_point':rho,'n128_to_n64':block_index(64,2),'n256_to_n64':block_index(64,4)}
        for k,v in source.items():membership[k]=v
        np.savez_compressed(out/'trajectory-source.npz',**membership)
        b={'ids':ids}
        for K in (16,32):
            p=sample_points(a,ids,rho,np.linspace(0.,.05,2*K+1))
            for key in ('time','positions','rho'):b[f'k{K}_{key}']=p[key]
            b[f'k{K}_boundary_time']=p['time'][::2];b[f'k{K}_midpoint_time']=p['time'][1::2]
            (out/f'k{K}').mkdir()
        np.savez_compressed(out/'builder-inputs.npz',**b)
        save(out/'input-matching.json',{'pure_sources':{name:digest(SOURCE/name) for name in ('coarse-trajectory-source.npz','trajectory-source.npz')},'old_density_checks':checks,
          'builder_packet_sha256':digest(out/'builder-inputs.npz'),'actual_mass_c0_reference_loaded':False,
          'new_label_center_record':record(a),'new_point_density_record':record(rho)})
        del source,membership;gc.collect();index={};guard('pure-union-points-sealed')
        for K in (16,32):
            for step in range(K):
                frame=2*step+1
                graph=build_periodic_faces(ids,b[f'k{K}_positions'][frame],b[f'k{K}_rho'][frame],.02,(1,1),1,guard)
                probes=verify_probe_cells(graph,protocol['fixed_geometry_probe_ids']);packet=coefficient_packet(graph)
                name=f'k{K}/geometry-step{step:02d}.json';save(out/name,packet)
                index[f'k{K}-step{step:02d}']={'N':64,'K':K,'step':step,'frame':frame,'file':name,'sha256':digest(out/name),
                  'midpoint_time_saved':float(b[f'k{K}_midpoint_time'][step]),'geometry_diagnostics':graph.diagnostics,'independent_fixed_probes':probes}
                del graph,packet;gc.collect();guard(f'sealed-geometry-k{K}-step{step:02d}')
        save(out/'graph-index.json',index);save(out/'coefficient-seal.json',{'graphs':48,'graph_index_sha256':digest(out/'graph-index.json'),
          'builder_packet_sha256':digest(out/'builder-inputs.npz'),'all48_coefficients_sealed_before_mass_c0_reference_loading':True,
          'builder_received_mass_c0_reference_true_faces_or_F_C':False})
        del b;gc.collect();guard('all48-coefficients-sealed-before-state')
        sources=[]
        for name,p in COPIES.items():
            rel=str(p.relative_to(ROOT));assert digest(p)==prior[rel],rel;shutil.copyfile(p,out/name)
            sources.append({'copy':name,'source':rel,'sha256':digest(p)})
        save(out/'source-copies.json',sources)
        d=arrays(out/'original-fine.npz');ref=arrays(out/'original-parent-reference.npz');data=native_data(d,ref)
        u=data[64];np.savez_compressed(out/'union-definition.npz',**u,
          n128_mass_sum=group_sum(d['parent_mass'],64,2),n256_mass_sum=group_sum(d['mass'],64,4))
        runs,paths=load_old(out)
        for K in (16,32):
            r,p=native_run(out,K,u,index,guard);runs[64,K]=r;paths[64,K]=p
            save(out/f'k{K}/native-results.json',r);np.savez_compressed(out/f'k{K}/trajectory-states.npz',**p)
        result,common=compare(data,runs,paths);save(out/'results.json',result);np.savez_compressed(out/'common-observations.npz',**common)
        guard('union64-comparison-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','previous_recorded_checks':466,'new_software_checks':2,
          'recorded_checks_total':468,'previous_checks_rerun':0,'new_candidate_graphs':48,'new_candidate_transient_runs':2,
          'new_independent_uniformization_sequences':2,'new_reference_PDE_solves':0,'old_candidate_reruns':0,'new_spatial_levels':[64],
          'registered_N64_union_control_executed':True,'source_gate_already_executed':True,'comparison_decision':result['comparison_decision'],
          'no_automatic_K64_or_grid_extension':True,'shock_extension_deferred':True,'general_smooth_flow_admitted':False,
          'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
          'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':True,
          'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'resource_caps_passed':True,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
          'threads':4,'numerical_norm_estimator_seed':SEED,'network_used':False,'environment_configuration_changed':False,
          'committed_or_pushed':False,'published':False})
        print(json.dumps({'comparison_decision':result['comparison_decision'],'final_grid_observations':{N:v[-1] for N,v in result['per_grid_observations'].items()}},indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
          'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();data=native_data(arrays(out/'original-fine.npz'),arrays(out/'original-parent-reference.npz'));runs,paths=load_old(out);index=read(out/'graph-index.json')
    saved=arrays(out/'union-definition.npz')
    for k,v in data[64].items():assert np.array_equal(v,saved[k]),k
    for K in (16,32):
        r,p=native_run(out,K,data[64],index,replay=True);assert r==read(out/f'k{K}/native-results.json')
        old=arrays(out/f'k{K}/trajectory-states.npz')
        for k,v in p.items():assert np.array_equal(v,old[k]),k
        runs[64,K]=r;paths[64,K]=p
    r,c=compare(data,runs,paths);assert r==read(out/'results.json')
    saved=arrays(out/'common-observations.npz')
    for k,v in c.items():assert np.array_equal(v,saved[k]),k
    print('Sealed48 graphs: actual N64 K16/32 plus independent complete sequences replayed bitwise; original128/256 paths reused; all common/native/H comparisons reproduced. No geometry/PDE/old candidate rerun, no normalization or ORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
