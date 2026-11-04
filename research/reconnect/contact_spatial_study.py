"""One registered static-contact space experiment; add N16/N64, reuse N32 only."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save as raw_save,digest
from affine_shear_completion import plain_json
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from audit_shear_voronoi_time import audit_geometry
from contact_oracle_core import pure_contact,face_control,derive_material,cartesian_control
from contact_oracle_reporting import evaluate_saved
from contact_spatial_core import compare
from shear_union64_study import arrays,native_run
from t2_d0_core import bitwise_same

SOURCE=ROOT/'runs/reconnect-contact-oracle';NEW=(16,64);LEVELS=(16,32,64)
PROTOCOL=ROOT/'research/reconnect/contact_spatial_protocol.json'


def save(path,obj):raw_save(path,plain_json(obj))


def load_all(out):
    data={};meta={};runs={};paths={};controls={};cm={}
    for N in LEVELS:
        folder=out/f'n{N}';data[N]=arrays(folder/'material-reference.npz');meta[N]=read(folder/'reference-parameters.json')
        runs[N]=read(folder/'k16/native-results.json');paths[N]=arrays(folder/'k16/trajectory-states.npz')
        controls[N]=arrays(folder/'Cartesian-control.npz');cm[N]=read(folder/'Cartesian-parameters.json')
    return data,meta,runs,paths,controls,cm


def main(output,tests,test_log):
    pre=time.monotonic();prior=read(SOURCE/'sha256.json');assert len(prior)==6459
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json');assert not gate['already_executed'] and gate['new_N_values']==list(NEW) and gate['new_candidate_graphs']==2
    assert read(tests)=={'successful':True,'checks_run':3,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,path in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(path,out/name)
    names=['contact_spatial_core.py','contact_spatial_study.py','test_contact_spatial.py','contact_oracle_core.py','contact_oracle_reporting.py',
      'periodic_voronoi_fv.py','audit_shear_voronoi_time.py','cartesian_fv_modes.py','contact_spectral_reference.py','voronoi_time_action.py',
      'voronoi_time16_core.py','shear_union64_study.py','material_union.py','t1_uniform_core.py']
    save(out/'implementation-lock.json',{'files':{n:digest(ROOT/'research/reconnect'/n) for n in names},'protocol_sha256':digest(PROTOCOL),'fixed_before_new_scientific_inputs':True})
    save(out/'config.json',{'updated':'2026-11-04','timezone':'Asia/Shanghai','parent_manifest_entries_verified':6459,'parent_manifest_sha256':digest(SOURCE/'sha256.json'),
      'previous_recorded_checks':493,'new_software_checks':3,'recorded_checks_total':496,'previous_checks_rerun':0,'N_values':list(LEVELS),'new_N_values':list(NEW),
      'new_candidate_graphs':2,'new_candidate_transient_runs':2,'new_complete_independent_uniformization_sequences':2,'N32_paths_loaded_only':1,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_spectral_projection_fits':0,'new_material_data_not_recovered_originals':True})
    preflight=time.monotonic()-pre;started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>900 or peak>1024:raise RuntimeError(f'registered resource cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(*args):raise TimeoutError('registered900s contact space cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(900)
    try:
        pure_source=ROOT/gate['sources'][0]['path'];assert digest(pure_source)==gate['sources'][0]['sha256'];shutil.copyfile(pure_source,out/'original-pure-inputs.npz')
        z=arrays(out/'original-pure-inputs.npz');copies=[{'source':str(pure_source.relative_to(ROOT)),'copy':'original-pure-inputs.npz','sha256':digest(pure_source),'phase':'purebeforegraphs'}];index={}
        for N in NEW:
            folder=out/f'n{N}';folder.mkdir();(folder/'k16').mkdir();p=pure_contact(z[f'n{N}_positions'],z[f'n{N}_ids']);np.savez_compressed(folder/'builder-inputs.npz',**p)
            packet=coefficient_packet(build_periodic_faces(p['ids'],p['positions'],p['rho'],.02,progress=guard));save(folder/'geometry.json',packet)
            index[str(N)]={'file':f'n{N}/geometry.json','sha256':digest(folder/'geometry.json'),'builder_inputs_sha256':digest(folder/'builder-inputs.npz')}
            save(folder/'face-laws.json',face_control(packet,p,N))
            _,_,_,cert=audit_geometry(packet,p['positions'],p['rho'],p['ids'],[0,N//2,N-1,N*N//2,N*N-1])
            save(folder/'geometry-only-audit.json',{'certificate':cert,'cache_sha256':digest(folder/'geometry.json'),'audit_source_sha256':digest(ROOT/'research/reconnect/audit_shear_voronoi_time.py'),
              'mass_c0_reference_or_oldstates_loaded':False,'builder_called_by_auditor':False});guard(f'sealed-and-pure-audited-static-n{N}')
        save(out/'graph-index.json',index);save(out/'coefficient-seal.json',{'graphs':2,'graph_index_sha256':digest(out/'graph-index.json'),
          'both_new_static_graphs_sealed_before_any_material_c0_reference_or_reused_state':True,'builder_received_true_material_faces_F_C_or_precomputed_g':False})
        guard('both-graphs-sealed-before-all-material-and-reference')
        ref_source=ROOT/gate['sources'][1]['path'];assert digest(ref_source)==gate['sources'][1]['sha256'];shutil.copyfile(ref_source,out/'source-reference-K128-p90.json');case=read(out/'source-reference-K128-p90.json')
        copies.append({'source':str(ref_source.relative_to(ROOT)),'copy':'source-reference-K128-p90.json','sha256':digest(ref_source),'phase':'afterbothsealed'})
        for N in NEW:
            folder=out/f'n{N}';p=arrays(folder/'builder-inputs.npz');d,meta=derive_material(case,p,N)
            np.savez_compressed(folder/'material-reference.npz',**d);save(folder/'reference-parameters.json',meta)
            localindex={f'k16-step{i:02d}':{'N':N,'K':16,'step':i,'file':'geometry.json','sha256':digest(folder/'geometry.json'),'midpoint_time_saved':float(F(2*i+1,640))} for i in range(16)}
            save(folder/'graph-index.json',localindex);run,path=native_run(folder,16,d,localindex,guard)
            save(folder/'k16/native-results.json',run);np.savez_compressed(folder/'k16/trajectory-states.npz',**path)
            control,cm=cartesian_control(d,N);np.savez_compressed(folder/'Cartesian-control.npz',**control);save(folder/'Cartesian-parameters.json',cm)
            save(folder/'results.json',evaluate_saved(d,meta,run,path,control,cm));guard(f'n{N}-full-path-and-Cartesian-control-saved')
        folder=out/'n32';folder.mkdir();(folder/'k16').mkdir()
        sources={item['path']:item['sha256'] for item in gate['sources']}
        for name in ('material-reference.npz','reference-parameters.json','Cartesian-control.npz','Cartesian-parameters.json','results.json','k16/native-results.json','k16/trajectory-states.npz'):
            source=SOURCE/name;rel=str(source.relative_to(ROOT));assert digest(source)==sources[rel]
            shutil.copyfile(source,folder/name);copies.append({'source':rel,'copy':f'n32/{name}','sha256':digest(source),'phase':'afterbothsealed,reusednoaction'})
        save(out/'source-copies.json',copies);args=load_all(out);result,common=compare(*args)
        assert plain_json(result['per_grid_controls']['32'])==read(folder/'results.json')
        for N in NEW:assert plain_json(result['per_grid_controls'][str(N)])==read(out/f'n{N}/results.json')
        save(out/'results.json',result);np.savez_compressed(out/'common-observations.npz',**common);guard('three-grid-native-common-concentration-H-control-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-04','timezone':'Asia/Shanghai','recorded_checks_total':496,'new_software_checks':3,'previous_checks_rerun':0,
          'registered_three_space_level_contact_oracle_control_executed':True,'registered_three_space_level_contact_oracle_control_passed':result['registered_three_space_level_contact_oracle_control_passed'],
          'comparison_decision':result['comparison_decision'],'new_candidate_graphs':2,'new_candidate_transient_runs':2,'new_complete_independent_uniformization_sequences':2,
          'N32_paths_loaded_only':1,'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_spectral_projection_fits':0,'original_backends_changed':False,
          'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False,
          'PDF_authorized_by_user':True,'software_release_created':False,'environment_published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':True,
          'preflight_manifest_verification_seconds':preflight,'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'resource_caps_passed':True,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4,'norm_estimator_seed':20261101,
          'network_used':False,'environment_configuration_changed':False,'published':False})
        print(json.dumps(plain_json({'decision':result['comparison_decision'],'observations':result['per_grid_observations'],'comparisons':result['adjacent_spatial_comparisons']}),indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();data,meta,runs,paths,controls,cm=load_all(out);case=read(out/'source-reference-K128-p90.json')
    for N in NEW:
        folder=out/f'n{N}';d2,m2=derive_material(case,arrays(folder/'builder-inputs.npz'),N);assert meta[N]==m2
        for key,value in data[N].items():assert bitwise_same(value,d2[key]),key
        r2,p2=native_run(folder,16,data[N],read(folder/'graph-index.json'),replay=True);assert r2==runs[N]
        for key,value in paths[N].items():assert bitwise_same(value,p2[key]),key
        c2,cm2=cartesian_control(data[N],N);assert plain_json(cm2)==cm[N]
        for key,value in controls[N].items():assert bitwise_same(value,c2[key]),key
    result,common=compare(data,meta,runs,paths,controls,cm);assert plain_json(result)==read(out/'results.json')
    for N in LEVELS:assert plain_json(result['per_grid_controls'][str(N)])==read(out/f'n{N}/results.json')
    saved=arrays(out/'common-observations.npz')
    for key,value in common.items():assert bitwise_same(value,saved[key]),key
    print('Two saved static point-FV graphs: newN16/N64 actual/full independent K16 paths,halfstates,rawCSR/weights,Cartesian allmodes,finite-cell references andfixed-parent observations replayed bitwise; N32loadedonly,original decision unchanged. No geometry/PDE/oldcandidate/ORC rerun.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or--output/--tests/--test-log required')
