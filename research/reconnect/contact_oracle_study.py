"""One registered same-backend static contact; one graph and one K16 path/control."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save as raw_save,digest
from affine_shear_completion import plain_json
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from audit_shear_voronoi_time import audit_geometry
from shear_union64_study import arrays,native_run
from contact_oracle_core import pure_contact,face_control,derive_material,cartesian_control,evaluate
from t2_d0_core import bitwise_same

SOURCE=ROOT/'runs/reconnect-affine-shear-spatial';PROTOCOL=ROOT/'research/reconnect/contact_oracle_protocol.json'


def save(path,obj):raw_save(path,plain_json(obj))


def load_all(out):
    return (arrays(out/'material-reference.npz'),read(out/'reference-parameters.json'),read(out/'k16/native-results.json'),
      arrays(out/'k16/trajectory-states.npz'),arrays(out/'Cartesian-control.npz'),read(out/'Cartesian-parameters.json'))


def main(output,tests,test_log):
    before=time.monotonic();prior=read(SOURCE/'sha256.json');assert len(prior)==6374
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json');assert not gate['already_executed'] and gate['N']==32 and gate['new_candidate_graphs']==1 and gate['ordered_intervals']==16
    assert read(tests)=={'successful':True,'checks_run':3,'failures':0,'errors':0,'skipped':0}
    assert digest(ROOT/'exports/ltn-affine-shear-spatial.zip')=='7466b3e90291dbb1132858d0f9521dda6f06f7d87b9c234d7d66e13ab496d183'
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);(out/'k16').mkdir()
    for name,path in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(path,out/name)
    for name in ('ltn-contact-oracle-tests-attempt1.json','ltn-contact-oracle-tests-attempt1.log','ltn-test-contact-oracle-attempt1.py'):shutil.copyfile(Path('/tmp')/name,out/name)
    save(out/'test-fixture-correction.json',{'first_attempt_passed':False,'first_attempt_failures':1,
      'reason':'N4 square has rebuilt L1radius=1/4,at existing independent auditor strictR<1/4 boundary',
      'change':'only geometric test fixture N4 toN8 andcorresponding fixedIDs/facecount; other two fixtures unchanged',
      'scientific_N32_or_audit_guard_formula_threshold_changed':False})
    names=['contact_oracle_core.py','contact_oracle_study.py','test_contact_oracle.py','periodic_voronoi_fv.py','audit_shear_voronoi_time.py',
      'voronoi_time_action.py','voronoi_time16_core.py','shear_union64_study.py','contact_spectral_reference.py','cartesian_fv_modes.py','t1_uniform_core.py']
    save(out/'implementation-lock.json',{'files':{name:digest(ROOT/'research/reconnect'/name) for name in names},'protocol_sha256':digest(PROTOCOL),'fixed_before_new_scientific_inputs':True})
    save(out/'config.json',{'updated':'2026-11-04','timezone':'Asia/Shanghai','parent_manifest_entries_verified':6374,'parent_manifest_sha256':digest(SOURCE/'sha256.json'),
      'previous_recorded_checks':490,'new_software_checks':3,'recorded_checks_total':493,'previous_checks_rerun':0,'N':32,'K':16,
      'new_candidate_graphs':1,'new_candidate_transient_runs':1,'new_complete_independent_uniformization_sequences':1,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_spectral_projection_fits':0,'material_data_newly_derived_not_recovered':True,
      'PDF_requested_by_user':True,'environment_or_software_publication_requested':False})
    preflight=time.monotonic()-before;started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>900 or peak>1024:raise RuntimeError(f'registered resource cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(*args):raise TimeoutError('registered900s contact cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(900)
    try:
        pure_source=ROOT/gate['pure_source']['path'];assert digest(pure_source)==gate['pure_source']['sha256'];shutil.copyfile(pure_source,out/'original-pure-inputs.npz')
        z=arrays(out/'original-pure-inputs.npz');p=pure_contact(z['n32_positions'],z['n32_ids']);np.savez_compressed(out/'builder-inputs.npz',**p)
        packet=coefficient_packet(build_periodic_faces(p['ids'],p['positions'],p['rho'],.02,progress=guard));save(out/'geometry.json',packet)
        save(out/'coefficient-seal.json',{'graphs':1,'geometry_sha256':digest(out/'geometry.json'),'builder_inputs_sha256':digest(out/'builder-inputs.npz'),
          'static_graph_sealed_before_any_material_c0_reference_or_oldstate':True,'builder_received_true_material_faces_or_F_C_or_precomputed_g':False})
        guard('single-static-coefficient-cache-sealed-before-material-and-reference')
        face=face_control(packet,p,32);save(out/'face-laws.json',face)
        _,_,_,cert=audit_geometry(packet,p['positions'],p['rho'],p['ids'],[0,16,31,512,1023]);save(out/'geometry-only-audit.json',{
          'certificate':cert,'cache_sha256':digest(out/'geometry.json'),'audit_source_sha256':digest(ROOT/'research/reconnect/audit_shear_voronoi_time.py'),
          'mass_c0_reference_or_oldstates_loaded':False,'builder_called_by_auditor':False})
        source=ROOT/gate['sources'][0]['path'];assert digest(source)==gate['sources'][0]['sha256'];shutil.copyfile(source,out/'source-reference-K128-p90.json')
        data,meta=derive_material(read(out/'source-reference-K128-p90.json'),p,32);np.savez_compressed(out/'material-reference.npz',**data);save(out/'reference-parameters.json',meta)
        copies=[{'source':str(pure_source.relative_to(ROOT)),'copy':'original-pure-inputs.npz','sha256':digest(pure_source),'phase':'purebeforegeometry'},
          {'source':str(source.relative_to(ROOT)),'copy':'source-reference-K128-p90.json','sha256':digest(source),'phase':'aftersealonly'}];save(out/'source-copies.json',copies)
        index={f'k16-step{i:02d}':{'N':32,'K':16,'step':i,'file':'geometry.json','sha256':digest(out/'geometry.json'),'midpoint_time_saved':float(F(2*i+1,640))} for i in range(16)}
        save(out/'graph-index.json',index);run,path=native_run(out,16,data,index,guard);save(out/'k16/native-results.json',run);np.savez_compressed(out/'k16/trajectory-states.npz',**path)
        control,cm=cartesian_control(data,32);np.savez_compressed(out/'Cartesian-control.npz',**control);save(out/'Cartesian-parameters.json',cm)
        result=evaluate(data,meta,run,path,control,cm);save(out/'results.json',result);guard('contact-full-path-independent-Cartesian-and-PDE-comparison-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-04','timezone':'Asia/Shanghai','recorded_checks_total':493,'new_software_checks':3,'previous_checks_rerun':0,
          'registered_same_backend_contact_oracle_control_executed':True,'registered_same_backend_contact_oracle_control_passed':result['registered_same_backend_contact_oracle_control_passed'],
          'comparison_decision':result['comparison_decision'],'new_candidate_graphs':1,'new_candidate_transient_runs':1,'new_complete_independent_uniformization_sequences':1,
          'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_spectral_projection_fits':0,'original_backends_changed':False,
          'one_N_not_spatial_convergence':True,'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,'finite_node_candidate_admitted':False,
          'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False,'PDF_authorized_by_user':True,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':False,'tests_second_attempt_passed':True,'main_first_attempt_passed':True,
          'preflight_manifest_verification_seconds':preflight,'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'resource_caps_passed':True,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4,'norm_estimator_seed':20261101,
          'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps(plain_json(result),indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
          'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();data,meta,run,path,control,cm=load_all(out)
    p=arrays(out/'builder-inputs.npz');d2,m2=derive_material(read(out/'source-reference-K128-p90.json'),p,32)
    for key,value in data.items():assert bitwise_same(value,d2[key]),key
    assert meta==m2
    r2,p2=native_run(out,16,data,read(out/'graph-index.json'),replay=True);assert r2==run
    for key,value in path.items():assert bitwise_same(value,p2[key]),key
    c2,cm2=cartesian_control(data,32);assert plain_json(cm2)==cm
    for key,value in control.items():assert bitwise_same(value,c2[key]),key
    assert plain_json(evaluate(data,meta,run,path,control,cm))==read(out/'results.json')
    print('One saved point-FV graph: actual/full independent K16 path,halfstates,CSR/raw weights,Cartesian full modes,new finite-cell means,stored-coefficient references andallpositive decisions replayed bitwise; no geometry/oldcandidate/PDE solve/ORC rerun.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
