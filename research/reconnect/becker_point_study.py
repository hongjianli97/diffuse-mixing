"""One registered moving Becker point-FV oracle with explicit duration and outer walls."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save as raw_save,digest
from affine_shear_completion import plain_json
from bounded_periodic_voronoi import build_bounded_faces
from audit_bounded_geometry import audit_geometry
from becker_point_core import Becker,COUNTS,T,sample_frames,extrude,observe_columns,native_run,evaluate
from shear_union64_study import arrays
from voronoi_time_action import metrics
from t2_d0_core import bitwise_same

SOURCE=ROOT/'runs/reconnect-contact-spatial';PROTOCOL=ROOT/'research/reconnect/becker_point_protocol.json'


def save(path,obj):raw_save(path,plain_json(obj))


def load_all(out):
    d=arrays(out/'material-inputs.npz');col=arrays(out/'source-column-inputs.npz');runs={K:read(out/f'k{K}/native-results.json') for K in COUNTS}
    paths={K:arrays(out/f'k{K}/trajectory-states.npz') for K in COUNTS};return d,col,runs,paths,read(out/'reference-indicators.json')


def reference_indicators(col):
    m,c0=col['mass'],col['c0'];norm=lambda v:float(np.sqrt(m@(v*v)))
    parts={'saved_reference_space_difference':norm(col['reference_n4096-k4096']-col['reference_n8192-k4096']),
      'saved_reference_time_difference':norm(col['reference_n8192-k4096']-col['reference_n8192-k8192'])}
    return {'components_absolute':parts,'reference_space_time_indicator_absolute':sum(parts.values()),'empirical_refinement_differences_not_certified_error_bound':True}


def density_diagnostics(out,flow,labels,col,world):
    rows=[];m=col['mass'];bounds=labels
    for i in range(16):
        frame=world['cases']['16']['frames'][2*i+1];t=float(F(frame['time_exact']));faces=flow.x_mu(bounds,t);volume=np.diff(faces);rhoavg=m/volume
        rp=np.array(frame['rho']).reshape(8,32)[0];packet=read(out/f'k16/geometry-step{i:02d}.json');vor=np.array(list(map(float,packet['cell_areas_exact']))).reshape(8,32)[0]*8
        rows.append({'time':t,'maximum_point_vs_true_material_average_density_relative_difference':float(np.max(abs(rp/rhoavg-1))),
          'maximum_point_Voronoi_vs_true_material_volume_relative_difference':float(np.max(abs(vor/volume-1))),
          'maximum_mass_over_point_rho_vs_true_volume_relative_difference':float(np.max(abs((m/rp)/volume-1))),
          'true_faces_used_by_builder':False,'true_faces_used_only_after_sealing_for_diagnostics':True})
    return rows


def main(output,tests,test_log):
    pre=time.monotonic();prior=read(SOURCE/'sha256.json');assert len(prior)==6576
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json');assert not gate['already_executed'] and gate['new_candidate_graphs']==28 and gate['nodes']==256 and gate['T_exact']=='1/10'
    assert read(tests)=={'successful':True,'checks_run':4,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,path in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),('boundary-duration-contract.md',ROOT/'research/reconnect/BECKER_BOUNDARY_DURATION_CONTRACT.md'),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(path,out/name)
    names=['bounded_periodic_voronoi.py','audit_bounded_geometry.py','explicit_duration_action.py','becker_point_core.py','becker_point_study.py','test_becker_point.py',
      'periodic_voronoi_fv.py','voronoi_time_action.py','voronoi_time16_core.py','t1_uniform_core.py']
    save(out/'implementation-lock.json',{'files':{name:digest(ROOT/'research/reconnect'/name) for name in names},'Becker_source_sha256':digest(ROOT/'research/rebuild/becker.py'),
      'protocol_sha256':digest(PROTOCOL),'contract_sha256':digest(out/'boundary-duration-contract.md'),'fixed_before_new_scientific_points_graphs_mass_scalar_reference':True})
    save(out/'config.json',{'updated':'2026-11-04','timezone':'Asia/Shanghai','parent_manifest_entries_verified':6576,'parent_manifest_sha256':digest(SOURCE/'sha256.json'),
      'previous_recorded_checks':496,'new_software_checks':4,'recorded_checks_total':500,'previous_checks_rerun':0,'columns':32,'rows':8,'nodes':256,'T_exact':'1/10','D':.02,'delta':.08,
      'K_values':[4,8,16],'new_candidate_graphs':28,'new_candidate_transient_runs':3,'new_complete_independent_uniformization_sequences':3,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_spectral_fits':0,'old_sources_not_recovered_v080':True})
    preflight=time.monotonic()-pre;started=time.monotonic()
    def guard(stage,i=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>900 or peak>1024:raise RuntimeError(f'registered resource limit exceeded:{elapsed}s/{peak}MiB')
        if not total or i==total:print(f'{stage}:{i}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(*args):raise TimeoutError('registered900s moving Becker cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(900)
    try:
        source=ROOT/gate['sources'][0]['path'];assert digest(source)==gate['sources'][0]['sha256']
        # Only given mass-coordinate labels opened here. Actual mass/c0/reference stay unopened.
        with np.load(source,allow_pickle=False) as z:labels=z['mass_edges'].copy()
        assert labels.shape==(33,);np.save(out/'source-mass-labels.npy',labels);flow=Becker(delta=.08);world=sample_frames(labels,flow);save(out/'pure-inputs.json',world)
        save(out/'input-matching.json',{'source_input_sha256':digest(source),'only_given_mass_coordinate_labels_parsed_before_geometry':True,
          'mass_c0_reference_or_oldstates_parsed':False,'pure_inputs_sha256':digest(out/'pure-inputs.json')})
        index={}
        for K in COUNTS:
            (out/f'k{K}').mkdir()
            for i in range(K):
                f=world['cases'][str(K)]['frames'][2*i+1];packet=build_bounded_faces(world['ids'],f['positions'],f['rho'],f['D'],f['x_bounds'],progress=guard)
                rel=f'k{K}/geometry-step{i:02d}.json';save(out/rel,packet)
                index[f'k{K}-step{i:02d}']={'K':K,'step':i,'file':rel,'sha256':digest(out/rel),'midpoint_time_saved':float(F(f['time_exact']))}
                _,_,_,cert=audit_geometry(packet,f['positions'],f['rho'],world['ids'],f['D'],f['x_bounds'])
                save(out/f'k{K}/geometry-audit-step{i:02d}.json',{'cache_sha256':digest(out/rel),'pure_inputs_sha256':digest(out/'pure-inputs.json'),
                  'audit_source_sha256':digest(ROOT/'research/reconnect/audit_bounded_geometry.py'),'mass_c0_reference_loaded':False,'certificate':cert})
                guard(f'sealed-and-pure-audited-bounded-k{K}-step{i:02d}')
        save(out/'graph-index.json',index);save(out/'coefficient-seal.json',{'graphs':28,'graph_index_sha256':digest(out/'graph-index.json'),
          'all28_graphs_sealed_before_actual_mass_c0_reference_or_oldstate_load':True,'internal_material_faces_or_F_C_received_by_builder':False})
        guard('all28-bounded-graphs-sealed-before-material-and-reference')
        shutil.copyfile(source,out/'source-column-inputs.npz');col=arrays(out/'source-column-inputs.npz');data=extrude(col);np.savez_compressed(out/'material-inputs.npz',**data)
        fine=ROOT/gate['sources'][1]['path'];assert digest(fine)==gate['sources'][1]['sha256'];shutil.copyfile(fine,out/'source-reference-n8192-k8192.npz')
        save(out/'source-copies.json',[{'source':str(source.relative_to(ROOT)),'copy':'source-column-inputs.npz','sha256':digest(source),'phase':'allfieldsafter28sealed;masslabelsonlybefore'},
          {'source':str(fine.relative_to(ROOT)),'copy':'source-reference-n8192-k8192.npz','sha256':digest(fine),'phase':'after28sealed'}])
        ref_meta=reference_indicators(col);save(out/'reference-indicators.json',ref_meta)
        runs={};paths={}
        for K in COUNTS:
            run,path=native_run(out,K,data,index,guard);runs[K]=run;paths[K]=path;save(out/f'k{K}/native-results.json',run);np.savez_compressed(out/f'k{K}/trajectory-states.npz',**path)
        result=evaluate(data,col,runs,paths,ref_meta);save(out/'results.json',result);save(out/'density-volume-diagnostics.json',density_diagnostics(out,flow,labels,col,world))
        save(out/'stage3-status.json',{'updated':'2026-11-04','timezone':'Asia/Shanghai','recorded_checks_total':500,'new_software_checks':4,'previous_checks_rerun':0,
          'registered_moving_Becker_discriminator_executed':True,'registered_moving_Becker_oracle_discriminator_passed':result['registered_moving_Becker_oracle_discriminator_passed'],
          'comparison_decision':result['comparison_decision'],'new_candidate_graphs':28,'new_candidate_transient_runs':3,'new_complete_independent_uniformization_sequences':3,
          'old_candidate_reruns':0,'new_reference_PDE_solves':0,'original_periodic_geometry_exponential_uniformization_and_H_backends_changed':False,
          'optional_outer_boundary_and_duration_interfaces_added':True,'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,
          'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False,'environment_or_software_version_published':False})
        guard('moving-Becker-source-time-and-physical-comparison-saved')
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':True,
          'preflight_manifest_verification_seconds':preflight,'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'resource_caps_passed':True,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4,'norm_estimator_seed':20261101,
          'network_used':False,'environment_configuration_changed':False,'published':False})
        print(json.dumps(plain_json(result),indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();data,col,runs,paths,rm=load_all(out);index=read(out/'graph-index.json')
    d2=extrude(col)
    for key,value in data.items():assert bitwise_same(value,d2[key]),key
    assert reference_indicators(col)==rm
    for K in COUNTS:
        r2,p2=native_run(out,K,data,index,replay=True);assert r2==runs[K]
        for key,value in paths[K].items():assert bitwise_same(value,p2[key]),key
    assert plain_json(evaluate(data,col,runs,paths,rm))==read(out/'results.json')
    print('28 bounded-world caches: K4/8/16 actual/full independent explicitT.1 paths,halfstates,CSR/rawweights replayed bitwise; original32column means/ref,transverseintegrals,time/Hdecisions reproduced. No geometry/oldcandidate/PDE/ORC rerun.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or--output/--tests/--test-log required')
