"""One registered three-space-level affine pair; 112 new graphs, old N32 paths reused."""
from pathlib import Path
from fractions import Fraction as F
import argparse,gc,json,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save,digest
from affine_compression_inputs import MODELS,COUNTS,pure_inputs,directional_certificate
from oriented_periodic_boundary import OrientedPeriodicRectangle,build_oriented_periodic_faces
from shear_union64_study import arrays,native_run
from t3_affine_spatial_core import modal_certificate,compare
from t3_affine_study import save_references
from t2_d0_core import bitwise_same

SOURCE=ROOT/'runs/reconnect-t3-affine-review'
OLD=ROOT/'runs/reconnect-t3-affine'
PROTOCOL=ROOT/'research/reconnect/t3_affine_spatial_protocol.json'
NEW=(16,64);LEVELS=(16,32,64)


def saved_rows(folder,data,world,N,model,K):
    return [modal_certificate(data,world['models'][model][str(K)]['frames'][2*s+1],
      arrays(folder/model/f'k{K}/action-step{s:02d}.npz'),N,s) for s in range(K)]


def load_all(out):
    data={};worlds={};runs={};paths={};residues={}
    for N in LEVELS:
        folder=out/f'n{N}';data[N]=arrays(folder/'material-inputs.npz');worlds[N]=read(folder/'pure-inputs.json')
        for model in MODELS:
            for K in COUNTS:
                case=folder/model/f'k{K}';runs[N,model,K]=read(case/'native-results.json');paths[N,model,K]=arrays(case/'trajectory-states.npz')
                residues[N,model,K]=[v['mode_residue_absolute']+v['constant_residue_absolute'] for v in read(case/'modal-residues.json')]
    return data,worlds,runs,paths,residues


def check_saved_references(folder,b):
    z=arrays(folder/'analytic-references.npz')
    for model in MODELS:
        for field in ('reference','continuous_grid_symbol'):assert bitwise_same(z[model+'_'+field],b[field][model])
        for K in COUNTS:
            for field in ('frozen_wire_symbol','frozen_nominal_symbol'):assert bitwise_same(z[f'{model}_k{K}_{field}'],b[field][model,K])


def main(output,tests,test_log):
    preflight=time.monotonic();prior=read(SOURCE/'sha256.json');assert len(prior)==4545
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json');proto=read(PROTOCOL)
    assert not gate['already_executed'] and gate['new_N_values']==list(NEW) and gate['new_candidate_graphs']==112
    assert gate['main_wall_seconds_cap']==900 and gate['main_peak_RSS_MiB_cap']==1024
    assert read(tests)=={'successful':True,'checks_run':3,'failures':0,'errors':0,'skipped':0}
    assert digest(ROOT/'exports/ltn-t3-affine-review.zip')=='4e1639b16f73424dcdbba03a8b57f8391a2ad33b06ae535b89f6d3f67abf1482'
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,path in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(path,out/name)
    names=['t3_affine_spatial_core.py','t3_affine_spatial_study.py','test_t3_affine_spatial.py','audit_t3_affine_spatial_geometry.py',
      'affine_compression_inputs.py','t3_affine_core.py','t3_affine_study.py','oriented_periodic_boundary.py','periodic_voronoi_fv.py',
      'audit_oriented_world_geometry.py','voronoi_time_action.py','voronoi_time16_core.py','shear_union64_study.py','material_union.py','t1_uniform_core.py','t2_translation_core.py']
    save(out/'implementation-lock.json',{'files':{name:digest(ROOT/'research/reconnect'/name) for name in names},'protocol_sha256':digest(PROTOCOL),'fixed_before_new_scientific_inputs':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':4545,'parent_manifest_sha256':digest(SOURCE/'sha256.json'),
      'previous_recorded_checks':480,'new_software_checks':3,'recorded_checks_total':483,'previous_checks_rerun':0,'N_values':list(LEVELS),'new_N_values':list(NEW),
      'new_candidate_graphs':112,'new_candidate_transient_runs':12,'new_independent_uniformization_sequences':12,'reused_N32_saved_paths':6,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_material_definitions':0})
    preflight_elapsed=time.monotonic()-preflight;started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>900 or peak>1024:raise RuntimeError(f'registered cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(*args):raise TimeoutError('registered900s study cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(900)
    try:
        pure=ROOT/gate['pure_source']['path'];assert digest(pure)==gate['pure_source']['sha256'];shutil.copyfile(pure,out/'original-pure-inputs.npz')
        z=arrays(out/'original-pure-inputs.npz');index={};copies=[{'source':str(pure.relative_to(ROOT)),'copy':'original-pure-inputs.npz','sha256':digest(pure),'phase':'pureonlybefore112graphs'}]
        for N in NEW:
            folder=out/f'n{N}';folder.mkdir();world=pure_inputs(z[f'n{N}_positions'],z[f'n{N}_ids']);ids=np.array(world['ids'],dtype=np.int64)
            save(folder/'pure-inputs.json',world);save(folder/'input-matching.json',{'N':N,'pure_inputs_sha256':digest(folder/'pure-inputs.json'),
              'mass_c0_reference_or_oldstates_loaded':False,'density_or_diffusivity_renormalized':False})
            for model in MODELS:
                (folder/model).mkdir();localindex={}
                for K in COUNTS:
                    (folder/model/f'k{K}').mkdir()
                    for step in range(K):
                        frame=world['models'][model][str(K)]['frames'][2*step+1]
                        packet,wp,cert=build_oriented_periodic_faces(ids,frame['world_positions_exact'],frame['rho'],frame['D_float'],OrientedPeriodicRectangle.from_packet(frame['boundary']),guard)
                        direction=directional_certificate(packet,frame,ids,N)
                        if not direction['all_directional_conductances_exact']:raise ArithmeticError('directional model control failed; no repair')
                        base=f'n{N}/{model}/k{K}';localname=base+f'/geometry-step{step:02d}.json';worldname=base+f'/world-geometry-step{step:02d}.json'
                        save(out/localname,packet);save(out/worldname,wp);save(out/(base+f'/isometry-step{step:02d}.json'),cert);save(out/(base+f'/direction-step{step:02d}.json'),direction)
                        item={'N':N,'model':model,'K':K,'step':step,'frame':2*step+1,'file':localname,'sha256':digest(out/localname),
                          'world_file':worldname,'world_sha256':digest(out/worldname),'midpoint_time_saved':float(F(frame['time_exact']))}
                        index[f'n{N}-{model}-k{K}-step{step:02d}']=item;localindex[f'k{K}-step{step:02d}']={**item,'file':f'k{K}/geometry-step{step:02d}.json'}
                        del packet,wp,cert;gc.collect();guard(f'sealed-spatial-n{N}-{model}-k{K}-step{step:02d}')
                save(folder/model/'graph-index.json',localindex)
            del world;gc.collect()
        del z;gc.collect();save(out/'graph-index.json',index)
        save(out/'coefficient-seal.json',{'graphs':len(index),'graph_index_sha256':digest(out/'graph-index.json'),
          'all112_new_local_and_world_graphs_sealed_before_material_reference_or_reused_states':True,
          'builder_received_mass_c0_reference_true_internal_faces_or_F_C':False})
        guard('all112-new-graphs-sealed-before-any-material-or-oldstate')
        for N in NEW:
            folder=out/f'n{N}';path=ROOT/gate['material_sources'][str(N)]['path'];assert digest(path)==gate['material_sources'][str(N)]['sha256']
            shutil.copyfile(path,folder/'original-material-reference.npz')
            with np.load(folder/'original-material-reference.npz',allow_pickle=False) as zz:data={key:zz[key].copy() for key in ('mass','c0')}
            world=read(folder/'pure-inputs.json');data['ids']=np.array(world['ids'],dtype=np.int64);np.savez_compressed(folder/'material-inputs.npz',**data)
            copies.append({'source':str(path.relative_to(ROOT)),'copy':f'n{N}/original-material-reference.npz','sha256':digest(path),'phase':'after112sealed,mass-c0onlyparsed'})
            for model in MODELS:
                localindex=read(folder/model/'graph-index.json')
                for K in COUNTS:
                    run,pathvalues=native_run(folder/model,K,data,localindex,guard);case=folder/model/f'k{K}'
                    save(case/'native-results.json',run);np.savez_compressed(case/'trajectory-states.npz',**pathvalues)
                    save(case/'modal-residues.json',saved_rows(folder,data,world,N,model,K))
            del world,data;gc.collect()
        folder=out/'n32';folder.mkdir()
        names=['material-inputs.npz','pure-inputs.json','analytic-references.npz','analytic-parameters.json','results.json']
        for model in MODELS:
            for K in COUNTS:
                (folder/model/f'k{K}').mkdir(parents=True)
                names.extend(f'{model}/k{K}/{name}' for name in ('native-results.json','trajectory-states.npz','modal-residues.json'))
        for name in names:
            path=OLD/name;target=folder/('old-case-results.json' if name=='results.json' else name);shutil.copyfile(path,target)
            copies.append({'source':str(path.relative_to(ROOT)),'copy':str(target.relative_to(out)),'sha256':digest(path),'phase':'after112sealed,reusednoaction'})
        save(out/'source-copies.json',copies)
        args=load_all(out);result,common,bundles=compare(*args)
        assert result['per_grid_controls']['32']==read(folder/'old-case-results.json'),'old N32 decision changed'
        for N in NEW:save_references(out/f'n{N}',bundles[N])
        check_saved_references(folder,bundles[32]);save(out/'results.json',result);np.savez_compressed(out/'common-observations.npz',**common)
        guard('three-space-level-native-common-H-and-fine-model-comparison-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','previous_recorded_checks':480,'new_software_checks':3,'recorded_checks_total':483,
          'previous_checks_rerun':0,'new_candidate_graphs':112,'new_candidate_transient_runs':12,'new_independent_uniformization_sequences':12,'reused_N32_paths':6,
          'original_N32_decision':result['per_grid_controls']['32']['comparison_decision'],'original_N32_case_passed':False,
          'registered_three_space_level_affine_control_passed':result['T3_registered_three_space_level_affine_control_passed'],'comparison_decision':result['comparison_decision'],
          'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_material_definitions':0,'original_backends_changed':False,
          'general_T3_admitted':False,'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete',
          'fluid_orc_enabled':False,'kappa_computed':False,'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':True,
          'preflight_manifest_verification_seconds':preflight_elapsed,'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'resource_caps_passed':True,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4,'norm_estimator_seed':20261101,
          'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps({'decision':result['comparison_decision'],'spatial_gains_resolved':result['all_native_common_and_H_spatial_gains_resolved'],
          'grid_controls_passed':result['all_grid_symbol_reference_and_time_controls_passed'],'fine_model_response_resolved':result['fine_grid_model_response_resolved'],
          'observations':result['per_grid_observations'],'fine_model_comparisons':result['per_grid_controls']['64']['model_comparisons']},indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
          'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();args=load_all(out);data,worlds,runs,paths,residues=args
    for N in NEW:
        folder=out/f'n{N}'
        for model in MODELS:
            index=read(folder/model/'graph-index.json')
            for K in COUNTS:
                run,path=native_run(folder/model,K,data[N],index,replay=True);assert run==runs[N,model,K]
                for key,value in path.items():assert bitwise_same(value,paths[N,model,K][key]),key
                assert saved_rows(folder,data[N],worlds[N],N,model,K)==read(folder/model/f'k{K}/modal-residues.json')
    result,common,bundles=compare(*args);assert result==read(out/'results.json')
    assert result['per_grid_controls']['32']==read(out/'n32/old-case-results.json')
    saved=arrays(out/'common-observations.npz')
    for key,value in common.items():assert bitwise_same(value,saved[key]),key
    for N in LEVELS:check_saved_references(out/f'n{N}',bundles[N])
    print('112 sealed new graphs:12 actual/complete independent paths and halfstates/CSR/weights replayed bitwise; six N32 paths loaded only, not rerun; all references, common integrals, positive budgets and original N32 decision reproduced. No geometry/PDE/old-candidate rerun or ORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
