"""Registered N16/32/64 affine shear: 56 new graphs, three N32 paths reused only."""
from pathlib import Path
from fractions import Fraction as F
import argparse,gc,json,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save as raw_save,digest
from affine_shear_core import COUNTS,pure_inputs,directional_certificate,modal_certificate
from sliding_periodic_voronoi import SlidingPeriodicDomain,build_sliding_faces
from affine_shear_completion import plain_json
from affine_shear_spatial_core import compare
from affine_shear_study import save_references
from shear_union64_study import arrays,native_run
from t2_d0_core import bitwise_same

SOURCE=ROOT/'runs/reconnect-affine-shear';NEW=(16,64);LEVELS=(16,32,64)
PROTOCOL=ROOT/'research/reconnect/affine_shear_spatial_protocol.json'


def save(path,obj):raw_save(path,plain_json(obj))


def saved_rows(folder,data,labels,world,N,K):
    return [modal_certificate(data,labels,world['cases'][str(K)]['frames'][2*i+1],
      arrays(folder/f'k{K}/action-step{i:02d}.npz'),N,i) for i in range(K)]


def load_all(out):
    pure=arrays(out/'original-pure-inputs.npz');data={};labels={};worlds={};runs={};paths={};res={}
    for N in LEVELS:
        folder=out/f'n{N}';data[N]=arrays(folder/'material-inputs.npz');labels[N]=pure[f'n{N}_positions'];worlds[N]=read(folder/'pure-inputs.json')
        for K in COUNTS:
            runs[N,K]=read(folder/f'k{K}/native-results.json');paths[N,K]=arrays(folder/f'k{K}/trajectory-states.npz')
            res[N,K]=[sum(r['mode_residues_absolute'])+r['constant_residue_absolute'] for r in read(folder/f'k{K}/modal-residues.json')]
    return data,labels,worlds,runs,paths,res


def check_references(folder,b):
    z=arrays(folder/'analytic-references.npz')
    for key in ('weights','modes','reference','continuous_grid_symbol'):assert bitwise_same(z[key],b[key]),key
    for K in COUNTS:assert bitwise_same(z[f'frozen_k{K}'],b['frozen_symbols'][K])
    assert read(folder/'analytic-parameters.json')==b['meta']


def main(output,tests,test_log):
    pre=time.monotonic();prior=read(SOURCE/'sha256.json');assert len(prior)==5830
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json')
    assert not gate['already_executed'] and gate['new_N_values']==list(NEW) and gate['new_candidate_graphs']==56
    assert gate['main_wall_seconds_cap']==1500 and gate['main_peak_RSS_MiB_cap']==1024
    assert read(tests)=={'successful':True,'checks_run':3,'failures':0,'errors':0,'skipped':0}
    assert digest(ROOT/'exports/ltn-affine-shear.zip')=='470fd77d903b240f0752bc43238df967ccdb809b052487b4550de8c676f47c0e'
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,path in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(path,out/name)
    for name in ('ltn-affine-shear-spatial-tests-attempt1.json','ltn-affine-shear-spatial-tests-attempt1.log','ltn-test-affine-shear-spatial-attempt1.py'):
        shutil.copyfile(Path('/tmp')/name,out/name)
    save(out/'test-correction.json',{'first_attempt_passed':False,'first_attempt_failures':1,'reason':'test incorrectly inherited unresolved compression-model expectation; analytic shear fixture already resolves direction',
      'change':'two expected booleans False to True only; adverse full-sequence rejection unchanged','scientific_criteria_or_thresholds_changed':False,'prior_checks_rerun':0})
    names=['affine_shear_spatial_core.py','affine_shear_spatial_study.py','test_affine_shear_spatial.py','audit_affine_shear_spatial_geometry.py',
      'sliding_periodic_voronoi.py','audit_sliding_world_geometry.py','affine_shear_core.py','affine_shear_completion.py','affine_shear_study.py',
      'periodic_voronoi_fv.py','voronoi_time_action.py','voronoi_time16_core.py','shear_union64_study.py','material_union.py','t1_uniform_core.py']
    save(out/'implementation-lock.json',{'files':{n:digest(ROOT/'research/reconnect'/n) for n in names},'protocol_sha256':digest(PROTOCOL),'fixed_before_new_scientific_inputs':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':5830,'parent_manifest_sha256':digest(SOURCE/'sha256.json'),
      'previous_recorded_checks':487,'new_software_checks':3,'recorded_checks_total':490,'previous_checks_rerun':0,'N_values':list(LEVELS),'new_N_values':list(NEW),
      'new_candidate_graphs':56,'new_candidate_transient_runs':6,'new_complete_independent_uniformization_sequences':6,'reused_N32_saved_paths':3,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_material_definitions':0})
    pre_elapsed=time.monotonic()-pre;started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>1500 or peak>1024:raise RuntimeError(f'registered cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(*args):raise TimeoutError('registered1500s affine-shear spatial cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(1500)
    try:
        pure=ROOT/gate['pure_source']['path'];assert digest(pure)==gate['pure_source']['sha256'];shutil.copyfile(pure,out/'original-pure-inputs.npz')
        z=arrays(out/'original-pure-inputs.npz');index={};copies=[{'source':str(pure.relative_to(ROOT)),'copy':'original-pure-inputs.npz','sha256':digest(pure),'phase':'pureonlybefore56graphs'}]
        for N in NEW:
            folder=out/f'n{N}';folder.mkdir();world=pure_inputs(z[f'n{N}_positions'],z[f'n{N}_ids']);ids=np.array(world['ids'],dtype=np.int64)
            save(folder/'pure-inputs.json',world);save(folder/'input-matching.json',{'N':N,'pure_inputs_sha256':digest(folder/'pure-inputs.json'),'mass_c0_reference_or_oldstates_loaded':False})
            localindex={}
            for K in COUNTS:
                (folder/f'k{K}').mkdir()
                for step in range(K):
                    frame=world['cases'][str(K)]['frames'][2*step+1]
                    packet,wp,cert=build_sliding_faces(ids,frame['world_positions_exact'],frame['rho'],frame['D_float'],SlidingPeriodicDomain(frame['shear_exact']),guard)
                    local=f'k{K}/geometry-step{step:02d}.json';worldname=f'k{K}/world-geometry-step{step:02d}.json'
                    save(folder/local,packet);save(folder/worldname,wp);save(folder/f'k{K}/boundary-step{step:02d}.json',cert)
                    direction=directional_certificate(packet,frame,N);save(folder/f'k{K}/direction-step{step:02d}.json',direction)
                    if not direction['all_geometry_controls_passed']:raise ArithmeticError('exact direction/second-moment control failed; no repair')
                    item={'N':N,'K':K,'step':step,'frame':2*step+1,'file':local,'sha256':digest(folder/local),
                      'world_file':worldname,'world_sha256':digest(folder/worldname),'midpoint_time_saved':float(F(frame['time_exact']))}
                    localindex[f'k{K}-step{step:02d}']=item;index[f'n{N}-k{K}-step{step:02d}']={**item,'file':f'n{N}/'+local,'world_file':f'n{N}/'+worldname}
                    del packet,wp;gc.collect();guard(f'sealed-spatial-sliding-n{N}-k{K}-step{step:02d}')
            save(folder/'graph-index.json',localindex);del world;gc.collect()
        del z;gc.collect();save(out/'graph-index.json',index)
        save(out/'coefficient-seal.json',{'graphs':len(index),'graph_index_sha256':digest(out/'graph-index.json'),
          'all56_new_world_euclidean_graphs_sealed_before_any_material_reference_or_reused_state':True,
          'builder_received_mass_c0_reference_true_internal_faces_F_C_or_precomputed_g':False})
        guard('all56-new-sliding-graphs-sealed-before-any-material-or-oldstate')
        for N in NEW:
            folder=out/f'n{N}';source=ROOT/gate['material_sources'][str(N)]['path'];assert digest(source)==gate['material_sources'][str(N)]['sha256']
            shutil.copyfile(source,folder/'original-material-reference.npz')
            with np.load(folder/'original-material-reference.npz',allow_pickle=False) as zz:data={key:zz[key].copy() for key in ('mass','c0')}
            world=read(folder/'pure-inputs.json');data['ids']=np.array(world['ids'],dtype=np.int64);np.savez_compressed(folder/'material-inputs.npz',**data)
            labels=arrays(out/'original-pure-inputs.npz')[f'n{N}_positions'];localindex=read(folder/'graph-index.json')
            copies.append({'source':str(source.relative_to(ROOT)),'copy':f'n{N}/original-material-reference.npz','sha256':digest(source),'phase':'after56sealed,mass-c0onlyparsed'})
            for K in COUNTS:
                run,path=native_run(folder,K,data,localindex,guard);save(folder/f'k{K}/native-results.json',run);np.savez_compressed(folder/f'k{K}/trajectory-states.npz',**path)
                save(folder/f'k{K}/modal-residues.json',saved_rows(folder,data,labels,world,N,K))
            del world,data;gc.collect()
        folder=out/'n32';folder.mkdir()
        names=['material-inputs.npz','pure-inputs.json','analytic-references.npz','analytic-parameters.json','results-completed.json']
        for K in COUNTS:
            (folder/f'k{K}').mkdir();names.extend(f'k{K}/{name}' for name in ('native-results.json','trajectory-states.npz','modal-residues.json'))
        expected={i['path']:i['sha256'] for i in gate['reused_N32_records']};expected[gate['N32_valid_result']['path']]=gate['N32_valid_result']['sha256']
        for name in names:
            source=SOURCE/name;rel=str(source.relative_to(ROOT))
            if rel in expected:assert digest(source)==expected[rel]
            target=folder/('old-case-results.json' if name=='results-completed.json' else name);shutil.copyfile(source,target)
            copies.append({'source':rel,'copy':str(target.relative_to(out)),'sha256':digest(source),'phase':'after56sealed,reusedwithoutaction'})
        save(out/'source-copies.json',copies)
        args=load_all(out);result,common,bundles=compare(*args);assert plain_json(result['per_grid_controls']['32'])==read(folder/'old-case-results.json')
        for N in NEW:save_references(out/f'n{N}',bundles[N])
        check_references(folder,bundles[32]);save(out/'results.json',result);np.savez_compressed(out/'common-observations.npz',**common)
        guard('three-space-level-native-common-H-and-fine-direction-controls-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','previous_recorded_checks':487,'new_software_checks':3,'recorded_checks_total':490,
          'previous_checks_rerun':0,'registered_three_space_level_affine_shear_control_executed':True,'registered_three_space_level_affine_shear_control_passed':result['registered_three_space_level_affine_shear_control_passed'],
          'comparison_decision':result['comparison_decision'],'original_N32_case_passed':True,'new_candidate_graphs':56,'new_candidate_transient_runs':6,
          'new_complete_independent_uniformization_sequences':6,'reused_N32_paths':3,'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_material_definitions':0,
          'original_backends_changed':False,'general_T3_admitted':False,'general_smooth_flow_admitted':False,'general_nonorthogonal_domains_admitted':False,'finite_node_candidate_admitted':False,
          'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False,'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':False,'tests_second_attempt_passed':True,'main_first_attempt_passed':True,
          'preflight_manifest_verification_seconds':pre_elapsed,'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'resource_caps_passed':True,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4,'norm_estimator_seed':20261101,
          'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps(plain_json({'decision':result['comparison_decision'],'observations':result['per_grid_observations'],'comparisons':result['adjacent_spatial_comparisons']}),indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
          'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();args=load_all(out);data,labels,worlds,runs,paths,res=args
    for N in NEW:
        folder=out/f'n{N}';index=read(folder/'graph-index.json')
        for K in COUNTS:
            run,path=native_run(folder,K,data[N],index,replay=True);assert run==runs[N,K]
            for key,value in path.items():assert bitwise_same(value,paths[N,K][key]),key
            assert saved_rows(folder,data[N],labels[N],worlds[N],N,K)==read(folder/f'k{K}/modal-residues.json')
    result,common,bundles=compare(*args);assert plain_json(result)==read(out/'results.json')
    assert plain_json(result['per_grid_controls']['32'])==read(out/'n32/old-case-results.json')
    saved=arrays(out/'common-observations.npz')
    for key,value in common.items():assert bitwise_same(value,saved[key]),key
    for N in LEVELS:check_references(out/f'n{N}',bundles[N])
    print('56 sealed sliding-world graphs: six new actual/complete independent paths,halfstates,rawCSR/weights replayed bitwise; three N32 paths loaded only; original material integrals,references,bridges,positive budgets andN32 decisions reproduced. No geometry/PDE/old-candidate/ORC rerun.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
