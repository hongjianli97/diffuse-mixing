"""One registered N32 exact sliding-periodic affine-shear material experiment."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save,digest
from sliding_periodic_voronoi import SlidingPeriodicDomain,build_sliding_faces
from affine_shear_core import COUNTS,pure_inputs,directional_certificate,modal_certificate,evaluate
from shear_union64_study import arrays,native_run
from t2_d0_core import bitwise_same

SOURCE=ROOT/'runs/reconnect-t3-coverage'
PROTOCOL=ROOT/'research/reconnect/affine_shear_protocol.json'


def saved_modal_rows(out,data,labels,world,K):
    return [modal_certificate(data,labels,world['cases'][str(K)]['frames'][2*i+1],arrays(out/f'k{K}/action-step{i:02d}.npz'),32,i) for i in range(K)]


def save_references(out,b):
    values={key:b[key] for key in ('weights','modes','reference','continuous_grid_symbol')}
    for K,value in b['frozen_symbols'].items():values[f'frozen_k{K}']=value
    np.savez_compressed(out/'analytic-references.npz',**values);save(out/'analytic-parameters.json',b['meta'])


def inputs(out):
    data=arrays(out/'material-inputs.npz')
    with np.load(out/'original-material-reference.npz',allow_pickle=False) as z:
        for key in ('mass','c0'):assert bitwise_same(data[key],z[key])
    return data


def load_all(out):
    d=inputs(out);z=arrays(out/'original-pure-inputs.npz');labels=z['n32_positions'];world=read(out/'pure-inputs.json');runs={};paths={};res={}
    for K in COUNTS:
        runs[K]=read(out/f'k{K}/native-results.json');paths[K]=arrays(out/f'k{K}/trajectory-states.npz')
        res[K]=[sum(row['mode_residues_absolute'])+row['constant_residue_absolute'] for row in read(out/f'k{K}/modal-residues.json')]
    return d,labels,world,runs,paths,res


def main(output,tests,test_log):
    pre=time.monotonic();prior=read(SOURCE/'sha256.json');assert len(prior)==5540
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json');assert not gate['already_executed'] and gate['N']==32 and gate['new_candidate_graphs']==28
    assert gate['main_wall_seconds_cap']==900 and gate['main_peak_RSS_MiB_cap']==1024
    assert read(tests)=={'successful':True,'checks_run':4,'failures':0,'errors':0,'skipped':0}
    assert digest(ROOT/'exports/ltn-t3-coverage.zip')=='dac706856f97f44be0cb037b85b5476908baca6bb0fc473d9e64032a0588ace0'
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,path in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),
      ('contract-at-lock.md',ROOT/gate['written_contract']['path']),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(path,out/name)
    names=['sliding_periodic_voronoi.py','audit_sliding_world_geometry.py','affine_shear_core.py','affine_shear_study.py','test_affine_shear.py',
      'audit_affine_shear_geometry.py','periodic_voronoi_fv.py','oriented_periodic_boundary.py','voronoi_time_action.py','voronoi_time16_core.py','shear_union64_study.py','t1_uniform_core.py']
    save(out/'implementation-lock.json',{'files':{name:digest(ROOT/'research/reconnect'/name) for name in names},'protocol_sha256':digest(PROTOCOL),
      'contract_sha256':digest(out/'contract-at-lock.md'),'fixed_before_scientific_inputs':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':5540,'parent_manifest_sha256':digest(SOURCE/'sha256.json'),
      'previous_recorded_checks':483,'new_software_checks':4,'recorded_checks_total':487,'previous_checks_rerun':0,
      'N':32,'T_exact':'1/20','D':.02,'rho':1.,'K_values':list(COUNTS),'new_candidate_graphs':28,'new_candidate_transient_runs':3,
      'new_independent_uniformization_sequences':3,'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_material_definitions':0})
    preflight=time.monotonic()-pre;started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>900 or peak>1024:raise RuntimeError(f'registered resource cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(*args):raise TimeoutError('registered900s affine shear cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(900)
    try:
        pure=ROOT/gate['pure_source']['path'];assert digest(pure)==gate['pure_source']['sha256'];shutil.copyfile(pure,out/'original-pure-inputs.npz')
        z=arrays(out/'original-pure-inputs.npz');labels=z['n32_positions'];ids=z['n32_ids'];world=pure_inputs(labels,ids);save(out/'pure-inputs.json',world)
        save(out/'input-matching.json',{'pure_inputs_sha256':digest(out/'pure-inputs.json'),'source_pure_sha256':digest(pure),
          'mass_c0_reference_or_oldstates_loaded':False,'world_points_inverse_sheared_for_geometry':False})
        del z;index={}
        for K in COUNTS:
            (out/f'k{K}').mkdir()
            for step in range(K):
                frame=world['cases'][str(K)]['frames'][2*step+1];boundary=SlidingPeriodicDomain(frame['shear_exact'])
                packet,wp,cert=build_sliding_faces(ids,frame['world_positions_exact'],frame['rho'],frame['D_float'],boundary,guard)
                local=f'k{K}/geometry-step{step:02d}.json';worldname=f'k{K}/world-geometry-step{step:02d}.json'
                save(out/local,packet);save(out/worldname,wp);save(out/f'k{K}/boundary-step{step:02d}.json',cert)
                # Diagnostic formulas only after both coefficient/geometry files have closed.
                control=directional_certificate(packet,frame,32);save(out/f'k{K}/direction-step{step:02d}.json',control)
                if not control['all_geometry_controls_passed']:raise ArithmeticError('declared three-direction geometry/moment control failed; no repair')
                index[f'k{K}-step{step:02d}']={'N':32,'K':K,'step':step,'frame':2*step+1,'file':local,'sha256':digest(out/local),
                  'world_file':worldname,'world_sha256':digest(out/worldname),'midpoint_time_saved':float(F(frame['time_exact']))}
                del packet,wp;gc.collect();guard(f'sealed-sliding-k{K}-step{step:02d}')
        save(out/'graph-index.json',index);save(out/'coefficient-seal.json',{'graphs':28,'graph_index_sha256':digest(out/'graph-index.json'),
          'pure_inputs_sha256':digest(out/'pure-inputs.json'),'all28_world_euclidean_graphs_sealed_before_material_scalar_reference':True,
          'builder_received_mass_c0_reference_true_internal_faces_F_C_or_precomputed_g':False})
        guard('all28-sliding-graphs-sealed-before-material-and-reference')
        material=ROOT/gate['material_source']['path'];assert digest(material)==gate['material_source']['sha256'];shutil.copyfile(material,out/'original-material-reference.npz')
        with np.load(out/'original-material-reference.npz',allow_pickle=False) as zz:data={key:zz[key].copy() for key in ('mass','c0')}
        data['ids']=ids.copy();np.savez_compressed(out/'material-inputs.npz',**data)
        save(out/'source-copies.json',[{'source':str(pure.relative_to(ROOT)),'copy':'original-pure-inputs.npz','sha256':digest(pure),'phase':'pureonlybeforegeometry'},
          {'source':str(material.relative_to(ROOT)),'copy':'original-material-reference.npz','sha256':digest(material),'phase':'after28sealed,mass-c0onlyparsed'}])
        runs={};paths={};res={}
        for K in COUNTS:
            run,path=native_run(out,K,data,index,guard);runs[K]=run;paths[K]=path;save(out/f'k{K}/native-results.json',run)
            np.savez_compressed(out/f'k{K}/trajectory-states.npz',**path);rows=saved_modal_rows(out,data,labels,world,K);save(out/f'k{K}/modal-residues.json',rows)
            res[K]=[sum(row['mode_residues_absolute'])+row['constant_residue_absolute'] for row in rows]
        result,b=evaluate(data,labels,world,runs,paths,res);save_references(out,b);save(out/'results.json',result);guard('affine-shear-reference-time-and-direction-controls-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','previous_recorded_checks':483,'new_software_checks':4,'recorded_checks_total':487,
          'previous_checks_rerun':0,'registered_affine_shear_control_executed':True,'registered_affine_shear_control_passed':result['registered_affine_shear_numerical_control_passed'],
          'comparison_decision':result['comparison_decision'],'sliding_periodic_geometry_implemented':True,'original_backends_changed':False,
          'new_candidate_graphs':28,'new_candidate_transient_runs':3,'new_independent_uniformization_sequences':3,'old_candidate_reruns':0,
          'new_reference_PDE_solves':0,'new_material_definitions':0,'general_T3_admitted':False,'general_smooth_flow_admitted':False,
          'general_nonorthogonal_domains_admitted':False,'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete',
          'fluid_orc_enabled':False,'kappa_computed':False,'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':True,
          'preflight_manifest_verification_seconds':preflight,'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'resource_caps_passed':True,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4,'norm_estimator_seed':20261101,
          'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps({'decision':result['comparison_decision'],'observations':result['observations'],'time_controls':result['time_controls'],
          'direction_observables':result['direction_observables']},indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
          'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();args=load_all(out);data,labels,world,runs,paths,res=args;index=read(out/'graph-index.json')
    for K in COUNTS:
        run,path=native_run(out,K,data,index,replay=True);assert run==runs[K]
        for key,value in path.items():assert bitwise_same(value,paths[K][key]),key
        assert saved_modal_rows(out,data,labels,world,K)==read(out/f'k{K}/modal-residues.json')
    result,b=evaluate(*args);assert result==read(out/'results.json');saved=arrays(out/'analytic-references.npz')
    for key in ('weights','modes','reference','continuous_grid_symbol'):assert bitwise_same(saved[key],b[key]),key
    for K in COUNTS:assert bitwise_same(saved[f'frozen_k{K}'],b['frozen_symbols'][K])
    print('28 exact sliding-world caches: three actual and complete independent paths,halfstates,rawCSR/weights replayed bitwise; original c0 bridge,two-mode references,raw observables andunchanged H/time/direction decisions reproduced. No geometry/PDE/old candidate/ORC rerun.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
