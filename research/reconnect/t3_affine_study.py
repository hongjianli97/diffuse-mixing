"""One preregistered paired constantD/Gamma affine material benchmark, original backends."""
from pathlib import Path
from fractions import Fraction as F
import argparse,gc,json,math,platform,resource,shutil,signal,time
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save,digest
from affine_compression_inputs import MODELS,COUNTS,pure_inputs,directional_certificate
from oriented_periodic_boundary import OrientedPeriodicRectangle,build_oriented_periodic_faces
from shear_union64_study import arrays,native_run
from t3_affine_core import evaluate
from t2_d0_core import bitwise_same

SOURCE=ROOT/'runs/reconnect-t2-rotation'
PROTOCOL=ROOT/'research/reconnect/t3_affine_protocol.json'
CONTROL=ROOT/'research/reconnect/T3_AFFINE_CONTROL.md'


def inputs(out):
    d=arrays(out/'material-inputs.npz')
    with np.load(out/'original-material-reference.npz',allow_pickle=False) as z:
        for k in ('mass','c0'):assert bitwise_same(d[k],z[k])
    return d


def residues_from_saved(out,data,world,model,K):
    m,c0=data['mass'],data['c0'];root=np.sqrt(m);phi=c0-.5;qh=4*32**2*math.sin(math.pi/32)**2;rows=[]
    norm=lambda v:float(np.sqrt(m@(v*v)))
    for step in range(K):
        frame=world['models'][model][str(K)]['frames'][2*step+1];scale=F(frame['a_exact'])
        lam=qh*float(F(frame['Gamma_actual_exact'])*(scale+1/scale));action=arrays(out/model/f'k{K}/action-step{step:02d}.npz')
        A=csr_matrix((action['symmetric_data'],action['symmetric_indices'],action['symmetric_indptr']),shape=tuple(action['shape']))
        eigen=norm(A@(root*phi)/root+lam*phi);constant=norm(.5*(A@root)/root);limit=1e-10*max(1.,lam*norm(phi))
        if eigen+constant>limit:raise ArithmeticError('original modal algebra scale failed; residual cannot donate arbitrarybudget')
        rows.append({'step':step,'declared_frozen_eigenvalue':lam,'mode_residue_absolute':eigen,'constant_residue_absolute':constant,
          'original_algebra_scale':limit,'original_algebra_scale_passed':True})
    return rows


def save_references(out,bundle):
    values={}
    for model in MODELS:
        values[model+'_reference']=bundle['reference'][model];values[model+'_continuous_grid_symbol']=bundle['continuous_grid_symbol'][model]
        for K in COUNTS:
            values[f'{model}_k{K}_frozen_wire_symbol']=bundle['frozen_wire_symbol'][model,K]
            values[f'{model}_k{K}_frozen_nominal_symbol']=bundle['frozen_nominal_symbol'][model,K]
    np.savez_compressed(out/'analytic-references.npz',**values)
    save(out/'analytic-parameters.json',{k:v for k,v in bundle.items() if k not in ('reference','continuous_grid_symbol','frozen_wire_symbol','frozen_nominal_symbol')})


def main(output,tests,test_log):
    prior=read(SOURCE/'sha256.json');assert len(prior)==4015
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json');proto=read(PROTOCOL)
    assert not gate['already_executed'] and gate['N']==32 and gate['ordered_interval_counts']==[4,8,16] and gate['new_candidate_graphs']==56
    assert gate['models']==list(MODELS) and gate['main_wall_seconds_cap']==600 and gate['main_peak_RSS_MiB_cap']==1024
    assert read(tests)=={'successful':True,'checks_run':3,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,p in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),('control-at-lock.md',CONTROL),
      ('tests.json',Path(tests)),('tests.log',Path(test_log)),('tests-first-attempt.json',Path('/tmp/ltn-t3-affine-tests-attempt1.json')),
      ('tests-first-attempt.log',Path('/tmp/ltn-t3-affine-tests-attempt1.log'))]:shutil.copyfile(p,out/name)
    names=['affine_compression_inputs.py','t3_affine_core.py','t3_affine_study.py','test_t3_affine.py','oriented_periodic_boundary.py',
      'audit_oriented_world_geometry.py','periodic_voronoi_fv.py','voronoi_time_action.py','voronoi_time16_core.py','shear_union64_study.py','t1_uniform_core.py','t2_translation_core.py']
    save(out/'implementation-lock.json',{'files':{n:digest(ROOT/'research/reconnect'/n) for n in names},'protocol_sha256':digest(PROTOCOL),
      'control_note_sha256':digest(CONTROL),'fixed_before_new_scientific_inputs':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':4015,
      'parent_manifest_sha256':digest(SOURCE/'sha256.json'),'new_software_checks':3,'recorded_checks_total':480,'previous_checks_rerun':0,
      'N':32,'models':list(MODELS),'K_values':list(COUNTS),'T_exact':'1/20','new_reference_PDE_solves':0,'old_candidate_reruns':0,'new_material_definitions':0})
    started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>600 or peak>1024:raise RuntimeError(f'registered cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('registered600second main cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    try:
        pure=ROOT/gate['pure_source'];assert digest(pure)==gate['pure_source_sha256'];shutil.copyfile(pure,out/'original-pure-inputs.npz')
        z=arrays(out/'original-pure-inputs.npz');world=pure_inputs(z['n32_positions'],z['n32_ids']);ids=np.array(world['ids'],dtype=np.int64)
        save(out/'pure-inputs.json',world);save(out/'input-matching.json',{'source_pure_sha256':digest(pure),'pure_inputs_sha256':digest(out/'pure-inputs.json'),
          'mass_c0_reference_or_oldstates_loaded':False,'density_or_diffusivity_renormalized':False,'actual_constitutive_rounding_defects_saved':True})
        del z;index={};per={M:{} for M in MODELS};gc.collect()
        for model in MODELS:
            (out/model).mkdir()
            for K in COUNTS:
                (out/model/f'k{K}').mkdir()
                for step in range(K):
                    frame=world['models'][model][str(K)]['frames'][2*step+1];boundary=OrientedPeriodicRectangle.from_packet(frame['boundary'])
                    local,wp,cert=build_oriented_periodic_faces(ids,frame['world_positions_exact'],frame['rho'],frame['D_float'],boundary,guard)
                    direction=directional_certificate(local,frame,ids,32)
                    base=f'{model}/k{K}';localname=base+f'/geometry-step{step:02d}.json';worldname=base+f'/world-geometry-step{step:02d}.json'
                    save(out/localname,local);save(out/worldname,wp);save(out/(base+f'/isometry-step{step:02d}.json'),cert);save(out/(base+f'/direction-step{step:02d}.json'),direction)
                    if not direction['all_directional_conductances_exact']:raise ArithmeticError('declared direction/model geometry failed; no correction')
                    item={'N':32,'model':model,'K':K,'step':step,'frame':2*step+1,'file':localname,'sha256':digest(out/localname),
                      'world_file':worldname,'world_sha256':digest(out/worldname),'midpoint_time_saved':float(F(frame['time_exact']))}
                    index[f'{model}-k{K}-step{step:02d}']=item;per[model][f'k{K}-step{step:02d}']={**item,'file':f'k{K}/geometry-step{step:02d}.json'}
                    del local,wp,cert;gc.collect();guard(f'sealed-affine-{model}-k{K}-step{step:02d}')
            save(out/model/'graph-index.json',per[model])
        save(out/'graph-index.json',index);save(out/'coefficient-seal.json',{'graphs':56,'graph_index_sha256':digest(out/'graph-index.json'),
          'pure_inputs_sha256':digest(out/'pure-inputs.json'),'all56_model_coefficients_and_world_graphs_sealed_before_material_or_reference':True,
          'builder_received_mass_scalar_reference_internalfaces_or_F_C':False})
        guard('all56-coefficients-sealed-before-material-and-reference')
        material=ROOT/gate['material_source'];assert digest(material)==gate['material_source_sha256'];shutil.copyfile(material,out/'original-material-reference.npz')
        with np.load(out/'original-material-reference.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in ('mass','c0')}
        d['ids']=ids;np.savez_compressed(out/'material-inputs.npz',**d)
        save(out/'source-copies.json',[{'copy':'original-pure-inputs.npz','source':str(pure.relative_to(ROOT)),'sha256':digest(pure),'phase':'pureonlybeforegeometry'},
          {'copy':'original-material-reference.npz','source':str(material.relative_to(ROOT)),'sha256':digest(material),'phase':'after56sealed,mass-c0onlyparsed'}])
        runs={};paths={};residues={}
        for model in MODELS:
            for K in COUNTS:
                run,path=native_run(out/model,K,d,per[model],guard);runs[model,K]=run;paths[model,K]=path
                save(out/model/f'k{K}/native-results.json',run);np.savez_compressed(out/model/f'k{K}/trajectory-states.npz',**path)
                rows=residues_from_saved(out,d,world,model,K);save(out/model/f'k{K}/modal-residues.json',rows)
                residues[model,K]=[v['mode_residue_absolute']+v['constant_residue_absolute'] for v in rows]
        result,bundle=evaluate(d,world,runs,paths,residues);save_references(out,bundle);save(out/'results.json',result);guard('paired-affine-model-comparison-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','previous_recorded_checks':477,'new_software_checks':3,'recorded_checks_total':480,
          'previous_checks_rerun':0,'registered_affine_pair_executed':True,'T3_registered_affine_case_passed':result['T3_registered_affine_case_passed'],
          'comparison_decision':result['comparison_decision'],'new_candidate_graphs':56,'new_candidate_transient_runs':6,'new_independent_uniformization_sequences':6,
          'new_reference_PDE_solves':0,'old_candidate_reruns':0,'new_material_definitions':0,'original_geometry_resistance_time_backends_changed':False,
          'general_T3_admitted':False,'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete',
          'fluid_orc_enabled':False,'kappa_computed':False,'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':False,'tests_failed_attempts_preserved':1,'main_first_attempt_passed':True,
          'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'resource_caps_passed':True,
          'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4,'numerical_norm_estimator_seed':20261101,
          'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps({'decision':result['comparison_decision'],'final_models':{M:v[-1] for M,v in result['per_model_observations'].items()},
          'model_response':[{k:v for k,v in row.items() if k!='indicator_components_absolute'} for row in result['model_comparisons']]},indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();d=inputs(out);world=read(out/'pure-inputs.json');runs={};paths={};residues={}
    for model in MODELS:
        index=read(out/model/'graph-index.json')
        for K in COUNTS:
            run,path=native_run(out/model,K,d,index,replay=True);assert run==read(out/model/f'k{K}/native-results.json')
            saved=arrays(out/model/f'k{K}/trajectory-states.npz')
            for key,v in path.items():assert bitwise_same(v,saved[key]),key
            rows=residues_from_saved(out,d,world,model,K);assert rows==read(out/model/f'k{K}/modal-residues.json')
            runs[model,K]=run;paths[model,K]=path;residues[model,K]=[v['mode_residue_absolute']+v['constant_residue_absolute'] for v in rows]
    result,b=evaluate(d,world,runs,paths,residues);assert result==read(out/'results.json');saved=arrays(out/'analytic-references.npz')
    for M in MODELS:
        assert bitwise_same(saved[M+'_reference'],b['reference'][M]);assert bitwise_same(saved[M+'_continuous_grid_symbol'],b['continuous_grid_symbol'][M])
        for K in COUNTS:
            assert bitwise_same(saved[f'{M}_k{K}_frozen_wire_symbol'],b['frozen_wire_symbol'][M,K]);assert bitwise_same(saved[f'{M}_k{K}_frozen_nominal_symbol'],b['frozen_nominal_symbol'][M,K])
    print('All56 sealed affine modelgraphs: six actual/complete independent paths andmodalresidues replayed bitwise; distinct PDEreferences,time/input/spatial/numerical budgets andmodeldecision reproduced. No geometry,PDEsolver,oldcandidate,normalization orORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
