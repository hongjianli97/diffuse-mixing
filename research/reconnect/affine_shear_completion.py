"""Append-only reporting recovery; scientific caches and frozen implementation stay unchanged."""
from pathlib import Path
import argparse,json,platform,resource,shutil,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from affine_shear_study import load_all,saved_modal_rows
from affine_shear_core import COUNTS,evaluate
from shear_union64_study import arrays,native_run
from t2_d0_core import bitwise_same


def plain_json(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:plain_json(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [plain_json(item) for item in value]
    return value


def main(directory,first_log):
    out=Path(directory).resolve();start=time.monotonic()
    assert not (out/'results-completed.json').exists() and not (out/'execution.json').exists()
    failure=read(out/'failed-execution.json');assert failure['type']=='TypeError' and 'JSON serializable' in failure['error']
    assert failure['elapsed_seconds']<900 and failure['peak_RSS_MiB']<1024
    assert read(out/'coefficient-seal.json')['graphs']==28
    assert read(out/'geometry-audit-execution.json')['all28_independent_sliding_world_geometry_audits_passed']
    before={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    save(out/'pre-completion-files-sha256.json',before)
    save(out/'completion-lock.json',{'completion_code_sha256':digest(Path(__file__)),'purpose':'scalar serialization only; no threshold/formula or scientific cache changes',
      'frozen_core_sha256':digest(ROOT/'research/reconnect/affine_shear_core.py'),'original_implementation_lock_sha256':digest(out/'implementation-lock.json')})
    shutil.copyfile(first_log,out/'reporting-first-attempt.log')
    args=load_all(out);r,b=evaluate(*args);r=plain_json(r);save(out/'results-completed.json',r)
    saved=arrays(out/'analytic-references.npz')
    for key in ('weights','modes','reference','continuous_grid_symbol'):assert bitwise_same(saved[key],b[key]),key
    for K in COUNTS:assert bitwise_same(saved[f'frozen_k{K}'],b['frozen_symbols'][K])
    for name,h in before.items():assert digest(out/name)==h,name
    save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','previous_recorded_checks':483,'new_software_checks':4,'recorded_checks_total':487,
      'previous_checks_rerun':0,'registered_affine_shear_control_executed':True,'registered_affine_shear_control_passed':r['registered_affine_shear_numerical_control_passed'],
      'comparison_decision':r['comparison_decision'],'sliding_periodic_geometry_implemented':True,'original_backends_changed':False,
      'new_candidate_graphs':28,'new_candidate_transient_runs':3,'new_independent_uniformization_sequences':3,'old_candidate_reruns':0,
      'new_reference_PDE_solves':0,'new_material_definitions':0,'reporting_failure_preserved':True,'completed_results_file':'results-completed.json',
      'general_T3_admitted':False,'general_smooth_flow_admitted':False,'general_nonorthogonal_domains_admitted':False,'finite_node_candidate_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False,'pdf_generated':False,'software_release_created':False,'published':False})
    save(out/'execution.json',{'tests_exit_code':0,'original_study_exit_code':1,'completion_exit_code':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':False,
      'scientific_caches_complete_before_reporting_failure':True,'failure_reason':'NumPy bool is not JSON serializable; plain scalar cast in separate append-only completion stage',
      'original_failure_log_and_incomplete_JSON_retained':True,'elapsed_main_to_reporting_failure_seconds':failure['elapsed_seconds'],'peak_main_RSS_MiB':failure['peak_RSS_MiB'],
      'elapsed_completion_seconds':time.monotonic()-start,'peak_completion_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      'all_pre_completion_files_unchanged':True,'no_scientific_path_rerun_during_completion':True,'criteria_and_thresholds_unchanged':True,
      'python':platform.python_version(),'threads':4,'norm_estimator_seed':20261101,'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
    print(json.dumps({'decision':r['comparison_decision'],'observations':r['observations'],'time_controls':r['time_controls'],
      'direction_observables':r['direction_observables'],'pre_completion_files_verified':len(before)},indent=2),flush=True)


def replay(directory):
    out=Path(directory).resolve();args=load_all(out);data,labels,world,runs,paths,res=args;index=read(out/'graph-index.json')
    for K in COUNTS:
        run,path=native_run(out,K,data,index,replay=True);assert run==runs[K]
        for key,value in path.items():assert bitwise_same(value,paths[K][key]),key
        assert saved_modal_rows(out,data,labels,world,K)==read(out/f'k{K}/modal-residues.json')
    r,b=evaluate(*args);assert plain_json(r)==read(out/'results-completed.json');saved=arrays(out/'analytic-references.npz')
    for key in ('weights','modes','reference','continuous_grid_symbol'):assert bitwise_same(saved[key],b[key]),key
    for K in COUNTS:assert bitwise_same(saved[f'frozen_k{K}'],b['frozen_symbols'][K])
    for name,h in read(out/'pre-completion-files-sha256.json').items():assert digest(out/name)==h,name
    print('Three actual/full independent affine-shear paths and all halfstates/rawCSR/weights replayed bitwise from28 caches; references,unmodified c0 bridge,raw observables andsame numerical decisions reproduced; original serialization failure/incomplete JSON remain byte-identical. No geometry/PDE/old-candidate/ORC rerun.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--first-log');p.add_argument('--replay',action='store_true');a=p.parse_args()
    if a.replay:replay(a.directory)
    elif a.first_log:main(a.directory,a.first_log)
    else:p.error('--first-log or --replay required')
