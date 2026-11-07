"""Append-only rational-area diagnostic reporting recovery; scientific source/data unchanged."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json,resource,shutil,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from becker_point_study import load_all
from becker_point_core import Becker,evaluate
from affine_shear_completion import plain_json


def density_diagnostics(out,flow,labels,col,world):
    rows=[];m=col['mass'];bounds=labels
    for i in range(16):
        frame=world['cases']['16']['frames'][2*i+1];t=float(F(frame['time_exact']));faces=flow.x_mu(bounds,t);volume=np.diff(faces);rhoavg=m/volume
        rp=np.array(frame['rho']).reshape(8,32)[0];packet=read(out/f'k16/geometry-step{i:02d}.json');vor=np.array([float(F(value)) for value in packet['cell_areas_exact']]).reshape(8,32)[0]*8
        rows.append({'time':t,'maximum_point_vs_true_material_average_density_relative_difference':float(np.max(abs(rp/rhoavg-1))),
          'maximum_point_Voronoi_vs_true_material_volume_relative_difference':float(np.max(abs(vor/volume-1))),
          'maximum_mass_over_point_rho_vs_true_volume_relative_difference':float(np.max(abs((m/rp)/volume-1))),
          'true_faces_used_by_builder':False,'true_faces_used_only_after_sealing_for_diagnostics':True})
    return rows


def main(directory,first_log):
    out=Path(directory).resolve();started=time.monotonic();assert not (out/'execution.json').exists()
    failure=read(out/'failed-execution.json');assert failure['type']=='ValueError' and 'could not convert string to float' in failure['error']
    assert failure['elapsed_seconds']<900 and failure['peak_RSS_MiB']<1024 and read(out/'coefficient-seal.json')['graphs']==28
    before={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()};save(out/'pre-completion-files-sha256.json',before)
    shutil.copyfile(first_log,out/'diagnostics-first-attempt.log')
    save(out/'completion-lock.json',{'completion_code_sha256':digest(Path(__file__)),'frozen_study_sha256':digest(ROOT/'research/reconnect/becker_point_study.py'),
      'frozen_core_sha256':digest(ROOT/'research/reconnect/becker_point_core.py'),'purpose':'rational-area parsing in diagnostic output only; no propagation/geometry/criterion changes'})
    args=load_all(out);d,col,runs,paths,rm=args;r=plain_json(evaluate(*args));assert r==read(out/'results.json')
    labels=np.load(out/'source-mass-labels.npy',allow_pickle=False);world=read(out/'pure-inputs.json')
    save(out/'density-volume-diagnostics.json',density_diagnostics(out,Becker(delta=.08),labels,col,world))
    save(out/'stage3-status.json',{'updated':'2026-11-04','timezone':'Asia/Shanghai','recorded_checks_total':500,'new_software_checks':4,'previous_checks_rerun':0,
      'registered_moving_Becker_discriminator_executed':True,'registered_moving_Becker_oracle_discriminator_passed':r['registered_moving_Becker_oracle_discriminator_passed'],
      'comparison_decision':r['comparison_decision'],'new_candidate_graphs':28,'new_candidate_transient_runs':3,'new_complete_independent_uniformization_sequences':3,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'original_periodic_geometry_exponential_uniformization_and_H_backends_changed':False,
      'optional_outer_boundary_and_duration_interfaces_added':True,'diagnostic_parser_failure_preserved':True,'append_only_reporting_completion':True,
      'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False,
      'environment_or_software_version_published':False})
    for rel,h in before.items():assert digest(out/rel)==h,rel
    save(out/'execution.json',{'tests_exit_code':0,'original_study_exit_code':1,'completion_exit_code':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':False,
      'scientific_geometry_paths_and_results_complete_before_diagnostic_parser_failure':True,'all_pre_completion_files_unchanged':True,
      'scientific_paths_rerun_during_completion':0,'failure_reason':'float(rational string) at volume diagnostic output; newparser uses float(Fraction(value)) only',
      'elapsed_main_to_diagnostic_failure_seconds':failure['elapsed_seconds'],'peak_main_RSS_MiB':failure['peak_RSS_MiB'],
      'elapsed_completion_seconds':time.monotonic()-started,'peak_completion_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      'network_used':False,'environment_configuration_changed':False,'published':False})
    print(json.dumps({'decision':r['comparison_decision'],'pre_completion_files_preserved':len(before),'time_control':r['time_control'],'scientific_paths_rerun':0},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--first-log',required=True);a=p.parse_args();main(a.directory,a.first_log)


