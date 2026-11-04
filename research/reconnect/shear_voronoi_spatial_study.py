"""One registered N256 FV shear refinement observed on original N128 materials."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,math,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save,digest
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from shear_voronoi_fv_study import record,verify_probe_cells
from shear_voronoi_time_study import source_data,propagate,relative
from voronoi_time_action import metrics
from voronoi_spatial_projection import validate_record,project_path

SOURCE=ROOT/'runs/reconnect-shear-voronoi-time'
PROTOCOL=ROOT/'research/reconnect/voronoi_spatial_protocol.json'
CONTROL=ROOT/'research/reconnect/SHEAR_VORONOI_SPATIAL_CONTROL.md'
SEED=20261101


def compare(data,cases,states,coarse_controls):
    child_sum,rounding,mass_audit=validate_record(data)
    mp,cp0=data['parent_mass'],data['parent_c0'];M=float(mp.sum())
    E0=metrics(mp,cp0,cp0)['variance'];sigma=math.sqrt(E0)
    norm=lambda v:relative(mp,v,sigma)
    H=lambda c:metrics(mp,c,cp0)['H']
    p_factor=math.sqrt(float(np.max(child_sum/mp)))
    projected={}
    for K in (4,8):
        projected[f'k{K}_parent_states']=project_path(states[f'k{K}_states'],data)
        projected[f'k{K}_parent_independent_states']=project_path(states[f'k{K}_independent_states'],data)
    projected['projected_reference_states']=project_path(data['reference_states'],data)
    projected['reference_parent_difference']=projected['projected_reference_states']-data['parent_reference_states']
    projected['reference_projection_replication_difference']=projected['reference_parent_difference']-data['reference_parent_projection_difference']
    initial=project_path(np.array([data['c0']]),data)[0]
    projected['initial_bridge_vectors']=np.array([data['parent_reference_states'][0]-cp0,initial-cp0,
        projected['projected_reference_states'][0]-initial])
    bridge=sum(norm(v) for v in projected['initial_bridge_vectors'])
    H_bridge=abs(H(data['parent_reference_states'][0])-H(cp0))+abs(H(initial)-H(cp0))+abs(H(projected['projected_reference_states'][0])-H(initial))
    h_machine=lambda a:(2*math.sqrt(M)*max(1.,float(np.max(abs(cp0))))*a+a*a)/E0
    rows=[]
    for k,t in enumerate(data['parent_reference_times']):
        i4,i8=2*k,4*k
        fine=projected['k8_parent_states'][i8];fine4=projected['k4_parent_states'][i4]
        coarse=data['parent_FV_k8_states'][i8];coarse4=data['parent_FV_k4_states'][i4]
        ref=data['parent_reference_states'][k]
        solver_abs=observed=observed_H=0.
        for K,idx in ((4,i4),(8,i8)):
            if idx:
                f=cases[str(K)]['steps'][idx-1];c=coarse_controls['cases'][str(K)]['steps'][idx-1]
                solver_abs+=p_factor*(f['cumulative_machine_indicator_absolute']+f['cumulative_tail_indicator_absolute'])
                solver_abs+=c['cumulative_machine_indicator_absolute']+c['cumulative_tail_indicator_absolute']
                a=projected[f'k{K}_parent_states'][idx];b=projected[f'k{K}_parent_independent_states'][idx]
                observed+=norm(a-b)+c['primary_control_mass_norm_difference']/sigma
                observed_H+=abs(H(a)-H(b))+abs(c['state_metrics']['H']-c['independent_metrics']['H'])
        components={'coarse_4_8':norm(coarse4-coarse),'fine_4_8_common':norm(fine4-fine),
            'twice_parent_reference_space':2*norm(data['parent_reference_space_difference'][k]),
            'twice_parent_reference_time':2*norm(data['parent_reference_time_difference'][k]),
            'reference_parent_projection':norm(projected['reference_parent_difference'][k]),
            'reference_projection_replication':norm(projected['reference_projection_replication_difference'][k]),
            'initial_bridges':bridge,'solver_machine_and_tails':solver_abs/sigma,'observed_solvers':observed}
        indicator=sum(components.values());ec=norm(coarse-ref);ef=norm(fine-ref);gain=ec-ef
        classify=lambda g,b:'fine_better_resolvably' if g>b else ('fine_worse_resolvably' if g < -b else 'comparison_inconclusive')
        coarse_time_ref=ref+data['parent_reference_time_difference'][k]
        H_space=abs(H(coarse_time_ref+data['parent_reference_space_difference'][k])-H(coarse_time_ref))
        H_time=abs(H(coarse_time_ref)-H(ref))
        H_parts={'coarse_4_8':abs(H(coarse4)-H(coarse)),'fine_4_8_common':abs(H(fine4)-H(fine)),
            'twice_reference_space':2*H_space,'twice_reference_time':2*H_time,
            'reference_parent_projection':abs(H(projected['projected_reference_states'][k])-H(ref)),
            'initial_H_bridges':H_bridge,'observed_solver_H':observed_H,'solver_machine_to_H':h_machine(solver_abs)}
        H_indicator=sum(H_parts.values());H_gain=abs(H(coarse)-H(ref))-abs(H(fine)-H(ref))
        row={'time':float(t),'coarse_error_common':ec,'fine_error_common':ef,'coarse_over_fine_error_ratio':None if ef==0 else ec/ef,
            'absolute_error_gain':gain,'state_indicator_components':components,'combined_state_indicator':indicator,
            'state_decision':classify(gain,indicator),'coarse_H':H(coarse),'fine_H_common':H(fine),'reference_H':H(ref),
            'coarse_H_signed_error':H(coarse)-H(ref),'fine_H_signed_error':H(fine)-H(ref),
            'H_absolute_error_gain':H_gain,'H_indicator_components':H_parts,'combined_H_indicator':H_indicator,
            'H_decision':classify(H_gain,H_indicator),
            'native_fine_error':cases['8']['observations'][k]['relative_concentration_error'],
            'native_fine_H':cases['8']['observations'][k]['candidate_metrics']['H'],
            'native_fine_reference_H':cases['8']['observations'][k]['reference_metrics']['H'],
            'common_raw_mass_change_from_original_parent_c0':float(mp@(fine-cp0)),
            'projected_minimum':float(fine.min()),'projected_maximum':float(fine.max())}
        assert ec==coarse_controls['cases']['8']['observations'][k]['relative_concentration_error']
        rows.append(row)
    both_state=all(r['state_decision']=='fine_better_resolvably' for r in rows[1:])
    both_H=all(r['H_decision']=='fine_better_resolvably' for r in rows[1:])
    decision='spatial_improvement_resolved_at_both_checkpoints' if both_state and both_H else (
        'adverse_at_a_checkpoint' if any(r['state_decision']=='fine_worse_resolvably' or r['H_decision']=='fine_worse_resolvably' for r in rows[1:]) else 'spatial_comparison_inconclusive')
    result={'scope':'one same smooth flow,N128/N256 common original parent material means',
        'parent_initial_fluctuation_norm':sigma,'projection_solver_norm_factor':p_factor,'parent_mass_audit':mass_audit,
        'saved_child_mass_sum_matches':bool(np.array_equal(child_sum,data['child_mass_sum'])),
        'saved_parent_mass_rounding_matches':bool(np.array_equal(rounding,data['parent_mass_rounding_relative'])),
        'observations':rows,'state_improvement_resolved_at_both_checkpoints':both_state,'H_improvement_resolved_at_both_checkpoints':both_H,
        'comparison_decision':decision,'automatic_further_simulations_authorized_by_this_gate':False,
        'time_and_reference_differences_are_certified_bounds':False,'two_levels_prove_asymptotic_order':False,
        'coarse_candidate_or_reference_rerun':False,'general_smooth_flow_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    return result,projected


def main(output,tests,test_log):
    prior=read(SOURCE/'sha256.json')
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    assert len(prior)==1905
    spec=read(SOURCE/'next-gate-spec.json');protocol=read(PROTOCOL)
    assert not spec['already_executed'] and (spec['N_fine'],spec['N_parent'],spec['new_midpoint_graphs'])==(256,128,12)
    assert spec['ordered_interval_counts']==[4,8] and spec['main_wall_seconds_cap']==2400 and spec['main_peak_RSS_MiB_cap']==2048
    assert read(tests)=={'successful':True,'checks_run':4,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,p in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),
        ('control-at-lock.md',CONTROL),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(p,out/name)
    locked=['periodic_voronoi_fv.py','voronoi_time_action.py','shear_voronoi_time_study.py','variable_parent_observation.py',
            'voronoi_spatial_projection.py','test_voronoi_spatial_projection.py','shear_voronoi_spatial_study.py']
    save(out/'implementation-lock.json',{'files':{n:digest(ROOT/'research/reconnect'/n) for n in locked},
        'protocol_sha256':digest(PROTOCOL),'control_note_sha256':digest(CONTROL),'all_options_fixed_before_new_geometry':True})
    save(out/'config.json',{'source_manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'source_manifest_sha256':digest(SOURCE/'sha256.json'),
        'source_manifest_entries_verified':len(prior),'source_gate_sha256':digest(SOURCE/'next-gate-spec.json'),
        'new_software_checks':4,'recorded_checks_total':449,'previous_checks_rerun':0,'one_original_flow':True,'no_parameter_matrix':True})
    assert digest(ROOT/spec['builder_input_packet'])==spec['builder_packet_sha256']
    assert digest(ROOT/spec['comparison_packet'])==spec['comparison_packet_sha256']
    shutil.copyfile(ROOT/spec['builder_input_packet'],out/'builder-inputs.npz')
    b=source_data(out/'builder-inputs.npz');index={};started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>2400 or rss>2048:raise RuntimeError(f'fixed resource cap exceeded:{elapsed:.3f}s/{rss:.2f}MiB')
        if not total or k%8192==0 or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{rss:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('registered2400second main cap reached')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(2400)
    try:
        for K in (4,8):
            for step in range(K):
                frame=2*step+1
                graph=build_periodic_faces(b['ids'],b[f'k{K}_positions'][frame],b[f'k{K}_rho'][frame],.02,(1,1),1,guard)
                probes=verify_probe_cells(graph,protocol['fine_geometry_probe_ids']);packet=coefficient_packet(graph)
                name=f'geometry-k{K}-step{step:02d}.json';save(out/name,packet)
                index[f'k{K}-step{step:02d}']={'file':name,'sha256':digest(out/name),'K':K,'step':step,'frame':frame,
                    'midpoint_time_saved':float(b[f'k{K}_midpoint_time'][step]),'positions':record(b[f'k{K}_positions'][frame]),
                    'rho':record(b[f'k{K}_rho'][frame]),'geometry_diagnostics':graph.diagnostics,'independent_fixed_probes':probes}
                del packet,graph;gc.collect();guard(f'sealed-geometry-k{K}-step{step:02d}')
        save(out/'graph-index.json',index)
        save(out/'coefficient-seal.json',{'graph_index_sha256':digest(out/'graph-index.json'),'graphs':12,
            'builder_packet_sha256':digest(out/'builder-inputs.npz'),'all12_coefficients_sealed_before_mass_c0_reference_loading':True,
            'builder_received_mass_state_reference_true_faces_or_F_C':False,'builder_array_records':{k:record(v) for k,v in b.items()}})
        del b;gc.collect();guard('all12coefficients-sealed-before-state')
        shutil.copyfile(ROOT/spec['comparison_packet'],out/'original-comparison.npz');data=source_data(out/'original-comparison.npz')
        validate_record(data);fine=dict(data);fine['reference_times']=data['parent_reference_times']
        save(out/'input-provenance.json',{'comparison_packet_sha256':digest(out/'original-comparison.npz'),
            'original_array_records':{k:record(v) for k,v in data.items()},'fine_reference_times_is_only_an_in_memory_alias':True})
        np.random.seed(SEED)
        cases,arrays,sigma=propagate(out,fine,index,guard)
        save(out/'native-results.json',{'cases':cases,'native_initial_mass_fluctuation_norm':sigma,'scope':'native N256 cells only'})
        shutil.copyfile(SOURCE/'results.json',out/'source-coarse-results.json')
        coarse_controls=read(out/'source-coarse-results.json');result,projected=compare(data,cases,arrays,coarse_controls)
        save(out/'results.json',result);np.savez_compressed(out/'trajectory-states.npz',**arrays,**projected)
        guard('spatial-observation-and-comparison-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-01','new_software_checks':4,'previous_recorded_checks':445,'recorded_checks_total':449,
            'previous_checks_rerun':0,'new_candidate_graphs':12,'new_candidate_transient_runs':2,'independent_uniformization_sequences':2,
            'new_physical_flow_definitions':0,'new_PDE_reference_solves':0,'new_kernel_builds':0,'coarse_candidate_rerun':False,
            'registered_N256_spatial_gate_executed':True,'comparison_decision':result['comparison_decision'],
            'stop_after_registered_case':True,'shock_extension_deferred':True,'general_smooth_flow_admitted':False,
            'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
            'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':True,
            'scientific_main_first_attempt_passed':True,'elapsed_main_seconds':time.monotonic()-started,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'resource_caps_passed':True,
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4,
            'norm_estimator_seed':SEED,'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'error_type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_retained':True,'no_threshold_or_state_correction':True})
        raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();data=source_data(out/'original-comparison.npz');fine=dict(data)
    fine['reference_times']=data['parent_reference_times'];index=read(out/'graph-index.json')
    np.random.seed(SEED);cases,arrays,sigma=propagate(out,fine,index,replay=True)
    assert {'cases':cases,'native_initial_mass_fluctuation_norm':sigma,'scope':'native N256 cells only'}==read(out/'native-results.json')
    result,projected=compare(data,cases,arrays,read(out/'source-coarse-results.json'))
    assert result==read(out/'results.json')
    with np.load(out/'trajectory-states.npz',allow_pickle=False) as z:
        assert set(z.files)==set(arrays)|set(projected)
        for k,v in {**arrays,**projected}.items():assert np.array_equal(z[k],v),k
    print('Same12 sealed N256 graphs: chronological4/8 exponential and independent uniformization, original parent-mass projection and fixed spatial comparison replay; no geometry/PDE/coarse rerun or ORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
