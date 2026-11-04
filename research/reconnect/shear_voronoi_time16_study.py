"""One registered16-interval check on the original two smooth-shear material grids."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,math,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save,digest
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from shear_voronoi_fv_study import record,verify_probe_cells
from shear_voronoi_time_study import source_data,relative
from voronoi_spatial_projection import validate_record,project_path
from voronoi_time_action import operators,metrics
from voronoi_time16_core import sample_points,ordered_path

SOURCE=ROOT/'runs/reconnect-shear-voronoi-spatial'
PROTOCOL=ROOT/'research/reconnect/voronoi_time16_protocol.json'
CONTROL=ROOT/'research/reconnect/SHEAR_VORONOI_TIME16_CONTROL.md'
SEED=20261101
LABEL_SOURCES={128:'runs/reconnect-shear-frozen/shear-frozen.npz',256:'runs/reconnect-shear-transient-spatial/shear-transient-spatial.npz'}
OLD_POINT_SOURCES={128:'runs/reconnect-shear-voronoi-time/builder-inputs.npz',256:'runs/reconnect-shear-voronoi-spatial/builder-inputs.npz'}


def grid_data(d,N):
    p='parent_' if N==128 else ''
    return {'mass':d[p+'mass'],'c0':d[p+'c0'],'ids':d[p+'ids'],
        'reference_states':d[p+'reference_states'],'reference_times':d['parent_reference_times']}


def native_run(out,N,data,index,guard=lambda *args:None,replay=False):
    folder=out/f'n{N}';m=data['mass'];c0=data['c0']
    def load(step):
        item=index[f'n{N}-step{step:02d}'];p=out/item['file'];assert digest(p)==item['sha256']
        packet=read(p);op=operators(m,np.array(packet['edges'],dtype=np.int64),map(F,packet['conductances_exact']))
        del packet;gc.collect();return op
    def persist(step,op,weights):
        arrays={'symmetric_data':op.symmetric_generator.data,'symmetric_indices':op.symmetric_generator.indices,
            'symmetric_indptr':op.symmetric_generator.indptr,'uniformization_data':op.uniformization_matrix.data,
            'uniformization_indices':op.uniformization_matrix.indices,'uniformization_indptr':op.uniformization_matrix.indptr,
            'shape':np.array(op.symmetric_generator.shape),'edges':op.edges,'g_float':op.conductance,'weights':weights}
        p=folder/f'action-step{step:02d}.npz'
        if replay:
            with np.load(p,allow_pickle=False) as z:
                for k,v in arrays.items():assert np.array_equal(v,z[k]),k
        else:np.savez_compressed(p,**arrays)
    np.random.seed(SEED)
    run,states=ordered_path(m,c0,load,16,persist,lambda stage,k,total:guard(f'n{N}-{stage}',k,total))
    sigma=run['initial_mass_fluctuation_norm'];obs=[]
    for k,t in enumerate(data['reference_times']):
        c=states['states'][8*k];ref=data['reference_states'][k]
        obs.append({'time':float(t),'relative_concentration_error':relative(m,c-ref,sigma),
            'candidate_metrics':metrics(m,c,c0),'reference_metrics':metrics(m,ref,c0)})
    run['observations']=obs
    for step,row in enumerate(run['steps']):
        item=index[f'n{N}-step{step:02d}'];row['coefficient_cache_sha256']=item['sha256'];row['midpoint_time_saved']=item['midpoint_time_saved']
    return run,states


def compare(d,runs,states,old_fine,old_coarse,old_states):
    child_sum,rounding,mass_audit=validate_record(d);mp,cp0=d['parent_mass'],d['parent_c0']
    E0=metrics(mp,cp0,cp0)['variance'];sigma=math.sqrt(E0);M=float(mp.sum())
    norm=lambda v:relative(mp,v,sigma);H=lambda c:metrics(mp,c,cp0)['H']
    factor=math.sqrt(float(np.max(child_sum/mp)))
    projected={'fine_parent_states16':project_path(states['256']['states'],d),
        'fine_parent_independent_states16':project_path(states['256']['independent_states'],d),
        'projected_reference_states':project_path(d['reference_states'],d)}
    projected['reference_parent_difference']=projected['projected_reference_states']-d['parent_reference_states']
    projected['reference_projection_replication_difference']=projected['reference_parent_difference']-d['reference_parent_projection_difference']
    initial=project_path(np.array([d['c0']]),d)[0]
    projected['initial_bridge_vectors']=np.array([d['parent_reference_states'][0]-cp0,initial-cp0,projected['projected_reference_states'][0]-initial])
    bridge=sum(norm(v) for v in projected['initial_bridge_vectors'])
    H_bridge=abs(H(d['parent_reference_states'][0])-H(cp0))+abs(H(initial)-H(cp0))+abs(H(projected['projected_reference_states'][0])-H(initial))
    h_machine=lambda a:(2*math.sqrt(M)*max(1.,float(np.max(abs(cp0))))*a+a*a)/E0
    rows=[];time_rows=[]
    for k,t in enumerate(d['parent_reference_times']):
        i4,i8,i16=2*k,4*k,8*k
        coarse=states['128']['states'][i16];coarse8=d['parent_FV_k8_states'][i8];coarse4=d['parent_FV_k4_states'][i4]
        fine=projected['fine_parent_states16'][i16];fine8=old_states['k8_parent_states'][i8];fine4=old_states['k4_parent_states'][i4]
        ref=d['parent_reference_states'][k]
        solver_abs=observed=observed_H=0.
        if k:
            for native,fact,N,prior in [(False,1.,'128',old_coarse),(True,factor,'256',old_fine)]:
                old=prior['cases']['8']['steps'][i8-1];new=runs[N]['steps'][i16-1]
                solver_abs+=fact*(old['cumulative_machine_indicator_absolute']+old['cumulative_tail_indicator_absolute'])
                solver_abs+=fact*(new['cumulative_machine_indicator_absolute']+new['cumulative_tail_indicator_absolute'])
                if native:
                    a8=fine8;b8=old_states['k8_parent_independent_states'][i8]
                    a16=fine;b16=projected['fine_parent_independent_states16'][i16]
                    observed+=norm(a8-b8)+norm(a16-b16)
                    observed_H+=abs(H(a8)-H(b8))+abs(H(a16)-H(b16))
                else:
                    observed+=old['primary_control_mass_norm_difference']/sigma+new['primary_control_mass_norm_difference']/sigma
                    observed_H+=abs(old['state_metrics']['H']-old['independent_metrics']['H'])+abs(new['state_metrics']['H']-new['independent_metrics']['H'])
        parts={'coarse_8_16':norm(coarse8-coarse),'fine_8_16_common':norm(fine8-fine),
            'twice_parent_reference_space':2*norm(d['parent_reference_space_difference'][k]),
            'twice_parent_reference_time':2*norm(d['parent_reference_time_difference'][k]),
            'reference_parent_projection':norm(projected['reference_parent_difference'][k]),
            'reference_projection_replication':norm(projected['reference_projection_replication_difference'][k]),
            'initial_bridges':bridge,'solver_machine_and_tails':solver_abs/sigma,'observed_solvers':observed}
        budget=sum(parts.values());ec=norm(coarse-ref);ef=norm(fine-ref);gain=ec-ef
        classify=lambda g,b:'fine_better_resolvably' if g>b else ('fine_worse_resolvably' if g < -b else 'comparison_inconclusive')
        coarse_time=ref+d['parent_reference_time_difference'][k]
        hs=abs(H(coarse_time+d['parent_reference_space_difference'][k])-H(coarse_time));ht=abs(H(coarse_time)-H(ref))
        H_parts={'coarse_8_16':abs(H(coarse8)-H(coarse)),'fine_8_16_common':abs(H(fine8)-H(fine)),
            'twice_reference_space':2*hs,'twice_reference_time':2*ht,
            'reference_parent_projection':abs(H(projected['projected_reference_states'][k])-H(ref)),
            'initial_H_bridges':H_bridge,'observed_solver_H':observed_H,'solver_machine_to_H':h_machine(solver_abs)}
        H_budget=sum(H_parts.values());H_gain=abs(H(coarse)-H(ref))-abs(H(fine)-H(ref))
        rows.append({'time':float(t),'coarse16_error_common':ec,'fine16_error_common':ef,'coarse_over_fine_error_ratio':None if ef==0 else ec/ef,
            'absolute_error_gain':gain,'state_indicator_components':parts,'combined_state_indicator':budget,'state_decision':classify(gain,budget),
            'coarse16_H':H(coarse),'fine16_H_common':H(fine),'reference_H':H(ref),'coarse16_H_signed_error':H(coarse)-H(ref),
            'fine16_H_signed_error':H(fine)-H(ref),'H_absolute_error_gain':H_gain,'H_indicator_components':H_parts,
            'combined_H_indicator':H_budget,'H_decision':classify(H_gain,H_budget),
            'native_fine16_error':runs['256']['observations'][k]['relative_concentration_error'],
            'native_fine16_H':runs['256']['observations'][k]['candidate_metrics']['H'],
            'fine_common_raw_mass_change_from_parent_c0':float(mp@(fine-cp0))})
        cr48=norm(coarse4-coarse8);cr816=norm(coarse8-coarse);fr48=norm(fine4-fine8);fr816=norm(fine8-fine)
        time_rows.append({'time':float(t),'coarse4_8':cr48,'coarse8_16':cr816,'fine4_8_common':fr48,'fine8_16_common':fr816,
            'coarse_difference_ratio':None if cr816==0 else cr48/cr816,'fine_difference_ratio':None if fr816==0 else fr48/fr816,
            'ratios_certify_order_or_total_error':False})
    for N in (128,256):
        if N==128:a4=d['parent_FV_k4_states'][[0,2,4]];a8=d['parent_FV_k8_states'][[0,4,8]];a16=states['128']['states'][[0,8,16]]
        else:a4=old_states['k4_parent_states'][[0,2,4]];a8=old_states['k8_parent_states'][[0,4,8]];a16=projected['fine_parent_states16'][[0,8,16]]
        projected[f'n{N}_time4_8_difference_common']=a4-a8;projected[f'n{N}_time8_16_difference_common']=a8-a16
    both_state=all(r['state_decision']=='fine_better_resolvably' for r in rows[1:]);both_H=all(r['H_decision']=='fine_better_resolvably' for r in rows[1:])
    decision='spatial_improvement_resolved_after_time_control' if both_state and both_H else ('adverse_at_a_checkpoint' if any(r['state_decision']=='fine_worse_resolvably' or r['H_decision']=='fine_worse_resolvably' for r in rows[1:]) else 'spatial_comparison_still_inconclusive')
    return {'parent_initial_fluctuation_norm':sigma,'parent_mass_audit':mass_audit,'projection_solver_norm_factor_raw_float':factor,
        'observations':rows,'time_difference_controls':time_rows,'comparison_decision':decision,
        'state_improvement_resolved_at_both_checkpoints':both_state,'H_improvement_resolved_at_both_checkpoints':both_H,
        'original4_8_inconclusive_result_unchanged':True,'time_indicators_are_certified_bounds':False,'Richardson_or_fitted_state_used':False,
        'two_spatial_levels_prove_general_order':False,'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False},projected


def main(output,tests,test_log):
    prior=read(SOURCE/'sha256.json')
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    assert len(prior)==1993;spec=read(SOURCE/'next-gate-spec.json');protocol=read(PROTOCOL)
    assert not spec['already_executed'] and spec['new_ordered_interval_counts']==[16] and spec['new_midpoint_graphs']==32
    assert spec['main_wall_seconds_cap']==4000 and spec['main_peak_RSS_MiB_cap']==2048
    assert read(tests)=={'successful':True,'checks_run':4,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,p in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),('control-at-lock.md',CONTROL),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(p,out/name)
    locked=['periodic_voronoi_fv.py','voronoi_time_action.py','variable_parent_observation.py','voronoi_spatial_projection.py',
        'shear_probe.py','voronoi_time16_core.py','test_voronoi_time16.py','shear_voronoi_time16_study.py']
    save(out/'implementation-lock.json',{'files':{n:digest(ROOT/'research/reconnect'/n) for n in locked},'protocol_sha256':digest(PROTOCOL),
        'control_note_sha256':digest(CONTROL),'fixed_before_new_time_input_validation_and_geometry':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':1993,
        'parent_manifest_sha256':digest(SOURCE/'sha256.json'),'source_gate_sha256':digest(SOURCE/'next-gate-spec.json'),
        'new_software_checks':4,'recorded_checks_total':453,'previous_checks_rerun':0,'new_reference_PDE_solves':0,'new_spatial_levels':0})
    started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>4000 or peak>2048:raise RuntimeError(f'registered main resource cap exceeded:{elapsed:.3f}s/{peak:.2f}MiB')
        if not total or k%8192==0 or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('registered4000second main cap reached')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(4000)
    try:
        b={};source_arrays={};matches={}
        for N in (128,256):
            p=ROOT/LABEL_SOURCES[N];assert digest(p)==prior[LABEL_SOURCES[N]]
            with np.load(p,allow_pickle=False) as z:labels=z['material_label_centers'].copy();ids=z['ids'].copy();rho=z['rho0_point'].copy()
            generated=sample_points(labels,ids,rho,np.linspace(0.,.05,33));old_path=ROOT/OLD_POINT_SOURCES[N]
            assert digest(old_path)==prior[OLD_POINT_SOURCES[N]]
            checks=[]
            with np.load(old_path,allow_pickle=False) as z:
                assert record(z['ids'])==record(ids)
                for K in (4,8):
                    old_times=z[f'k{K}_time'];repeat=sample_points(labels,ids,rho,old_times);stride=16//K
                    for key in ['time','positions','rho']:
                        assert record(repeat[key])==record(z[f'k{K}_{key}'])==record(generated[key][::stride])
                    checks.append({'old_intervals':K,'all_time_position_density_bytes_match':True})
            for key,v in generated.items():b[f'n{N}_{key}']=v
            b[f'n{N}_boundary_time']=generated['time'][::2];b[f'n{N}_midpoint_time']=generated['time'][1::2]
            for key,v in [('labels',labels),('ids',ids),('rho0_point',rho)]:source_arrays[f'n{N}_{key}']=v
            matches[str(N)]={'label_source':LABEL_SOURCES[N],'label_source_sha256':digest(p),'old_point_source':OLD_POINT_SOURCES[N],
                'old_point_source_sha256':digest(old_path),'checks':checks,'new_array_records':{k:record(v) for k,v in generated.items()}}
            (out/f'n{N}').mkdir();del generated,repeat,labels,ids,rho;gc.collect()
        np.savez_compressed(out/'builder-inputs.npz',**b);np.savez_compressed(out/'trajectory-source.npz',**source_arrays)
        save(out/'input-matching.json',{'grids':matches,'all_old4_8_records_and_new_nested_subsets_match_by_bytes':True,
            'mass_c0_reference_or_derivative_arrays_loaded':False,'builder_packet_sha256':digest(out/'builder-inputs.npz')})
        del source_arrays;index={};guard('new-point-inputs-validated-before-geometry')
        for N in (128,256):
            for step in range(16):
                frame=2*step+1
                graph=build_periodic_faces(b[f'n{N}_ids'],b[f'n{N}_positions'][frame],b[f'n{N}_rho'][frame],.02,(1,1),1,guard)
                probes=verify_probe_cells(graph,protocol['fixed_probe_ids'][str(N)]);packet=coefficient_packet(graph)
                name=f'n{N}/geometry-step{step:02d}.json';save(out/name,packet)
                index[f'n{N}-step{step:02d}']={'N':N,'K':16,'step':step,'frame':frame,'file':name,'sha256':digest(out/name),
                    'midpoint_time_saved':float(b[f'n{N}_midpoint_time'][step]),'geometry_diagnostics':graph.diagnostics,'independent_fixed_probes':probes}
                del graph,packet;gc.collect();guard(f'sealed-geometry-n{N}-step{step:02d}')
        save(out/'graph-index.json',index);save(out/'coefficient-seal.json',{'graphs':32,'graph_index_sha256':digest(out/'graph-index.json'),
            'builder_packet_sha256':digest(out/'builder-inputs.npz'),'all32_coefficients_sealed_before_mass_c0_reference_loading':True,
            'builder_received_mass_c0_reference_true_faces_or_F_C':False})
        del b;gc.collect();guard('all32coefficients-sealed-before-state')
        for name,p in [('original-comparison.npz',SOURCE/'original-comparison.npz'),('source-spatial-states.npz',SOURCE/'trajectory-states.npz'),
            ('source-coarse-results.json',SOURCE/'source-coarse-results.json'),('source-fine-results.json',SOURCE/'native-results.json'),
            ('source-spatial-results.json',SOURCE/'results.json')]:shutil.copyfile(p,out/name)
        d=source_data(out/'original-comparison.npz');old=source_data(out/'source-spatial-states.npz');validate_record(d)
        runs={};states={}
        for N in (128,256):
            run,path=native_run(out,N,grid_data(d,N),index,guard);runs[str(N)]=run;states[str(N)]=path
            save(out/f'n{N}/native-results.json',run);np.savez_compressed(out/f'n{N}/trajectory-states.npz',**path)
            guard(f'n{N}-sixteen-interval-path-saved')
        result,projected=compare(d,runs,states,read(out/'source-fine-results.json'),read(out/'source-coarse-results.json'),old)
        save(out/'results.json',result);np.savez_compressed(out/'common-observations.npz',**projected);guard('time16-comparison-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','new_software_checks':4,'previous_recorded_checks':449,
            'recorded_checks_total':453,'previous_checks_rerun':0,'new_candidate_graphs':32,'new_candidate_transient_runs':2,
            'new_independent_uniformization_sequences':2,'new_reference_PDE_solves':0,'new_spatial_levels':0,'new_flow_definitions':0,
            'old4_8_candidate_reruns':0,'registered16_interval_control_executed':True,'comparison_decision':result['comparison_decision'],
            'old_spatial_inconclusive_result_unchanged':True,'no_automaticK32_or_grid_expansion':True,'shock_extension_deferred':True,
            'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
            'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':True,
            'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'resource_caps_passed':True,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'threads':4,'numerical_norm_estimator_seed':SEED,'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();d=source_data(out/'original-comparison.npz');old=source_data(out/'source-spatial-states.npz')
    index=read(out/'graph-index.json');runs={};states={}
    for N in (128,256):
        run,path=native_run(out,N,grid_data(d,N),index,replay=True);assert run==read(out/f'n{N}/native-results.json')
        with np.load(out/f'n{N}/trajectory-states.npz',allow_pickle=False) as z:
            for key,v in path.items():assert np.array_equal(v,z[key]),key
        runs[str(N)]=run;states[str(N)]=path
    r,p=compare(d,runs,states,read(out/'source-fine-results.json'),read(out/'source-coarse-results.json'),old)
    assert r==read(out/'results.json')
    with np.load(out/'common-observations.npz',allow_pickle=False) as z:
        for key,v in p.items():assert np.array_equal(v,z[key]),key
    print('Same32 sealed graphs: both original material grids, actual16-interval and independent full sequences, original parent projection and unchanged8/16 comparison replay; no geometry/PDE/old4_8 rerun or ORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
