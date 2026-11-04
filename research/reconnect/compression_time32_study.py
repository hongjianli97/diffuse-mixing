"""One registered32-interval control; immutable original4/8/16 paths are reused."""
from pathlib import Path
import argparse,gc,json,math,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save,digest
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from shear_voronoi_fv_study import record,verify_probe_cells
from shear_union64_study import arrays,native_run,classify
from compression_voronoi_study import original_data
from compression_voronoi_core import sample_points,kernel_local_controls
from voronoi_time_action import metrics

SOURCE=ROOT/'runs/reconnect-compression-voronoi-time'
PROTOCOL=ROOT/'research/reconnect/compression_time32_protocol.json'
CONTROL=ROOT/'research/reconnect/COMPRESSION_TIME32_CONTROL.md'
COPIES={name:SOURCE/name for name in ('original-reference.npz','original-kernel.npz','original-kernel-results.json','original-reference-results.json')}
COPIES['source-results.json']=SOURCE/'results.json'
for K in (4,8,16):
    COPIES[f'old-k{K}-states.npz']=SOURCE/f'k{K}/trajectory-states.npz'
    COPIES[f'old-k{K}-results.json']=SOURCE/f'k{K}/native-results.json'


def load_previous(out):
    return ({K:read(out/f'old-k{K}-results.json') for K in (4,8,16)},
      {K:arrays(out/f'old-k{K}-states.npz') for K in (4,8,16)})


def compare(ref,old,old_result,runs,paths):
    m,c0=ref['mass'],ref['c0'];E0=metrics(m,c0,c0)['variance'];sigma=math.sqrt(E0)
    norm=lambda v:float(np.sqrt(m@(v*v)))/sigma;H=lambda c:metrics(m,c,c0)['H']
    hm=lambda a:(2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))*a+a*a)/E0
    bridge=norm(ref['best_projected_states'][0]-c0);Hb=abs(H(ref['best_projected_states'][0])-H(c0))
    observations={};rows=[];temporal=[];vectors={}
    for K in (4,8,16,32):
        observations[str(K)]=[]
        for k,t in enumerate(ref['snapshot_time']):
            c=paths[K]['states'][k*K//2];r=ref['best_projected_states'][k]
            observations[str(K)].append({'time':float(t),'relative_concentration_error':norm(c-r),
              'candidate_metrics':metrics(m,c,c0),'reference_metrics':metrics(m,r,c0),'H_signed_error':H(c)-H(r),
              'minimum':float(c.min()),'maximum':float(c.max()),'raw_mass_change_from_c0':float(m@(c-c0))})
    for K in (4,8):
        for row in old_result['cases'][str(K)]['steps']:
            a=row['sparse_graph_audit'];assert a['graph_algebra_eligible'] and a['linear_algebra_tolerance']==1e-10
            assert not a['clipping_or_renormalization'] and not row['operator_diagnostics']['clipping_fitting_or_normalization']
    for k,t in enumerate(ref['snapshot_time']):
        u4,u8,u16,u32=[paths[K]['states'][k*K//2] for K in (4,8,16,32)]
        b4,b8=[old[f'k{K}_states'][k*K//2] for K in (4,8)];r=ref['best_projected_states'][k]
        local=kernel_local_controls(old,old_result,m,c0,k);machine=observed=observedH=0.
        for K in (16,32):
            i=k*K//2
            if i:
                row=runs[K]['steps'][i-1];machine+=row['cumulative_machine_indicator_absolute']+row['cumulative_tail_indicator_absolute']
                u=paths[K]['states'][i];v=paths[K]['independent_states'][i];observed+=norm(u-v);observedH+=abs(H(u)-H(v))
        components={'FV16_32':norm(u16-u32),'saved_kernel4_8':norm(b4-b8),
          'twice_reference_space':2*norm(ref['space_difference'][k]),'twice_reference_fine_time':2*norm(ref['time_fine_difference'][k]),
          'initial_reference_bridge':bridge,'FV_machine_and_tails':machine/sigma,'FV_independent_sequences':observed,
          'all_saved_kernel_local_controls':local['relative_sum']}
        hp={'FV16_32':abs(H(u16)-H(u32)),'saved_kernel4_8':abs(H(b4)-H(b8)),
          'twice_reference_space':2*abs(H(ref['n512-k256_projected_states'][k])-H(ref['n1024-k256_projected_states'][k])),
          'twice_reference_fine_time':2*abs(H(ref['n1024-k512_projected_states'][k])-H(r)),
          'initial_reference_bridge':Hb,'FV_independent_sequences':observedH,'all_saved_kernel_local_H_controls':local['local_H_absolute_sum'],
          'machine_tails_and_saved_local_controls_to_H':hm(machine+local['absolute_sum'])}
        e,b=norm(u32-r),norm(b8-r);gainH=abs(H(b8)-H(r))-abs(H(u32)-H(r))
        row={'time':float(t),'FV32_error':e,'saved_kernel8_error':b,'state_gain':b-e,
          'state_indicator_components':components,'state_indicator':sum(components.values()),'FV32_H':H(u32),
          'saved_kernel8_H':H(b8),'reference_H':H(r),'FV32_H_signed_error':H(u32)-H(r),'H_gain':gainH,
          'H_indicator_components':hp,'H_indicator':sum(hp.values()),'saved_kernel8_over_FV32_error_ratio':None if e==0 else b/e,
          'saved_kernel_local_controls':local}
        for label in ('state','H'):row[label+'_decision']=classify(row[label+'_gain'],row[label+'_indicator'])
        rows.append(row)
        d48,d816,d1632=norm(u4-u8),norm(u8-u16),norm(u16-u32)
        h48,h816,h1632=abs(H(u4)-H(u8)),abs(H(u8)-H(u16)),abs(H(u16)-H(u32))
        ratio=lambda a,b:None if b==0 else a/b
        temporal.append({'time':float(t),'FV4_8_state_difference':d48,'FV8_16_state_difference':d816,'FV16_32_state_difference':d1632,
          'state_old_difference_ratio':ratio(d48,d816),'state_new_difference_ratio':ratio(d816,d1632),
          'FV4_8_H_difference':h48,'FV8_16_H_difference':h816,'FV16_32_H_difference':h1632,
          'H_new_difference_ratio':ratio(h816,h1632),'state_difference_decreased':d1632<d816,'H_difference_decreased':h1632<h816,
          'new_difference_over_FV32_error':ratio(d1632,e),'difference_is_actual_remaining_time_error':False})
    for C,N in ((4,8),(8,16),(16,32)):
        vectors[f'FV{C}_{N}_difference']=paths[C]['states'][[0,C//2,C]]-paths[N]['states'][[0,N//2,N]]
    vectors['saved_kernel4_8_difference']=old['k4_states'][[0,2,4]]-old['k8_states'][[0,4,8]]
    vectors['initial_reference_bridge']=ref['best_projected_states'][0]-c0
    decreasing=all(v['state_difference_decreased'] and v['H_difference_decreased'] for v in temporal[1:])
    resolved=all(v[label+'_decision']=='fine_better_resolvably' for v in rows[1:] for label in ('state','H'))
    adverse=any(v[label+'_decision']=='fine_worse_resolvably' for v in rows[1:] for label in ('state','H'))
    decision='compression32_time_control_decreased_and_improvement_resolved' if decreasing and resolved else ('adverse_at_a_checkpoint' if adverse else 'compression32_comparison_inconclusive')
    return {'original_initial_fluctuation_norm':sigma,'observations_by_intervals':observations,'comparison_observations':rows,
      'time_controls':temporal,'both_state_and_H_time_differences_decreased':decreasing,'state_and_H_gains_resolved_at_both_times':resolved,
      'comparison_decision':decision,'old4_8_16_candidates_rerun':False,'time_indicators_are_certified_bounds':False,
      'observed_ratios_certify_general_order':False,'Richardson_or_fitted_state_used':False,'general_smooth_flow_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False},vectors


def main(output,tests,test_log,initial_tests=None,initial_test_log=None):
    prior=read(SOURCE/'sha256.json');assert len(prior)==2500
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    spec=read(SOURCE/'next-gate-spec.json');proto=read(PROTOCOL)
    assert not spec['already_executed'] and spec['new_ordered_interval_counts']==[32] and spec['new_midpoint_graphs']==32
    assert spec['main_wall_seconds_cap']==1400 and spec['main_peak_RSS_MiB_cap']==1024
    assert read(tests)=={'successful':True,'checks_run':2,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for n,p in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),
      ('control-at-lock.md',CONTROL),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(p,out/n)
    for name,p in [('tests-first-attempt.json',initial_tests or tests),('tests-first-attempt.log',initial_test_log or test_log)]:shutil.copyfile(p,out/name)
    locked=['periodic_voronoi_fv.py','voronoi_time_action.py','voronoi_time16_core.py','shear_union64_study.py','compression_probe.py',
      'compression_voronoi_core.py','compression_voronoi_study.py','compression_time32_study.py','test_compression_time32.py']
    save(out/'implementation-lock.json',{'files':{n:digest(ROOT/'research/reconnect'/n) for n in locked},
      'protocol_sha256':digest(PROTOCOL),'control_note_sha256':digest(CONTROL),'fixed_before_new_pure_input_and_geometry':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':2500,
      'parent_manifest_sha256':digest(SOURCE/'sha256.json'),'new_software_checks':2,'previous_recorded_checks':461,
      'recorded_checks_total':463,'previous_checks_rerun':0,'new_reference_PDE_solves':0,'old_candidate_reruns':0})
    started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>1400 or peak>1024:raise RuntimeError(f'registered resource cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k%8192==0 or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('registered1400second main cap reached')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(1400)
    try:
        pure=arrays(SOURCE/'trajectory-source.npz');assert set(pure)=={'labels','ids','rho0_point'}
        assert digest(SOURCE/'trajectory-source.npz')==spec['pure_point_source_sha256']
        fine=sample_points(pure['labels'],pure['ids'],pure['rho0_point'],np.linspace(0.,.05,65));b={'ids':fine['ids']}
        for key in ('time','positions','rho'):b[f'k32_{key}']=fine[key]
        b['k32_boundary_time']=fine['time'][::2];b['k32_midpoint_time']=fine['time'][1::2];checks=[]
        assert digest(SOURCE/'builder-inputs.npz')==spec['previous_builder_packet_sha256']
        with np.load(SOURCE/'builder-inputs.npz',allow_pickle=False) as z:
            assert record(z['ids'])==record(pure['ids'])
            for K in (4,8,16):
                generated=sample_points(pure['labels'],pure['ids'],pure['rho0_point'],np.linspace(0.,.05,2*K+1))
                for key in ('time','positions','rho'):assert record(generated[key])==record(z[f'k{K}_{key}'])==record(fine[key][::(32//K)])
                checks.append({'K':K,'all_old_records_and_nested32_subsets_bitwise_match':True})
        shutil.copyfile(SOURCE/'trajectory-source.npz',out/'trajectory-source.npz');np.savez_compressed(out/'builder-inputs.npz',**b)
        save(out/'input-matching.json',{'source_packet_sha256':digest(SOURCE/'builder-inputs.npz'),'builder_packet_sha256':digest(out/'builder-inputs.npz'),
          'checks':checks,'actual_mass_c0_reference_or_old_states_loaded':False,'J_F_C_true_faces_or_gradients_in_builder':False})
        del fine,generated;gc.collect();(out/'k32').mkdir();index={};guard('all-old-pure-records-and-new-subsets-verified')
        for step in range(32):
            graph=build_periodic_faces(b['ids'],b['k32_positions'][2*step+1],b['k32_rho'][2*step+1],.02,(1,1),1,guard)
            probes=verify_probe_cells(graph,proto['fixed_geometry_probe_ids']);packet=coefficient_packet(graph)
            name=f'k32/geometry-step{step:02d}.json';save(out/name,packet)
            index[f'k32-step{step:02d}']={'N':128,'K':32,'step':step,'frame':2*step+1,'file':name,'sha256':digest(out/name),
              'midpoint_time_saved':float(b['k32_midpoint_time'][step]),'geometry_diagnostics':graph.diagnostics,'independent_fixed_probes':probes}
            del graph,packet;gc.collect();guard(f'sealed-geometry-k32-step{step:02d}')
        save(out/'graph-index.json',index);save(out/'coefficient-seal.json',{'graphs':32,'graph_index_sha256':digest(out/'graph-index.json'),
          'builder_packet_sha256':digest(out/'builder-inputs.npz'),'all32_coefficients_sealed_before_mass_c0_reference_or_old_state_loading':True,
          'builder_received_mass_c0_reference_true_faces_or_F_C':False})
        del b;gc.collect();guard('all32-coefficients-sealed-before-state');copied=[]
        for name,p in COPIES.items():
            rel=str(p.relative_to(ROOT));assert digest(p)==prior[rel];shutil.copyfile(p,out/name);copied.append({'copy':name,'source':rel,'sha256':digest(p)})
        save(out/'source-copies.json',copied);ref,old=original_data(out);runs,paths=load_previous(out)
        assert record(ref['ids'])==record(pure['ids']) and record(ref['material_label_centers'])==record(pure['labels'])
        run,path=native_run(out,32,ref,index,guard);runs[32]=run;paths[32]=path
        save(out/'k32/native-results.json',run);np.savez_compressed(out/'k32/trajectory-states.npz',**path)
        result,vectors=compare(ref,old,read(out/'original-kernel-results.json'),runs,paths)
        previous=read(out/'source-results.json')
        for K in (4,8,16):assert result['observations_by_intervals'][str(K)]==previous['observations_by_intervals'][str(K)]
        save(out/'results.json',result);np.savez_compressed(out/'comparison-vectors.npz',**vectors);guard('time32-comparison-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','previous_recorded_checks':461,'new_software_checks':2,
          'recorded_checks_total':463,'previous_checks_rerun':0,'new_candidate_graphs':32,'new_candidate_transient_runs':1,
          'new_independent_uniformization_sequences':1,'new_reference_PDE_solves':0,'old_candidate_reruns':0,'new_spatial_levels':[],
          'registered32_interval_control_executed':True,'comparison_decision':result['comparison_decision'],'no_automatic_K64_or_grid_extension':True,
          'shock_extension_deferred':True,'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,
          'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':read(out/'tests-first-attempt.json')['successful'],'main_first_attempt_passed':True,
          'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'resource_caps_passed':True,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4,
          'numerical_norm_estimator_seed':20261101,'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
          'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();ref,old=original_data(out);runs,paths=load_previous(out)
    run,path=native_run(out,32,ref,read(out/'graph-index.json'),replay=True);assert run==read(out/'k32/native-results.json')
    saved=arrays(out/'k32/trajectory-states.npz')
    for key,v in path.items():np.testing.assert_array_equal(v,saved[key])
    runs[32]=run;paths[32]=path;r,v=compare(ref,old,read(out/'original-kernel-results.json'),runs,paths);assert r==read(out/'results.json')
    saved=arrays(out/'comparison-vectors.npz')
    for key,value in v.items():np.testing.assert_array_equal(value,saved[key])
    print('Same32 sealed graphs: actual N128 compression32 and independent complete uniformization path replayed bitwise; old4/8/16 paths reused without reruns; all controls and comparisons reproduced. No geometry/PDE/normalization/ORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--initial-tests');p.add_argument('--initial-test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log,a.initial_tests,a.initial_test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
