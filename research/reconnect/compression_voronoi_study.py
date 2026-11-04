"""One registered smooth compression: same Voronoi FV, actual4/8/16 ordered paths."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,math,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save,digest
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from shear_voronoi_fv_study import record,verify_probe_cells
from shear_union64_study import arrays,native_run,classify
from voronoi_time_action import metrics
from compression_voronoi_core import sample_points,kernel_local_controls

SOURCE=ROOT/'runs/reconnect-shear-union64'
PROTOCOL=ROOT/'research/reconnect/compression_voronoi_protocol.json'
CONTROL=ROOT/'research/reconnect/COMPRESSION_VORONOI_TIME_CONTROL.md'
POINT_SOURCE=ROOT/'runs/reconnect-ordered-compression/ordered-compression.npz'
REFERENCE_SOURCE=ROOT/'runs/reconnect-compression-reference/compression-reference.npz'
COPIES={'original-reference.npz':REFERENCE_SOURCE,'original-kernel.npz':POINT_SOURCE,
 'original-kernel-results.json':ROOT/'runs/reconnect-ordered-compression/results.json',
 'original-reference-results.json':ROOT/'runs/reconnect-compression-reference/results.json'}


def original_data(out):
    ref=arrays(out/'original-reference.npz');old=arrays(out/'original-kernel.npz')
    for name in ('mass','c0','ids','material_label_centers'):assert record(ref[name])==record(old[name]),name
    assert record(ref['best_projected_states'])==record(old['reference_states'])
    assert record(ref['snapshot_time'])==record(old['reference_times'])
    return ref,old


def compare(ref,old,old_result,runs,paths):
    m,c0=ref['mass'],ref['c0'];E0=metrics(m,c0,c0)['variance'];sigma=math.sqrt(E0)
    norm=lambda v:float(np.sqrt(m@(v*v)))/sigma
    H=lambda c:metrics(m,c,c0)['H']
    hm=lambda a:(2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))*a+a*a)/E0
    bridge=norm(ref['best_projected_states'][0]-c0);Hbridge=abs(H(ref['best_projected_states'][0])-H(c0))
    rows=[];temporal=[];observations={};vectors={}
    for K in (4,8,16):
        observations[str(K)]=[]
        for k,t in enumerate(ref['snapshot_time']):
            u=paths[K]['states'][k*K//2];r=ref['best_projected_states'][k]
            observations[str(K)].append({'time':float(t),'relative_concentration_error':norm(u-r),
              'candidate_metrics':metrics(m,u,c0),'reference_metrics':metrics(m,r,c0),
              'H_signed_error':H(u)-H(r),'minimum':float(u.min()),'maximum':float(u.max()),'raw_mass_change_from_c0':float(m@(u-c0))})
    for K in (4,8):
        for row in old_result['cases'][str(K)]['steps']:
            a=row['sparse_graph_audit'];assert a['graph_algebra_eligible'] and a['linear_algebra_tolerance']==1e-10
            assert not a['clipping_or_renormalization'] and not row['operator_diagnostics']['clipping_fitting_or_normalization']
    for k,t in enumerate(ref['snapshot_time']):
        u4,u8,u16=[paths[K]['states'][k*K//2] for K in (4,8,16)]
        b4,b8=[old[f'k{K}_states'][k*K//2] for K in (4,8)];r=ref['best_projected_states'][k]
        local=kernel_local_controls(old,old_result,m,c0,k)
        solver_abs=observed=observedH=0.
        for K in (8,16):
            i=k*K//2
            if not i:continue
            row=runs[K]['steps'][i-1]
            solver_abs+=row['cumulative_machine_indicator_absolute']+row['cumulative_tail_indicator_absolute']
            other=paths[K]['independent_states'][i];u=paths[K]['states'][i]
            observed+=norm(u-other);observedH+=abs(H(u)-H(other))
        components={'FV8_16':norm(u8-u16),'saved_kernel4_8':norm(b4-b8),
          'twice_reference_space':2*norm(ref['space_difference'][k]),'twice_reference_fine_time':2*norm(ref['time_fine_difference'][k]),
          'initial_reference_bridge':bridge,'FV_machine_and_tails':solver_abs/sigma,'FV_independent_sequences':observed,
          'all_saved_kernel_local_controls':local['relative_sum']}
        r256=ref['n1024-k256_projected_states'][k];r512=ref['n512-k256_projected_states'][k];rt512=ref['n1024-k512_projected_states'][k]
        hp={'FV8_16':abs(H(u8)-H(u16)),'saved_kernel4_8':abs(H(b4)-H(b8)),
          'twice_reference_space':2*abs(H(r512)-H(r256)),'twice_reference_fine_time':2*abs(H(rt512)-H(r)),
          'initial_reference_bridge':Hbridge,'FV_independent_sequences':observedH,'all_saved_kernel_local_H_controls':local['local_H_absolute_sum'],
          'machine_tails_and_saved_local_controls_to_H':hm(solver_abs+local['absolute_sum'])}
        e,b=norm(u16-r),norm(b8-r);hg=abs(H(b8)-H(r))-abs(H(u16)-H(r))
        row={'time':float(t),'FV16_error':e,'saved_kernel8_error':b,'state_gain':b-e,
          'state_indicator_components':components,'state_indicator':sum(components.values()),
          'FV16_H':H(u16),'saved_kernel8_H':H(b8),'reference_H':H(r),'FV16_H_signed_error':H(u16)-H(r),
          'H_gain':hg,'H_indicator_components':hp,'H_indicator':sum(hp.values()),
          'saved_kernel8_over_FV16_error_ratio':None if e==0 else b/e,'saved_kernel_local_controls':local}
        for label in ('state','H'):row[label+'_decision']=classify(row[label+'_gain'],row[label+'_indicator'])
        rows.append(row)
        d48,d816=norm(u4-u8),norm(u8-u16);h48,h816=abs(H(u4)-H(u8)),abs(H(u8)-H(u16))
        temporal.append({'time':float(t),'FV4_8_state_difference':d48,'FV8_16_state_difference':d816,
          'state_difference_ratio':None if d816==0 else d48/d816,'FV4_8_H_difference':h48,'FV8_16_H_difference':h816,
          'H_difference_ratio':None if h816==0 else h48/h816,'state_difference_decreased':d816<d48,'H_difference_decreased':h816<h48})
    vectors['FV4_8_difference']=paths[4]['states'][[0,2,4]]-paths[8]['states'][[0,4,8]]
    vectors['FV8_16_difference']=paths[8]['states'][[0,4,8]]-paths[16]['states'][[0,8,16]]
    vectors['saved_kernel4_8_difference']=old['k4_states'][[0,2,4]]-old['k8_states'][[0,4,8]]
    vectors['initial_reference_bridge']=ref['best_projected_states'][0]-c0
    improving=all(v['state_difference_decreased'] and v['H_difference_decreased'] for v in temporal[1:])
    resolved=all(v[label+'_decision']=='fine_better_resolvably' for v in rows[1:] for label in ('state','H'))
    adverse=any(v[label+'_decision']=='fine_worse_resolvably' for v in rows[1:] for label in ('state','H'))
    decision='smooth_compression_improvement_resolved_with_time_control' if improving and resolved else ('adverse_at_a_checkpoint' if adverse else 'smooth_compression_comparison_inconclusive')
    return {'original_initial_fluctuation_norm':sigma,'observations_by_intervals':observations,'comparison_observations':rows,
      'time_controls':temporal,'both_state_and_H_time_differences_decreased':improving,'state_and_H_gains_resolved_at_both_times':resolved,
      'comparison_decision':decision,'time_reference_and_local_controls_are_certified_bounds':False,
      'ratios_certify_general_order':False,'Richardson_or_fitted_state_used':False,'new_reference_PDE_solves':0,'old_candidate_reruns':0,
      'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False},vectors


def main(output,tests,test_log):
    prior=read(SOURCE/'sha256.json');assert len(prior)==2331
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    spec=read(SOURCE/'next-gate-spec.json');proto=read(PROTOCOL)
    assert not spec['already_executed'] and spec['new_ordered_interval_counts']==[4,8,16] and spec['new_midpoint_graphs']==28
    assert spec['N']==128 and spec['main_wall_seconds_cap']==2000 and spec['main_peak_RSS_MiB_cap']==1024
    assert digest(POINT_SOURCE)==spec['pure_point_source_sha256'] and digest(REFERENCE_SOURCE)==spec['original_state_source_sha256']
    assert read(tests)=={'successful':True,'checks_run':4,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for n,p in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),
      ('control-at-lock.md',CONTROL),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(p,out/n)
    locked=['periodic_voronoi_fv.py','voronoi_time_action.py','voronoi_time16_core.py','shear_union64_study.py',
      'compression_probe.py','compression_voronoi_core.py','test_compression_voronoi.py','compression_voronoi_study.py']
    save(out/'implementation-lock.json',{'files':{n:digest(ROOT/'research/reconnect'/n) for n in locked},
      'protocol_sha256':digest(PROTOCOL),'control_note_sha256':digest(CONTROL),'fixed_before_new_pure_point_validation_and_geometry':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':2331,
      'parent_manifest_sha256':digest(SOURCE/'sha256.json'),'new_software_checks':4,'previous_recorded_checks':457,
      'recorded_checks_total':461,'previous_checks_rerun':0,'new_reference_PDE_solves':0,'new_spatial_levels':[]})
    started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>2000 or peak>1024:raise RuntimeError(f'registered resource cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k%8192==0 or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('registered2000second main cap reached')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(2000)
    try:
        with np.load(POINT_SOURCE,allow_pickle=False) as z:
            labels=z['material_label_centers'].copy();ids=z['ids'].copy();rho0=z['rho0_point'].copy()
        axis=(np.arange(128)+.5)/128;assert record(labels)==record(np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2))
        assert record(rho0)==record(1+.25*np.cos(2*np.pi*labels[:,0]))
        fine=sample_points(labels,ids,rho0,np.linspace(0.,.05,33));b={'ids':ids};checks=[]
        with np.load(POINT_SOURCE,allow_pickle=False) as z:
            for K in (4,8,16):
                generated=sample_points(labels,ids,rho0,np.linspace(0.,.05,2*K+1));stride=16//K
                for key in ('time','positions','rho'):
                    assert record(generated[key])==record(fine[key][::stride])
                    if K in (4,8):assert record(generated[key])==record(z[f'k{K}_{key}'])
                    b[f'k{K}_{key}']=generated[key]
                b[f'k{K}_boundary_time']=generated['time'][::2];b[f'k{K}_midpoint_time']=generated['time'][1::2]
                if K in (4,8):checks.append({'K':K,'all_time_position_density_records_and_nested_K16_subsets_bitwise_match':True})
                (out/f'k{K}').mkdir()
        np.savez_compressed(out/'trajectory-source.npz',labels=labels,ids=ids,rho0_point=rho0)
        np.savez_compressed(out/'builder-inputs.npz',**b)
        save(out/'input-matching.json',{'source':str(POINT_SOURCE.relative_to(ROOT)),'source_sha256':digest(POINT_SOURCE),
          'builder_packet_sha256':digest(out/'builder-inputs.npz'),'original_array_records':{'labels':record(labels),'ids':record(ids),'rho0':record(rho0)},
          'checks':checks,'mass_c0_reference_or_old_kernel_states_loaded':False,'J_F_C_true_faces_or_gradients_in_builder':False})
        del fine,generated;gc.collect();index={};guard('all-original-pure-inputs-verified')
        for K in (4,8,16):
            for step in range(K):
                frame=2*step+1
                graph=build_periodic_faces(ids,b[f'k{K}_positions'][frame],b[f'k{K}_rho'][frame],.02,(1,1),1,guard)
                probes=verify_probe_cells(graph,proto['fixed_geometry_probe_ids']);packet=coefficient_packet(graph)
                name=f'k{K}/geometry-step{step:02d}.json';save(out/name,packet)
                index[f'k{K}-step{step:02d}']={'N':128,'K':K,'step':step,'frame':frame,'file':name,'sha256':digest(out/name),
                  'midpoint_time_saved':float(b[f'k{K}_midpoint_time'][step]),'geometry_diagnostics':graph.diagnostics,'independent_fixed_probes':probes}
                del graph,packet;gc.collect();guard(f'sealed-geometry-k{K}-step{step:02d}')
        save(out/'graph-index.json',index);save(out/'coefficient-seal.json',{'graphs':28,'graph_index_sha256':digest(out/'graph-index.json'),
          'builder_packet_sha256':digest(out/'builder-inputs.npz'),'all28_coefficients_sealed_before_mass_c0_reference_or_kernel_state_loading':True,
          'builder_received_mass_c0_reference_true_faces_or_F_C':False})
        del b;gc.collect();guard('all28-coefficients-sealed-before-state')
        copied=[]
        for name,p in COPIES.items():
            rel=str(p.relative_to(ROOT));assert digest(p)==prior[rel];shutil.copyfile(p,out/name)
            copied.append({'copy':name,'source':rel,'sha256':digest(p)})
        save(out/'source-copies.json',copied);ref,old=original_data(out)
        assert record(ids)==record(ref['ids']) and record(labels)==record(ref['material_label_centers'])
        runs={};paths={}
        for K in (4,8,16):
            run,path=native_run(out,K,ref,index,guard);runs[K]=run;paths[K]=path
            save(out/f'k{K}/native-results.json',run);np.savez_compressed(out/f'k{K}/trajectory-states.npz',**path)
        result,vectors=compare(ref,old,read(out/'original-kernel-results.json'),runs,paths)
        save(out/'results.json',result);np.savez_compressed(out/'comparison-vectors.npz',**vectors);guard('compression-comparison-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','previous_recorded_checks':457,'new_software_checks':4,
          'recorded_checks_total':461,'previous_checks_rerun':0,'new_candidate_graphs':28,'new_candidate_transient_runs':3,
          'new_independent_uniformization_sequences':3,'new_reference_PDE_solves':0,'old_candidate_reruns':0,'new_spatial_levels':[],
          'registered_compression_time_control_executed':True,'source_gate_already_executed':True,'comparison_decision':result['comparison_decision'],
          'no_automatic_K32_or_grid_extension':True,'shock_extension_deferred':True,'general_smooth_flow_admitted':False,
          'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
          'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':True,
          'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'resource_caps_passed':True,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
          'threads':4,'numerical_norm_estimator_seed':20261101,'network_used':False,'environment_configuration_changed':False,
          'committed_or_pushed':False,'published':False})
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
          'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();ref,old=original_data(out);runs={};paths={};index=read(out/'graph-index.json')
    for K in (4,8,16):
        r,p=native_run(out,K,ref,index,replay=True);assert r==read(out/f'k{K}/native-results.json')
        saved=arrays(out/f'k{K}/trajectory-states.npz')
        for key,v in p.items():np.testing.assert_array_equal(v,saved[key])
        runs[K]=r;paths[K]=p
    r,v=compare(ref,old,read(out/'original-kernel-results.json'),runs,paths);assert r==read(out/'results.json')
    saved=arrays(out/'comparison-vectors.npz')
    for key,value in v.items():np.testing.assert_array_equal(value,saved[key])
    print('Sealed28 graphs: actual original N128 compression K4/8/16 and complete independent uniformization paths reproduced bitwise; all time/reference/saved-kernel controls and comparisons reproduced. No old candidate,geometry,PDE,normalization or ORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
