"""One registered smooth-shear scalar transient, fixed 4/8 midpoint controls."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import gc
import json
import math
import platform
import resource
import shutil
import signal
import time
import numpy as np
import scipy

from audit_route_decision import ROOT, read, save, digest
from periodic_voronoi_fv import build_periodic_faces, coefficient_packet
from shear_voronoi_fv_study import record, verify_probe_cells
from voronoi_time_action import operators, exponential, uniformization, arithmetic_allowance, metrics, dissipation

SOURCE = ROOT/'runs/reconnect-shear-voronoi-FV-frozen'
PROTOCOL = ROOT/'research/reconnect/voronoi_time_protocol.json'
CONTROL = ROOT/'research/reconnect/SHEAR_VORONOI_TIME_CONTROL.md'
OUT_REL = 'runs/reconnect-shear-voronoi-time'


def relative(m, vector, sigma):
    return float(np.sqrt(m @ (vector*vector))/sigma)


def source_data(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def propagate(out, data, index, guard=lambda *args:None, replay=False):
    m,c0 = data['mass'],data['c0']
    sigma = math.sqrt(metrics(m,c0,c0)['variance'])
    if not (c0.min()>=0 and c0.max()<=1):
        raise ValueError('registered scalar not in[0,1]; no state repair')
    saved = {};cases = {}
    total_mass = float(m.sum());upper0=float(np.max(abs(c0)))
    for K in (4,8):
        dt=F(1,20*K);state=c0.copy();control=c0.copy()
        states=[state.copy()];controls=[control.copy()];halves=[];rows=[]
        allowance_sum=0.;tail_sum=0.
        for step in range(K):
            item=index[f'k{K}-step{step:02d}']
            path=out/item['file'];assert digest(path)==item['sha256']
            packet=read(path)
            op=operators(m,np.array(packet['edges'],dtype=np.int64),map(F,packet['conductances_exact']))
            del packet;gc.collect()
            diag=op.diagnostics
            assert diag['uniformization_minimum_stored_coefficient']>=0
            assert diag['uniformization_row_sum_residual_Linf']<1e-10
            assert diag['uniformization_invariant_mass_residual_Linf']/float(m.max())<1e-10
            following,half=exponential(op,state,dt)
            independent,weights,info=uniformization(op,control,dt)
            allowance=arithmetic_allowance(op,c0,dt,info['M'])
            allowance_sum+=allowance
            tail_sum+=float(F(info['tail_upper_exact']))*math.sqrt(total_mass)*upper0
            absolute_delta=float(np.sqrt(m@((following-independent)**2)))
            if not np.isfinite(following).all() or not np.isfinite(independent).all() or absolute_delta>allowance_sum+tail_sum:
                raise ArithmeticError('full-sequence independent solver control failed; no repair')
            before=metrics(m,state,c0);after=metrics(m,following,c0)
            drift=float(m@(following-c0))
            if abs(drift)/float(m@c0)>1e-10 or following.min()<c0.min()-1e-10 or following.max()>c0.max()+1e-10:
                raise ArithmeticError('raw mass/extrema check failed; no correction')
            if after['variance']>before['variance']+1e-10*sigma*sigma:
                raise ArithmeticError('raw variance check failed; no correction')
            chi=[dissipation(op,u) for u in (state,half,following)]
            if min(chi)<0:
                raise ArithmeticError('negative shared-face dissipation')
            loss=.5*(before['variance']-after['variance'])
            simpson=float(dt)*(chi[0]+4*chi[1]+chi[2])/6
            row={'step':step,'dt_exact':str(dt),'midpoint_time_saved':item['midpoint_time_saved'],
                'coefficient_cache_sha256':item['sha256'],'operator':diag,'poisson':info,
                'state_metrics':after,'independent_metrics':metrics(m,independent,c0),
                'primary_control_mass_norm_difference':absolute_delta,
                'primary_control_relative_difference':absolute_delta/sigma,
                'machine_indicator_absolute_this_interval':allowance,
                'cumulative_machine_indicator_absolute':allowance_sum,
                'cumulative_tail_indicator_absolute':tail_sum,
                'raw_mass_change_from_c0':drift,'minimum':float(following.min()),'maximum':float(following.max()),
                'independent_raw_mass_change_from_c0':float(m@(independent-c0)),
                'chi_start_half_end':chi,'half_variance_loss':loss,
                'Simpson_chi_integral':simpson,'raw_energy_balance_difference':loss-simpson,
                'Simpson_is_empirical_not_exact_balance':True}
            if not replay:
                np.savez_compressed(out/f'action-k{K}-step{step:02d}.npz',
                    symmetric_data=op.symmetric_generator.data,symmetric_indices=op.symmetric_generator.indices,
                    symmetric_indptr=op.symmetric_generator.indptr,
                    uniformization_data=op.uniformization_matrix.data,uniformization_indices=op.uniformization_matrix.indices,
                    uniformization_indptr=op.uniformization_matrix.indptr,
                    shape=np.array(op.symmetric_generator.shape),edges=op.edges,g_float=op.conductance,weights=weights)
            else:
                with np.load(out/f'action-k{K}-step{step:02d}.npz',allow_pickle=False) as z:
                    for name,v in [('symmetric_data',op.symmetric_generator.data),('symmetric_indices',op.symmetric_generator.indices),
                        ('symmetric_indptr',op.symmetric_generator.indptr),('uniformization_data',op.uniformization_matrix.data),
                        ('uniformization_indices',op.uniformization_matrix.indices),('uniformization_indptr',op.uniformization_matrix.indptr),
                        ('edges',op.edges),('g_float',op.conductance),('weights',weights)]:
                        assert np.array_equal(z[name],v),name
            state=following;control=independent
            states.append(state.copy());controls.append(control.copy());halves.append(half);rows.append(row)
            guard(f'propagated-k{K}',step+1,K)
            del op;gc.collect()
        states=np.array(states);controls=np.array(controls)
        snapshots=states[[0,K//2,K]];control_snapshots=controls[[0,K//2,K]]
        observations=[]
        for k,t in enumerate(data['reference_times']):
            u=snapshots[k];reference=data['reference_states'][k]
            cm=metrics(m,u,c0);rm=metrics(m,reference,c0)
            observations.append({'time':float(t),'relative_concentration_error':relative(m,u-reference,sigma),
                'primary_control_relative_difference':relative(m,u-control_snapshots[k],sigma),
                'candidate_metrics':cm,'reference_metrics':rm,'H_minus_reference':cm['H']-rm['H']})
        cases[str(K)]={'steps':rows,'observations':observations}
        saved[f'k{K}_states']=states;saved[f'k{K}_independent_states']=controls
        saved[f'k{K}_exponential_half_states']=np.array(halves)
    return cases,saved,sigma


def compare(data,cases,saved,sigma):
    m,c0=data['mass'],data['c0'];E0=sigma*sigma
    M=float(m.sum());maxc=max(1.,float(np.max(abs(c0))))
    norm=lambda v:relative(m,v,sigma)
    H=lambda u:metrics(m,u,c0)['H']
    h_allow=lambda absolute:(2*math.sqrt(M)*maxc*absolute+absolute**2)/E0
    bridge=norm(data['reference_states'][0]-c0)
    bridge_H=abs(H(data['reference_states'][0])-H(c0))
    old_machine=64*np.finfo(float).eps*(4+8+2)*max(1.,float(np.sqrt(m@(c0*c0))))/sigma
    rows=[]
    for k,t in enumerate(data['reference_times']):
        i4=k*2;i8=k*4
        a=saved['k8_states'][i8];a4=saved['k4_states'][i4]
        b=data['k8_states'][i8];b4=data['k4_states'][i4];ref=data['reference_states'][k]
        new_time=norm(a4-a);old_time=norm(b4-b)
        ref_space=norm(data['reference_space_difference'][k]);ref_time=norm(data['reference_time_difference'][k])
        solver_abs=0.;solver_observed=0.;solver_H_observed=0.
        for K,idx in ((4,i4),(8,i8)):
            if idx:
                r=cases[str(K)]['steps'][idx-1]
                solver_abs+=r['cumulative_machine_indicator_absolute']+r['cumulative_tail_indicator_absolute']
                solver_observed+=r['primary_control_relative_difference']
                solver_H_observed+=abs(r['state_metrics']['H']-r['independent_metrics']['H'])
        indicator=new_time+old_time+2*(ref_space+ref_time+bridge)+solver_abs/sigma+solver_observed+old_machine
        efv=norm(a-ref);ek=norm(b-ref);gain=ek-efv
        dHnew=abs(H(a4)-H(a));dHold=abs(H(b4)-H(b))
        # Time difference is M127-k128 minus best M127-k256. Space difference
        # is M63-k128 minus M127-k128, so use that same time base for both sides.
        # These rounded reconstructions only estimate saved H differences;
        # they never replace the original best reference used for errors.
        coarse_time=ref+data['reference_time_difference'][k]
        ref_H_space=abs(H(coarse_time+data['reference_space_difference'][k])-H(coarse_time))
        ref_H_time=abs(H(coarse_time)-H(ref))
        H_indicator=dHnew+dHold+2*(ref_H_space+ref_H_time+bridge_H)+solver_H_observed+h_allow(solver_abs+old_machine*sigma)
        fv_H_error=abs(H(a)-H(ref));old_H_error=abs(H(b)-H(ref));H_gain=old_H_error-fv_H_error
        classify=lambda gain,budget:'FV_better_resolvably' if gain>budget else ('FV_worse_resolvably' if gain < -budget else 'comparison_inconclusive')
        row={'time':float(t),'FV_relative_concentration_error':efv,'saved_kernel_relative_concentration_error':ek,
            'kernel_over_FV_error_ratio':None if efv==0 else ek/efv,'concentration_error_gain':gain,
            'FV_4_8_state_difference':new_time,'saved_kernel_4_8_state_difference':old_time,
            'reference_space_indicator':ref_space,'reference_time_indicator':ref_time,
            'initial_reference_bridge_relative':bridge,'new_solver_machine_indicator_relative':solver_abs/sigma,
            'new_solver_observed_difference_sum_relative':solver_observed,'old_solver_machine_indicator_relative':old_machine,
            'combined_state_comparison_indicator':indicator,'state_decision':classify(gain,indicator),
            'FV_H':H(a),'saved_kernel_H':H(b),'reference_H':H(ref),
            'FV_H_signed_error':H(a)-H(ref),'saved_kernel_H_signed_error':H(b)-H(ref),
            'H_absolute_error_gain':H_gain,'FV_4_8_H_difference':dHnew,'saved_kernel_4_8_H_difference':dHold,
            'reference_H_space_indicator':ref_H_space,'reference_H_time_indicator':ref_H_time,
            'initial_reference_H_bridge':bridge_H,'combined_H_comparison_indicator':H_indicator,
            'H_decision':classify(H_gain,H_indicator)}
        rows.append(row)
    nonzero=rows[1:]
    decision='FV_better_at_both_checkpoints_resolvably' if all(r['state_decision']=='FV_better_resolvably' for r in nonzero) else (
        'FV_worse_at_a_checkpoint_resolvably' if any(r['state_decision']=='FV_worse_resolvably' for r in nonzero) else 'comparison_inconclusive')
    return {'original_initial_mass_fluctuation_norm':sigma,'observations':rows,'concentration_comparison_decision':decision,
        'H_better_at_both_checkpoints_resolvably':all(r['H_decision']=='FV_better_resolvably' for r in nonzero),
        'reference_and_time_indicators_are_not_certified_error_bounds':True,
        'two_time_levels_prove_asymptotic_order':False,'extrapolated_or_corrected_state_used':False,
        'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}


def main(output,tests,test_log):
    prior=read(SOURCE/'sha256.json')
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    spec=read(SOURCE/'next-gate-spec.json');protocol=read(PROTOCOL)
    assert not spec['already_executed'] and spec['ordered_interval_counts']==[4,8] and spec['expected_new_midpoint_graphs']==12
    assert read(tests)=={'successful':True,'checks_run':6,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,path in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),
                      ('control-at-lock.md',CONTROL),('tests.json',Path(tests)),('tests.log',Path(test_log))]:
        shutil.copyfile(path,out/name)
    locked=['periodic_voronoi_fv.py','voronoi_time_action.py','test_voronoi_time_action.py','shear_voronoi_time_study.py']
    save(out/'implementation-lock.json',{'files':{name:digest(ROOT/'research/reconnect'/name) for name in locked},
        'protocol_sha256':digest(PROTOCOL),'control_note_sha256':digest(CONTROL),'real_midpoint_geometries_not_evaluated_before_lock':True})
    save(out/'config.json',{'source_manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
        'source_manifest_sha256':digest(SOURCE/'sha256.json'),'source_manifest_entries_verified':len(prior),
        'protocol_sha256':digest(SOURCE/'next-gate-spec.json'),'new_checks':6,'recorded_checks_total':445,
        'scope':'one original smooth shear actual scalar; necessary4/8 time control, no shock/scan/reference solve'})
    for key in ('builder_input_packet','comparison_packet'):
        assert digest(ROOT/spec[key])==spec[key+'_sha256']
    shutil.copyfile(ROOT/spec['builder_input_packet'],out/'builder-inputs.npz')
    b=source_data(out/'builder-inputs.npz');index={};started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>600 or rss>512:raise RuntimeError(f'fixed resource cap exceeded:{elapsed:.3f}s/{rss:.2f}MiB')
        if not total or k%4096==0 or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{rss:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('registered600second main cap reached')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    try:
        for K in (4,8):
            for step in range(K):
                frame=2*step+1
                graph=build_periodic_faces(b['ids'],b[f'k{K}_positions'][frame],b[f'k{K}_rho'][frame],
                    spec['D'],spec['periods'],spec['unit_thickness'],guard)
                probes=verify_probe_cells(graph,protocol['fixed_geometry_probe_ids'])
                packet=coefficient_packet(graph);name=f'geometry-k{K}-step{step:02d}.json'
                save(out/name,packet)
                index[f'k{K}-step{step:02d}']={'file':name,'sha256':digest(out/name),'K':K,'step':step,'frame':frame,
                    'midpoint_time_saved':float(b[f'k{K}_midpoint_time'][step]),'positions':record(b[f'k{K}_positions'][frame]),
                    'rho':record(b[f'k{K}_rho'][frame]),'geometry_diagnostics':graph.diagnostics,'independent_fixed_probes':probes}
                del packet,graph;gc.collect();guard(f'sealed-geometry-k{K}-step{step:02d}')
        save(out/'graph-index.json',index)
        save(out/'coefficient-seal.json',{'graph_index_sha256':digest(out/'graph-index.json'),
            'graphs':12,'pure_builder_packet_sha256':digest(out/'builder-inputs.npz'),
            'all12_geometry_and_conductance_caches_sealed_before_mass_c0_reference_load':True,
            'geometry_builder_received_mass_scalar_reference_or_true_faces':False,
            'original_builder_array_records':{k:record(v) for k,v in b.items()}})
        del b;gc.collect();guard('all12coefficients-sealed-before-state')
        shutil.copyfile(ROOT/spec['comparison_packet'],out/'original-comparison.npz')
        data=source_data(out/'original-comparison.npz')
        save(out/'input-provenance.json',{'source_builder_packet':spec['builder_input_packet'],
            'source_comparison_packet':spec['comparison_packet'],'source_comparison_sha256':digest(out/'original-comparison.npz'),
            'comparison_array_records':{k:record(v) for k,v in data.items()},
            'all_original_inputs_reused_no_new_PDE_solve_or_flow_definition':True})
        cases,arrays,sigma=propagate(out,data,index,guard)
        comparison=compare(data,cases,arrays,sigma)
        save(out/'results.json',{'cases':cases,**comparison})
        np.savez_compressed(out/'trajectory-states.npz',**arrays)
        guard('actual-scalar-and-comparison-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-01','new_software_checks':6,'previous_recorded_checks':439,
            'recorded_checks_total':445,'previous_checks_rerun':0,'new_candidate_graphs':12,'new_actual_flow_cases':1,
            'new_candidate_transient_runs':2,'independent_uniformization_sequences':2,'new_PDE_reference_solves':0,
            'new_physical_flow_definitions':0,'new_kernel_builds':0,'new_full_dense_generators_or_propagators':0,
            'source_registered_smooth_shear_time_gate_executed':True,'shock_extension_deferred':True,
            'concentration_comparison_decision':comparison['concentration_comparison_decision'],
            'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,
            'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
            'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'test_first_attempt_passed':True,
            'scientific_main_first_attempt_passed':True,'elapsed_main_seconds':time.monotonic()-started,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'resource_caps_passed':True,
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'threads':4,'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps(comparison,indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'error_type':type(e).__name__,'error':str(e),
            'elapsed_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'partial_evidence_retained':True,'no_threshold_or_state_correction':True})
        raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();data=source_data(out/'original-comparison.npz');index=read(out/'graph-index.json')
    cases,arrays,sigma=propagate(out,data,index,replay=True)
    assert {'cases':cases,**compare(data,cases,arrays,sigma)}==read(out/'results.json')
    with np.load(out/'trajectory-states.npz',allow_pickle=False) as z:
        assert set(z.files)==set(arrays)
        for k,v in arrays.items():assert np.array_equal(z[k],v),k
    print('Same12 sealed exact-coefficient graphs: original-mass chronological4/8 exponential and full independent uniformization sequences, states and metrics replay; no geometry/PDE solve, shock or ORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');args=p.parse_args()
    if args.replay:replay(args.replay)
    elif args.output and args.tests and args.test_log:main(args.output,args.tests,args.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
