"""One preregistered smooth compression snapshot: original-mass auxiliary FV."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import resource
import shutil
import signal
import time
import numpy as np

from audit_route_decision import ROOT,read,save,digest
from auxiliary_tensor_fv import build_auxiliary_faces,coefficient_packet,exact_action,independent_float_action

SOURCE=ROOT/'runs/reconnect-capacity-defect'


def record(a):
    return {'shape':list(a.shape),'dtype':a.dtype.str,'C_order_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def rational(q):return {'numerator_hex':hex(q.numerator),'denominator_hex':hex(q.denominator),'display':float(q)}
def norm(m,v):return math.sqrt(math.fsum(float(a)*float(b)**2 for a,b in zip(m,v)))


def evaluate(graph,data):
    m=data['mass'];u=data['c0'];reference=data['reference_rate'];kernel=data['kernel_rate']
    q=exact_action(graph,m,u);primary=np.array(list(map(float,q)))
    control=independent_float_action(graph,m,u,data['rho'])
    mm=tuple(F(float(v)) for v in m);uu=tuple(F(float(v)) for v in u)
    degree=[F(0)]*len(m)
    for (i,j),g in zip(graph.edges,graph.conductances):degree[i]+=g;degree[j]+=g
    row_L1=max(2*a/b for a,b in zip(degree,mm));M=sum(mm,F(0))
    # Fixed before the real comparison; numerical allowance, not PDE tolerance.
    allowance=64*np.finfo(float).eps*max(1.,math.sqrt(float(M)))*max(1.,float(row_L1))*max(1.,float(np.max(abs(u))))
    delta=norm(m,primary-control)
    if not np.isfinite(primary).all() or delta>allowance:raise ArithmeticError('independent action control failed; no coefficient or tolerance repair')
    reference_norm=norm(m,reference)
    if reference_norm==0:raise ValueError('fixed source full reference norm is zero; no substitute denominator')
    efv=norm(m,primary-reference)/reference_norm;ek=norm(m,kernel-reference)/reference_norm
    rule=norm(m,data['gauss8_rate']-data['gauss16_rate'])/reference_norm
    representation=norm(m,reference-data['gauss16_rate'])/reference_norm
    roundmax=max(abs(F(float(v))-w) for v,w in zip(primary,q))
    rounding_indicator=math.sqrt(float(M))*math.nextafter(float(roundmax),math.inf)
    candidate_arithmetic=(allowance+delta+rounding_indicator)/reference_norm
    kernel_arithmetic=64*np.finfo(float).eps*max(1.,norm(m,kernel))/reference_norm
    reference_indicator=rule+representation
    comparison_indicator=2*reference_indicator+candidate_arithmetic+kernel_arithmetic
    gain=ek-efv
    decision='FV_better_than_saved_kernel_resolvably' if gain>comparison_indicator else (
        'FV_worse_than_saved_kernel_resolvably' if gain < -comparison_indicator else 'comparison_inconclusive')
    raw_inventory=sum((a*F(float(b)) for a,b in zip(mm,primary)),F(0))
    exact_inventory=sum((a*b for a,b in zip(mm,q)),F(0));assert exact_inventory==0
    chi=sum((g*(uu[j]-uu[i])**2 for (i,j),g in zip(graph.edges,graph.conductances)),F(0))
    projected=-sum((a*b*c for a,b,c in zip(mm,uu,q)),F(0));assert projected==chi>=0
    raw_chi=-math.fsum(float(a)*float(b)*float(c) for a,b,c in zip(m,u,primary))
    results={'scope':'one frozen smooth nonshock compression state, not an evolved scalar or general-flow acceptance',
        'N':128,'time_label_exact':'1/40','state':'unchanged saved true material mass means',
        'normalization':'mass norm of same saved full PDE reference rate, including density drift',
        'reference_rate_mass_norm':reference_norm,'FV_relative_rate_error':efv,'saved_kernel_relative_rate_error_recomputed':ek,
        'saved_kernel_over_FV_error_ratio':None if efv==0 else ek/efv,'error_reduction':gain,
        'saved_reference_rule_difference_indicator':rule,'saved_reference_formula_difference_indicator':representation,
        'candidate_arithmetic_indicator_relative':candidate_arithmetic,'kernel_arithmetic_indicator_relative':kernel_arithmetic,
        'combined_comparison_indicator_relative':comparison_indicator,'comparison_decision':decision,
        'control_indicators_are_not_certified_total_PDE_error_bounds':True,
        'maximum_absolute_rate_error':float(np.max(abs(primary-reference))),
        'exact_shared_mass_rate_sum':rational(exact_inventory),'raw_float_output_mass_rate_sum':rational(raw_inventory),
        'exact_graph_dissipation':rational(chi),'exact_energy_pairing_residual':rational(projected-chi),
        'raw_float_output_graph_projection':raw_chi,'raw_float_projection_minus_exact_pair_display':raw_chi-float(chi),
        'PDE_material_mean_projection_not_continuous_dissipation':-math.fsum(float(a)*float(b)*float(c) for a,b,c in zip(m,u,reference)),
        'saved_kernel_material_mean_projection':-math.fsum(float(a)*float(b)*float(c) for a,b,c in zip(m,u,kernel)),
        'minimum_state_target_rate':float(primary[int(np.argmin(u))]),
        'maximum_state_target_rate':float(primary[int(np.argmax(u))]),
        'no_scalar_propagation_no_I_H_prediction':True,'no_new_absolute_physical_threshold':True,
        'ordinary_central_resistance_not_mass_average_resistance_flux':True,
        'auxiliary_mass_not_used_in_generator':True,'general_nontensor_smooth_flow_admitted':False,
        'finite_node_candidate_admitted':False,'contact_transient_physically_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    controls={'float_action_difference_mass_norm':delta,'predeclared_machine_allowance_mass_norm':float(allowance),
        'predeclared_allowance_formula':'64*epsilon*max(1,sqrt(total actual mass))*max(1,max row L1)*max(1,max abs c0)',
        'maximum_row_L1_exact':str(row_L1),'primary_conversion_maximum_error':rational(roundmax),
        'primary_conversion_mass_norm_indicator':rounding_indicator,'exact_mass_energy_pairing_verified':True,
        'edge_count':len(graph.edges),'minimum_conductance':float(min(graph.conductances)),
        'maximum_conductance':float(max(graph.conductances)),
        'new_global_matrices':0,'full_propagators':0}
    arrays={'FV_rate':primary,'independent_float_rate':control,'FV_error':primary-reference,'kernel_error':kernel-reference}
    return results,controls,arrays,q


def main(output,tests,test_log):
    parent=read(SOURCE/'sha256.json')
    for rel,h in parent.items():assert digest(ROOT/rel)==h,rel
    spec=read(SOURCE/'next-gate-spec.json');assert not spec['already_executed'] and spec['candidate_state_actions_max']==1
    checks=read(tests);assert checks=={'successful':True,'checks_run':5,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',{'protocol':spec,'protocol_sha256':digest(SOURCE/'next-gate-spec.json'),
        'source_manifest_entries_verified':len(parent),'user_scope':'smooth nonshock feasibility first; shock extension deferred',
        'builder_source_sha256':digest(ROOT/'research/reconnect/auxiliary_tensor_fv.py'),
        'new_software_checks':5,'recorded_checks_total':433,'old_checks_not_rerun':428,
        'predeclared_roundoff_allowance_formula':'64*epsilon*max(1,sqrt(total actual mass))*max(1,max row L1)*max(1,max abs c0)',
        'input_rows_and_original_labels_not_reordered_internal_geometry_order_only':True})
    for dest,path in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('tests.json',Path(tests)),('tests.log',Path(test_log)),
        ('tests-first-attempt.json',Path('/tmp/ltn-auxiliary-tensor-tests.json')),
        ('tests-first-attempt.log',Path('/tmp/ltn-auxiliary-tensor-tests.log'))]:shutil.copyfile(path,out/dest)
    packet=read(SOURCE/'saved-input.json')
    builder={'ids':np.asarray(packet['ids'],dtype=np.int64),'positions':np.asarray(packet['positions'],dtype=float),
        'rho':np.asarray(packet['rho'],dtype=float)}
    np.savez_compressed(out/'builder-inputs.npz',**builder)
    started=time.monotonic();stage=['construction']
    def guard(label):
        stage[0]=label;elapsed=time.monotonic()-started;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>600 or rss>512:raise RuntimeError(f'predeclared resource cap exceeded: {elapsed}s, {rss}MiB')
        print(f'{label}; {elapsed:.3f}s; peak {rss:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('predeclared600second limit reached')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    try:
        graph=build_auxiliary_faces(builder['ids'],builder['positions'],builder['rho'],spec['D'],spec['periods'],spec['unit_thickness'])
        save(out/'coefficient-cache.json',coefficient_packet(graph));guard('coefficients-built')
        save(out/'coefficient-seal.json',{'coefficient_cache_sha256':digest(out/'coefficient-cache.json'),
            'builder_source_sha256':digest(ROOT/'research/reconnect/auxiliary_tensor_fv.py'),
            'builder_array_records':{k:record(v) for k,v in builder.items()},
            'actual_scalar_and_reference_not_loaded_until_after_this_seal':True,
            'builder_received_mass_scalar_reference_true_geometry':False})
        guard('coefficient-seal-written-before-scalar')
        assert digest(ROOT/spec['source_npz'])==spec['source_npz_sha256']
        with np.load(ROOT/spec['source_npz'],allow_pickle=False) as z:
            data={k:z[k].copy() for k in ['ids','mass','c0','reference_rate','kernel_rate','gauss8_rate','gauss16_rate']}
            data['positions']=z['positions'][spec['snapshot_index']].copy();data['rho']=z['rho'][spec['snapshot_index']].copy()
        for k,v in builder.items():assert record(v)==record(data[k]),k
        assert record(data['mass'])==read(SOURCE/'input-provenance.json')['copied_array_records']['mass']
        np.savez_compressed(out/'original-state.npz',**data)
        save(out/'input-provenance.json',{'source_npz':spec['source_npz'],'source_npz_sha256':digest(ROOT/spec['source_npz']),
            'parent_input_screen_packet_sha256':digest(SOURCE/'saved-input.json'),
            'original_state_npz_sha256':digest(out/'original-state.npz'),
            'array_records':{k:record(v) for k,v in data.items()},'no_state_mass_density_or_reference_replacement':True})
        guard('original-state-matched-after-coefficient-seal')
        result,controls,arrays,q=evaluate(graph,data);guard('one-state-and-independent-control-completed')
        save(out/'results.json',result);save(out/'action-controls.json',controls)
        save(out/'rates-exact.json',{'exact_rates':list(map(str,q))})
        np.savez_compressed(out/'actions.npz',**arrays)
        shutil.copyfile(ROOT/'runs/reconnect-kernel2d-compression-midpoint/results.json',out/'source-kernel-results.json')
        save(out/'stage3-status.json',{'updated':'2026-11-01','timezone':'Etc/UTC','new_software_checks':5,
            'previous_recorded_checks':428,'recorded_checks_total':433,'previous_checks_rerun':0,
            'new_candidate_graphs':1,'new_candidate_conductances':len(graph.edges),'new_actual_frozen_state_cases':1,
            'new_candidate_primary_actions':1,'independent_float_action_controls':1,'new_physical_flow_definitions':0,
            'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,'new_kernel_builds':0,'new_density_path_builds':0,
            'new_global_matrices':0,'new_exact_algebra_certificate_groups':0,
            'comparison_decision':result['comparison_decision'],'user_scope':'smooth nonshock flows before shock extension',
            'general_nontensor_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,
            'contact_transient_physically_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
            'next_work':'interpret fixed smooth-flow action and preregister at most one necessary smooth-flow control; no shock work or automatic scan',
            'pdf_generated':False,'software_release_created':False,'published':False})
        guard('saved-outputs');signal.alarm(0)
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'test_attempts':2,
            'first_test_attempt_passed':False,'test_expectation_correction':'unexpected c0 is rejected by Python signature TypeError, not ValueError',
            'backend_modified_after_initial_test_error':False,'main_study_first_attempt_passed':True,
            'source_manifest_entries_verified':len(parent),'elapsed_main_seconds':time.monotonic()-started,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'resource_caps_passed':True,'wall_clock_cap_seconds':600,'peak_RSS_MiB_cap':512,
            'python':platform.python_version(),'numpy':np.__version__,'new_software_checks':5,'recorded_checks_total':433,
            'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
        print(json.dumps({k:result[k] for k in ['reference_rate_mass_norm','FV_relative_rate_error',
            'saved_kernel_relative_rate_error_recomputed','saved_kernel_over_FV_error_ratio','error_reduction',
            'combined_comparison_indicator_relative','comparison_decision']},indent=2),flush=True)
    except BaseException as error:
        signal.alarm(0);save(out/'failure.json',{'type':type(error).__name__,'reason':str(error),'last_stage':stage[0],
            'elapsed_seconds':time.monotonic()-started,'partial_evidence_retained':True,
            'no_input_coefficient_or_threshold_repair':True,'fluid_orc_enabled':False});raise


def replay(directory):
    out=Path(directory);s=read(out/'source-gate-spec.json')
    with np.load(out/'builder-inputs.npz',allow_pickle=False) as z:b={k:z[k] for k in z.files}
    graph=build_auxiliary_faces(b['ids'],b['positions'],b['rho'],s['D'],s['periods'],s['unit_thickness'])
    assert coefficient_packet(graph)==read(out/'coefficient-cache.json')
    with np.load(out/'original-state.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    result,controls,arrays,q=evaluate(graph,data)
    assert list(map(str,q))==read(out/'rates-exact.json')['exact_rates']
    assert result==read(out/'results.json') and controls==read(out/'action-controls.json')
    with np.load(out/'actions.npz',allow_pickle=False) as z:
        for k,v in arrays.items():assert np.array_equal(z[k],v),k
    print('Same original smooth-state coefficients, exact action, independent float control and metrics replay; no shock or transient case and ORC disabled.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');args=p.parse_args()
    if args.replay:replay(args.replay)
    elif all([args.output,args.tests,args.test_log]):main(args.output,args.tests,args.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
