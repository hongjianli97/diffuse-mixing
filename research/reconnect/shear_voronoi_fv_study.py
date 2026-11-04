"""One preregistered nonproduct smooth shear: exact periodic auxiliary-face FV."""
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
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet,exact_action,rate_digest,enclosure,dot,frac

SOURCE=ROOT/'runs/reconnect-smooth-auxiliary-FV-frozen'
PROTOCOL=ROOT/'research/reconnect/periodic_voronoi_protocol.json'
CONSTRUCTION=ROOT/'research/reconnect/PERIODIC_VORONOI_CONSTRUCTION.md'


def record(a):return {'shape':list(a.shape),'dtype':a.dtype.str,'C_order_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}
def number(q):return {'exact':str(q),'display':float(q)}
def norm(m,v):return math.sqrt(math.fsum(float(a)*float(b)**2 for a,b in zip(m,v)))


def verify_probe_cells(graph,probe_ids):
    out=[];id_lookup={v:k for k,v in enumerate(graph.ids)}
    for ident in probe_ids:
        i=id_lookup[ident];poly=graph.polygons[i];R=max(abs(q[0])+abs(q[1]) for q,_ in poly)
        if R>=min(graph.periods)/4:raise ValueError('independent probe nearest-image condition unsupported')
        possible=0;worst=None
        for j,p in enumerate(graph.points):
            if j==i:continue
            delta=[]
            for k,L in enumerate(graph.periods):
                q=p[k]-graph.points[i][k]
                if q>L/2:q-=L
                elif q<-L/2:q+=L
                delta.append(q)
            if abs(delta[0])>2*R or abs(delta[1])>2*R:continue
            v=tuple(delta);d2=dot(v,v);possible+=1
            value=max(2*dot(v,q)-d2 for q,_ in poly)
            if value>0:raise ArithmeticError('independent all-point halfplane probe failed')
            worst=value if worst is None else max(worst,value)
        out.append({'id':ident,'other_points_checked':len(graph.ids)-1,'possible_cutters_tested':possible,
            'maximum_signed_halfplane_residual_exact':str(worst),'final_radius_L1_exact':str(R)})
    return {'fixed_ids':probe_ids,'all_selected_cells_passed':True,'records':out}


def evaluate(graph,data):
    m=data['mass'];u=data['c0'];rho=data['rho'];n=len(m);mm=tuple(map(frac,m));uu=tuple(map(frac,u))
    q=exact_action(graph,m,u);primary=np.array(list(map(float,q)));rows=[[] for _ in range(n)];degree=[F(0)]*n
    energy_intervals=[]
    def energies():
        for (i,j),g in zip(graph.edges,graph.conductances):
            d=uu[j]-uu[i];f=g*d;energy=g*d*d
            if -uu[i]*f+uu[j]*f!=energy:raise ArithmeticError('exact paired energy identity failed')
            yield energy
    energy=enclosure(energies())
    for k,((i,j),g) in enumerate(zip(graph.edges,graph.conductances)):
        degree[i]+=g;degree[j]+=g
        d=math.sqrt(float(dot(graph.displacements[k],graph.displacements[k])))
        face=float(graph.ratios[k])*d;gi=float(rho[i])*float(graph.D);gj=float(rho[j])*float(graph.D)
        control_g=0. if graph.D==0 else float(graph.thickness)*face/(d/(2*gi)+d/(2*gj))
        if not math.isfinite(control_g) or control_g<0:raise ArithmeticError('invalid independent half-resistance coefficient')
        rows[i].append((j,control_g));rows[j].append((i,control_g))
    control=np.array([math.fsum(w*(float(u[j])-float(u[i])) for j,w in row)/float(m[i]) for i,row in enumerate(rows)])
    M=sum(mm,F(0));rowL1=max(2*w/a for w,a in zip(degree,mm));maxdegree=max(map(len,rows))
    allowance=float(64*np.finfo(float).eps*(maxdegree+1)*max(1.,math.sqrt(float(M)))*max(1.,float(rowL1))*max(1.,float(np.max(abs(u)))))
    delta=norm(m,primary-control)
    if not np.isfinite(primary).all() or delta>allowance:raise ArithmeticError('independent action control failed; no tolerance repair')
    R=norm(m,data['reference_rate'])
    if R==0:raise ValueError('original full reference norm is zero; no replacement denominator')
    efv=norm(m,primary-data['reference_rate'])/R;ek=norm(m,data['kernel_rate']-data['reference_rate'])/R
    rule=norm(m,data['gauss8_rate']-data['gauss16_rate'])/R
    formula=norm(m,data['reference_rate']-data['gauss16_rate'])/R
    conversion=max(abs(frac(v)-w) for v,w in zip(primary,q))
    arithmetic=(allowance+delta+math.sqrt(float(M))*math.nextafter(float(conversion),math.inf))/R
    old_arithmetic=64*np.finfo(float).eps*max(1.,norm(m,data['kernel_rate']))/R
    comparison=2*(rule+formula)+arithmetic+old_arithmetic;gain=ek-efv
    decision='FV_better_than_saved_kernel_resolvably' if gain>comparison else (
        'FV_worse_than_saved_kernel_resolvably' if gain < -comparison else 'comparison_inconclusive')
    rawmass=sum((a*frac(v) for a,v in zip(mm,primary)),F(0))
    auxiliary=tuple(graph.thickness*r*a for r,a in zip(graph.rho,graph.areas));ratios=tuple(a/b for a,b in zip(auxiliary,mm))
    rmin,rmax=min(ratios),max(ratios);eta=max(abs(v-1) for v in ratios);spread=rmax/rmin
    mass_interval=enclosure(auxiliary)
    capacity={'minimum_ratio':number(rmin),'maximum_ratio':number(rmax),'eta':number(eta),'ratio_spread':number(spread),
        'same_fixed_L_capacity_comparison_coefficient':number(eta*(2+spread)),
        'auxiliary_total_mass_interval':mass_interval,'actual_total_mass_exact':str(M),
        'auxiliary_minus_actual_mass_interval':{'lower_exact':str(F(mass_interval['lower_exact'])-M),
            'upper_exact':str(F(mass_interval['upper_exact'])-M),
            'raw_float_display':mass_interval['raw_float_fsum']-float(M)},
        'not_a_total_PDE_error_bound_no_mass_replacement':True}
    epsilon_c=epsilon_2=F(0)
    for (i,j),v in zip(graph.edges,graph.displacements):
        lower=max(abs(v[0]),abs(v[1]))
        epsilon_c=max(epsilon_c,(sum(map(abs,graph.centroids[i]))+sum(map(abs,graph.centroids[j])))/lower)
        epsilon_2=max(epsilon_2,(graph.second_moments[i]+graph.second_moments[j])/lower)
    result={'scope':'one original nonproduct periodic smooth shear frozen state, not a transient or general-flow acceptance',
        'N':128,'time_label_exact':'1/20','original_actual_material_means_and_masses':True,
        'reference_rate_mass_norm':R,'normalization':'same full saved shear PDE rate including mixed/shear/density terms',
        'FV_relative_rate_error':efv,'saved_kernel_relative_rate_error_recomputed':ek,
        'saved_kernel_over_FV_error_ratio':None if efv==0 else ek/efv,'error_reduction':gain,
        'reference_rule_indicator':rule,'reference_formula_indicator':formula,
        'candidate_arithmetic_indicator_relative':arithmetic,'kernel_arithmetic_indicator_relative':old_arithmetic,
        'combined_comparison_indicator_relative':comparison,'comparison_decision':decision,
        'indicators_are_not_certified_total_PDE_error_bounds':True,
        'raw_float_mass_rate_sum':number(rawmass),'mass_pair_identity_exact_and_global_sum_zero_by_shared_construction':True,
        'exact_face_energy_pairings_verified':True,'graph_dissipation_exact_sum_enclosure':energy,
        'raw_float_graph_projection':-math.fsum(float(a)*float(b)*float(c) for a,b,c in zip(m,u,primary)),
        'PDE_material_mean_projection_not_continuous_dissipation':-math.fsum(float(a)*float(b)*float(c) for a,b,c in zip(m,u,data['reference_rate'])),
        'epsilon_c_L1_over_Linf_upper':number(epsilon_c),'epsilon_2_over_Linf_upper':number(epsilon_2),
        'true_material_observation_error_not_inferred_from_auxiliary_geometry':True,
        'new_scalar_propagation_or_I_H_prediction':False,'new_absolute_physical_threshold':False,
        'nonproduct_geometry_evaluated':True,'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    controls={'independent_action_difference_mass_norm':delta,'predeclared_machine_allowance_mass_norm':allowance,
        'maximum_degree':maxdegree,'maximum_row_L1_exact':str(rowL1),
        'primary_conversion_maximum_error_exact':str(conversion),'edge_count':len(graph.edges),
        'minimum_conductance':float(min(graph.conductances)),'maximum_conductance':float(max(graph.conductances)),
        'exact_rate_digest':rate_digest(q),'exact_rate_hash_schema':'hex(numerator)/hex(denominator) newline per original input row',
        'global_rational_sums_use_certified_dyadic_enclosures_not_huge_LCM':True,'new_full_global_matrices':0}
    arrays={'FV_rate':primary,'independent_float_rate':control,'FV_error':primary-data['reference_rate'],
        'kernel_error':data['kernel_rate']-data['reference_rate']}
    return result,controls,capacity,arrays


def main(output,tests,test_log):
    parent=read(SOURCE/'sha256.json')
    for rel,h in parent.items():assert digest(ROOT/rel)==h,rel
    spec=read(SOURCE/'next-gate-spec.json');protocol=read(PROTOCOL)
    assert not spec['already_executed'] and spec['one_saved_state_max']==1
    checks=read(tests);assert checks=={'successful':True,'checks_run':6,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for dest,src in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),
        ('construction-at-lock.md',CONSTRUCTION),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(src,out/dest)
    save(out/'config.json',{'parent_manifest_entries_verified':len(parent),'parent_protocol_sha256':digest(SOURCE/'next-gate-spec.json'),
        'new_software_checks':6,'recorded_checks_total':439,'previous_checks_rerun':0,
        'scope':'smooth nonshock first, one existing nonproduct shear snapshot',
        'all_implementation_options_locked_before_real_geometry':True})
    save(out/'implementation-lock.json',{'source_protocol_sha256':digest(PROTOCOL),'construction_sha256':digest(CONSTRUCTION),
        'builder_source_sha256':digest(ROOT/'research/reconnect/periodic_voronoi_fv.py'),
        'test_source_sha256':digest(ROOT/'research/reconnect/test_periodic_voronoi_fv.py'),
        'real_geometry_or_scalar_not_evaluated_before_lock':True})
    assert digest(ROOT/spec['source_npz'])==spec['source_npz_sha256']
    with np.load(ROOT/spec['source_npz'],allow_pickle=False) as z:
        assert float(z['time'][1])==spec['source_time_display']
        b={'ids':z['ids'].copy(),'positions':z['positions'][1].copy(),'rho':z['rho'][1].copy()}
    np.savez_compressed(out/'builder-inputs.npz',**b)
    started=time.monotonic();last=['start']
    def guard(stage,k=0,total=0):
        last[0]=f'{stage}-{k}/{total}';elapsed=time.monotonic()-started;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>600 or rss>512:raise RuntimeError(f'predeclared resource limit exceeded: {elapsed}s {rss}MiB')
        if not total or k%1024==0 or k==total:print(f'{last[0]}; {elapsed:.3f}s; peak {rss:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('predeclared600second cap reached')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    try:
        graph=build_periodic_faces(b['ids'],b['positions'],b['rho'],spec['D'],spec['periods'],spec['unit_thickness'],guard)
        guard('geometry-built');save(out/'geometry-controls.json',graph.diagnostics)
        probes=verify_probe_cells(graph,protocol['independent_actual_geometry_probe_ids']);save(out/'independent-geometry-probes.json',probes)
        guard('independent-global-halfplane-probes-passed')
        packet=coefficient_packet(graph);save(out/'coefficient-cache.json',packet);del packet;guard('coefficients-saved')
        save(out/'coefficient-seal.json',{'coefficient_cache_sha256':digest(out/'coefficient-cache.json'),
            'builder_array_records':{k:record(v) for k,v in b.items()},
            'geometry_and_all_coefficients_sealed_before_scalar_reference_access':True,
            'actual_mass_scalar_reference_true_geometry_received_by_builder':False})
        with np.load(ROOT/spec['source_npz'],allow_pickle=False) as z:
            data={k:z[k].copy() for k in ['ids','mass','c0','reference_rate','kernel_rate','gauss8_rate','gauss16_rate']}
            data['positions']=z['positions'][1].copy();data['rho']=z['rho'][1].copy()
        for k,v in b.items():assert record(v)==record(data[k]),k
        np.savez_compressed(out/'original-state.npz',**data)
        save(out/'input-provenance.json',{'source_npz':spec['source_npz'],'source_npz_sha256':digest(ROOT/spec['source_npz']),
            'original_state_sha256':digest(out/'original-state.npz'),'array_records':{k:record(v) for k,v in data.items()},
            'original_mass_means_points_rho_and_reference_preserved':True})
        result,controls,capacity,arrays=evaluate(graph,data);guard('fixed-state-action-completed')
        save(out/'results.json',result);save(out/'action-controls.json',controls);save(out/'capacity-diagnostics.json',capacity)
        np.savez_compressed(out/'actions.npz',**arrays)
        shutil.copyfile(ROOT/'runs/reconnect-shear-frozen/results.json',out/'source-kernel-results.json')
        save(out/'stage3-status.json',{'updated':'2026-11-01','timezone':'Etc/UTC','new_software_checks':6,
            'previous_recorded_checks':433,'recorded_checks_total':439,'previous_checks_rerun':0,
            'new_candidate_graphs':1,'new_actual_frozen_state_cases':1,'new_candidate_conductances':len(graph.edges),
            'new_candidate_primary_actions':1,'independent_float_action_controls':1,'new_physical_flow_definitions':0,
            'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,'new_kernel_builds':0,'new_density_path_builds':0,
            'new_full_global_matrices':0,'new_exact_algebra_certificate_groups':0,
            'comparison_decision':result['comparison_decision'],'nonproduct_smooth_geometry_evaluated':True,
            'shock_extension_deferred':True,'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,
            'contact_transient_physically_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
            'next_work':'read fixed smooth shear result and select at most one necessary smooth convergence/transient control; no shock or automatic scan',
            'pdf_generated':False,'software_release_created':False,'published':False})
        guard('outputs-saved');signal.alarm(0)
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'first_attempts_passed':True,
            'source_manifest_entries_verified':len(parent),'elapsed_main_seconds':time.monotonic()-started,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'resource_caps_passed':True,
            'wall_clock_cap_seconds':600,'peak_RSS_MiB_cap':512,'python':platform.python_version(),'numpy':np.__version__,
            'new_software_checks':6,'recorded_checks_total':439,'network_used':False,'environment_configuration_changed':False,
            'committed_or_pushed':False,'published':False})
        print(json.dumps({k:result[k] for k in ['reference_rate_mass_norm','FV_relative_rate_error',
            'saved_kernel_relative_rate_error_recomputed','saved_kernel_over_FV_error_ratio',
            'error_reduction','combined_comparison_indicator_relative','comparison_decision']},indent=2),flush=True)
    except BaseException as error:
        signal.alarm(0);save(out/'failure.json',{'type':type(error).__name__,'reason':str(error),'last_stage':last[0],
            'elapsed_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'partial_evidence_retained':True,'no_input_geometry_coefficient_or_threshold_repair':True,'fluid_orc_enabled':False});raise


def replay(directory):
    out=Path(directory);spec=read(out/'source-gate-spec.json')
    with np.load(out/'builder-inputs.npz',allow_pickle=False) as z:b={k:z[k] for k in z.files}
    graph=build_periodic_faces(b['ids'],b['positions'],b['rho'],spec['D'],spec['periods'],spec['unit_thickness'])
    assert coefficient_packet(graph)==read(out/'coefficient-cache.json')
    assert verify_probe_cells(graph,read(out/'implementation-protocol.json')['independent_actual_geometry_probe_ids'])==read(out/'independent-geometry-probes.json')
    with np.load(out/'original-state.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    result,controls,capacity,arrays=evaluate(graph,data)
    assert result==read(out/'results.json') and controls==read(out/'action-controls.json') and capacity==read(out/'capacity-diagnostics.json')
    with np.load(out/'actions.npz',allow_pickle=False) as z:
        for k,v in arrays.items():assert np.array_equal(z[k],v),k
    print('Same exact periodic geometry, counterpart faces, fixed probes, original-mass action and metrics replay; no propagation, shock case or ORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');args=p.parse_args()
    if args.replay:replay(args.replay)
    elif all([args.output,args.tests,args.test_log]):main(args.output,args.tests,args.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
