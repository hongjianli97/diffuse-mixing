"""Read saved Becker budgets only; never build a graph or call a PDE/action solver."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal, localcontext
import argparse,csv,json,math,platform,resource,shutil,signal,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from voronoi_time_action import metrics
from t1_uniform_core import h_indicator

SOURCE=ROOT/'runs/reconnect-becker-point'
PROTOCOL=ROOT/'research/reconnect/becker_budget_review_protocol.json'
COUNTS=(4,8,16)


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def record(x):return {'exact':str(x),'float':float(x)}


def overlap(edges,values,target):
    """Independent exact integration of stored piecewise-constant reference means."""
    answer=[]
    for lo,hi in zip(target[:-1],target[1:]):
        lo,hi=F(float(lo)),F(float(hi));total=F(0)
        first=max(0,int(np.searchsorted(edges,float(lo),side='right'))-1)
        last=min(len(values),int(np.searchsorted(edges,float(hi),side='left'))+1)
        for j in range(first,last):
            left=max(lo,F(float(edges[j])));right=min(hi,F(float(edges[j+1])))
            if right>left:total+=(right-left)*F(float(values[j]))
        answer.append(float(total/(hi-lo)))
    return np.array(answer)


def analyze(folder):
    folder=Path(folder);r=read(folder/'results.json');ref=read(folder/'reference-indicators.json')
    original=r['time_control'];assert r['comparison_decision']=='moving_Becker_control_failed_or_unresolved'
    assert not r['registered_moving_Becker_oracle_discriminator_passed']
    data=arrays(folder/'material-inputs.npz');col=arrays(folder/'source-column-inputs.npz')
    m,c0=data['mass'],data['c0'];mc=col['mass'];norm=lambda mass,v:float(np.sqrt(mass@(v*v)))
    sigma=r['original_column_initial_fluctuation_norm'];assert math.sqrt(metrics(mc,col['c0'],col['c0'])['variance'])==sigma
    variance=metrics(m,c0,c0)['variance'];A=2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))
    parts=original['components_absolute'];b=sum(parts.values());hb=h_indicator(m,c0,b)
    assert b/sigma==original['fixed_state_decrease_indicator_relative'] and hb==original['fixed_H_decrease_indicator']
    groups={'twice_reference_space':2*F(ref['components_absolute']['saved_reference_space_difference']),
      'twice_reference_time':2*F(ref['components_absolute']['saved_reference_time_difference']),
      'twice_machine':F(0),'twice_tail':F(0),'twice_observed_full_sequence':F(0)}
    sources={'twice_reference_space':'source-column-inputs.npz: reference_n4096-k4096 minus reference_n8192-k4096',
      'twice_reference_time':'source-column-inputs.npz: reference_n8192-k4096 minus reference_n8192-k8192',
      'twice_machine':'k4,k8,k16/native-results.json: final cumulative_machine_indicator_absolute',
      'twice_tail':'k4,k8,k16/native-results.json: final cumulative_tail_indicator_absolute',
      'twice_observed_full_sequence':'k4,k8,k16/trajectory-states.npz: final actual minus complete independent state'}
    numeric=[];saved_H={};paths={};numeric_rows=[]
    for K in COUNTS:
        run=read(folder/f'k{K}/native-results.json');path=arrays(folder/f'k{K}/trajectory-states.npz');paths[K]=path
        step=run['steps'][-1];machine=step['cumulative_machine_indicator_absolute'];tail=step['cumulative_tail_indicator_absolute']
        delta=norm(m,path['states'][-1]-path['independent_states'][-1]);assert delta==step['primary_control_mass_norm_difference']
        numeric.append(machine+tail+delta)
        for label,value in [('machine',machine),('tail',tail),('observed_full_sequence',delta)]:groups['twice_'+label]+=2*F(value)
        H=metrics(m,path['states'][-1],c0)['H'];saved_H[K]=H;assert H==r['observations'][COUNTS.index(K)]['H']
        numeric_rows.append({'K':K,'machine_absolute':machine,'tail_absolute':tail,'observed_full_sequence_absolute':delta})
    assert 2*sum(numeric)==parts['twice_all_path_numeric']
    rs=norm(mc,col['reference_n4096-k4096']-col['reference_n8192-k4096'])
    rt=norm(mc,col['reference_n8192-k4096']-col['reference_n8192-k8192'])
    assert rs==ref['components_absolute']['saved_reference_space_difference'] and rt==ref['components_absolute']['saved_reference_time_difference']
    assert rs+rt==ref['reference_space_time_indicator_absolute'] and 2*(rs+rt)==parts['twice_saved_reference_space_time']
    for name,value in parts.items():
        if 'bridge' in name:groups[name]=F(value);sources[name]='results.json: time_control.components_absolute.'+name
    exact_total=sum(groups.values(),F(0));rounding=F(b)-exact_total
    dH=original['H4_8']-original['H8_16'];assert abs(saved_H[4]-saved_H[8])==original['H4_8'] and abs(saved_H[8]-saved_H[16])==original['H8_16']
    nonref=F(b)-F(parts['twice_saved_reference_space_time'])
    floor=h_indicator(m,c0,float(nonref));space_floor=h_indicator(m,c0,float(groups['twice_reference_space']))
    rows=[]
    for name,value in groups.items():
        rows.append({'component':name,'absolute':float(value),'exact':str(value),'fraction_of_original_state_budget':float(value/F(b)),
          'original_H_formula_for_component_only_diagnostic':h_indicator(m,c0,float(value)),'source':sources[name]})
    with localcontext() as ctx:
        ctx.prec=80
        D=lambda x:Decimal(F(x).numerator)/Decimal(F(x).denominator)
        a,e,d=D(A),D(variance),D(dH)
        critical=2*e*d/(a+(a*a+4*e*d).sqrt())
        rlimit=(critical-D(nonref))/2;slimit=rlimit-D(rt)
        feasibility={'same_formula_total_budget_at_equality':str(critical),'same_formula_reference_space_time_ceiling':str(rlimit),
          'space_ceiling_if_saved_time_unchanged':str(slimit),'saved_reference_space_time_over_ceiling':str(D(rs+rt)/rlimit),
          'saved_reference_space_over_ceiling_if_time_unchanged':str(D(rs)/slimit),
          'planning_diagnostic_only_not_new_acceptance_threshold':True}
    fine=arrays(folder/'source-reference-n8192-k8192.npz');projected=overlap(fine['mass_edges'],fine['c'],col['mass_edges'])
    projection=norm(mc,projected-col['reference']);old_audit=read(folder/'saved-reference-material-audit.json')
    assert projection==old_audit['independent_exact_overlap_projection_mass_norm_difference']
    refH={key:metrics(mc,col[key],col['c0'])['H'] for key in ('reference_n4096-k4096','reference_n8192-k4096','reference_n8192-k8192')}
    assert col['reference'].tobytes()==col['reference_n8192-k8192'].tobytes()
    diagnostics=read(folder/'density-volume-diagnostics.json');geometry={key:max(row[key] for row in diagnostics) for key in diagnostics[0] if key.startswith('maximum_')}
    assert floor<dH and space_floor>dH and original['state_time_decrease_resolved'] and not original['H_time_decrease_resolved']
    # Formal O(h^2) planning only; no new reference vector, extrapolation or acceptance.
    plans=[]
    for factor in (4,16):
        estimate=rs/factor+rt;estimate_b=float(nonref)+2*estimate
        plans.append({'spatial_difference_reduction_for_formal_second_order_planning_only':factor,
          'reference_space_time_estimate_absolute':estimate,'original_formula_H_indicator_estimate':h_indicator(m,c0,estimate_b),
          'estimate_below_saved_H_decline':h_indicator(m,c0,estimate_b)<dH,'not_measured_or_certified':True})
    result={'decision':'saved_reference_space_dominates_reference_only_check_justified',
      'source_decision_retained':r['comparison_decision'],'source_case_passed':False,
      'original_binary64_budget_absolute':b,'original_binary64_H_indicator':hb,'original_H_prefactor':A,'original_initial_variance_binary64':variance,
      'saved_H_decline':dH,'unchanged_original_H_indicator_over_decline':hb/dH,
      'positive_components_exact_sum':record(exact_total),'retained_accumulation_roundoff_not_a_scientific_cancellation':record(rounding),
      'components':rows,'numeric_by_path':numeric_rows,'component_only_H_values_are_not_additive':True,
      'same_formula_nonreference_only_floor_diagnostic':floor,'same_formula_reference_space_only_floor_diagnostic':space_floor,
      'reference_space_time':{'space_absolute':rs,'time_absolute':rt,'sum_absolute':rs+rt,'space_fraction_of_reference_sum':rs/(rs+rt)},
      'projection':{'independent_exact_overlap_vs_saved_means_absolute':projection,'original_budget_changed_to_include_it':False,
        'exact_overlap_is_a_projection_check_not_continuum_reference_error_certificate':True},
      'saved_reference_H_observables':refH,'saved_reference_H_space_difference_diagnostic':abs(refH['reference_n4096-k4096']-refH['reference_n8192-k4096']),
      'saved_reference_H_time_difference_diagnostic':abs(refH['reference_n8192-k4096']-refH['reference_n8192-k8192']),
      'H_observables_not_used_to_shrink_original_indicator':True,'feasibility':feasibility,'formal_second_order_planning_only':plans,
      'density_volume_diagnostics_no_causal_error_allocation':geometry,'source_arrays_or_original_case_indicators_changed':False,
      'new_candidate_or_reference_states_generated':False,'indicators_are_certified_total_error_bounds':False,
      'recorded_checks_total':500,'new_software_checks':0,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}
    return result,rows


def next_gate(result):
    timing=read(ROOT/'runs/rebuild-stage3-space/results.json')['reference_runs'][-1]['wall_seconds']
    inputs=['research/rebuild/reference.py','research/rebuild/becker.py','research/reconnect/becker_budget_review_protocol.json',
      'runs/reconnect-becker-point/source-column-inputs.npz','runs/reconnect-becker-point/source-reference-n8192-k8192.npz',
      'runs/reconnect-becker-point/results.json','runs/reconnect-becker-point/implementation-protocol.json']
    for K in COUNTS:inputs+=['runs/reconnect-becker-point/'+f'k{K}/{name}' for name in ('native-results.json','trajectory-states.npz')]
    return {'already_executed':False,'type':'one_fixed_reference_only_Becker_space_time_L_shape_check',
      'reason':'reference-space part dominates fixed H indicator; same-formula nonreference floor is below observed decline. One more space level alone is not predicted sufficient under formal O(h^2); fixed L-shape independently controls space and time without candidate changes',
      'source_review_decision':result['decision'],'sources':[{'path':p,'sha256':digest(ROOT/p)} for p in inputs],
      'physical_case':{'delta':.08,'D_original_float':.02,'T_original_float':.1,'material_columns':32,'transverse_rows':8,'candidate_K_values_unchanged':[4,8,16]},
      'reference_levels':[{'cells':16384,'steps':8192},{'cells':32768,'steps':8192},{'cells':32768,'steps':16384}],
      'reference_method':'unchanged independent fine uniform-mass conservative centered-face midpoint-time Crank-Nicolson; reference.py and Becker source frozen; no candidate FV import or coefficient fitting',
      'reference_initial_and_boundary':'original analytic finite mass-cell average initial; original mass domain and zero scalar flux at external material endpoints; shock is not a wall',
      'reference_guards':'same preexisting max_relative_mass_drift <=1e-9; min/max stay within original fine initial extrema +/-1e-10; nonfinite or guard failure stops without clipping/normalization',
      'projections':'independently exact Fraction overlap onto original32 mass cells; retain direct-sum versus old primitive-interpolation projection discrepancy as extra positive term, do not replace original mass/c0 or old reference',
      'new_reference_indicator':'mass-norm projected space difference (16384,8192)-(32768,8192) plus time difference (32768,8192)-(32768,16384), both positive; empirical differences,not certified error',
      'source_case_policy':'retain original unresolved result and all original files. New saved-state/reference comparison is a separate predeclared result,never a retrospective pass',
      'new_comparison':'only stored candidate K4/8/16 endpoint states, same original mass/c0/sigma/H. Retain original numeric and observation bridges; replace reference input by new independent indicator,add projection and initial-mean rounding bridges positively; use unchanged original H helper/formula and strict greater-than rule',
      'planning_diagnostic_not_a_new_threshold':result['feasibility'],
      'formal_order_estimates_not_guarantees':result['formal_second_order_planning_only'],
      'old_reference_consistency':'report old8192 versus new32768 reference means and H separately; do not count signed cancellation as acceptance',
      'new_candidate_graphs':0,'new_edge_weights':0,'new_operator_actions':0,'new_candidate_transient_runs':0,'old_candidate_reruns':0,
      'new_complete_independent_uniformization_sequences':0,'new_reference_PDE_solves':3,'new_software_checks':0,'recorded_checks_total':500,
      'main_wall_seconds_cap':300,'main_peak_RSS_MiB_cap':1024,
      'resource_basis':{'saved8192_by8192_wall_seconds':timing,'linear_cells_times_steps_planning_seconds':14*timing,'not_measured_new_runtime':True},
      'stop':'on failure/unresolved retain it; no automatic higher N/K/rows/centres/parameter or additional reference solve, H-formula or threshold retune,clipping/normalization/F_Cfit,Richardson state,or full-matrix scan',
      'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}


def preserve():
    parent=read(SOURCE/'sha256.json');assert len(parent)==6756
    for rel,h in parent.items():assert digest(ROOT/rel)==h,rel
    originals=read('/workspace/original-handoff-sha256.json');assert len(originals)==782
    for rel,h in originals.items():assert digest(ROOT/rel)==h,rel
    attachments=read('/workspace/setup-ltn/attachment-manifest.json');assert len(attachments)==15
    for item in attachments:
        p=Path('/workspace')/item['path'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
    saved=read(SOURCE/'preservation.json');formal=saved['formal_scientific_manifests_verified']+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(parent)}]
    assert len(formal)==102
    for item in formal:
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
    historical=saved['historical_diagnostic_manifests_not_claimed_as_current_hash_matches'];assert len(historical)==4
    for item in historical:
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p)
        differences=[{'path':rel,'saved':h,'current':digest(ROOT/rel)} for rel,h in entries.items() if digest(ROOT/rel)!=h]
        assert sorted(differences,key=lambda v:v['path'])==sorted(item['historical_paths_differing_from_later_formal_versions'],key=lambda v:v['path'])
    return {'original_files_verified':782,'attachments_verified':15,'parent_entries_verified':len(parent),'formal_manifests_verified':formal,
      'historical_diagnostic_differences_retained':historical,'all_existing_scientific_entries_unchanged':True}


def main(output):
    started=time.monotonic()
    def timeout(*args):raise TimeoutError('registered120s saved-budget review cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    protocol=read(PROTOCOL);gate=read(SOURCE/'next-gate-spec.json');assert not gate['already_executed']
    assert gate['type']=='read_only_saved_moving_Becker_H_reference_budget_review'
    for item in protocol['sources']:assert digest(ROOT/item['path'])==item['sha256'],item['path']
    for item in gate['sources']:assert digest(ROOT/item['path'])==item['sha256']
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(PROTOCOL,out/'review-protocol.json');shutil.copyfile(SOURCE/'next-gate-spec.json',out/'source-gate-spec.json')
    save(out/'implementation-lock.json',{'review_code_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),'fixed_before_budget_output':True})
    # Small self-contained copies of already-saved data, not regenerated scientific input.
    copied=[]
    for item in protocol['sources']:
        if not item['path'].startswith(str(SOURCE.relative_to(ROOT))+'/'):continue
        rel=Path(item['path']).relative_to(SOURCE.relative_to(ROOT));dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/item['path'],dest)
        copied.append({'source':item['path'],'copy':str(dest.relative_to(out)),'sha256':item['sha256']})
    save(out/'source-copies.json',copied)
    result,rows=analyze(out/'source');save(out/'results.json',result)
    with (out/'budget-components.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(out/'next-gate-spec.json',next_gate(result));save(out/'preservation.json',preserve())
    zeros={k:0 for k in protocol if k.startswith('new_') or k=='old_candidate_reruns'}
    save(out/'stage3-status.json',{'updated':'2026-11-04','timezone':'Asia/Shanghai','review_decision':result['decision'],
      'original_case_decision':result['source_decision_retained'],'original_case_passed':False,'recorded_checks_total':500,**zeros,
      'next_reference_only_check_registered':True,'next_gate_executed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert elapsed<120 and peak<1024
    save(out/'execution.json',{'review_exit_code':0,'review_first_attempt_passed':True,'elapsed_main_seconds':elapsed,'peak_RSS_MiB':peak,
      'python':platform.python_version(),'new_candidate_or_reference_states_generated':False,'network_used':False,'published':False})
    signal.alarm(0);print(json.dumps({'decision':result['decision'],'space_share':rows[0]['fraction_of_original_state_budget'],
      'nonreference_H_floor':result['same_formula_nonreference_only_floor_diagnostic'],'feasibility':result['feasibility'],'checks':500},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
