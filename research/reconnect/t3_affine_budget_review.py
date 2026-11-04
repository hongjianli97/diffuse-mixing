"""Saved-data accounting only: standard library, no graph/solver/reference generation."""
from fractions import Fraction as F
from pathlib import Path
import argparse,ast,csv,json,math,platform,resource,shutil,signal,struct,time,zipfile
from audit_route_decision import ROOT,read,save,digest

SOURCE=ROOT/'runs/reconnect-t3-affine'
PROTOCOL=ROOT/'research/reconnect/t3_affine_budget_review_protocol.json'
MODELS=('constant_D','constant_Gamma')
GROUPS=('spatial_symbol_error','midpoint_time_error_symbol','observed8_16','machine_tail','observed_complete_sequence',
 'integrated_mode_and_constant_residues','twice_analytic_evaluation','initial_bridge','input_rounding_Duhamel')
ZERO_FIELDS=['new_points','new_candidate_graphs','new_edge_weights','new_operator_actions','new_candidate_transient_runs',
 'new_complete_independent_uniformization_sequences','new_reference_PDE_solves','new_reference_states','old_candidate_reruns',
 'new_material_definitions','new_software_checks','new_exact_algebra_certificate_groups','LP_solves']
COPIES={name:name for name in ('results.json','analytic-parameters.json','analytic-reference-audit.json','numerical-audit.json','implementation-protocol.json',
 'material-inputs.npz','analytic-references.npz')}


def stored_array(path,name):
    """Read only a named, fixed-size float64 NPY member; never execute pickle/header code."""
    with zipfile.ZipFile(path) as z:raw=z.read(name+'.npy')
    assert raw[:6]==b'\x93NUMPY'
    version=tuple(raw[6:8]);assert version in ((1,0),(2,0))
    count=2 if version==(1,0) else 4;length=int.from_bytes(raw[8:8+count],'little');start=8+count
    header=ast.literal_eval(raw[start:start+length].decode('latin1'))
    assert not header['fortran_order'] and header['descr']=='<f8'
    shape=header['shape'];assert shape in ((1024,),(3,1024))
    body=raw[start+length:];assert len(body)==8*math.prod(shape)
    values=tuple(F(x[0]) for x in struct.iter_unpack('<d',body));assert all(math.isfinite(float(x)) for x in values)
    return values if len(shape)==1 else tuple(values[i*1024:(i+1)*1024] for i in range(3))


def exact_variance(m,c):
    total=sum(m,F(0));mu=sum((a*b for a,b in zip(m,c)),F(0))/total
    return sum((a*(b-mu)**2 for a,b in zip(m,c)),F(0))


def qrecord(q):return {'exact':str(q),'float':float(q)}


def analyze(folder):
    """Every number refers to old N32 inputs/arrays; no new state is synthesized."""
    folder=Path(folder);r=read(folder/'results.json');p=read(folder/'analytic-parameters.json');refaudit=read(folder/'analytic-reference-audit.json')
    assert r['comparison_decision']=='T3_affine_model_response_inconclusive' and not r['T3_registered_affine_case_passed']
    m=stored_array(folder/'material-inputs.npz','mass');c0=stored_array(folder/'material-inputs.npz','c0')
    assert set(m)=={F(1,1024)};M=sum(m,F(0));v0=exact_variance(m,c0);assert v0==F(refaudit['exact_initial_variance'])
    sigma=r['initial_mass_fluctuation_norm'];assert sigma==p['initial_fluctuation_norm']
    # Retain the original saved formula and its binary64 evaluation, separately from exact diagnostics.
    E0saved=sigma*sigma;prefactor=2*math.sqrt(float(M))*max(1.,max(map(float,map(abs,c0))))
    # Squaring a rounded sqrt of a rounded reduction need not be within one ulp
    # of the exact variance. Use the existing certified upper bound instead.
    upper=F(p['initial_fluctuation_upper'])
    assert prefactor==2. and F(sigma)<=upper and v0<=upper*upper and E0saved>0
    vlo,vhi=min(v0,F(E0saved)),max(v0,F(E0saved))
    H=lambda c:1-exact_variance(m,c)/v0
    Hfloor=lambda b:(2*b+b*b)/vhi
    rows=[];component_rows=[];observables={}
    for model in MODELS:
        ref=stored_array(folder/'analytic-references.npz',model+'_reference');grid=stored_array(folder/'analytic-references.npz',model+'_continuous_grid_symbol')
        observables[model]=[{'time':t,'saved_PDE_H_exact':str(H(ref[k])),'saved_grid_H_exact':str(H(grid[k])),
          'saved_PDE_H':float(H(ref[k])),'saved_grid_H':float(H(grid[k])),'grid_minus_PDE_H_signed':float(H(grid[k])-H(ref[k]))}
          for k,t in enumerate((0.,.025,.05))]
    for k,old in enumerate(r['model_comparisons']):
        t=old['time'];groups={name:F(0) for name in GROUPS};original=0.
        for model in MODELS:
            case=r['reference_error_components'][model+'-k16'][k];parts=case['components_absolute']
            assert all(v>=0 for v in parts.values()) and sum(parts.values())==case['total_absolute']
            original+=case['total_absolute'];obs=old['indicator_components_absolute'][model+'_observed8_16_absolute'];original+=obs
            assert case['total_absolute']==old['indicator_components_absolute'][model+'_reference_time_input_numerical_absolute']
            for name,value in parts.items():groups[name]+=F(value)
            groups['observed8_16']+=F(obs)
        assert original/sigma==old['fixed_state_indicator_relative']
        assert (prefactor*original+original**2)/E0saved==old['fixed_H_indicator']
        exact_total=sum(groups.values(),F(0));roundoff=F(original)-exact_total;spatial=groups['spatial_symbol_error']
        actualgap=F(old['model_H_difference']);gridgap=F(observables['constant_D'][k]['saved_grid_H_exact'])-F(observables['constant_Gamma'][k]['saved_grid_H_exact'])
        pdegap=F(observables['constant_D'][k]['saved_PDE_H_exact'])-F(observables['constant_Gamma'][k]['saved_PDE_H_exact'])
        floor=Hfloor(spatial);upper=(2*spatial+spatial*spatial)/vlo
        row={'time':t,'source_model_response_resolved':old['model_response_resolved'],'source_model_H_response_resolved':old['model_H_response_resolved'],
          'source_actual_H_gap':old['model_H_difference'],'source_fixed_H_indicator':old['fixed_H_indicator'],
          'source_state_indicator_absolute':original,'positive_component_sum':qrecord(exact_total),'retained_accumulation_roundoff':qrecord(roundoff),
          'groups':{name:qrecord(value) for name,value in groups.items()},
          'state_indicator_fractions':{name:float(value/F(original)) if original else 0. for name,value in groups.items()},
          'spatial_only_original_formula_diagnostic_lower':qrecord(floor),'spatial_only_original_formula_diagnostic_upper':qrecord(upper),
          'spatial_floor_exceeds_saved_actual_gap':floor>actualgap,'spatial_floor_exceeds_saved_continuous_grid_gap':floor>gridgap,
          'saved_continuous_grid_H_gap':qrecord(gridgap),'saved_PDE_H_gap':qrecord(pdegap),
          'signed_model_H_spatial_bias':qrecord(gridgap-pdegap),
          'sum_absolute_individual_H_spatial_bias':qrecord(sum((abs(F(observables[model][k]['saved_grid_H_exact'])-F(observables[model][k]['saved_PDE_H_exact'])) for model in MODELS),F(0))),
          'fixed_H_indicator_minus_saved_actual_gap':old['fixed_H_indicator']-old['model_H_difference'],
          'spatial_floor_minus_saved_grid_gap':qrecord(floor-gridgap),
          'component_shares_are_additive_H_shares':False,'signed_cancellation_used_for_acceptance':False,
          'source_indicator_or_acceptance_replaced':False}
        rows.append(row)
        for name,value in groups.items():component_rows.append({'time':t,'component':name,'absolute':float(value),'absolute_exact':str(value),
          'fraction_of_original_state_indicator':row['state_indicator_fractions'][name]})
    critical=rows[1];assert critical['spatial_floor_exceeds_saved_actual_gap'] and critical['spatial_floor_exceeds_saved_continuous_grid_gap']
    return {'decision':'saved_N32_spatial_budget_dominates_time_only_expansion_not_selected',
      'original_decision_retained':r['comparison_decision'],'original_T3_case_passed':False,
      'saved_initial_variance_exact':str(v0),'saved_sigma_squared_binary64':E0saved,
      'diagnostic_variance_enclosure_exact':[str(vlo),str(vhi)],'original_H_prefactor':prefactor,
      'checkpoints':rows,'saved_array_observables':observables,
      'saved_arrays_only_no_new_reference_states':True,'time_only_expansion_selected':False,
      'diagnostic_floor_is_not_a_new_error_certificate':True,'source_results_indicators_and_decision_changed':False,
      'general_T3_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False},component_rows


def next_gate():
    parent=read(SOURCE/'sha256.json');base=ROOT/'runs/reconnect-t1-uniform'
    materials={}
    for N in (16,64):
        path=base/f'n{N}/material-reference.npz';rel=str(path.relative_to(ROOT));assert digest(path)==parent[rel]
        materials[str(N)]={'path':rel,'sha256':parent[rel],'old_T1_reference_history_only':True,'mass_c0_not_parsed_this_review':True}
    paths=[]
    for model in MODELS:
        for K in (4,8,16):
            for name in ('trajectory-states.npz','native-results.json','modal-residues.json'):
                path=SOURCE/model/f'k{K}'/name;paths.append({'path':str(path.relative_to(ROOT)),'sha256':digest(path)})
    return {'already_executed':False,'type':'one_three_space_level_same_affine_pair_control_after_saved_budget_review',
      'reason':'saved N32 spatial-only original-formula floor exceeds actual and stored continuous-time-grid H model gaps; separate space control is necessary, time-only expansion not selected',
      'scope':'same physical affine pair and T=.05; three original material space levels only, not a new flow/model/parameter scan',
      'N_values':[16,32,64],'new_N_values':[16,64],'reused_N_values':[32],'models':list(MODELS),'T_exact':'1/20','ordered_interval_counts':[4,8,16],
      'new_candidate_graphs':112,'new_candidate_transient_runs':12,'new_complete_independent_uniformization_sequences':12,
      'reused_saved_candidate_paths':6,'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_material_definitions':0,
      'motion_and_constitutive_protocol':{'path':str((SOURCE/'implementation-protocol.json').relative_to(ROOT)),'sha256':digest(SOURCE/'implementation-protocol.json')},
      'pure_source':{'path':str((SOURCE/'original-pure-inputs.npz').relative_to(ROOT)),'sha256':digest(SOURCE/'original-pure-inputs.npz')},
      'material_sources':materials,'N32_source_seal':{'path':str((SOURCE/'sha256.json').relative_to(ROOT)),'sha256':digest(SOURCE/'sha256.json')},
      'reused_N32_records':paths,'common_material_observation_N':16,
      'material_observation':'original child component integrals sum divided by original N16 mass; retain initial/reference projection bridges, no capacity or scalar reweighting',
      'coefficient_inputs':'only pure labels transformed into actual points,IDs,given rho/D and boundary; all112 new local/world graphs sealed before any actual m/c0/reference or reused N32 state parse',
      'references':'same separate physical clocks at N16/N64 with original finite-cell means; independent80/100digit checks; old T1 reference never an affine target',
      'backends':'unchanged original geometry,ordinary center resistance,exact boundary adapter and chronological action; generalized study/diagnostics only in new files',
      'time_and_symbol_controls':'all N and K4/8/16,original frozen-symbol/residue/input/machine/tail indicators and original H formula; actual4/8 to8/16 state/H decrease must be resolved at both nonzero times',
      'spatial_comparison':'at fixed K16,adjacent16/32 and32/64,each physical model separately; native and common N16 mass-L2/reference error and H/reference error gains must exceed predetermined positive comparison budgets at both times',
      'spatial_budget':'both observed8/16 differences,analytic midpoint-to-continuous time errors,both input/numerical/residue/evaluation terms and all relevant initial/reference projection bridges; report components before decision, no signed cancellation or Richardson replacement',
      'model_comparison':'fine N64 state and H model response at both nonzero times must exceed unchanged original-formula both-model spatial/time/input/numerical/reference plus observed8/16 indicators and agree with physical references',
      'coarse_case_policy':'retain N32 original inconclusive and report N16 decisions without requiring every coarse grid to resolve model H; no retroactive pass or universal T3 admission',
      'planned_new_software_checks':3,'check_categories':['multi-N exact directional laws and modal guards','direct material child-integral/common observation and nonzero bridges','native/common/time/model budget classification and failure rejection'],
      'recorded_checks_total_if_all_pass':483,'tail_per_interval_exact':'1/10000000000000000','original_algebra_tolerance':1e-10,'norm_estimator_seed':20261101,
      'fixed_geometry_probe_formula':'unique IDs [0,N//2,N-1,N^2//2,N^2-1]',
      'main_wall_seconds_cap':900,'main_peak_RSS_MiB_cap':1024,
      'resource_basis':'56N32 graphs took85.285s before material load; node-linear planning estimate(.25+4)*85.285=362.461s for56N16+56N64,900s conservative envelope; not measured speed or accuracy',
      'stop':'preserve adverse/unresolved, no H-indicator/threshold change,clipping,pointrepair,normalization,F/Cfit,automaticN128/K32/newflow/model or full-matrix scan',
      'new_points_graphs_states_or_references_generated_this_review':False,'next_experiment_executed':False,
      'general_spatial_convergence_claimed':False,'general_T3_admitted':False,'general_smooth_flow_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}


def main(output):
    started=time.monotonic();signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('registered120s review cap')));signal.alarm(120)
    prior=read(SOURCE/'sha256.json');assert len(prior)==4483
    for i,(rel,h) in enumerate(prior.items(),1):
        assert digest(ROOT/rel)==h,rel
        if i%1000==0:print(f'prior sealed entries:{i}/{len(prior)}',flush=True)
    gate=read(SOURCE/'next-gate-spec.json');assert not gate['already_executed'] and gate['type']=='read_only_saved_T3_affine_control_and_error_budget_review'
    assert all(gate[k]==0 for k in ('new_candidate_graphs','new_candidate_transient_runs','new_independent_uniformization_sequences','new_reference_PDE_solves','old_candidate_reruns','planned_new_software_checks'))
    for item in gate['sources']:assert digest(ROOT/item['path'])==item['sha256']
    protocol=read(PROTOCOL);assert all(protocol[k]==0 for k in ZERO_FIELDS)
    assert digest(ROOT/'exports/ltn-t3-affine.zip')==protocol['source_delivery_sha256']
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    failed=out.with_name('reconnect-t3-affine-review-attempt1')
    assert read(failed/'failed-execution.json')['review_exit_code']==1
    shutil.copytree(failed,out/'review-first-attempt')
    for name,path in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('review-protocol.json',PROTOCOL)]:shutil.copyfile(path,out/name)
    save(out/'implementation-lock.json',{'fixed_before_budget_output':True,'code_sha256':digest(Path(__file__)),'review_protocol_sha256':digest(PROTOCOL),
      'imports_only_standard_library_and_file_provenance_helpers':True})
    (out/'source').mkdir();copied=[]
    for original,name in COPIES.items():
        rel=str((SOURCE/original).relative_to(ROOT));assert digest(SOURCE/original)==prior[rel]
        shutil.copyfile(SOURCE/original,out/'source'/name);copied.append({'source':rel,'copy':'source/'+name,'sha256':prior[rel]})
    save(out/'source-copies.json',copied);result,rows=analyze(out/'source');save(out/'results.json',result)
    with (out/'budget-components.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/'next-gate-spec.json',next_gate())
    save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','review_executed':True,'decision':result['decision'],
      'original_case_decision':result['original_decision_retained'],'original_T3_case_passed':False,'recorded_checks_total':480,'previous_checks_rerun':0,
      **dict.fromkeys(ZERO_FIELDS,0),'time_only_expansion_selected':False,'separate_spatial_control_registered':True,'next_experiment_executed':False,
      'original_backends_changed':False,'finite_node_candidate_admitted':False,'general_T3_admitted':False,'general_smooth_flow_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert elapsed<120 and peak<1024
    save(out/'execution.json',{'review_exit_code':0,'review_first_attempt_passed':False,'review_failed_attempts_preserved':1,'final_review_passed':True,'elapsed_main_seconds':elapsed,'peak_RSS_MiB':peak,
      'python':platform.python_version(),'standard_library_only':True,'new_points_graphs_states_or_reference_evaluations':0,
      'source_manifest_entries_verified':len(prior),'network_used':False,'environment_configuration_changed':False,'published':False})
    signal.alarm(0);print(json.dumps({'decision':result['decision'],'critical_checkpoint':result['checkpoints'][1],'recorded_checks_total':480},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
