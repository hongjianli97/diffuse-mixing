"""Read-only original-criteria and immutable-code lineage review; standard library only."""
from pathlib import Path
import argparse,ast,csv,json,platform,resource,shutil,signal,time
from audit_route_decision import ROOT,read,save,digest

SOURCE=ROOT/'runs/reconnect-t3-affine-spatial'
PROTOCOL=ROOT/'research/reconnect/t3_coverage_protocol.json'
CONTRACT=ROOT/'research/reconnect/AFFINE_SHEAR_DOMAIN_CONTRACT.md'
BACKENDS=('periodic_voronoi_fv.py','voronoi_time_action.py','voronoi_time16_core.py')
CASES=('reconnect-shear-union64','reconnect-compression-union64','reconnect-t1-uniform','reconnect-t2-translation','reconnect-t2-d0','reconnect-t2-rotation','reconnect-t3-affine-spatial')
ZERO_FIELDS=('new_points','new_candidate_graphs','new_edge_weights','new_operator_actions','new_candidate_transient_runs',
 'new_complete_independent_uniformization_sequences','new_reference_PDE_solves','new_reference_states','old_candidate_reruns',
 'new_material_definitions','new_software_checks','new_exact_algebra_certificate_groups','LP_solves')
TEXT_SOURCES=('docs/METHOD.md','docs/CODEX_TASK.md','docs/PHASE3_CURRENT.md','docs/PHASE3_REBUILD.md',
 'research/reconnect/T1_UNIFORM_COMPARISON.md','research/reconnect/T2_TRANSLATION_COMPARISON.md','research/reconnect/T2_D0_COMPARISON.md',
 'research/reconnect/T2_ROTATION_COMPARISON.md','research/reconnect/T3_AFFINE_SPATIAL_COMPARISON.md','research/reconnect/T3_AFFINE_COMPARISON.md',
 'research/reconnect/SHEAR_UNION64_COMPARISON.md','research/reconnect/COMPRESSION_UNION64_COMPARISON.md',
 'research/reconnect/CAPACITY_INPUT_AND_STAGE3.md','research/rebuild/SPATIAL_CONVERGENCE.md',
 'research/reconnect/FV_CONTACT_FIXED_TIME_COMPARISON.md','research/reconnect/FV_CONTACT_COMMON_OBSERVATION.md',
 'research/reconnect/shear_probe.py','research/reconnect/oriented_periodic_boundary.py','research/reconnect/fv_contact_time_study.py')


def all_sources():
    names=set(TEXT_SOURCES)|{'research/reconnect/'+name for name in BACKENDS}
    for case in CASES:
        names.update(f'runs/{case}/{name}' for name in ('results.json','implementation-lock.json','stage3-status.json'))
    names.update(['runs/reconnect-t3-affine-spatial/next-gate-spec.json','runs/reconnect-t3-affine-spatial/numerical-audit.json'])
    return sorted(names)


def quotes(resolve):
    specs=(('T1','docs/METHOD.md','|T1|'),('T2','docs/METHOD.md','|T2|'),('T3','docs/METHOD.md','|T3|'),('T4','docs/METHOD.md','|T4|'),
      ('levels','docs/METHOD.md','每个空间实验至少三档'),('density_sources','docs/CODEX_TASK.md','oracle、kernel、ledger'),
      ('physical_error','docs/METHOD.md','该容差不是 PDE 精度门槛'),('no_DNS_wait','docs/CODEX_TASK.md','没有原始 DNS 数据时'),
      ('default_IE','docs/METHOD.md','隐式 Euler 为默认'),('fluid_ORC_gate','docs/METHOD.md','生产后端尚未通过 T1–T4'))
    result=[]
    for key,path,needle in specs:
        matches=[(i+1,line) for i,line in enumerate(resolve(path).read_text().splitlines()) if needle in line]
        assert len(matches)==1,(key,matches)
        line,text=matches[0];result.append({'id':key,'path':path,'sha256':digest(resolve(path)),'line':line,'text':text})
    return result


def inspect_code(resolve):
    probes=(('axis_periods','research/reconnect/periodic_voronoi_fv.py','if len(periods)!=2:'),
      ('axis_image_distance','research/reconnect/periodic_voronoi_fv.py','point[j][0]+sx*box[0]-point[i][0]'),
      ('proper_orthogonal_only','research/reconnect/oriented_periodic_boundary.py','exact proper orthogonality required'),
      ('given_sine_shear','research/reconnect/shear_probe.py','return a+s/(2*np.pi)*np.sin(2*np.pi*b)+.2*time,b-.1*time'),
      ('chronological_exponential_research_path','research/reconnect/voronoi_time16_core.py','after,half=exponential(op,state,dt)'),
      ('Cartesian_contact_prior','research/reconnect/fv_contact_time_study.py','modes=build_modes(rho,mass'))
    records=[]
    for key,path,needle in probes:
        text=resolve(path).read_text();ast.parse(text)
        found=[{'line':i+1,'text':line} for i,line in enumerate(text.splitlines()) if needle in line];assert found,key
        records.append({'id':key,'path':path,'sha256':digest(resolve(path)),'evidence':found,'source_executed':False})
    return records


def coverage(resolve):
    q=quotes(resolve);code=inspect_code(resolve);lineage=[]
    for case in CASES:
        lock=read(resolve(f'runs/{case}/implementation-lock.json'))
        record={'case':case,'lock_path':f'runs/{case}/implementation-lock.json','backends':{}}
        for name in BACKENDS:
            current=digest(resolve('research/reconnect/'+name));assert lock['files'][name]==current,(case,name)
            record['backends'][name]=current
        lineage.append(record)
    saved={case:read(resolve(f'runs/{case}/results.json')) for case in CASES}
    assert saved['reconnect-t1-uniform']['T1_registered_case_passed']
    assert saved['reconnect-t2-translation']['T2_registered_translation_case_passed']
    assert saved['reconnect-t2-d0']['T2_registered_D0_case_passed']
    assert saved['reconnect-t2-rotation']['T2_registered_rotation_case_passed']
    assert saved['reconnect-t3-affine-spatial']['T3_registered_three_space_level_affine_control_passed']
    assert saved['reconnect-shear-union64']['all_adjacent_times_common_native_H_resolved']
    assert saved['reconnect-compression-union64']['all_adjacent_times_common_native_H_resolved']
    affine=saved['reconnect-t3-affine-spatial'];assert affine['per_grid_controls']['32']['comparison_decision']=='T3_affine_model_response_inconclusive'
    rows=[]
    def row(key,requirement,state,evidence,scope,gap):
        rows.append({'id':key,'original_requirement_ids':requirement,'state':state,'evidence':evidence,'covered_scope':scope,'remaining_gap':gap})
    row('T1_uniform',['T1','levels'],'registered_control_passed',['research/reconnect/T1_UNIFORM_COMPARISON.md','runs/reconnect-t1-uniform/results.json'],
      'N16/32/64 periodic uniform mass-mean sine; K4/8/16 on unchanged research exponential action','not a production-default implicit-Euler or general space-order declaration')
    for key,case,label in [('T2_translation','reconnect-t2-translation','positiveD rigid translation'),('T2_D0','reconnect-t2-d0','given nonrigid D0 path and conditional Q0 identity'),('T2_rotation','reconnect-t2-rotation','positiveD continuous exact co-rotating orthogonal torus')]:
        row(key,['T2'],'registered_control_passed',[f'runs/{case}/results.json'],label,
          'only declared boundary/input mode; no fluid kappa run and no arbitrary-boundary admission')
    row('T3_orthogonal_affine_compression',['T3','levels'],'registered_three_grid_control_passed',
      ['research/reconnect/T3_AFFINE_SPATIAL_COMPARISON.md','runs/reconnect-t3-affine-spatial/results.json'],
      'same original N16/32/64 material means; both constantD/constantGamma models; native/common concentration and H gains; K4/8/16; original N32 inconclusive retained',
      'not affine shear or general T3 admission; no further compression grid/time test selected')
    row('T3_direction_and_constitutive',['T3'],'covered_in_registered_orthogonal_case',['runs/reconnect-t3-affine-spatial/results.json'],
      'exact gx=Gamma/a and gy=Gamma*a; fine model response resolved under original full H indicator',
      'off-diagonal affine-shear two-mode response not executed; do not demand arbitrary angle/flow scans')
    row('T3_periodic_sinusoidal_shear',['T3','levels'],'positive_same_backend_nonaffine_evidence',
      ['research/reconnect/SHEAR_UNION64_COMPARISON.md','runs/reconnect-shear-union64/results.json','research/reconnect/shear_probe.py'],
      'actual periodic sine shear, nonuniform oracle density, N64/128/256; native/common concentration/H gains resolved; N128/256 K4/8/16 and N64 necessary8/16',
      'map is nonlinear in labels; cannot relabel as literal affine-shear execution or both constitutive laws; do not rerun passed sine shear')
    row('T3_affine_shear',['T3'],'missing_registered_execution_and_sliding_lattice_mode',
      ['research/reconnect/shear_probe.py','research/reconnect/oriented_periodic_boundary.py','research/reconnect/periodic_voronoi_fv.py'],
      'specific gap identified from original case and current orthogonal boundary API',
      'one optional exact sliding-periodic Euclidean geometry mode plus one fixed N32 shear control selected; not implemented or run')
    row('T4_Becker_baseline',['T4'],'existing_material_FV_baseline_not_final_point_backend_acceptance',
      ['research/rebuild/SPATIAL_CONVERGENCE.md'],'same Becker delta.08/D.02/T.1 with32/64/128 true material columns and separate time/reference controls',
      'true material faces/centers and scalar1D boundary prior differ from final point-geometry/time pipeline; final-backend T4 and layer-thickness controls incomplete')
    row('T4_contact_baseline',['T4'],'existing_aligned_Cartesian_baseline_not_final_pipeline_acceptance',
      ['research/reconnect/FV_CONTACT_FIXED_TIME_COMPARISON.md','research/reconnect/FV_CONTACT_COMMON_OBSERVATION.md','research/reconnect/fv_contact_time_study.py'],
      'independent stationary density contact, finite-cell reference, ordinary resistance; N128/256 spatial comparison and semidiscrete/BE time diagnostics',
      'useful conditional same-flux baseline on aligned geometry, not a completed final point-builder/trajectory/boundary pipeline three-level T4 record')
    row('density_sources',['density_sources'],'same_final_backend_three_source_comparison_incomplete',
      ['docs/CODEX_TASK.md','research/reconnect/CAPACITY_INPUT_AND_STAGE3.md'],
      'oracle isolation and prior density/interface theory retained','oracle/particle_reconstruction/ledger_experimental with same m,c0,D,boundary/backend not jointly accepted; no F fitting or mass repair')
    row('integrator_and_delivery_scope',['default_IE'],'research_integrator_declared_not_production_default_migration',
      ['docs/METHOD.md','research/reconnect/voronoi_time16_core.py'],
      'current studies explicitly use chronological frozen exponential actions and independent raw uniformization',
      'do not claim original implicit-Euler default/CLI/schema migration was delivered; production packaging is separate from this missing physics control')
    row('ORC_DNS_scope',['fluid_ORC_gate','no_DNS_wait'],'fluid_ORC_off_no_new_DNS_prerequisite',['docs/METHOD.md','docs/CODEX_TASK.md'],
      'T5/T6,DNS and dynamicORC are not added Stage3 waiting conditions; Q0/isometry statements kept conditional',
      'Stage3 remains incomplete; do not use absent oldv080/246 recovery or fluid kappa computation as a new prerequisite to run the selected analytic control')
    return {'original_requirement_quotes':q,'same_backend_lineage':lineage,'code_inspection':code,'acceptance_ledger':rows,
      'review_decision':'retain_supported_controls_select_one_affine_shear_sliding_domain_control',
      'T3_affine_compression_registered_control_passed':True,'periodic_sine_shear_positive_evidence_retained':True,
      'T3_affine_shear_executed':False,'general_T3_admitted':False,'source_N32_inconclusive_retained':True,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}


def next_gate():
    pure=ROOT/'runs/reconnect-t3-affine-spatial/original-pure-inputs.npz';material=ROOT/'runs/reconnect-t1-uniform/n32/material-reference.npz'
    return {'already_executed':False,'type':'one_optional_exact_sliding_periodic_geometry_mode_and_one_N32_affine_shear_control',
      'reason':'original METHOD T3 affine shear lacks execution; saved sine shear is nonlinear and present given-boundary API is orthogonal only',
      'written_contract':{'path':str(CONTRACT.relative_to(ROOT)),'sha256':digest(CONTRACT)},
      'implementation':'new opt-in exact sliding-periodic world-Euclidean Voronoi code and independent world auditor; original rectangle/isometry/action code unchanged; s0 delegates original rectangle builder',
      'given_boundary_family':'origin0,B columns[(1,0),(s,1)],exact rational0<=s<=1/4; not fitted localF/C and not general lattice admission',
      'world_motion':'X(a,t)=B(t)a; no original axisbox modulo or inverse-shear Euclidean shortcut',
      'N':32,'T_exact':'1/20','shear_slope_exact':'5','s_final_exact':'1/4','sample_times_exact':'j/(40K),j0..2K','ordered_interval_counts':[4,8,16],
      'rho':1.,'D':.02,'D_exact_wire':'Fraction(float(.02))','model_scope':'rho1 makesconstantD andconstantGamma identical; one model only, previous compression distinction retained',
      'pure_source':{'path':str(pure.relative_to(ROOT)),'sha256':digest(pure)},
      'material_source':{'path':str(material.relative_to(ROOT)),'sha256':digest(material),'old_T1_reference_history_only':True},
      'material_state':'original N32 mass/c0/IDs byte-identical; do not replace original c0 by an ideal two-mode reconstruction',
      'new_candidate_graphs':28,'new_candidate_transient_runs':3,'new_complete_independent_uniformization_sequences':3,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_material_definitions':0,
      'coefficient_inputs':'only actual world points,IDs,givenrho,D andglobalboundary; all28 local/world coefficients sealed before actualm/c0/reference/oldstates parse',
      'geometry_requirement':'world-Euclidean bisectors,reciprocalfaces,positivehalf-resistances,area andcomplete conservativeimage/bin search; fractional coordinates only canonicalize/index,not change distances',
      'independent_geometry_controls':'afterseal,verify allworldfaces andfixed9image probes; uniformpointlattice g1=D(1-s+s^2),g2=D(1-s),g3=Ds,countsN^2each,area1/N^2 andthree-direction secondmoment; never feedthese formulas tobuilder',
      'far_image_control':'canonicalfractiondifferences(-1,1)^2 imply worlddistance>1/(1+s) outside9images; require percell2R<1/(1+s); reject absent completeness certificate',
      'fixed_geometry_probe_ids':[0,16,31,512,1023],
      'analytic_material_reference':'eta=+/-1,w_eta=sin(2pi(ax+eta ay)),A=.1sinc(pi/N)^2; cbar=.5+A sum exp(-4pi^2 D(2t-eta*5t^2+25t^3/3))*w_eta',
      'discrete_symbol_controls':'postseal independent lambda_plus=4DN^2 sin^2(pi/N)(2-2s+s^2);lambda_minus adds4DN^2*s*sin^2(2pi/N); preserve original c0 bridge and per-mode/constant residual guards before using indicators',
      'direction_observables':'raw I_eta=sum m(c-mu0)w_eta; retaininitialvalues andcompare changes toreference; positive CauchySchwarz bounds fromwhole-state PDE indicators plus observed8/16,never signed error cancellation or state projection',
      'reference_evaluation':'independent80/100digit finite-cell/two-clock/frozen-continuous symbols; same64eps sqrt(M)max(1,max|c0|) evaluator allowance andoriginalinitialbridge',
      'acceptance':'all declaredgeometry/moment/residue/input/numericalcontrols; actualstate/H vsfrozen andphysicalreference withinfixedpositive numerical/time/spatial/reference/bridge indicators; actual4/8-to8/16 state/H decrease resolvedatbothnonzerotimes; two-mode response contrast agreeswithreference andexceeds combinedpositiveindicators plus observed8/16',
      'H_indicator':'unchanged originalquadratic mass-norm perturbation indicator; actualH fromwhole unmodified originalc0',
      'planned_new_software_checks':4,'check_categories':['exact slidingbasis/canonicalization ands0 originaldispatch,invalidinputrejection',
        'independentworldfaces/search/moment/resistance andcorruptionrejection','two-mode truecellmeans/clocks andinitialbridge preserved',
        'time/direction/whole-state/H budgets andadverse independentsequence rejection'],
      'recorded_checks_total_if_all_pass':487,'tail_per_interval_exact':'1/10000000000000000','original_algebra_tolerance':1e-10,'norm_estimator_seed':20261101,
      'main_wall_seconds_cap':900,'main_peak_RSS_MiB_cap':1024,
      'resource_basis':'old28N32rectangulargraphs about43s asbackground only; newnonorthogonalcost unmeasured,hard900s/1024MiB stop preservespartialfailure',
      'stop':'retainunsupported/failed/unresolved; no pointrepair,positivefacedeletion,clipping,mass/rho/state/row/weightnormalization,F/Cfit,Hformulachange,thresholdrelaxation,automaticnewN/K/angle/flow/modelscan',
      'proofs_are_written_not_machine_verified':True,'new_geometry_implemented_this_round':False,'new_shear_points_graphs_references_or_states_generated_this_round':False,
      'new_spatial_convergence_claimed':False,'general_nonorthogonal_domain_admitted':False,'general_T3_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}


def main(output):
    start=time.monotonic()
    def alarm(*args):raise TimeoutError('registered120s coverage review cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
    prior=read(SOURCE/'sha256.json');assert len(prior)==5450
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json');assert not gate['already_executed'] and gate['type']=='read_only_same_backend_T1_T4_and_T3_direction_shear_coverage_review'
    for item in gate['sources']:assert digest(ROOT/item['path'])==item['sha256']
    protocol=read(PROTOCOL);assert all(protocol[key]==0 for key in ZERO_FIELDS)
    assert digest(ROOT/'exports/ltn-t3-affine-spatial.zip')==protocol['source_delivery_sha256']
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    failed=out.with_name('reconnect-t3-coverage-attempt1')
    assert read(failed/'failed-execution.json')['review_exit_code']==1
    shutil.copytree(failed,out/'review-first-attempt')
    for name,path in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('review-protocol.json',PROTOCOL),('contract-at-review.md',CONTRACT)]:shutil.copyfile(path,out/name)
    save(out/'review-lock.json',{'code_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),'contract_sha256':digest(CONTRACT),'fixed_before_review_output':True})
    gate_sources={item['path']:item['sha256'] for item in gate['sources']}
    gate_rel=str((SOURCE/'next-gate-spec.json').relative_to(ROOT))
    assert prior[gate_rel]==digest(SOURCE/'next-gate-spec.json')
    items=[]
    for rel in all_sources():
        p=ROOT/rel;h=digest(p)
        if rel in prior:assert prior[rel]==h,rel
        else:assert rel=='docs/PHASE3_REBUILD.md' and gate_sources[rel]==h,rel
        target=out/'evidence'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        items.append({'path':rel,'copy':str(target.relative_to(out)),'sha256':h,'source_SHA_bound_by_formal_parent':True,
          'cumulative_content_entry_present':rel in prior,'binding':'cumulative_content_entry' if rel in prior else 'immutable_parent_gate_source_SHA'})
    save(out/'evidence-index.json',{'source_manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'source_manifest_sha256':digest(SOURCE/'sha256.json'),
      'source_manifest_entries_verified':len(prior),'sources':items,'NPZ_materials_hashed_only_for_future_gate_not_parsed':True})
    result=coverage(lambda rel:ROOT/rel);save(out/'coverage-review.json',result)
    save(out/'acceptance-ledger.json',{'original_requirement_quotes':result['original_requirement_quotes'],'rows':result['acceptance_ledger'],
      'physical_acceptance_percentage_computed':False,'new_acceptance_criteria_added':False,'source_generals_or_original_N32_decision_changed':False})
    with (out/'acceptance-ledger.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=('id','state','covered_scope','remaining_gap'));w.writeheader();w.writerows({k:row[k] for k in w.fieldnames} for row in result['acceptance_ledger'])
    save(out/'next-gate-spec.json',next_gate())
    save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','review_executed':True,'decision':result['review_decision'],
      **dict.fromkeys(ZERO_FIELDS,0),'recorded_checks_total':483,'previous_checks_rerun':0,'registered_affine_compression_pass_retained':True,
      'registered_sine_shear_positive_evidence_retained':True,'affine_shear_execution_missing':True,'next_adapter_or_experiment_executed':False,
      'original_backends_changed':False,'original_N32_inconclusive_retained':True,'general_T3_admitted':False,'finite_node_candidate_admitted':False,
      'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    elapsed=time.monotonic()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert elapsed<120 and peak<1024
    save(out/'execution.json',{'review_exit_code':0,'review_first_attempt_passed':False,'review_failed_attempts_preserved':1,'final_review_passed':True,'elapsed_main_seconds':elapsed,'peak_RSS_MiB':peak,
      'python':platform.python_version(),'standard_library_only':True,'new_numerical_or_flow_tests':0,'old_numerical_paths_replayed':False,
      'source_manifest_entries_verified':len(prior),'network_used':False,'environment_configuration_changed':False,'published':False})
    signal.alarm(0);print(json.dumps({'decision':result['review_decision'],'ledger_rows':len(result['acceptance_ledger']),
      'same_backend_locks_verified':len(result['same_backend_lineage']),'new_software_checks':0,'recorded_checks_total':483},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
