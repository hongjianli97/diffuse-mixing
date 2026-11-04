"""Read-only T2 rotation/domain contract review; no points, graphs or numerical runs."""
from pathlib import Path
import argparse,ast,json,platform,resource,shutil,time
from audit_route_decision import ROOT,read,save,digest

SOURCE=ROOT/'runs/reconnect-t2-d0'
PROTOCOL=ROOT/'research/reconnect/t2_rotation_domain_protocol.json'
REPORT=ROOT/'research/reconnect/T2_ROTATION_DOMAIN_CONTRACT.md'
ZERO_FIELDS=['new_points','new_candidate_graphs','new_edge_weights','new_operator_actions','new_candidate_transient_runs',
 'new_complete_independent_uniformization_sequences','new_reference_PDE_solves','old_candidate_reruns','new_material_definitions',
 'new_software_checks','new_exact_algebra_certificate_groups','LP_solves']
STATIC=ROOT/'runs/reconnect-t1-uniform'


def inspect_backend(path):
    text=path.read_text();tree=ast.parse(text)
    fn=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='build_periodic_faces')
    arguments=[v.arg for v in fn.args.args];assert arguments==['ids','positions','rho','D','periods','thickness','progress']
    evidence={
      'two_period_lengths':"if len(periods)!=2:raise ValueError('two periodic lengths required')",
      'axis_image_displacement':'return (point[j][0]+sx*box[0]-point[i][0],point[j][1]+sy*box[1]-point[i][1])',
      'axis_nearest_image_x':'sx=-int(((point[j][0]-point[i][0])/box[0]+F(1,2))//1)',
      'axis_nearest_image_y':'sy=-int(((point[j][1]-point[i][1])/box[1]+F(1,2))//1)',
      'float_input_conversion':'x=np.asarray(positions,dtype=float)',
      'axis_canonical_vertex':'(p[0]+q[0])%graph.periods[0]'}
    rows=[];lines=text.splitlines()
    for role,needle in evidence.items():
        matches=[{'line':i+1,'text':line} for i,line in enumerate(lines) if needle in line];assert matches,role
        rows.append({'role':role,'exact_source_evidence':matches})
    return {'path':str(path.relative_to(ROOT)),'sha256':digest(path),'function':'build_periodic_faces','line':fn.lineno,
      'arguments':arguments,'boundary_mode':'axis_aligned_rectangular_periodic','world_periodic_basis_argument_present':False,
      'general_continuously_rotating_lattice_representable_directly':False,'evidence':rows,
      'inspection_is_numerical_or_flow_test':False,'api_or_backend_changed':False}


def next_gate():
    sources={
      'pure_points':STATIC/'builder-inputs.npz','material_mass_c0_reference':STATIC/'n32/material-reference.npz',
      'static_geometry':STATIC/'n32/geometry.json','static16_states':STATIC/'n32/k16/trajectory-states.npz',
      'static16_results':STATIC/'n32/k16/native-results.json','static_analytic_parameters':STATIC/'n32/analytic-parameters.json'}
    actions={f'step{i:02d}':{'path':str((STATIC/f'n32/k16/action-step{i:02d}.npz').relative_to(ROOT)),
      'sha256':digest(STATIC/f'n32/k16/action-step{i:02d}.npz')} for i in range(16)}
    return {'already_executed':False,'type':'one_optional_exact_oriented_rectangle_adapter_and_one_continuous_rotation_control',
      'scope':'implement separate research boundary isometry adapter and independent world-geometry audit; oneN32 positiveD rotating periodic domain',
      'implementation_location':'new files only inresearch/reconnect; original periodic_voronoi_fv.py andtime backends unchanged',
      'original_backend_sha256':digest(ROOT/'research/reconnect/periodic_voronoi_fv.py'),
      'boundary_data':'given globaloriginb,positiveperiodlengthsL,orthonormalpreservingdirectionsE,worldperiodbasisB=Ediag(L);not fitted from trajectories',
      'adapter_inputs':'worldpositions_exact,IDs,point_rho,D,givenboundary; no source local labels,m,c0,reference,true internal faces orF/C',
      'adapter_exact_contract':'rationalE,detE=1,E^TE=I,B^TB=diag(L^2); exacty=E^T(x-b); require Fraction(float(y))==y andcanonical domain membership before originalbuilder',
      'unsupported_input_handling':'reject nonisometry,lossyfloatconversion,invalidIDs/density/domain; no orthonormalization,clipping,jitter,roundtrip correction or source-label fallback',
      'general_float_world_trajectories_or_nonorthogonal_lattices_admitted':False,
      'N':32,'D':.02,'T_exact':'1/20','K':16,'origin_exact':['0','0'],'period_lengths_exact':['1','1'],
      'sample_times_exact':'j/640,j0..32; midpointoddj','rotation_parameter':'s(t)=t/(2T)',
      'rotation_frame_exact':'E=(1/(1+s^2))*[[1-s^2,-2s],[2s,1-s^2]]','rotation_angle':'theta=2atan(s),final2atan(1/2)',
      'angular_speed':'1/[T*(1+s^2)],not constant','world_motion':'x_i(t)=E(t)*originala_i,exact rational; no worldaxis modulo',
      'world_domain':'E(t)[0,1)^2,worldlatticeE(t)Z^2; rotates with trajectories','point_density':'same given oracle1',
      'material_state':'same true originalm/c0/IDs andmaterial reference; no new material definitions',
      'sources':{k:{'path':str(v.relative_to(ROOT)),'sha256':digest(v)} for k,v in sources.items()},'static_action_sources':actions,
      'coefficient_separation':'all16 freshoriginalgeometry builds+exactworldlifts sealed before m,c0,reference,staticgeometry oroldstate parse; do not copy staticg',
      'independent_world_audit':'worldpoints+B*n bisectors,reciprocalfaces,exactresistance,area andfixed probes; no reconstructing local cloud to validate same builder',
      'probe_ids':[0,16,31,512,1023],'world_probe_pruning':'enumerate{-1,0,1}^2images with exactdistance^2; prove farther images cannot cut when circumradius<minL/4',
      'new_candidate_graphs':16,'new_candidate_transient_runs':1,'new_complete_independent_uniformization_sequences':1,
      'new_reference_PDE_solves':0,'old_candidate_reruns':0,'new_material_definitions':0,
      'expected_objectivity':'IDedges/exactg agree with originalstatic afterseal; originalCSR actions agree; all17 actual/independent endpoints and16halfstates bitwise equal originalstatic16',
      'positiveD_behavior':'preserve nonzero baselineH andoriginalreferenceerror; do not forceH/kappa tozero or replace movingstates withreference',
      'time_and_space_scope':'singlecovariancecontrol,not convergence orangularscan; no newN/K levels',
      'planned_new_software_checks':3,'check_categories':['exactisometryroundtrip andinvalidboundary rejection','losslessfloat/IDs/canonicaldomain rejection','independentworldgeometry corruption rejection'],
      'tail_per_interval_exact':'1/10000000000000000','original_algebra_tolerance':1e-10,'norm_estimator_seed':20261101,
      'main_wall_seconds_cap':300,'main_peak_RSS_MiB_cap':1024,
      'resource_basis':'previous16N32 geometry run27.452s; envelope includes exactisometrycertificates/worldlifts; independentauditresources recorded separately',
      'stop':'retain failure; no scalarfit,pointrepair,mass/rho/state/row/weight normalization,F/Cfit,thresholdchange or automatic extraN/K/anglecase',
      'adapter_implemented_this_round':False,'rotation_points_graphs_states_generated_this_round':False,
      'kappa_computed':False,'fluid_ORC_run_required':False,'general_T2_admitted':False,'general_smooth_flow_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}


def main(output):
    started=time.monotonic();prior=read(SOURCE/'sha256.json');assert len(prior)==3825
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json');assert not gate['already_executed'] and gate['type']=='one_unexecuted_readonly_T2_rotation_domain_contract'
    for k in ('source_backend','source_method'):assert digest(ROOT/gate[k])==gate[k+'_sha256']
    assert all(gate[k]==0 for k in ('new_points','new_candidate_graphs','new_candidate_transient_paths','new_reference_PDE_solves','old_candidate_reruns','new_software_checks'))
    protocol=read(PROTOCOL);assert protocol['recorded_checks_total']==474
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,path in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('review-protocol.json',PROTOCOL),('contract-at-review.md',REPORT)]:shutil.copyfile(path,out/name)
    locked=['research/reconnect/t2_rotation_domain_review.py','research/reconnect/t2_rotation_domain_protocol.json','research/reconnect/T2_ROTATION_DOMAIN_CONTRACT.md']
    save(out/'review-lock.json',{'files':{p:digest(ROOT/p) for p in locked},'fixed_before_source_capability_inspection':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':3825,
      'parent_manifest_sha256':digest(SOURCE/'sha256.json'),'recorded_checks_total':474,'previous_checks_rerun':0,**dict.fromkeys(ZERO_FIELDS,0)})
    capability=inspect_backend(ROOT/gate['source_backend']);save(out/'backend-capability.json',capability)
    derivations=[
      {'id':'P1','claim':'mass/materialmeans andscalarisotropicPDE covariant under smoothrigidframe withcorrespondingvelocity anddomain','assumptions':['R inSO2,spatiallyconstant','scalarD','Ki,m,IDs pushforward together']},
      {'id':'P2','claim':'rotating lattice required;fixedsquare rotationsC4,fixednonsquare rectangleC2;nonzero rigidrotationvelocity not periodic onfixedtorus','assumptions':['positiveindependent rectangularperiods','samefixedlattice for automorphism statement']},
      {'id':'P3','claim':'currentaxisAPI cannotdirectlyexpress generalrotatedlattice','assumptions':['unchangedsourcehash andAST/exactsourceevidence']},
      {'id':'P4','claim':'sameIDg,M,Q,P,H andfixedreferenced0/ORC under exactisometry;D0identity implies fixedmetric kappa0','assumptions':['correspondingperiodiclattice','sharedordinarycenterresistance','samefixedIDs/m/c0','samevalidreferencecomponents']},
      {'id':'P5','claim':'separateopt-in boundaryadapter canreuseunmodifiedbuilder under exactisometry andlosslesslocalfloat contract','assumptions':['givenboundaryframe,notlocalfit','exactorthogonality','losslesslocalconversion','independentworldaudit required']},
      {'id':'P6','claim':'onecontinuousrationalrotationcase isphysicallyvalid onrotatingdomain,notfixedbox orquarterturn scan','assumptions':['specifiedtheta/velocity/lattice','sameoriginalmaterialstate','allcoefficientssealedbeforestate','actualimplementation/test notyetdone']}]
    save(out/'derivation-index.json',{'report':str(REPORT.relative_to(ROOT)),'report_sha256':digest(REPORT),'written_derivations':derivations,
      'machine_proof_verification_performed':False,'new_exact_algebra_certificate_groups':0,'not_new_numerical_test_results':True})
    save(out/'rotation-domain-decision.json',{'decision':'boundary_representation_gap_with_minimal_isometry_adapter_proposed',
      'current_api_boundary_mode':capability['boundary_mode'],'same_fixed_square_rotations':'multiplesofpi/2only','same_fixed_nonsquare_rectangle_rotations':'0andpionly',
      'continuousrigidrotation_requires_corresponding_periodic_basis':True,'axis_wrapping_of_general_rotated_worldpoints_is_valid_objectivity_case':False,
      'quarterturn_static_test_completes_continuous_rotation':False,'ordinary_center_resistance_disproved':False,
      'selected_minimal_proposal':'separate_exact_oriented_rectangle_boundary_isometry_adapter','global_boundary_orientation_is_local_F_C_fit':False,
      'adapter_implemented':False,'backend_or_api_changed':False,'rotation_test_executed':False,'positiveD_H_target':'sameasstaticpositiveDbaseline,notzero',
      'fixed_reference_metric_preserved_under_isometry':True,'D0_fixed_metric_kappa_zero_is_theoretical_not_a_fluid_ORC_run':True})
    save(out/'acceptance-ledger.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','T1_uniform_registered_case':'passed_prior_seal',
      'T2_translation':'passed_prior_seal','T2_D0_nonrigid':'bitwise_identity_passed_prior_seal','T2_D0_kappa':'fixedmetric_identity_theoretical_consequence,noORCrun',
      'T2_positiveD_rotation':'boundary_contract_completed,adapter_andphysicalrun_pending','T2_closeout_for_proposed_mode':'requireactualrotationpath+worldaudit+inputcontract+existingtranslation/D0 andfixedmetric interpretation',
      'arbitrary_float_world_inputs_or_nonorthogonal_lattices':'notadmittedbyfirstadapterproposal','T3_models_directions':'incomplete','T4_same_backend_shock_contact':'incomplete',
      'same_backend_density_source_comparison':'incomplete','new_DNS_or_dynamic_ORC_prerequisite_added':False,'completion_date_known':False,
      'recorded_checks_total':474,'new_software_checks':0,'general_T2_admitted':False,'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})
    save(out/'next-gate-spec.json',next_gate())
    sources={gate['source_backend']:'originalgeometryAPI,axisperiodicdisplacementevidence',gate['source_method']:'originalphysicalstate,T2,boundaryandORCrequirements',
      'docs/CODEX_TASK.md':'originalfixedstate/boundaryandbackendacceptancerequirements','runs/reconnect-t2-d0/sha256.json':'3825entrypriorseal',
      'runs/reconnect-t2-d0/next-gate-spec.json':'immutable read-only reviewregistration','runs/reconnect-t2-d0/results.json':'savedD0identity,no rerun',
      'runs/reconnect-t2-translation/results.json':'savedpositiveDtranslation,no rerun','research/reconnect/T1_UNIFORM_COMPARISON.md':'priorstaticmaterialreferenceevidence',
      'research/reconnect/T2_TRANSLATION_COMPARISON.md':'priortranslationreport','research/reconnect/T2_D0_COMPARISON.md':'priorD0report'}
    for row in next_gate()['sources'].values():sources[row['path']]='hash-only source for proposedfuture gate,noNPZparse'
    save(out/'evidence-index.json',{'sources':[{'path':p,'sha256':digest(ROOT/p),'role':role} for p,role in sources.items()],
      'NPZ_arrays_loaded':False,'source_numerical_states_recomputed':False})
    save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','registered_rotation_domain_review_executed':True,
      'decision':'boundary_representation_gap_with_minimal_isometry_adapter_proposed','recorded_checks_total':474,'previous_checks_rerun':0,
      **dict.fromkeys(ZERO_FIELDS,0),'adapter_implemented':False,'backend_or_api_changed':False,'rotation_test_executed':False,
      'next_gate_executed':False,'general_T2_admitted':False,'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False,'pdf_generated':False,'software_release_created':False,'published':False})
    save(out/'execution.json',{'review_exit_code':0,'review_first_attempt_passed':True,'elapsed_review_seconds':time.monotonic()-started,
      'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'python':platform.python_version(),'standard_library_only':True,
      'NPZ_arrays_loaded':False,'new_numerical_runs':0,'network_used':False,'environment_configuration_changed':False,'committed_or_pushed':False,'published':False})
    print(json.dumps({'decision':'boundary_representation_gap_with_minimal_isometry_adapter_proposed','new_points_graphs_states_tests':0,'recorded_checks_total':474,'next_gate_executed':False},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
