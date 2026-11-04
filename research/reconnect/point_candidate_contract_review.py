"""Read-only evidence synthesis and a frozen specification, not a candidate.

Standard library only. Does not import numerical/proof modules, construct edge
weights, evaluate operators, or run the predeclared future gate.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import ast
import json

from audit_route_decision import ROOT,digest,read,save


SOURCE=ROOT/'runs/reconnect-FV-contact-spatial'
ROLES={
    'research/reconnect/MAINLINE_RECONNECTION.md':'allowed point inputs, supplied graph and physical backend gap',
    'research/reconnect/CANDIDATE_ROUTE_DECISION.md':'historical route choice and restricted counterexample scope',
    'research/reconnect/CONDITIONAL_DENSITY_CLOSURE.md':'unchanged conditional constitutive contract',
    'research/reconnect/DENSITY_RECONSTRUCTION_WEAK_BOUND.md':'later completed conditional continuous density term',
    'research/reconnect/POINT_PAIR_QUADRATURE_BRIDGE.md':'later completed conditional material/point bridge; not evolution admission',
    'research/reconnect/PERIODIC_DENSITY_PATH_PRIMITIVE.md':'implemented inverse-density primitive only',
    'research/reconnect/SAVED_DENSITY_PATH_CHECK.md':'three exact saved path observations only',
    'research/reconnect/ORACLE_PATH_RESISTANCE_CONTROL.md':'continuous oracle result, not finite-node path exchange',
    'research/reconnect/ORACLE_INTERFACE_SECOND_ORDER.md':'known radial-path strong curvature limitation',
    'research/reconnect/MASS_AVERAGE_PROJECTION_WEAK_BUDGET.md':'material projection scope and diameter requirement',
    'research/reconnect/FV_GLOBAL_FROZEN_COMPARISON.md':'full-domain frozen FV baseline',
    'research/reconnect/FV_CONTACT_FIXED_TIME_COMPARISON.md':'limited ordinary FV time control',
    'research/reconnect/FV_CONTACT_COMMON_OBSERVATION.md':'bounded common-material spatial control; stop FV expansion',
    'research/reconnect/positive_kernel_2d.py':'unchanged B_h and original endpoint harmonic exchange',
    'research/reconnect/periodic_density_path.py':'implemented input-only path primitive',
    'research/reconnect/endpoint_density_path.py':'implemented exact input localization',
    'runs/reconnect-density-weak/stage3-status.json':'continuous perturbation proof is already complete conditionally',
    'runs/reconnect-point-quadrature/stage3-status.json':'point-pair proof is already complete conditionally',
    'runs/reconnect-route-decision/route-decision.json':'historical contract; old next-work status not current backlog',
    'runs/reconnect-oracle-path/results.json':'saved continuous oracle errors',
    'runs/reconnect-FV-global-frozen/results.json':'saved point kernel and full FV frozen errors',
    'runs/reconnect-FV-face-budget/source-main-c0-comparison.json':'original rows, rates, masses, c0 and fixed denominator',
    'runs/reconnect-FV-contact-time/results.json':'saved FV finite-time result, not a point-kernel transient',
    'runs/reconnect-FV-contact-spatial/results.json':'saved common-observation control',
    'runs/reconnect-contact-frozen/config.json':'fixed original N, h, D and periodic problem',
    'runs/reconnect-contact-frozen/contact-frozen.npz':'immutable original numerical input; not evaluated this round',
}


def signatures(path,names):
    tree=ast.parse(Path(path).read_text());result={}
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            result[node.name]=[arg.arg for arg in node.args.posonlyargs+node.args.args+node.args.kwonlyargs]
    assert set(result)==set(names)
    return result


def gate_spec(source_protocol,pair,config):
    p=config['protocol']
    return {
        'name':'reconstructed_density_path_full_frozen_pair','already_executed':False,
        'scope':'one original saved contact, complete support rows for IDs 63 and 64; no full graph or propagation',
        'source_snapshot':'runs/reconnect-contact-frozen/contact-frozen.npz',
        'source_snapshot_sha256':'9fe1d565ec4fabbe61bd1f4cf8d2dd8c0d730e6e1fc5ab56e2c185ac35f68a98',
        'frame':0,'N':p['N_per_direction'],'target_ids':[63,64],
        'D_exact':str(F(p['D_m2_per_s'])),'h_exact':str(F(p['h_m'])),
        'periods':p['periods_m'],'physical_thickness_exact':'1',
        'state':'original fixed material-cell mass averages; unchanged full original c0',
        'coefficient_builder_inputs':['point_positions','fixed_ID','fixed_mass','given_positive_point_density',
            'constant_D','predeclared_h','periodic_lengths','unit_thickness','requested_target_IDs'],
        'state_application_inputs':['prebuilt_row_coefficients','original_cbar_by_ID'],
        'forbidden_weight_dependencies':['c0','reference_scalar','PDE_rate','true_material_faces','true_material_adjacency',
            'true_cell_volumes','F_C','velocity_gradients','fitted_moments'],
        'constitutive_rule':{
            'rho_hat':'periodic Euclidean nearest-point density; ties use fixed minimum ID',
            'a_bar_rho':'integral_0^1 1/rho_hat(x_i+theta*d_ij) dtheta',
            'K_rec':'D/a_bar_rho',
            'q_i':'m_i/(rho_i*b); quadrature area, not inferred true material area',
            'B_h':'reuse unchanged positive_kernel_2d.slope_over_radius(r,h)',
            'g_ij':'2*b*q_i*q_j*K_rec*B_h(r_ij)',
            'rate_i':'sum_j g_ij*(cbar_j-cbar_i)/m_i',
            'D_zero':'zero exchange without evaluating infinite resistance'},
        'support':'all original geometric neighbors at r<=2h, unchanged B_h has zero value at and beyond r/h=2',
        'all_global_density_inputs_validated_before_exact_endpoint_ball_localization':True,
        'path_implementation':'existing exact inverse_density_path and proven endpoint-ball localization; no endpoint-only density shortcut',
        'all_support_directions_required_not_only_three_previously_checked_paths':True,
        'shared_target_edge_evaluated_once_and_reused':True,
        'open_pair_not_claimed_to_have_zero_total_mass_rate':True,
        'comparison':{
            'PDE_reference':'saved original rates; saved face/volume8/volume16 controls only',
            'old_kernel':'saved original endpoint-harmonic rates, same original positions and c0',
            'FV_context':'saved ordinary FV two-row result; extra geometry baseline, not candidate inputs',
            'normalization':'sqrt(sum_target m_i*(rate_i-reference_i)^2)/original_all_cell_PDE_rate_norm',
            'fixed_global_PDE_norm':pair['original_all_cell_mass_weighted_PDE_rate_norm'],
            'saved_old_kernel_pair_contribution':pair['saved_kernel_two_row_error_contribution_over_global_norm'],
            'saved_FV_pair_contribution':pair['FV_two_row_error_contribution_over_global_norm'],
            'raw_per_row_signed_rates_and_errors_required':True,
            'density_path_and_B_h_arithmetic_controls_and_c0_row_norm_indicator_required':True,
            'benefit_diagnostic':'saved old kernel pair error minus reconstructed pair error; interpret only beyond saved reference and arithmetic indicators',
            'no_new_absolute_physical_acceptance_threshold':True,
            'pair_contribution_not_a_global_error_or_finite_time_result':True},
        'stopping_rules':[
            'invalid original input or failed sharing/arithmetic control: retain failure and stop',
            'no resolvable benefit: retain outcome and stop this finite-configuration escalation',
            'even a resolvable pair improvement does not authorize a full graph, transient, production backend or ORC',
            'do not change h, D, density, c0, targets, kernel normalization or thresholds after observing results'],
        'prohibited_repairs':['coefficient_or_probability_clipping','mass_or_density_renormalization','c0_or_probe_fitted_weights',
            'common_cross_phase_scalar_search','new_auxiliary_scalar_cases_to_pick_weights'],
        'no_new_grid_h_time_or_flow_scan':True,'new_full_candidate_graphs':0,
        'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,'fluid_orc_enabled':False,
    }


def validate_review(decision,gate,ledger,status):
    assert decision['selected_next_gate']==gate['name'] and not gate['already_executed']
    assert gate['N']==128 and gate['frame']==0 and gate['target_ids']==[63,64]
    assert F(gate['D_exact'])==F(.02) and F(gate['h_exact'])==F(.04)
    assert gate['periods']==[1.,1.] and gate['physical_thickness_exact']=='1'
    assert gate['constitutive_rule']['K_rec']=='D/a_bar_rho'
    assert gate['constitutive_rule']['g_ij']=='2*b*q_i*q_j*K_rec*B_h(r_ij)'
    assert 'c0' not in gate['coefficient_builder_inputs'] and 'c0' in gate['forbidden_weight_dependencies']
    assert not decision['density_partition_is_material_geometry'] and not decision['ordinary_FV_is_general_point_backend']
    assert decision['older_contract_next_work_is_historical_not_current_backlog']
    assert ledger['continuous_density_weak_term']['current_state']=='proved_conditionally'
    assert ledger['point_pair_material_bridge']['current_state']=='proved_conditionally_not_finite_time_admission'
    assert ledger['finite_node_path_exchange']['current_state']=='not_implemented_or_physically_evaluated_in_reviewed_adapter'
    assert not ledger['point_pair_material_bridge']['fill_distance_alone_identifies_material_diameter']
    for obj in [decision,status]:
        assert obj['recorded_checks_total']==403 and obj['stage3_physical_acceptance']=='incomplete' and not obj['fluid_orc_enabled']
        assert not obj['finite_node_candidate_admitted']
        for key in ['new_software_checks','previous_checks_rerun','new_exact_algebra_certificate_groups','new_edge_weights',
                    'new_candidate_graphs','new_numerical_cases','new_candidate_transient_runs','new_reference_time_solutions']:
            assert obj[key]==0,key


def replay(output):
    out=Path(output).resolve()
    decision,gate,ledger,status=(read(out/name) for name in ['review-decision.json','gate-spec.json','assumption-ledger.json','stage3-status.json'])
    validate_review(decision,gate,ledger,status)
    for name,expected in read(out/'receipt-manifest.json').items():assert digest(out/name)==expected,name
    print(json.dumps({'standard_library_contract_review_replay_passed':True,'selected_gate':gate['name'],
        'gate_executed':False,'recorded_checks_total_unchanged':403,'new_weights_or_numerical_cases':0,'fluid_orc_enabled':False}),flush=True)


def main(output):
    manifest=read(SOURCE/'sha256.json')
    for rel,expected in manifest.items():assert digest(ROOT/rel)==expected,rel
    source_status=SOURCE/'stage3-status.json';protocol=read(source_status)['next_point_candidate_contract_review_predeclared']
    assert not protocol['already_executed'] and protocol['new_numerical_cases']==protocol['new_edge_weights']==0
    pair=read(ROOT/'runs/reconnect-FV-face-budget/source-main-c0-comparison.json')
    cfg=read(ROOT/'runs/reconnect-contact-frozen/config.json')
    weak=read(ROOT/'runs/reconnect-density-weak/stage3-status.json')
    bridge=read(ROOT/'runs/reconnect-point-quadrature/stage3-status.json')
    assert weak['density_weak_perturbation_proved_under_stated_assumptions'] and bridge['conditional_point_pair_quadrature_bound_proved']
    gate=gate_spec(protocol,pair,cfg)
    zero={'new_software_checks':0,'previous_checks_rerun':0,'new_exact_algebra_certificate_groups':0,'new_edge_weights':0,
        'new_candidate_graphs':0,'new_numerical_cases':0,'new_candidate_transient_runs':0,'new_reference_time_solutions':0,
        'recorded_checks_total':403,'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    decision={**zero,'decision':'retain limited FV baseline; select one fixed reconstructed-path pair diagnostic, not backend admission',
        'selected_next_gate':gate['name'],'ordinary_FV_is_general_point_backend':False,
        'density_partition_is_material_geometry':False,'older_contract_next_work_is_historical_not_current_backlog':True,
        'restricted_scalar_repair_stays_closed':True,'all_positive_methods_not_ruled_out':True,
        'continuous_oracle_result_is_not_a_discrete_candidate_result':True,
        'material_diameter_not_inferred_from_point_spacing':True,
        'no_further_FV_grid_or_time_expansion':True,'no_new_generic_true_face_F_C_or_gradient_inputs':True}
    ledger={
        'runtime_inputs':{'current_state':'available','inputs':gate['coefficient_builder_inputs'],
            'state':'fixed material-cell mass averages','forbidden_weight_dependencies':gate['forbidden_weight_dependencies']},
        'continuous_density_weak_term':{'current_state':'proved_conditionally','source':'research/reconnect/DENSITY_RECONSTRUCTION_WEAK_BOUND.md',
            'assumptions':['true global positive density bounds','phasewise regularity and interface tube control','fill distance and true point density'],
            'not_proved':'uniform accuracy on every sampled line or finite-node evolution'},
        'point_pair_material_bridge':{'current_state':'proved_conditionally_not_finite_time_admission',
            'source':'research/reconnect/POINT_PAIR_QUADRATURE_BRIDGE.md',
            'assumptions':['phase-constant flat contact model','representative point lies in actual material cell','true material diameter Delta/h tends to zero','fixed Lipschitz tests'],
            'fill_distance_alone_identifies_material_diameter':False,'path_superiority_proved':False},
        'nearest_point_density_partition':{'current_state':'declared_interpolation_model_not_identified_truth',
            'is_material_partition':False,'may_replace_mass_or_quadrature_area':False},
        'path_integral_primitive':{'current_state':'implemented_exactly_and_software_checked',
            'three_saved_paths_only_not_all_neighbor_operator':True,'outputs':'inverse-density average, not conductance'},
        'finite_node_path_exchange':{'current_state':'not_implemented_or_physically_evaluated_in_reviewed_adapter',
            'selected_gap':'complete finite-node mass-mean row action with the already declared constitutive rule'},
        'volume_quadrature_and_state_projection':{'current_state':'conditional_gap_for_general_point_inputs',
            'q_i':'m_i/(rho_i*b) is a quadrature area; equality to true area is benchmark-specific'},
        'evolution_and_ORC':{'current_state':'not_admitted_for_point_candidate','positive_shared_graph_algebra_is_not_physical_accuracy':True}}
    inventory={
        'periodic_density_path.py':signatures(ROOT/'research/reconnect/periodic_density_path.py',['inverse_density_path']),
        'endpoint_density_path.py':signatures(ROOT/'research/reconnect/endpoint_density_path.py',['endpoint_ball_selection','localized_inverse_density_path']),
        'positive_kernel_2d.py':signatures(ROOT/'research/reconnect/positive_kernel_2d.py',['slope_over_radius','build_periodic_exchange']),
        'future_adapter_proposed_path':'research/reconnect/reconstructed_path_rows.py',
        'future_adapter_present_at_review':(ROOT/'research/reconnect/reconstructed_path_rows.py').exists(),
        'code_inspection_method':'AST only, no numerical implementation imports'}
    assert not inventory['future_adapter_present_at_review']
    frozen=read(ROOT/'runs/reconnect-FV-global-frozen/results.json');time=read(ROOT/'runs/reconnect-FV-contact-time/results.json')
    spatial=read(SOURCE/'results.json');oracle=read(ROOT/'runs/reconnect-oracle-path/results.json')
    saved={'FV_global_frozen_error':frozen['ordinary_FV_global_relative_error'],
        'original_point_kernel_global_frozen_error':frozen['saved_kernel_global_relative_error'],
        'FV_semidiscrete_T0_05_error':time['spatial_semidiscrete_relative_error'],
        'FV_fine_common_observation_error':spatial['fine256_error_under_original128_observation'],
        'FV_two_level_error_ratio':spatial['coarse_error_divided_by_observed_fine_error'],
        'continuous_oracle_path_frozen_error':oracle['oracle_runs']['fine']['errors']['all'],
        'pair_controls':pair,'saved_values_only_no_scientific_recalculation':True}
    status={**zero,'updated':'2026-11-01','timezone':'Etc/UTC','scope':protocol['scope'],
        'review_completed':True,'selected_next_gate':gate['name'],
        'ordinary_central_resistance_FV_baseline_retained':True,'mass_average_resistance_remains_experimental':True,
        'next_reconstructed_path_pair_gate_predeclared':gate,'old_v080_or_246_tests_recovered':False,
        'report':'research/reconnect/POINT_CANDIDATE_INFORMATION_REVIEW.md',
        'pdf_generated':False,'software_version_created':False,'published':False}
    validate_review(decision,gate,ledger,status)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',{'study':'read-only point-input information-contract review',
        'protocol':protocol,'protocol_source_sha256':digest(source_status),'source_manifest_entries_verified':len(manifest),
        'scientific_modules_imported_or_executed':False,'software_or_physical_test_suites_run':False})
    for name,value in [('review-decision.json',decision),('gate-spec.json',gate),('assumption-ledger.json',ledger),
        ('implementation-inventory.json',inventory),('saved-metrics.json',saved),('stage3-status.json',status)]:save(out/name,value)
    receipts={'source-review-protocol.json':source_status,
        'source-density-weak-status.json':ROOT/'runs/reconnect-density-weak/stage3-status.json',
        'source-point-pair-status.json':ROOT/'runs/reconnect-point-quadrature/stage3-status.json'}
    for name,path in receipts.items():
        with (out/name).open('xb') as f:f.write(path.read_bytes())
    save(out/'receipt-manifest.json',{name:digest(path) for name,path in receipts.items()})
    index=[]
    for rel,role in ROLES.items():
        assert manifest[rel]==digest(ROOT/rel),rel
        index.append({'path':rel,'sha256':manifest[rel],'role':role,'source_read_only':True,'scientific_recalculation':False})
    save(out/'evidence-index.json',{'source_manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
        'source_manifest_sha256':digest(SOURCE/'sha256.json'),'sources':index})
    replay(out)
    print(json.dumps({'review_completed':True,'selected_next_gate':gate['name'],'gate_executed':False,
        'recorded_checks_total':403,'new_numerical_cases':0,'new_weights':0},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--output');group.add_argument('--replay')
    args=parser.parse_args();replay(args.replay) if args.replay else main(args.output)
