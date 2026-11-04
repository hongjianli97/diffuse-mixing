"""Audit saved-vector identities, exact Gram controls and historical hashes."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np

from audit_route_decision import ROOT,read,save,digest,preservation
from global_residual_review import SOURCE,ORACLE,DIRECTIONAL,NORM,record,replay
from global_residual_scope_decision import next_gate


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    replay(out)
    provenance=read(out/'input-provenance.json');budget=read(out/'residual-budget.json')
    assert provenance['parent_input_sha256']==digest(SOURCE/'original-inputs.npz')
    assert provenance['parent_action_sha256']==digest(SOURCE/'global-action.npz')
    assert provenance['oracle_snapshot_sha256']==digest(ORACLE) and provenance['directional_snapshot_sha256']==digest(DIRECTIONAL)
    with np.load(out/'saved-vector-inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(out/'residual-vectors.npz',allow_pickle=False) as z:vectors={k:z[k] for k in z.files}
    with np.load(SOURCE/'original-inputs.npz',allow_pickle=False) as z:
        for k in ['ids','mass','c0','positions','rho','reference_rate','kernel_rate','near_contact','far_from_contact']:
            assert record(z[k])==record(data[k]),k
    with np.load(SOURCE/'global-action.npz',allow_pickle=False) as z:assert record(z['rate'])==record(data['reconstructed_rate'])
    with np.load(ORACLE,allow_pickle=False) as z:
        for item in provenance['original_common_member_matches']:
            assert item['identical_bytes']==record(data[item['current_member']])==record(z[item['oracle_member']])
        assert record(z['fine_rate'])==record(data['oracle_fine_rate'])
        assert record(z['coarse_rate'])==record(data['oracle_coarse_rate'])
    with np.load(DIRECTIONAL,allow_pickle=False) as z:
        assert record(z['fine_saved_total_rate'])==record(data['oracle_fine_rate'])
        assert np.array_equal(z['PDE_rate_a']+z['PDE_rate_b'],data['reference_rate'])
    assert all(provenance['source_array_records'][k]==record(v) for k,v in data.items())
    combined=data['reconstructed_rate']-data['oracle_fine_rate']
    continuous=data['oracle_fine_rate']-data['reference_rate']
    total=data['reconstructed_rate']-data['reference_rate']
    assert np.array_equal(combined,vectors['combined_point_pair_and_projection'])
    assert np.array_equal(continuous,vectors['continuous_oracle_minus_PDE'])
    assert np.array_equal(total,vectors['reconstructed_minus_PDE'])
    components=[combined,continuous,vectors['arithmetic_closure']]
    exact_gram=[];primary_minus_exact=[];den=F(NORM)**2
    for u in components:
        row=[];difference=[]
        for v in components:
            exact=sum((F(float(m))*F(float(a))*F(float(b)) for m,a,b in zip(data['mass'],u,v)),F(0))/den
            row.append(float(exact))
            computed=math.fsum(float(m)*float(a)*float(b) for m,a,b in zip(data['mass'],u,v))/(NORM*NORM)
            assert computed==budget['regions']['all']['signed_Gram'][len(exact_gram)][len(row)-1]
            difference.append(computed-float(exact))
        exact_gram.append(row);primary_minus_exact.append(difference)
    exact_binary_closure=np.array([float(F(float(t))-F(float(a))-F(float(b))-F(float(c)))
        for t,a,b,c in zip(total,combined,continuous,vectors['arithmetic_closure'])])
    binary_closure_norm=math.sqrt(math.fsum(float(m)*float(e)**2 for m,e in zip(data['mass'],exact_binary_closure)))/NORM
    assert budget['regions']['all']['combined_remainder_over_oracle_rule_indicator']>1
    assert budget['regions']['all']['main_component_correlation']>0
    assert budget['combined_remainder_not_named_pure_projection_error']
    assert budget['true_ideal_cell_pair_c0_action_not_available_in_saved_evidence']
    ideal=read(ROOT/'runs/reconnect-ideal-row/results.json')
    point=read(ROOT/'runs/reconnect-point-row/results.json')
    assert point['all_saved_row_points_same_pure_phase'] and point['signed_budget']['terms_have_opposite_sign']
    assert point['original_c0_saved_cosine_PDE_and_candidate_unchanged']
    assert ideal.get('new_edge_weights',0)==0 # Separate quadratic reference, not a missing original-c0 action.
    scope=read(out/'scope-ledger.json');gate=read(out/'next-gate-spec.json');status=read(out/'stage3-status.json')
    assert len(scope)==9 and gate==next_gate() and not gate['already_executed']
    assert gate['target_ids']==[63,64] and gate['two_predeclared_tensor_Gauss_orders']==[8,16]
    assert gate['candidate_electrical_coefficients_or_states_not_replaced'] and gate['not_new_required_candidate_inputs']
    assert gate['no_new_absolute_physical_acceptance_threshold']
    offsets=[tuple(v) for v in gate['neighbor_cell_offsets']]
    assert len(offsets)==len(set(offsets))==gate['neighbor_cells_per_target']
    assert (0,0) not in offsets and (11,1) in offsets
    assert F(11*11+1,128**2)>(2*F(.04))**2 # A cell can intersect support even if its representative point does not.
    for dx,dy in offsets:
        distance2=F(max(abs(dx)-1,0)**2+max(abs(dy)-1,0)**2,128**2)
        assert distance2<=(2*F(.04))**2
    assert status['next_ideal_original_c0_pair_reference_predeclared']==gate
    assert status['ideal_reference_result']=='not_computed' and status['global_point_pair_vs_projection_split']=='not_identified'
    assert status['recorded_checks_total']==417
    for k in ['new_software_checks','previous_checks_rerun','new_weights','new_path_integrals','new_kernel_or_operator_evaluations',
        'new_candidate_graphs','new_global_matrices','new_numerical_or_flow_cases','new_candidate_transient_runs',
        'new_PDE_reference_solves','new_exact_algebra_certificate_groups']:
        assert status[k]==0,k
    assert not status['finite_node_candidate_admitted'] and not status['fluid_orc_enabled']
    for name in ['scipy','cached_global_exchange','reconstructed_global_study','oracle_path_reference',
        'positive_kernel_2d','periodic_density_path','mass_projection_certificates','oracle_weak_certificates']:
        assert name not in sys.modules,name
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']
    prior=prior+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
        'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in prior:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256']
        entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==70
    save(out/'preservation.json',preserved)
    audit={'same_original_observation_point_inputs_state_reference_and_regions_verified':True,
        'old_fine_oracle_and_directional_total_identical_without_recomputation':True,
        'saved_vector_budget_and_nonadditive_signed_Gram_verified':True,
        'exact_binary_Gram_control':exact_gram,'primary_Gram_minus_exact_binary_control':primary_minus_exact,
        'exact_binary_vector_closure_norm_control':binary_closure_norm,
        'float_vector_closure_not_forced_to_zero':True,
        'quadratic_ideal_reference_not_misused_as_original_c0_projection_error':True,
        'one_missing_ideal_original_c0_pair_gate_selected_not_executed':True,
        'future_reference_cell_support_not_confused_with_point_neighbor_support':True,
        'no_operator_kernel_primitive_proof_or_old_test_suite_executed':True,
        'new_software_checks':0,'recorded_checks_total':417,'previous_checks_rerun':0,
        'new_weights':0,'new_numerical_or_flow_cases':0,'new_exact_algebra_certificate_groups':0,
        'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':70,
        'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'review-audit.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True)
    main(parser.parse_args().directory)
