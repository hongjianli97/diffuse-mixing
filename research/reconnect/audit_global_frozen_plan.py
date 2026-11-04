"""Independent count/contract and provenance audit; no candidate evaluation."""
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np

from audit_route_decision import ROOT,read,save,digest,preservation
from global_frozen_gate_plan import SOURCE,SNAPSHOT,SNAPSHOT_SHA,array_record,replay


def main(output):
    out=Path(output).resolve();assert not (out/'sha256.json').exists()
    replay(out)
    status=read(out/'stage3-status.json');cost=read(out/'cost-and-reuse.json');gate=read(out/'gate-spec.json')
    provenance=read(out/'input-provenance.json')
    assert (out/'source-stage3-status.json').read_bytes()==(SOURCE/'stage3-status.json').read_bytes()
    assert read(out/'config.json')['source_status_sha256']==digest(SOURCE/'stage3-status.json')
    assert digest(ROOT/SNAPSHOT)==SNAPSHOT_SHA==provenance['source_snapshot_sha256']
    with np.load(out/'planning-point-inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(ROOT/SNAPSHOT,allow_pickle=False) as z:
        for key,record in provenance['planning_members'].items():
            original=z[key][0] if key in ['positions','rho'] else z[key]
            assert array_record(original)==record==array_record(data[key]),key
        for key,record in provenance['future_state_reference_and_region_hashes_only'].items():
            assert array_record(z[key])==record,key
    # An independent bounded integer disk enumeration, rather than the
    # planner's signed-offset loop or a numerical neighbor-search library.
    N=128;cut=(2*F(.04)*N)**2;limit=cut.numerator//cut.denominator
    maximum=math.isqrt(limit)
    half=[(dx,dy) for dx in range(maximum+1) for dy in range(-maximum,maximum+1)
          if (dx>0 or dy>0) and dx*dx+dy*dy<=limit]
    half=sorted(half)
    assert half==[tuple(v) for v in cost['offsets_positive_half']]
    assert len(half)==166 and maximum==10 and limit==104
    expected_profiles={0:2,**{dx:2+2*dx for dx in range(1,maximum+1)}}
    actual_profiles=Counter(v['normal_offset_steps'] for v in cost['classes'])
    assert actual_profiles==expected_profiles
    expected_coefficients=2*maximum+sum((2+2*dx)*(math.isqrt(limit-dx*dx)+1) for dx in range(1,maximum+1))
    assert expected_coefficients==964==cost['distinct_coefficient_cache_slots_planned']
    assert sum(expected_profiles.values())==132==cost['representative_path_calls_planned']
    assert len({dx*dx+dy*dy for dx,dy in half})==45
    assert N*N*len(half)==2719744
    original=read(ROOT/'runs/reconnect-contact-frozen/results.json')
    assert original['operator_diagnostics']['geometric_support_pairs']==2719744
    assert gate['comparison']['saved_original_kernel_global_error']==original['regional_comparison']['all']['error_over_global_reference_norm']
    fv=read(ROOT/'runs/reconnect-FV-global-frozen/results.json')
    assert gate['comparison']['saved_ordinary_FV_global_error_context']==fv['ordinary_FV_global_relative_error']
    assert gate['comparison']['continuous_oracle_path_value_not_finite_node_target']==read(ROOT/'runs/reconnect-oracle-path/results.json')['oracle_runs']['fine']['errors']['all']
    assert gate['comparison']['locked_original_global_PDE_norm']==read(SOURCE/'main-c0-comparison.json')['fixed_global_PDE_norm']
    assert gate['comparison']['no_new_absolute_physical_acceptance_threshold']
    with np.load(out/'request-class-map.npz',allow_pickle=False) as z:
        path_map=z['path_class'];coefficient_map=z['coefficient_class']
    assert path_map.shape==coefficient_map.shape==(128,166)
    assert set(map(int,path_map.ravel()))==set(range(132))
    assert set(map(int,coefficient_map.ravel()))==set(range(964))
    rho=data['rho'].reshape(N,N)
    for column in range(N):
        for k,(dx,dy) in enumerate(half):
            sid=int(path_map[column,k]);record=cost['classes'][sid]
            assert record['normal_offset_steps']==dx
            assert record['given_density_sequence']==[str(F(float(rho[0,(column+j)%N]))) for j in range(dx+1)]
            coeff_key=cost['coefficient_cache_keys'][int(coefficient_map[column,k])]
            assert coeff_key==[sid,cost['radial_class_per_offset'][k]]
    for record in cost['classes']:
        i,j=record['representative_input_indices']
        assert record['representative_start_id']==int(data['ids'][i])
        assert record['representative_end_id']==int(data['ids'][j])
        assert record['representative_path_not_evaluated']
    assert status['next_global_frozen_gate_predeclared']==gate and not gate['already_executed']
    assert status['global_candidate_result']=='not_computed'
    for key in ['new_software_checks','previous_checks_rerun','new_paths_evaluated','new_edge_weights',
                'new_operator_actions','new_numerical_cases','new_candidate_graphs','new_global_matrices',
                'new_candidate_transient_runs','new_reference_solves','new_exact_algebra_certificate_groups']:
        assert status[key]==0,key
    assert status['recorded_checks_total']==410
    assert not status['finite_node_candidate_admitted'] and not status['fluid_orc_enabled']
    assert cost['only_integral_value_reusable_not_raw_ID_ownership_receipt']
    assert gate['integer_encoding_does_not_change_any_physical_coefficient_or_state']
    assert gate['c0_read_only_after_coefficient_cache_fixed']
    assert gate['resources']['caps_are_execution_limits_not_physical_acceptance_thresholds']
    forbidden={'c0','PDE_rate','true_material_faces','true_material_volumes','F_C','velocity_gradients'}
    assert forbidden<=set(gate['forbidden_weight_dependencies'])
    assert not forbidden.intersection(gate['coefficient_inputs'])
    for name in ['scipy','periodic_density_path','endpoint_density_path','positive_kernel_2d','reconstructed_path_rows',
                 'reconstructed_pair_study','contact_spectral_reference']:
        assert name not in sys.modules,name
    preserved=preservation()
    known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']
    prior=prior+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
        'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in prior:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256']
        entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==68
    save(out/'preservation.json',preserved)
    audit={'source_protocol_and_original_point_state_reference_region_hashes_verified':True,
        'independent_integer_support_and_profile_count_formula_agree':True,
        'full_coverage_counts_not_obtained_by_constructing_global_edges':True,
        'all_21248_normal_column_offset_slots_accounted_for':True,
        '132_future_primitive_requests_45_radius_keys_964_coefficient_slots_verified':True,
        'only_numeric_path_integral_reusable_ID_receipts_not_transplanted':True,
        'integer_representation_is_exact_arithmetic_not_mass_density_or_probability_renormalization':True,
        'old_pair_equality_and_original_global_norm_required_before_new_comparison':True,
        'new_global_error_not_computed_and_no_acceptance_claim':True,
        'geometry_cost_record_replay_passed':True,'no_candidate_primitive_kernel_or_solver_imported':True,
        'new_software_checks':0,'recorded_checks_total':410,'previous_checks_rerun':0,
        'new_edge_weights':0,'new_numerical_cases':0,'new_exact_algebra_certificate_groups':0,
        'prior_formal_manifests_verified':68,'original_files_verified':782,'historical_attachments_verified':15,
        'finite_node_candidate_admitted':False,'fluid_orc_enabled':False}
    save(out/'plan-audit.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
