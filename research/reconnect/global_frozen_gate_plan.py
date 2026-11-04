"""Read-only geometry/cost review for one future original-snapshot global gate.

NumPy is used for saved-array IO and exact dyadic-grid equality only. This
module never imports the kernel, path primitive, row adapter or an integrator.
It emits requests and cache-index plans, not inverse-density values or weights.
"""
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import sys
import numpy as np

from audit_route_decision import ROOT,read,save,digest

SOURCE=ROOT/'runs/reconnect-reconstructed-pair'
SNAPSHOT='runs/reconnect-contact-frozen/contact-frozen.npz'
SNAPSHOT_SHA='9fe1d565ec4fabbe61bd1f4cf8d2dd8c0d730e6e1fc5ab56e2c185ac35f68a98'


def array_record(a):
    return {'shape':list(a.shape),'dtype':a.dtype.str,'C_order_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def plan_classes(positions,ids,mass,rho,h):
    """Classify requests using point inputs only; do not evaluate any path."""
    N=128
    expected=np.array([((2*c+1)/(2*N),(2*r+1)/(2*N)) for r in range(N) for c in range(N)])
    if positions.shape!=(N*N,2) or not np.array_equal(positions,expected):
        raise ValueError('this bounded acceleration requires the exact saved complete dyadic Cartesian point array')
    if ids.shape!=(N*N,) or len(set(map(int,ids)))!=N*N:
        raise ValueError('unique complete fixed identities required')
    if mass.shape!=ids.shape or rho.shape!=ids.shape:
        raise ValueError('all global mass and density inputs required')
    if not np.isfinite(mass).all() or not np.isfinite(rho).all() or np.any(mass<=0) or np.any(rho<=0):
        raise ValueError('positive finite global mass/density required')
    column_rho=tuple(F(float(v)) for v in rho[:N])
    if any(F(float(v))!=column_rho[i%N] for i,v in enumerate(rho)):
        raise ValueError('density must repeat exactly in the transverse direction')
    if column_rho!=(F(1),)*64+(F(4),)*64:
        raise ValueError('saved given point-density pattern differs; no repair or generalized claim')
    if any(F(float(m))/F(float(r))!=F(1,N*N) for m,r in zip(mass,rho)):
        raise ValueError('saved q input pattern differs; no mass adjustment')
    width=F(h)
    if width!=F(.04):raise ValueError('original binary h required')
    offsets=[]
    for dx in range(-N//2+1,N//2):
        for dy in range(-N//2+1,N//2):
            if (dx>0 or dx==0 and dy>0) and F(dx*dx+dy*dy,N*N)<=(2*width)**2:
                offsets.append((dx,dy))
    offsets=sorted(offsets)
    assert len(offsets)==166 and all(dx>=0 for dx,dy in offsets)
    signatures=sorted({(dx,tuple(column_rho[(c+j)%N] for j in range(dx+1)))
                       for c in range(N) for dx,dy in offsets})
    indices={key:i for i,key in enumerate(signatures)}
    table=np.array([[indices[(dx,tuple(column_rho[(c+j)%N] for j in range(dx+1)))]
                     for dx,dy in offsets] for c in range(N)],dtype=np.int32)
    counts=Counter(map(int,table.ravel()))
    radii=sorted({dx*dx+dy*dy for dx,dy in offsets})
    radial_index={r:i for i,r in enumerate(radii)}
    radial_ids=[radial_index[dx*dx+dy*dy] for dx,dy in offsets]
    coefficient_keys=sorted({(int(table[c,k]),radial_ids[k]) for c in range(N) for k in range(len(offsets))})
    coefficient_index={key:i for i,key in enumerate(coefficient_keys)}
    coefficient_map=np.array([[coefficient_index[(int(table[c,k]),radial_ids[k])]
                               for k in range(len(offsets))] for c in range(N)],dtype=np.int32)
    classes=[]
    for sid,(dx,pattern) in enumerate(signatures):
        column=next(c for c in range(N) if tuple(column_rho[(c+j)%N] for j in range(dx+1))==pattern)
        # A horizontal path has the same integral as each oblique member.
        # dx=0 uses one vertical step because its unique nearest column is fixed.
        endpoint=(column+dx)%N if dx else N+column
        classes.append({'class_id':sid,'normal_offset_steps':dx,'given_density_sequence':list(map(str,pattern)),
            'representative_start_id':int(ids[column]),'representative_end_id':int(ids[endpoint]),
            'representative_input_indices':[column,endpoint],
            'normal_column_offset_slots':counts[sid],
            'future_unordered_edge_multiplicity':counts[sid]*N,
            'representative_path_not_evaluated':True})
    # Safe denominator-size bound without evaluating a, B_h or a conductance.
    # rho in {1,4}, dx<=10 implies a has an unreduced numerator <=8*10.
    # Any finite binary64 B_h has denominator dividing 2**1074.
    universal_Q=(1<<(1074+58+28))*math.lcm(*range(1,81))
    result={'N':N,'global_nodes':N*N,'given_point_array_verified_exactly':True,
        'global_density_and_mass_validated_before_reuse':True,'density_transverse_repeat_verified':True,
        'q_equals_1_over_N_squared_exactly_in_this_input':True,
        'offsets_positive_half':[list(v) for v in offsets],
        'directed_support_neighbors_per_node':2*len(offsets),
        'undirected_edge_iterations_planned':N*N*len(offsets),
        'normal_column_offset_slots':N*len(offsets),
        'path_integral_equivalence_classes':len(signatures),
        'representative_path_calls_planned':len(signatures),
        'kernel_radius_squared_classes':len(radii),
        'radius_squared_lattice_numerators':radii,'radius_squared_denominator':N*N,
        'distinct_coefficient_cache_slots_planned':len(coefficient_keys),
        'coefficient_cache_keys':[list(v) for v in coefficient_keys],
        'radial_class_per_offset':radial_ids,
        'classes':classes,'no_integral_or_coefficient_values_computed':True,
        'only_integral_value_reusable_not_raw_ID_ownership_receipt':True,
        'worst_case_coefficient_common_denominator_bits':universal_Q.bit_length(),
        'worst_case_flux_common_denominator_bits':(universal_Q*(1<<1074)).bit_length(),
        'denominator_bound_is_not_a_physical_rescaling':True,
        'paths_computed_this_review':0,'new_edge_weights':0,'new_operator_actions':0}
    assert sum(v['future_unordered_edge_multiplicity'] for v in classes)==2719744
    assert len(signatures)==132 and len(radii)==45
    return result,table,coefficient_map


def gate_spec(cost):
    return {'name':'one_original_N128_full_domain_reconstructed_path_frozen_action','already_executed':False,
        'source_snapshot':SNAPSHOT,'source_snapshot_sha256':SNAPSHOT_SHA,
        'frame':0,'N':128,'D_exact':str(F(.02)),'h_exact':str(F(.04)),
        'periods':[1.,1.],'physical_thickness_exact':'1',
        'state':'all unchanged original fixed material-cell mass averages c0; fixed original mass/IDs',
        'coefficient_inputs':['positions','fixed_ids','fixed_mass','given_positive_point_rho','D','h','periods','unit_thickness'],
        'forbidden_weight_dependencies':['c0','PDE_rate','reference_scalar','true_material_faces','true_material_volumes','F_C','velocity_gradients','fitted_moments'],
        'formula':'a=integral_0^1 1/rho_hat, K=D/a, q=m/(rho*b), g=2*b*q_i*q_j*K*unchanged_B_h(r)',
        'rho_hat':'periodic Euclidean nearest input point; fixed minimum-ID ties',
        'optimization_scope':'exactly validated saved Cartesian point array and transverse constant density; reject unsupported inputs',
        'reuse_rule':'cache only exact inverse-density integral by (nonnegative normal offset, given column-density sequence); preserve representative ID receipts, never transplant ownership',
        'canonical_geometric_orientation':'dx>0, or dx==0 and dy>0, with unique short minimum image; reverse integral exactly if needed',
        'representative_path_calls':cost['representative_path_calls_planned'],
        'original_primitive_required_for_every_representative':True,
        'original_endpoint_ball_localization_and_global_validation_required':True,
        'kernel_radius_classes':cost['kernel_radius_squared_classes'],
        'unchanged_kernel_function':'positive_kernel_2d.slope_over_radius',
        'planned_coefficient_cache_slots':cost['distinct_coefficient_cache_slots_planned'],
        'all_undirected_edge_iterations':cost['undirected_edge_iterations_planned'],
        'streaming_action_without_sparse_matrix':True,
        'primary_arithmetic':'exact rational coefficients given binary B_h; common integer representations g=G/Q and c0=C/S; signed shared-edge integer flux accumulation, then divide by Q*S and original mass',
        'integer_encoding_does_not_change_any_physical_coefficient_or_state':True,
        'c0_read_only_after_coefficient_cache_fixed':True,
        'mandatory_old_pair_control':'all original 663 cached conductances and both exact rational c0 rates match; do not regenerate or alter saved receipts',
        'required_new_software_controls':['small complete point grid cached versus original per-edge primitive',
            'reject unsupported/nonpositive or incomplete global input',
            'fixed-ID tie semantics for integral-only reuse; no copied ownership',
            'exact integer action equals direct Fraction action; constants and shared mass exchange',
            'state permutation/variation cannot select or alter weights'],
        'software_check_count':'record actual completed count; no claimed increment at planning time',
        'comparison':{'locked_original_global_PDE_norm':0.2479286450687567,
            'recomputed_norm':'arithmetic control only; never replace locked norm or assert bit identity',
            'saved_original_kernel_global_error':0.14156491268558014,
            'saved_ordinary_FV_global_error_context':0.032630071085544846,
            'continuous_oracle_path_value_not_finite_node_target':0.031229167742325044,
            'metrics':['all-cell mass-weighted rate error over locked global norm',
                'saved near_contact and far_from_contact contributions over same norm',
                'raw per-ID rates and signed errors, maximum error IDs',
                'new row L1 norms times saved c0 mean difference',
                'exact shared flux/mass inventory and energy exchange; unrepaired float residuals'],
            'regions':'original saved near_contact/far_from_contact masks, byte-identical; no residual-selected regions',
            'uncertainty':'sum saved face/volume8/16 controls, new binary/60/100-digit B_h action and float action/row norms; keep original kernel reduction controls',
            'benefit':'saved original kernel global error minus new global error, compared with reference/arithmetic indicators only',
            'no_new_absolute_physical_acceptance_threshold':True},
        'resources':{'wall_clock_seconds_cap':600,'peak_RSS_MiB_cap':512,
            'caps_are_execution_limits_not_physical_acceptance_thresholds':True,
            'exceedance':'preserve partial receipts and stop; do not silently drop edges, loosen arithmetic or replace inputs',
            'current_review_contains_no_timed_candidate_benchmark':True},
        'stop_rules':['failed point-pattern proof, input, cached-pair equality, shared exchange or arithmetic control: preserve failure and stop',
            'no resolvable global benefit: retain result and stop escalation of this fixed configuration',
            'even resolvable global improvement does not authorize propagation, production backend, Stage3 acceptance or ORC',
            'no h/grid/state/region/density/kernel/fitted-scalar/threshold retuning after observation'],
        'new_physical_configurations_max':1,'new_reference_solves':0,'new_transients':0,
        'no_new_grid_h_time_scan':True,'fluid_orc_enabled':False,'finite_node_candidate_admitted':False}


def validate(cost,gate,status):
    assert cost['undirected_edge_iterations_planned']==2719744
    assert cost['path_integral_equivalence_classes']==gate['representative_path_calls']==132
    assert cost['kernel_radius_squared_classes']==45
    assert all(cost[k]==0 for k in ['paths_computed_this_review','new_edge_weights','new_operator_actions'])
    assert not gate['already_executed'] and gate['comparison']['locked_original_global_PDE_norm']==0.2479286450687567
    assert not gate['fluid_orc_enabled'] and not gate['finite_node_candidate_admitted']
    assert status['recorded_checks_total']==410 and status['new_software_checks']==0
    assert status['stage3_physical_acceptance']=='incomplete' and not status['fluid_orc_enabled']


def main(output):
    entries=read(SOURCE/'sha256.json')
    for rel,h in entries.items():assert digest(ROOT/rel)==h,rel
    source_status=read(SOURCE/'stage3-status.json')
    assert source_status['decision']=='resolvable_pair_improvement' and source_status['recorded_checks_total']==410
    assert source_status['next_work'].startswith('read-only scope and cost review')
    snapshot=ROOT/SNAPSHOT;assert digest(snapshot)==SNAPSHOT_SHA
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',{'scope':'read-only global frozen gate design and exact geometric reuse/cost review',
        'source_status_sha256':digest(SOURCE/'stage3-status.json'),
        'source_manifest_entries_verified':len(entries),'prior_tests_or_numerical_results_not_rerun':True})
    shutil.copyfile(SOURCE/'stage3-status.json',out/'source-stage3-status.json')
    copied={}
    with np.load(snapshot,allow_pickle=False) as z:
        data={k:z[k].copy() for k in ['ids','mass']}
        data['positions']=z['positions'][0].copy();data['rho']=z['rho'][0].copy()
        for key in ['c0','reference_rate','kernel_rate','near_contact','far_from_contact']:
            copied[key]=array_record(z[key]) # hashes only, never passed to equivalence keys
    input_records={k:array_record(v) for k,v in data.items()}
    cost,table,coefficient_map=plan_classes(**data,h=F(.04))
    gate=gate_spec(cost)
    status={'updated':'2026-11-01','timezone':'Etc/UTC','scope_and_cost_review_completed':True,
        'new_software_checks':0,'previous_checks_rerun':0,'recorded_checks_total':410,
        'new_paths_evaluated':0,'new_edge_weights':0,'new_operator_actions':0,'new_numerical_cases':0,
        'new_candidate_graphs':0,'new_global_matrices':0,'new_candidate_transient_runs':0,'new_reference_solves':0,
        'new_exact_algebra_certificate_groups':0,'finite_node_candidate_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'ordinary_central_resistance_FV_baseline_retained':True,'mass_average_resistance_remains_experimental':True,
        'next_global_frozen_gate_predeclared':gate,'global_candidate_result':'not_computed',
        'pdf_generated':False,'software_release_created':False,'published':False}
    validate(cost,gate,status)
    save(out/'cost-and-reuse.json',cost);save(out/'gate-spec.json',gate);save(out/'stage3-status.json',status)
    np.savez_compressed(out/'planning-point-inputs.npz',**data)
    np.savez_compressed(out/'request-class-map.npz',path_class=table,coefficient_class=coefficient_map)
    save(out/'input-provenance.json',{'source_snapshot':SNAPSHOT,'source_snapshot_sha256':digest(snapshot),
        'source_frame':0,'planning_members':input_records,'future_state_reference_and_region_hashes_only':copied,
        'no_c0_or_reference_used_to_select_cache_keys':True,
        'planning_input_artifact_sha256':digest(out/'planning-point-inputs.npz'),
        'class_map_artifact_sha256':digest(out/'request-class-map.npz')})
    assert input_records=={k:array_record(v) for k,v in data.items()}
    assert digest(snapshot)==SNAPSHOT_SHA
    for name in ['scipy','periodic_density_path','endpoint_density_path','positive_kernel_2d','reconstructed_path_rows']:
        assert name not in sys.modules,name
    print(json.dumps({k:v for k,v in cost.items() if k not in ['classes','offsets_positive_half','coefficient_cache_keys','radial_class_per_offset','radius_squared_lattice_numerators']},indent=2),flush=True)


def replay(directory):
    out=Path(directory);provenance=read(out/'input-provenance.json')
    assert digest(out/'planning-point-inputs.npz')==provenance['planning_input_artifact_sha256']
    assert digest(out/'request-class-map.npz')==provenance['class_map_artifact_sha256']
    with np.load(out/'planning-point-inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    cost,table,coefficient_map=plan_classes(**data,h=F(.04))
    assert cost==read(out/'cost-and-reuse.json')
    with np.load(out/'request-class-map.npz',allow_pickle=False) as z:
        assert np.array_equal(table,z['path_class']) and np.array_equal(coefficient_map,z['coefficient_class'])
    gate=read(out/'gate-spec.json');assert gate==gate_spec(cost)
    validate(cost,gate,read(out/'stage3-status.json'))
    for name in ['scipy','periodic_density_path','endpoint_density_path','positive_kernel_2d','reconstructed_path_rows']:
        assert name not in sys.modules,name
    print('Geometry/cost and future gate declarations replay exactly; no path integral, coefficient, action or global result evaluated.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output');parser.add_argument('--replay')
    args=parser.parse_args()
    if args.replay:replay(args.replay)
    elif args.output:main(args.output)
    else:parser.error('--output or --replay required')
