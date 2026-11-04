"""Record scope limits and one unexecuted reference-only two-row gate."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import math

from audit_route_decision import ROOT,read,save,digest


def next_gate():
    N=128;width=F(.04);cut=(2*width*N)**2;limit=cut.numerator//cut.denominator
    reach=math.isqrt(limit)+1
    offsets=[(dx,dy) for dx in range(-reach,reach+1) for dy in range(-reach,reach+1)
        if (dx or dy) and max(abs(dx)-1,0)**2+max(abs(dy)-1,0)**2<=limit]
    return {'name':'original_c0_two_row_ideal_material_pair_reference','already_executed':False,
        'purpose':'split the already measured combined point-pair/material-projection remainder on the original pair, without changing the candidate',
        'target_ids':[63,64],'N':128,'frame':0,'D_exact':str(F(.02)),'h_exact':str(F(.04)),
        'source_snapshot':'runs/reconnect-contact-frozen/contact-frozen.npz',
        'source_snapshot_sha256':'9fe1d565ec4fabbe61bd1f4cf8d2dd8c0d730e6e1fc5ab56e2c185ac35f68a98',
        'extra_information_reference_only':['known original stationary square material-cell boundaries',
            'full original analytic periodic piecewise density and contact locations'],
        'not_new_required_candidate_inputs':True,
        'state':'full unchanged original fixed material mass averages c0; no auxiliary polynomial or point-state substitution',
        'ideal_aggregate_rate':'r_star_i=(2/m_i)*sum_j integral_Ki integral_Kj K_path(x,y)*B_h(y-x)*(cbar_j-cbar_i) dy dx',
        'interpretation':'P_rho L_h P_rho on original c0; geometric oracle reference, not FV flux or a production backend',
        'candidate_electrical_coefficients_or_states_not_replaced':True,
        'target_and_neighbor_regions':'original aligned unit-torus material squares with local periodic lifts; original mass remains denominator',
        'cell_support_rule':'minimum distance between the two squares <= 2h; not restricted to saved point-support neighbors',
        'neighbor_cell_offsets':[list(v) for v in offsets],
        'neighbor_cells_per_target':len(offsets),
        'two_predeclared_tensor_Gauss_orders':[8,16],
        'four_dimensional_coordinates':['target_a','target_b','neighbor_a','neighbor_b'],
        'fine_raw_rule_is_primary':16,'no_extrapolation_or_closer_to_PDE_rule_selection':True,
        'kernel_and_path':'unchanged positive_kernel_2d.slope_over_radius and oracle_path_reference.path_gamma; no refit or moment normalization',
        'D_zero':'zero integral, no infinite resistance',
        'planned_max_kernel_node_pairs_two_targets_two_rules':2*len(offsets)*(8**4+16**4),
        'integration_blocking':'target tensor nodes in bounded blocks, all support cells retained; no full graph or cell-pair matrix',
        'saved_comparators':['current exact point-candidate original c0 two-row action',
            'unchanged saved fine continuous oracle and its coarse/fine difference','unchanged saved PDE mass-average rate'],
        'signed_budget':'point-PDE = (point-ideal) + (ideal-continuous) + (continuous-PDE) + uncorrected arithmetic closure',
        'normalization':'same locked original all-cell PDE rate norm 0.2479286450687567 for every two-row contribution',
        'controls':['raw coarse/fine ideal response vector difference',
            'saved continuous-oracle coarse/fine response vector difference on the same rows',
            'raw Gauss weights/volumes and independent necessary implementation controls without renormalization',
            'old point arithmetic and PDE differences copied, no old graph or PDE recomputation'],
        'open_pair_inventory':'preserve, do not force the two-row mass rate to zero',
        'stop_rules':['invalid original input/geometry/reference scope or implementation control: preserve and stop',
            'two fixed rules do not resolve a claimed signed separation: retain inconclusive result, do not extend orders or choose a favorable result',
            'even resolved two-row terms are not global causal error percentages or candidate improvements',
            'no new h/grid/D/state/target/region/threshold tuning and no propagation'],
        'wall_clock_seconds_cap':600,'peak_RSS_MiB_cap':512,'resource_limits_not_physical_thresholds':True,
        'new_reference_only_fixed_configuration_max':1,'new_candidate_weights':0,'new_global_matrices':0,
        'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,'no_new_absolute_physical_acceptance_threshold':True,
        'fluid_orc_enabled':False,'finite_node_candidate_admitted':False}


def main(directory):
    out=Path(directory).resolve();budget=read(out/'residual-budget.json')
    root=budget['regions']['all'];assert root['combined_remainder_over_oracle_rule_indicator']>1
    assert root['main_component_correlation']>0
    ledger=[
        {'claim':'original observation matches','status':'verified byte-identical mass, IDs, c0, reference, point positions/rho and region arrays across saved objects'},
        {'claim':'density-path reconstruction error','status':'at original node-to-node paths in this aligned product background, prior exact validation gives the same path resistance; no general trace or density-identifiability theorem inferred'},
        {'claim':'continuous residual','status':'measured from saved oracle after material averaging; known interface curvature and finite-h effects remain, not all quantitatively attributed'},
        {'claim':'point-pair versus projection split','status':'only their combined original-c0 remainder is measured; reviewed ideal-row evidence used another quadratic state, so it cannot supply the missing original-c0 intermediate'},
        {'claim':'old eta=0.0286101232','status':'pure-phase ideal quadratic closure response only, not original cosine-state projection error or a coefficient correction'},
        {'claim':'old continuous directional anti-correlation','status':'belongs to another decomposition inside the continuous component; not transferred to finite-node direction responses'},
        {'claim':'weak consistency','status':'requires stated fixed smooth tests and material diameter/scale assumptions; one frozen norm is not a joint limit or evolution proof'},
        {'claim':'energy/mixing','status':'same cbar projected frozen functional only; neither this scalar nor improved rate norm certifies finite-time mixing'},
        {'claim':'route decision','status':'retain current weights/results and FV baseline; use only the original pair for one missing ideal material-pair reference, no point-backend repair or scan'},
    ]
    gate=next_gate()
    status={'updated':'2026-11-01','timezone':'Etc/UTC','saved_residual_review_completed':True,
        'new_saved_vector_signed_budgets':1,'new_software_checks':0,'recorded_checks_total':417,'previous_checks_rerun':0,
        'new_weights':0,'new_path_integrals':0,'new_kernel_or_operator_evaluations':0,'new_candidate_graphs':0,
        'new_global_matrices':0,'new_numerical_or_flow_cases':0,'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,
        'new_exact_algebra_certificate_groups':0,'finite_node_candidate_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'ordinary_central_resistance_FV_baseline_retained':True,'mass_average_resistance_remains_experimental':True,
        'next_ideal_original_c0_pair_reference_predeclared':gate,
        'ideal_reference_result':'not_computed','global_point_pair_vs_projection_split':'not_identified',
        'pdf_generated':False,'software_release_created':False,'published':False}
    save(out/'scope-ledger.json',ledger);save(out/'next-gate-spec.json',gate);save(out/'stage3-status.json',status)
    print({'review_complete':True,'next_gate_executed':False,'neighbor_cells_per_target':gate['neighbor_cells_per_target'],
        'future_node_pairs_two_rules':gate['planned_max_kernel_node_pairs_two_targets_two_rules'],
        'recorded_checks_total':417,'new_weights':0,'fluid_orc_enabled':False},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True)
    main(parser.parse_args().directory)
