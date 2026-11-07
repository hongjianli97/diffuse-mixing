"""Append a correction to the single-grid planning inference; preserve all inputs."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json
from audit_route_decision import ROOT,read,save,digest


def derive(review):
    rows=[]
    for row in review['comparisons']:
        if row['decision']!='comparison_inconclusive':continue
        g=F(row['gain']);c=F(row['all_other_positive_terms']['exact_binary64_arithmetic'])
        # Obtain the original increments directly, not rounded planning summaries.
        parent=read(ROOT/'runs/reconnect-becker-spatial/results.json')
        pair=parent['adjacent_spatial_comparisons'][1]
        sigma=F(parent['common_initial_fluctuation_norm'])
        increments={n:F(pair['positive_components'][row['observable']][f'n{n}_observed8_16'])/sigma for n in (64,128)}
        choices=[]
        for active in ((64,),(128,),(64,128)):
            inactive=sum((value for n,value in increments.items() if n not in active),F(0))
            ceiling=(g-c-inactive)/2
            current=sum((value for n,value in increments.items() if n in active),F(0))
            choices.append({'new_K32_columns':list(active),'retained_inactive_time_indicator':float(inactive),
                'retained_other_budget':float(c),'sufficient_increment_sum_ceiling_if_new_numeric_bridge_zero':float(ceiling),
                'ceiling_exact':str(ceiling),'current_active_increment_sum':float(current),
                'observed_old_increment_over_ceiling':float(current/ceiling),'positive_room':ceiling>0,
                'sufficient_not_necessary_and_not_a_prediction':True})
        rows.append({'observable':row['observable'],'saved_gain':float(g),'choices':choices})
    # Exact scalar witness refutes the earlier necessity inference. These numbers
    # are a mathematical counterexample, not flow data or a software-test count.
    witness={'coarse_error_old':10,'fine_error_old':2,'fine_error_new':3,
             'coarse_old_time_indicator':5,'fine_old_time_indicator':5,'other':0}
    g=F(10)-F(2);new_gain=F(10)-F(3);new_indicator=F(5)+F(1)
    assert F(5)>g/2 and new_gain>new_indicator
    return {'date':'2026-11-07','timezone':'Asia/Shanghai','decision':'time_budget_dominance_confirmed_single_finest_time_probe_supported',
        'prior_review_results_and_first_audit_preserved':True,
        'correction':'The both-updated sufficient condition is valid only when both temporal indicators are replaced. Applying its half-gain ceiling to an inactive old indicator falsely inferred that both grids must be updated.',
        'prior_audit_scope_limit':'First audit verified exact arithmetic and the original spatial decision,not logical validity of the single-grid necessity inference.',
        'invalid_field':'comparisons[*].future_sufficient_condition_only.single_grid_update_cannot_satisfy_this_sufficient_condition_even_if_updated_increment_zero',
        'prior_next_step_scope_superseded':True,
        'corrected_inequality':'2*sum(active endpoint changes)/sigma32 + sum(inactive old time indicators)/sigma32 + C_retained + C_added < G_saved',
        'proof':'Each active reference-error norm changes by at most its endpoint-change norm (reverse triangle inequality). Inactive endpoints do not change the gain. Their old temporal indicators remain once in the new budget. Active increments enter once in the gain bound and once in the budget. No state is extrapolated and no acceptance threshold is replaced.',
        'exact_scalar_counterexample':{**witness,'old_gain':float(g),'new_gain':float(new_gain),'new_indicator':float(new_indicator),'earlier_both_required_inference_refuted':True},
        'corrected_planning':rows,
        'selected_next_probe':{'columns':128,'new_K':32,'rows':8,'new_graphs':32,'new_primary_paths':1,'new_independent_paths':1,'new_reference_PDE_solves':0,
            'reason':'One new path probes the finest-grid time increment and both native/common comparison bottlenecks;64 remains a saved control. Positive sufficient room exists. No estimated order guarantees resolution.',
            'spatial_pair_time_levels':[16,32],'new_full_spatial_or_Stage3_pass_allowed':False},
        'H_shifted_control_obstruction_preserved':True,'source_spatial_decision_unchanged':review['parent_decision_unchanged'],
        'new_graphs_or_states_this_turn':0,'new_software_checks':0,'recorded_checks_total':502,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}


def main(directory):
    out=Path(directory)
    before={name:digest(out/name) for name in ('results.json','audit.json','execution.json')}
    result=derive(read(out/'results.json'))
    save(out/'planning-correction.json',result)
    assert all(digest(out/name)==h for name,h in before.items())
    save(out/'planning-correction-preservation.json',{'prior_files':before,'prior_code_sha256':digest(ROOT/'research/reconnect/becker_spatial_budget_review.py'),
        'completion_code_sha256':digest(Path(__file__)),'prior_results_unchanged':True})
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
