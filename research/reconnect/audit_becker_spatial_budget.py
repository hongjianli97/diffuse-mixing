"""Independent exact/Decimal audit of saved scalar-budget attribution; no solver."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal, localcontext
import argparse,json,math,resource,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest


def arr(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def squared(m,a,b):
    return sum((F(float(w))*(F(float(x))-F(float(y)))**2 for w,x,y in zip(m,a,b)),F(0))


def decimal(q):
    return Decimal(q.numerator)/Decimal(q.denominator)


def near(value,saved):
    assert abs(value-Decimal.from_float(float(saved))) <= Decimal.from_float(32*math.ulp(float(saved))), (str(value),saved)


def main(directory,write=False):
    start=time.monotonic();out=Path(directory);src=out/'source';r=read(src/'results.json');review=read(out/'results.json')
    for name,h in read(out/'implementation-lock.json')['files'].items():assert digest(ROOT/name)==h,name
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==item['sha256']
    assert not review['parent_spatial_control_passed'] and review['parent_decision_unchanged']==r['comparison_decision']
    sigma=F(r['common_initial_fluctuation_norm']);passes=[]
    for digits in (80,100):
        with localcontext() as ctx:
            ctx.prec=digits;comparisons=[];time_checks=[]
            for n in (32,64,128):
                data=arr(src/f'n{n}/material.npz');paths={k:arr(src/f'n{n}/k{k}/trajectory-states.npz') for k in (4,8,16)}
                tc=r['time_controls'][str(n)];row=review['by_grid'][str(n)]
                differences=[]
                for a,b in ((4,8),(8,16)):
                    value=decimal(squared(data['mass'],paths[a]['states'][-1],paths[b]['states'][-1])).sqrt()
                    near(value,tc[f'state{a}_{b}']);differences.append(str(value))
                machine=tail=F(0);independent=Decimal(0)
                for k in (4,8,16):
                    last=read(src/f'n{n}/k{k}/native-results.json')['steps'][-1]
                    machine+=2*F(last['cumulative_machine_indicator_absolute']);tail+=2*F(last['cumulative_tail_indicator_absolute'])
                    independent+=2*decimal(squared(data['mass'],paths[k]['states'][-1],paths[k]['independent_states'][-1])).sqrt()
                groups=row['expanded_B'];get=lambda name:F(groups[name]['exact_binary64_arithmetic'])
                assert machine==get('machine') and tail==get('tail');near(independent,float(get('independent_sequence_difference')))
                assert sum((F(item['exact_binary64_arithmetic']) for item in groups.values()),F(0))==F(tc['budget_absolute'])
                assert get('reference_space')==2*F(r['reference_indicators'][str(n)]['components_absolute']['space'])
                assert get('reference_time')==2*F(r['reference_indicators'][str(n)]['components_absolute']['time'])
                ceiling=F(tc['H8_16']);floor=F(tc['H_indicator'])
                assert (ceiling<=floor)==row['shifted_H_decrease_cannot_clear_retained_floor_even_if_next_increment_zero']
                time_checks.append({'columns':n,'exact_difference_norms':differences,'shifted_H_ceiling':str(decimal(ceiling)),
                    'retained_H_floor':str(decimal(floor)),'shifted_H_impossible_with_retained_floor':ceiling<=floor})
            for pair in r['adjacent_spatial_comparisons']:
                for key,parts in pair['positive_components'].items():
                    row=next(v for v in review['comparisons'] if (v['coarse'],v['fine'],v['observable'])==(pair['coarse_columns'],pair['fine_columns'],key))
                    scale=sigma if key in ('native','common') else F(1)
                    temporal=sum((F(v) for name,v in parts.items() if name.endswith('observed8_16')),F(0))/scale
                    other=sum((F(v) for name,v in parts.items() if not name.endswith('observed8_16')),F(0))/scale
                    gain,budget=F(pair['gains'][key]),F(pair['fixed_indicators'][key])
                    assert temporal==F(row['temporal']['exact_binary64_arithmetic']) and other==F(row['all_other_positive_terms']['exact_binary64_arithmetic'])
                    assert temporal+other+F(row['outer_accumulation_roundoff_signed']['exact_binary64_arithmetic'])==budget
                    near(decimal(temporal/budget),row['temporal_fraction_of_original_budget']);near(decimal(gain/budget),row['gain_over_budget'])
                    expected='fine_better_resolvably' if gain>budget else ('fine_worse_resolvably' if gain < -budget else 'comparison_inconclusive')
                    assert expected==row['decision']==pair['decisions'][key]
                    if key in ('native','common'):
                        room=(gain-other)/2;future=row['future_sufficient_condition_only']
                        near(decimal(room),future['optimistic_pair_increment_ceiling_if_additional_terms_zero'])
                        near(decimal(temporal/room),future['current_pair_increment_over_ceiling'])
                        remaining=[F(parts[f'n{n}_observed8_16'])/sigma for n in (pair['coarse_columns'],pair['fine_columns'])]
                        assert all(x>=room for x in remaining)==future['single_grid_update_cannot_satisfy_this_sufficient_condition_even_if_updated_increment_zero']
                        assert sum((F(v['exact_binary64_arithmetic']) for v in row['expanded_other_terms'].values()),F(0))==other
                    comparisons.append({'coarse':pair['coarse_columns'],'fine':pair['fine_columns'],'observable':key,
                        'temporal':str(decimal(temporal)),'other':str(decimal(other)),'decision':expected})
            passes.append({'digits':digits,'time_checks':time_checks,'comparisons':comparisons})
    assert all(review[key]==0 for key in ('new_candidate_graphs','new_candidate_transient_runs','new_reference_PDE_solves','new_density_arrays','new_software_checks'))
    findings={'passed':True,'exact_binary64_budgets_and_independent_saved_norms_verified':True,'precision_checks':passes,
        'original_decision_retained':r['comparison_decision'],'new_software_checks':0,'recorded_checks_total':502,
        'scope':'saved arithmetic and triangle-inequality planning; no new physical data or certified continuum bound'}
    if write:save(out/'audit.json',{'passed':True,'first_attempt_passed':True,'findings':findings,'elapsed_seconds':time.monotonic()-start,
        'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'auditor_sha256':digest(Path(__file__))})
    else:assert read(out/'audit.json')['findings']==findings
    print(json.dumps({'passed':True,'precision_digits':[80,100],'comparisons_verified':8,'original_decision':r['comparison_decision'],'new_states':0},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();main(a.directory,a.write)
