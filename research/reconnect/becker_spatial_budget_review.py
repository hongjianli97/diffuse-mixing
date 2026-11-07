"""Read-only attribution of the saved spatial decision; no solver imports/calls."""
from pathlib import Path
from fractions import Fraction as F
import argparse, json, math, resource, shutil, signal, time
import numpy as np
from audit_route_decision import ROOT, read, save, digest
from becker_spatial_core import analyze as replay_parent

SOURCE = ROOT / 'runs/reconnect-becker-spatial'
PROTOCOL = ROOT / 'research/reconnect/becker_spatial_budget_review_protocol.json'


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def norm(m, v):
    return float(np.sqrt(m @ (v*v)))


def rational(value):
    return {'exact_binary64_arithmetic': str(value), 'float': float(value)}


def analyze(source):
    source = Path(source)
    r = read(source/'results.json')
    reproduced, _ = replay_parent(source)
    # Parent implementation is immutable. This is observation arithmetic only.
    from affine_shear_completion import plain_json
    assert plain_json(reproduced) == r
    sigma = F(r['common_initial_fluctuation_norm'])
    by_grid = {}
    for n in (32, 64, 128):
        tc = r['time_controls'][str(n)]
        data = arrays(source/f'n{n}/material.npz')
        paths = {k: arrays(source/f'n{n}/k{k}/trajectory-states.npz') for k in (4,8,16)}
        groups = {'machine': F(0), 'tail': F(0), 'independent_sequence_difference': F(0)}
        for k, path in paths.items():
            last = read(source/f'n{n}/k{k}/native-results.json')['steps'][-1]
            groups['machine'] += 2*F(last['cumulative_machine_indicator_absolute'])
            groups['tail'] += 2*F(last['cumulative_tail_indicator_absolute'])
            groups['independent_sequence_difference'] += 2*F(norm(data['mass'],path['states'][-1]-path['independent_states'][-1]))
        ref = r['reference_indicators'][str(n)]['components_absolute']
        groups['reference_space'] = 2*F(ref['space'])
        groups['reference_time'] = 2*F(ref['time'])
        groups['reference_initial_observation_geometry'] = 2*sum((F(v) for key,v in ref.items() if key not in ('space','time')), F(0))
        groups['transverse_observation'] = sum((F(v) for key,v in tc['components_absolute'].items() if 'bridge' in key),F(0))
        groups['saved_accumulation_roundoff_signed'] = F(tc['budget_absolute'])-sum(groups.values(),F(0))
        assert sum(groups.values(),F(0)) == F(tc['budget_absolute'])
        for key,value in groups.items():
            if key != 'saved_accumulation_roundoff_signed': assert value >= 0
        d48 = norm(data['mass'],paths[4]['states'][-1]-paths[8]['states'][-1])
        d816 = norm(data['mass'],paths[8]['states'][-1]-paths[16]['states'][-1])
        assert d48 == tc['state4_8'] and d816 == tc['state8_16']
        by_grid[str(n)] = {
            'B_absolute': tc['budget_absolute'],
            'expanded_B': {key:rational(value) for key,value in groups.items()},
            'state4_8': d48, 'state8_16': d816, 'observed_difference_ratio': d48/d816,
            'saved_time_control_passed': tc['time_control_resolved'],
            'H8_16': tc['H8_16'], 'original_H_indicator': tc['H_indicator'],
            'shifted_H_decrease_cannot_clear_retained_floor_even_if_next_increment_zero': tc['H8_16'] <= tc['H_indicator'],
            'shifted_H_inequality': '|H8-H16|-|H16-H32| <= |H8-H16|; retained B32>=B implies h(B32)>=h(B)',
            'new_H32_not_computed': True,
        }
    rows = []
    for pair in r['adjacent_spatial_comparisons']:
        c,n = pair['coarse_columns'],pair['fine_columns']
        for key,parts in pair['positive_components'].items():
            scale = sigma if key in ('native','common') else F(1)
            temporal = sum((F(v) for name,v in parts.items() if name.endswith('observed8_16')), F(0))/scale
            remaining = sum((F(v) for name,v in parts.items() if not name.endswith('observed8_16')), F(0))/scale
            gain, budget = F(pair['gains'][key]), F(pair['fixed_indicators'][key])
            decision = 'fine_better_resolvably' if gain>budget else ('fine_worse_resolvably' if gain < -budget else 'comparison_inconclusive')
            assert decision == pair['decisions'][key]
            row = {'coarse':c,'fine':n,'observable':key,'gain':float(gain),'original_budget':float(budget),'decision':decision,
                   'temporal':rational(temporal),'all_other_positive_terms':rational(remaining),
                   'outer_accumulation_roundoff_signed':rational(budget-temporal-remaining),
                   'temporal_fraction_of_original_budget':float(temporal/budget),
                   'gain_over_budget':float(gain/budget),'gain_positive':gain>0,
                   'temporal_alone_exceeds_gain':temporal>gain,
                   'all_other_terms_alone_below_gain':remaining<gain,
                   'diagnostic_removing_terms_is_not_acceptance':True}
            if key in ('native','common'):
                # Lipschitz continuity of the norm under an unknown NEW endpoint.
                # No state extrapolation or fitted time order is performed.
                room = (gain-remaining)/2
                per_grid = {str(q):F(parts[f'n{q}_observed8_16'])/sigma for q in (c,n)}
                row['future_sufficient_condition_only'] = {
                    'inequality':'2*d_new_pair/sigma32 + C_retained + C_additional < saved_gain',
                    'proof':'reverse triangle inequality: G_new >= G_saved-d_new_pair/sigma32; new budget=d_new_pair/sigma32+C_retained+C_additional',
                    'optimistic_pair_increment_ceiling_if_additional_terms_zero':float(room),
                    'current_pair_increment_over_ceiling':float(temporal/room),
                    'single_grid_update_cannot_satisfy_this_sufficient_condition_even_if_updated_increment_zero':all(value>=room for value in per_grid.values()),
                    'current_per_grid_relative_increments':{q:float(v) for q,v in per_grid.items()},
                    'not_necessary_condition_or_prediction_or_acceptance_threshold':True,
                }
                # Expand concentration budget by source, retaining floating accumulation bookkeeping.
                expanded = {}
                for q in (c,n):
                    factor = F(1) if key=='native' else F(next(x['norm_factor_upper'] for x in r['capacity_controls'] if x['columns']==q))
                    for label,value in by_grid[str(q)]['expanded_B'].items():
                        expanded[label] = expanded.get(label,F(0)) + factor*F(value['exact_binary64_arithmetic'])/sigma
                expanded['common_observation_capacity_bridges'] = sum((F(v) for name,v in parts.items()
                    if not name.endswith(('observed8_16','original_time_control_positive_budget','capacity_times_original_budget'))),F(0))/sigma
                expanded['remaining_multiplication_roundoff_signed'] = remaining-sum(expanded.values(),F(0))
                assert sum(expanded.values(),F(0))==remaining
                row['expanded_other_terms']={name:rational(value) for name,value in expanded.items()}
            else:
                row['H_non_temporal_polynomial_not_split_into_additive_causal_errors']=True
            rows.append(row)
    unresolved=[x for x in rows if x['decision']=='comparison_inconclusive']
    assert [(x['coarse'],x['fine'],x['observable']) for x in unresolved]==[(64,128,'native'),(64,128,'common')]
    assert all(x['temporal_alone_exceeds_gain'] and x['all_other_terms_alone_below_gain'] for x in unresolved)
    return {'date':'2026-11-07','timezone':'Asia/Shanghai',
        'decision':'saved_concentration_budget_temporal_dominated_paired_time_diagnostic_supported',
        'parent_decision_unchanged':r['comparison_decision'],'parent_spatial_control_passed':False,
        'parent_observation_arithmetic_reproduced':True,'comparisons':rows,'by_grid':by_grid,
        'interpretation':'budget attribution,not causal allocation of continuum error; no new state,extrapolation,order fit or replacement tolerance',
        'next_step_scope':'two K32 paths at saved64/128 grids only,diagnostic; shifted H decline cannot pass retained floor,so no automatic full spatial or Stage3 pass',
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_PDE_solves':0,'new_density_arrays':0,
        'new_software_checks':0,'recorded_checks_total':502,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}


def preserve():
    old=read(SOURCE/'preservation.json')
    formal=old['formal_manifests_verified']+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
        'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    assert len(formal)==108
    for item in formal:
        assert digest(ROOT/item['manifest'])==item['manifest_sha256']
        for rel,h in read(ROOT/item['manifest']).items(): assert digest(ROOT/rel)==h,rel
    originals=read('/workspace/original-handoff-sha256.json');attachments=read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals)==782 and len(attachments)==15
    for rel,h in originals.items(): assert digest(ROOT/rel)==h
    for item in attachments: assert digest(Path('/workspace')/item['path'])==item['sha256']
    return {'formal_manifests_verified':formal,'originals_verified':782,'attachments_verified':15,
        'historical_diagnostic_differences_retained':old['historical_diagnostic_differences_retained'],'all_parent_scientific_files_unchanged':True}


def main(output):
    started=time.monotonic()
    def timeout(*args): raise TimeoutError('registered120s read-only review cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    p=read(PROTOCOL);gate=read(SOURCE/'next-gate-spec.json')
    assert gate['type']=='read_only_saved_spatial_budget_obstruction_review'
    for item in p['sources']+gate['sources']: assert digest(ROOT/item['path'])==item['sha256'],item['path']
    shutil.copyfile(PROTOCOL,out/'protocol.json');shutil.copyfile(SOURCE/'next-gate-spec.json',out/'source-gate-spec.json')
    save(out/'implementation-lock.json',{'files':{rel:digest(ROOT/rel) for rel in p['implementation_files']},'no_new_simulations':True})
    copied=[]
    for rel in p['saved_data_files']:
        dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/rel,dest)
        copied.append({'source':str((SOURCE/rel).relative_to(ROOT)),'copy':str(dest.relative_to(out)),'sha256':digest(dest)})
    save(out/'source-copies.json',copied)
    result=analyze(out/'source');save(out/'results.json',result);save(out/'preservation.json',preserve())
    elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    assert elapsed<120 and peak<1024
    save(out/'execution.json',{'first_attempt_passed':True,'elapsed_seconds':elapsed,'peak_RSS_MiB':peak,
        'new_graphs_or_states':0,'published_or_pushed':False});signal.alarm(0)
    print(json.dumps({'decision':result['decision'],'unresolved':[x for x in result['comparisons'] if x['decision']=='comparison_inconclusive']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
