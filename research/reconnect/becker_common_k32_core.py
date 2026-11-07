"""Frozen common-K32 spatial diagnostic; ordinary resistance and old budgets retained."""
from pathlib import Path
from fractions import Fraction as F
import math
import numpy as np
from audit_route_decision import read
from becker_point_core import observe_columns
from becker_spatial_core import common_observe, exact_common
from aligned_becker_check import arrays, norm
from voronoi_time_action import metrics
from t1_uniform_core import h_indicator
from shear_union64_study import classify


def new_grid(out, N, old, base, reference):
    src = out / 'source/old'
    data = arrays(out / f'n{N}/material.npz')
    col = arrays(src / f'n{N}/column.npz')
    m, c0, mc = data['mass'], data['c0'], col['mass']
    mp, cp = base['mass'], base['c0']
    sigma = old['common_initial_fluctuation_norm']
    path = arrays(out / f'n{N}/k32/trajectory-states.npz')
    run = read(out / f'n{N}/k32/native-results.json')
    u, v = path['states'][-1], path['independent_states'][-1]
    u16 = arrays(src / f'n{N}/k16/trajectory-states.npz')['states'][-1]
    saved_common = arrays(src / 'common-observations.npz')
    pu = common_observe(u, m, mp)
    p16 = saved_common[f'n{N}_k16']
    np.testing.assert_array_equal(p16, common_observe(u16, m, mp))
    ref = np.tile(reference[f'n{N}_projected'], 8)
    pr = reference['n32_projected']
    last = run['steps'][-1]
    numeric = {'machine': last['cumulative_machine_indicator_absolute'],
               'tail': last['cumulative_tail_indicator_absolute'],
               'complete_sequence_difference': norm(m, u-v)}
    transverse = norm(mc, observe_columns(u, mc)-u.reshape(8, N)[0])
    Bparts = {f'retained_B{N}': old['time_controls'][str(N)]['budget_absolute'],
              'twice_new_path_numeric': 2*sum(numeric.values()), 'new_K32_transverse_bridge': transverse}
    B = sum(Bparts.values())
    p_exact, capacity = exact_common(u, m, mp)
    factor = math.nextafter(math.sqrt(float(max(capacity))), math.inf)
    defect = float(max(abs(q-1) for q in capacity))
    saved_capacity = next(row for row in old['capacity_controls'] if row['columns'] == N)
    assert factor == saved_capacity['norm_factor_upper'] and defect == saved_capacity['capacity_defect']
    oldparts = old['spatial_budget_by_grid'][str(N)]['common_parts']
    Jparts = {f'retained_{k}': value for k, value in oldparts.items()
              if k not in ('observed8_16', 'capacity_times_original_budget')}
    Jparts['new_K32_common_rounding'] = norm(mp, np.array([float(F(float(a))-b) for a,b in zip(pu,p_exact)]))
    Jparts['new_K32_capacity_final_bound'] = norm(mp, np.array([float(abs(q-1))*float(np.max(abs(u))) for q in capacity]))
    commonB = factor*B + sum(Jparts.values())
    d, dp = norm(m, u16-u), norm(mp, p16-pu)
    H, Hp = metrics(m,u,c0)['H'], metrics(mp,pu,cp)['H']
    obs = old['observations'][str(N)]
    hd, hpd = abs(H-obs['native_H']), abs(Hp-obs['common_H'])
    hb, hpb = h_indicator(m,c0,B), h_indicator(mp,cp,commonB)
    parts = {'native': {'observed16_32': d, 'positive_norm_budget': B},
             'common': {'observed16_32': dp, 'capacity_times_B': factor*B, **Jparts},
             'native_H': {'observed16_32': hd, 'original_H_indicator': hb},
             'common_H': {'observed16_32': hpd, 'original_H_indicator': hpb}}
    errors = {'native': norm(m,u-ref)/sigma, 'common': norm(mp,pu-pr)/sigma,
              'native_H': abs(H-obs['native_reference_H']), 'common_H': abs(Hp-obs['common_reference_H'])}
    tc = old['time_controls'][str(N)]
    state_decrease, H_decrease = tc['state8_16']-d, tc['H8_16']-hd
    sequence = all(row['primary_control_mass_norm_difference'] <=
                   row['cumulative_machine_indicator_absolute']+row['cumulative_tail_indicator_absolute'] for row in run['steps'])
    raw = all(abs(row['raw_mass_change_from_c0'])/float(m@c0) <= 1e-10
              and row['minimum'] >= float(c0.min())-1e-10 and row['maximum'] <= float(c0.max())+1e-10
              and min(row['chi_start_half_end']) >= 0 for row in run['steps'])
    V0 = metrics(m,c0,c0)['variance']
    variances = [metrics(m,x,c0)['variance'] for x in path['states']]
    raw = bool(raw and np.isfinite(path['states']).all() and np.isfinite(path['independent_states']).all()
               and all(b <= a+1e-10*V0 for a,b in zip(variances[:-1],variances[1:])))
    return {'columns': N, 'K': 32, 'newly_executed': True, 'errors': errors,
        'numeric_K32_parts': numeric, 'B_star_parts': Bparts, 'B_star': B,
        'J_star_parts': Jparts, 'common_norm_budget': commonB, 'spatial_parts': parts,
        'capacity': {'defect': defect, 'norm_factor_upper': factor, 'original_guard_passed': defect <= 1e-10},
        'state_time_control': {'old8_16': tc['state8_16'], 'new16_32': d, 'decrease': state_decrease,
                               'fixed_budget': B, 'resolved': state_decrease > B},
        'shifted_H_time_control': {'old8_16': tc['H8_16'], 'new16_32': hd, 'decrease': H_decrease,
            'retained_original_H_floor': tc['H_indicator'], 'current_original_H_indicator': hb,
            'resolved': H_decrease > hb, 'old_increment_below_retained_floor': tc['H8_16'] <= tc['H_indicator'],
            'no_small_H_exemption': True},
        'native_H_K32': H, 'common_H_K32': Hp, 'raw_all_steps_guards_passed': raw,
        'all_complete_sequence_controls_passed': sequence, 'native_K32_mass_change': float(m@(u-c0)),
        'min_all_steps': min(row['minimum'] for row in run['steps']),
        'max_all_steps': max(row['maximum'] for row in run['steps'])}, pu


def reused_grid(fine):
    parts = fine['positive_components']
    row = {k: fine[k] for k in ('errors', 'numeric_K32_parts', 'common_norm_budget', 'capacity',
        'state_time_control', 'shifted_H_time_control', 'native_H_K32', 'common_H_K32',
        'raw_all_steps_guards_passed', 'all_complete_sequence_controls_passed',
        'native_K32_mass_change', 'min_all_steps', 'max_all_steps')}
    row.update(columns=128, K=32, newly_executed=False, B_star=fine['B128_star'],
        B_star_parts=fine['B128_star_parts'], J_star_parts=fine['J128_star_parts'],
        spatial_parts={
            'native': {'observed16_32': parts['native']['new128_observed16_32'], 'positive_norm_budget': fine['B128_star']},
            'common': {'observed16_32': parts['common']['new128_observed16_32'],
                       'capacity_times_B': parts['common']['new128_capacity_B'], **fine['J128_star_parts']},
            'native_H': {'observed16_32': parts['native_H']['new128_observed16_32'],
                         'original_H_indicator': parts['native_H']['new128_original_H_indicator']},
            'common_H': {'observed16_32': parts['common_H']['new128_observed16_32'],
                         'original_H_indicator': parts['common_H']['new128_original_H_indicator']}})
    return row


def analyze(out):
    out = Path(out)
    old = read(out / 'source/old/results.json')
    fine = read(out / 'source/fine/results.json')
    base = arrays(out / 'source/old/n32/column.npz')
    ref = arrays(out / 'source/old/reference/observations-l1-k16384.npz')
    rows, common = {}, {}
    for N in (32,64):
        rows[str(N)], common[f'n{N}_k32'] = new_grid(out,N,old,base,ref)
    rows['128'] = reused_grid(fine)
    common['n128_k32'] = arrays(out/'source/fine/common-observations.npz')['n128_k32']
    adjacent = []
    sigma = old['common_initial_fluctuation_norm']
    for C,N in ((32,64),(64,128)):
        a,b = rows[str(C)],rows[str(N)]
        gains = {k: a['errors'][k]-b['errors'][k] for k in a['errors']}
        parts = {k: {f'n{Q}_{name}': v for Q in (C,N) for name,v in rows[str(Q)]['spatial_parts'][k].items()}
                 for k in gains}
        indicators = {k: sum(p.values())/(sigma if k in ('native','common') else 1.) for k,p in parts.items()}
        adjacent.append({'coarse_columns': C, 'fine_columns': N, 'K': 32, 'time': .1,
            'gains': gains, 'fixed_indicators': indicators, 'positive_components': parts,
            'decisions': {k: classify(gains[k],indicators[k]) for k in gains}})
    spatial = all(v == 'fine_better_resolvably' for r in adjacent for v in r['decisions'].values())
    states = all(r['state_time_control']['resolved'] for r in rows.values())
    guards = all(r['raw_all_steps_guards_passed'] and r['all_complete_sequence_controls_passed']
                 and r['capacity']['original_guard_passed'] for r in rows.values())
    passed = spatial and states and guards
    return {'date': '2026-11-07', 'timezone': 'Etc/UTC',
        'comparison_decision': 'common_K32_three_column_spatial_diagnostic_resolved' if passed else 'common_K32_spatial_diagnostic_failed_or_unresolved',
        'common_K32_spatial_diagnostic_resolved': passed, 'common_initial_fluctuation_norm': sigma,
        'by_grid': rows, 'adjacent_spatial_comparisons': adjacent,
        'all_eight_spatial_comparisons_resolved': spatial, 'all_state_time_controls_resolved': states,
        'all_raw_sequence_capacity_guards_passed': guards,
        'shifted_H_time_controls_resolved': {n:r['shifted_H_time_control']['resolved'] for n,r in rows.items()},
        'old_spatial_decision_unchanged': old['comparison_decision'],
        'old_K4_8_16_time_controls_preserved': old['all_time_controls_resolved'],
        'new_candidate_graphs': 64, 'new_candidate_transient_runs': 2,
        'new_complete_independent_uniformization_sequences': 2, 'old_candidate_reruns': 0,
        'new_reference_PDE_solves': 0, 'new_density_sources': 0, 'new_software_checks': 0, 'recorded_checks_total': 502,
        'one_dimensional_column_refinement_only': True, 'general_order_claimed': False,
        'indicators_are_certified_error_bounds': False, 'registered_H_check_waived': False,
        'new_full_spatial_or_Stage3_pass_allowed': False, 'all_three_density_sources_closed': False,
        'stage3_physical_acceptance': 'incomplete', 'fluid_orc_enabled': False, 'kappa_computed': False}, common
