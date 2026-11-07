"""Portable independent audit of saved evidence/quotes/Decimal comparisons only."""
from pathlib import Path
from decimal import Decimal, localcontext
import argparse
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(directory, write=False):
    started = time.monotonic()
    out = Path(directory).resolve()
    protocol = read(ROOT / 'research/reconnect/stage3_evidence_protocol.json')
    ledger = read(out / 'evidence-ledger.json')
    seal = read(ROOT / 'runs/reconnect-becker-fine-time/sha256.json')
    assert digest(ROOT / 'runs/reconnect-becker-fine-time/sha256.json') == protocol['parent_seal_sha256']
    for path, h in protocol['sources'].items():
        assert seal[path] == h == digest(ROOT / path) == digest(out / 'source' / path), path
    for path, h in read(out / 'implementation-lock.json').items():
        assert digest(ROOT / path) == h, path
    assert ledger['requirements'] == protocol['requirements']
    assert ledger['coverage'] == protocol['coverage']
    for q in ledger['requirements']:
        assert protocol['sources'][q['path']] == q['sha256']
        assert (out / 'source' / q['path']).read_text().splitlines()[q['line'] - 1] == q['text']
    assert any('分别细化 h、Δt、物理层厚度δ' in q['text'] for q in ledger['requirements'])
    assert any('避免同时变化无法归因' in q['text'] for q in ledger['requirements'])
    for row, expected in zip(ledger['case_rows'], protocol['cases'], strict=True):
        result = read(out / 'source' / row['source'])
        assert row['id'] == expected['id'] and row['expected'] is expected['expected']
        assert result[row['flag']] is expected['expected']
        assert row['saved_decision'] == result['comparison_decision']
        lock = read(out / 'source' / row['source'].replace('results.json', 'implementation-lock.json'))
        shared = {}
        for name, h in lock['files'].items():
            path = name if '/' in name else 'research/reconnect/' + name
            assert digest(ROOT / path) == h, path
            if Path(path).name in protocol['shared_core']:
                shared[Path(path).name] = h
                assert protocol['shared_core'][Path(path).name] == h
        assert shared == ledger['shared_core_hashes_by_case'][row['id']]
    old = read(out / 'source/runs/reconnect-becker-spatial/results.json')
    fine = read(out / 'source/runs/reconnect-becker-fine-time/results.json')
    assert not old['registered_spatial_control_passed'] and old['all_time_controls_resolved']
    assert fine['concentration_diagnostic_resolved'] and not fine['new_full_spatial_or_Stage3_pass_allowed']
    assert ledger['shifted_H_time'] == fine['shifted_H_time_control']
    decimal_runs = []
    for digits in (80, 100):
        with localcontext() as ctx:
            ctx.prec = digits
            D = lambda x: Decimal.from_float(float(x))
            # Only rounding comparison of saved binary scalars, never a physical tolerance.
            def near(a, b):
                assert abs(a - D(b)) <= Decimal('1e-14') * max(abs(a), abs(D(b)), Decimal('1e-40'))
            for row in ledger['old_K4_8_16_time_qualifications']:
                source = old['time_controls'][str(row['columns'])]
                for key in row:
                    if key != 'columns':
                        assert row[key] == source[key]
                assert D(source['state4_8']) - D(source['state8_16']) > D(source['budget_absolute'])
                assert D(source['H4_8']) - D(source['H8_16']) > D(source['H_indicator'])
            decisions = []
            for row in ledger['spatial_comparisons']:
                source = (next(x for x in old['adjacent_spatial_comparisons']
                               if x['coarse_columns'] == row['coarse_columns'])
                          if row['kind'] == 'original_common_K16' else fine)
                k = row['observable']
                assert row['gain'] == source['gains'][k] and row['indicator'] == source['fixed_indicators'][k]
                parts = source['positive_components'][k]
                assert all(x >= 0 for x in parts.values())
                budget = sum(map(D, parts.values()), Decimal(0))
                if k in ('native', 'common'):
                    budget /= D(old['common_initial_fluctuation_norm'])
                near(budget, row['indicator'])
                gain = D(row['gain'])
                decision = ('fine_better_resolvably' if gain > budget else
                            'fine_worse_resolvably' if gain < -budget else 'comparison_inconclusive')
                assert decision == row['decision'] == source['decisions'][k]
                decisions.append(decision)
            assert decisions[:8].count('fine_better_resolvably') == 6
            assert decisions[:8].count('comparison_inconclusive') == 2
            assert decisions[8:] == ['fine_better_resolvably'] * 4
            h = fine['shifted_H_time_control']
            change = D(h['old8_16']) - D(h['new16_32'])
            near(change, h['decrease'])
            assert D(h['new16_32']) >= 0
            assert change <= D(h['old8_16']) < D(h['retained_original_H_floor']) <= D(h['current_original_H_indicator'])
            assert not h['resolved']
            state = fine['state_time_control']
            assert D(state['old8_16']) - D(state['new16_32']) > D(state['fixed_budget'])
            before = abs(D(old['observations']['128']['native_H']) - D(old['observations']['128']['native_reference_H']))
            after = D(fine['errors']['native_H'])
            assert after > before
            decimal_runs.append({'digits': digits, 'original_spatial_resolved': 6,
                'original_spatial_inconclusive': 2, 'mixed_spatial_resolved': 4,
                'old_three_temporal_qualifications_passed': True,
                'shifted_H_ceiling_below_retained_floor': True,
                'same128_H_reference_error_change': str(after - before)})
    for field in ('new_graphs', 'new_candidate_paths', 'new_reference_PDE_solves', 'new_density_sources', 'new_software_checks'):
        assert ledger[field] == 0
    assert ledger['recorded_checks_total'] == 502 and ledger['stage3_physical_acceptance'] == 'incomplete'
    for field in ('fluid_orc_enabled', 'kappa_computed', 'original_spatial_passed', 'shifted_H_resolved',
                  'registered_H_check_waived', 'new_acceptance_criteria_added', 'physical_acceptance_percentage_computed'):
        assert ledger[field] is False
    assert len(ledger['case_rows']) == 10 and len(ledger['spatial_comparisons']) == 12
    result = {'passed': True, 'sources_verified': len(protocol['sources']), 'case_rows': 10,
        'exact_requirement_quotes': len(protocol['requirements']), 'decimal_scalar_checks': decimal_runs,
        'new_states_graphs_or_PDE_solutions': 0,
        'scope': 'source hashes, stored decisions, exact quotes and scalar budget arithmetic; no numerical re-solve',
        'full_historical_preservation_requires_original_workspace': True}
    if write:
        with (out / 'audit.json').open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write('\n')
        with (out / 'audit-execution.json').open('x') as f:
            json.dump({'elapsed_seconds': time.monotonic() - started}, f)
            f.write('\n')
    else:
        assert read(out / 'audit.json') == result
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default=ROOT / 'runs/reconnect-stage3-evidence')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    main(args.directory, args.write)
