"""Read-only Stage3 evidence ledger; stdlib only, no solver or array loading."""
from pathlib import Path
import argparse
import csv
import json
import resource
import shutil
import signal
import time

from audit_route_decision import ROOT, read, save, digest

OUT = ROOT / 'runs/reconnect-stage3-evidence'
PARENT = ROOT / 'runs/reconnect-becker-fine-time'
PROTOCOL = ROOT / 'research/reconnect/stage3_evidence_protocol.json'


def preserve():
    old = read(PARENT / 'preservation.json')
    formal = old['formal_manifests_verified'] + [{
        'manifest': str((PARENT / 'sha256.json').relative_to(ROOT)),
        'manifest_sha256': digest(PARENT / 'sha256.json'),
        'entries_verified': len(read(PARENT / 'sha256.json'))}]
    assert len(formal) == 110
    for item in formal:
        path = ROOT / item['manifest']
        assert digest(path) == item['manifest_sha256']
        entries = read(path)
        assert len(entries) == item['entries_verified']
        for name, expected in entries.items():
            assert digest(ROOT / name) == expected, name
    originals = read('/workspace/original-handoff-sha256.json')
    attachments = read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals) == 782 and len(attachments) == 15
    for name, expected in originals.items():
        assert digest(ROOT / name) == expected, name
    for item in attachments:
        path = Path('/workspace') / item['path']
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
    historical = old['historical_diagnostic_differences_retained']
    assert len(historical) == 4
    for item in historical:
        path = ROOT / item['manifest']
        assert digest(path) == item['manifest_sha256']
        entries = read(path)
        assert len(entries) == item['entries']
        differences = [{'path': p, 'saved': h, 'current': digest(ROOT / p)}
                       for p, h in entries.items() if digest(ROOT / p) != h]
        assert sorted(differences, key=lambda x: x['path']) == sorted(
            item['historical_paths_differing_from_later_formal_versions'], key=lambda x: x['path'])
    return {'formal_manifests_verified': formal, 'originals_verified': 782,
            'attachments_verified': 15, 'historical_diagnostic_differences_retained': historical,
            'all_prior_scientific_files_unchanged': True}


def main():
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('120s review cap')))
    signal.alarm(120)
    assert not OUT.exists(), 'append-only review requires fresh directory'
    protocol = read(PROTOCOL)
    seal = read(PARENT / 'sha256.json')
    assert digest(PARENT / 'sha256.json') == protocol['parent_seal_sha256']
    OUT.mkdir()
    save(OUT / 'implementation-lock.json', {p: digest(ROOT / p) for p in [
        'research/reconnect/stage3_evidence_review.py',
        'research/reconnect/audit_stage3_evidence.py',
        'research/reconnect/stage3_evidence_protocol.json']})
    for p, h in protocol['sources'].items():
        assert seal[p] == h and digest(ROOT / p) == h, p
        destination = OUT / 'source' / p
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / p, destination)
    for q in protocol['requirements']:
        assert (ROOT / q['path']).read_text().splitlines()[q['line'] - 1] == q['text']
    shared = {}
    case_rows = []
    for item in protocol['cases']:
        folder = 'runs/reconnect-' + item['id']
        result = read(ROOT / folder / 'results.json')
        assert result[item['flag']] is item['expected']
        assert result['stage3_physical_acceptance'] == 'incomplete' and not result['fluid_orc_enabled']
        lock = read(ROOT / folder / 'implementation-lock.json')['files']
        found = {}
        for name, h in lock.items():
            path = name if '/' in name else 'research/reconnect/' + name
            assert digest(ROOT / path) == h, (item['id'], path)
            if Path(path).name in protocol['shared_core']:
                found[Path(path).name] = h
        # K32 adds explicit-duration and bounded geometry wrappers; not an identical full API.
        for name, h in found.items():
            assert h == protocol['shared_core'][name]
        assert {'periodic_voronoi_fv.py', 'voronoi_time_action.py'} <= found.keys()
        shared[item['id']] = found
        case_rows.append({**item, 'source': folder + '/results.json',
                          'source_sha256': digest(ROOT / folder / 'results.json'),
                          'saved_decision': result['comparison_decision']})
    old = read(ROOT / 'runs/reconnect-becker-spatial/results.json')
    fine = read(PARENT / 'results.json')
    old_temporal = [{ 'columns': int(n), **{k: row[k] for k in (
        'state4_8', 'state8_16', 'state_decrease', 'budget_absolute',
        'H4_8', 'H8_16', 'H_decrease', 'H_indicator', 'time_control_resolved')}}
        for n, row in old['time_controls'].items()]
    comparisons = []
    for row in old['adjacent_spatial_comparisons']:
        for observable, decision in row['decisions'].items():
            comparisons.append({'kind': 'original_common_K16',
                'coarse_columns': row['coarse_columns'], 'fine_columns': row['fine_columns'],
                'observable': observable, 'gain': row['gains'][observable],
                'indicator': row['fixed_indicators'][observable], 'decision': decision})
    for observable, decision in fine['decisions'].items():
        comparisons.append({'kind': 'diagnostic_mixed_K16_K32', 'coarse_columns': 64,
            'fine_columns': 128, 'observable': observable, 'gain': fine['gains'][observable],
            'indicator': fine['fixed_indicators'][observable], 'decision': decision})
    h = fine['shifted_H_time_control']
    previous_h_error = abs(old['observations']['128']['native_H'] - old['observations']['128']['native_reference_H'])
    ledger = {'requirements': protocol['requirements'], 'case_rows': case_rows,
        'coverage': protocol['coverage'], 'shared_core_hashes_by_case': shared,
        'old_K4_8_16_time_qualifications': old_temporal,
        'spatial_comparisons': comparisons, 'new_128_state_time': fine['state_time_control'],
        'shifted_H_time': h,
        'same128_native_H_reference_errors': {'K16': previous_h_error, 'K32': fine['errors']['native_H']},
        'original_spatial_passed': False, 'mixed_time_diagnostic_resolved': True,
        'shifted_H_resolved': False, 'indefinite_shifted_triples_required_by_original_text': False,
        'registered_H_check_waived': False, 'new_acceptance_criteria_added': False,
        'general_spatial_order_claimed': False, 'physical_acceptance_percentage_computed': False,
        'new_graphs': 0, 'new_candidate_paths': 0, 'new_reference_PDE_solves': 0,
        'new_density_sources': 0, 'new_software_checks': 0, 'recorded_checks_total': 502,
        'stage3_physical_acceptance': 'incomplete', 'fluid_orc_enabled': False, 'kappa_computed': False}
    save(OUT / 'evidence-ledger.json', ledger)
    with (OUT / 'coverage.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'requirements', 'status', 'evidence', 'scope', 'gap'])
        writer.writeheader()
        writer.writerows(protocol['coverage'])
    save(OUT / 'preservation.json', preserve())
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    assert rss < 1024
    save(OUT / 'execution.json', {'elapsed_seconds': time.monotonic() - started,
        'peak_RSS_MiB': rss, 'source_files': len(protocol['sources']),
        'stdlib_only': True, 'arrays_or_solvers_loaded': False, 'completed': True})
    signal.alarm(0)
    print('Stage3 read-only evidence review complete; original decisions and H obstruction retained.')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
