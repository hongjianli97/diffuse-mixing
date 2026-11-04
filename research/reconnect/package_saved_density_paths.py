"""Seal the saved-path check and verify a standalone archived replay."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

from audit_route_decision import ROOT, digest, read, save


def main():
    source = ROOT / 'runs/reconnect-density-path'
    out = ROOT / 'runs/reconnect-saved-density-paths'
    exports = ROOT / 'exports'
    archive = exports / 'ltn-saved-density-paths-review.zip'
    delivery = exports / 'ltn-saved-density-paths-delivery.json'
    seal = out / 'sha256.json'
    assert not any(p.exists() for p in [archive, delivery, seal]), 'never overwrite sealed evidence'
    execution = read(out / 'execution.json')
    assert execution['tests_exit_code'] == execution['study_exit_code'] == execution['audit_exit_code'] == 0
    assert read(out / 'implementation-audit.json')['six_new_localization_unit_checks_passed']
    prior = read(source / 'sha256.json')
    for rel, expected in prior.items():
        assert digest(ROOT / rel) == expected, rel
    new_files = [
        'research/reconnect/SAVED_DENSITY_PATH_CHECK.md',
        'research/reconnect/endpoint_density_path.py',
        'research/reconnect/test_endpoint_density_path.py',
        'research/reconnect/saved_density_paths_study.py',
        'research/reconnect/audit_saved_density_paths.py',
        'research/reconnect/package_saved_density_paths.py',
    ]
    source_seal = str((source / 'sha256.json').relative_to(ROOT))
    entries = dict(prior)
    entries[source_seal] = digest(source / 'sha256.json')
    for rel in new_files:
        entries[rel] = digest(ROOT / rel)
    for p in out.iterdir():
        assert p.is_file(), p
        entries[str(p.relative_to(ROOT))] = digest(p)
    save(seal, dict(sorted(entries.items())))
    original_snapshot = read(out / 'config.json')['protocol']['source_snapshot']
    selected = set(new_files)
    selected.update(item['path'] for item in read(out / 'evidence-index.json')['sources']
                    if item['path'] != original_snapshot)
    selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([
        source_seal, 'research/reconnect/audit_route_decision.py',
        'research/reconnect/periodic_density_path.py', 'research/reconnect/test_periodic_density_path.py',
        'research/reconnect/density_path_study.py', 'docs/RESEARCH_STATUS.md',
    ])
    manifest = {rel: digest(ROOT / rel) for rel in sorted(selected)}
    note = (
        'LTN certified endpoint-ball localization and exactly three saved density paths. REVIEW ONLY.\n'
        'Includes current implementation, six new software checks, original immutable primitive, input-only NPZ, integer exclusion certificates and observations.\n'
        'Standalone checks: python research/reconnect/test_endpoint_density_path.py\n'
        'Standalone saved-input replay (NumPy required): python research/reconnect/saved_density_paths_study.py --replay runs/reconnect-saved-density-paths\n'
        'The original full contact graph snapshot is NOT included; its SHA and selected original member-byte hashes are recorded.\n'
        'Full historical preservation audit still needs the original workspace, 782 original files and 15 attachments.\n'
        'review-manifest.json covers included files; the cumulative scientific SHA manifest also covers unarchived history.\n'
        'Six new software checks passed on first execution; old 372 checks not rerun. Recorded total 378.\n'
        'Three saved-path records are not added as physical tests. No new conductance, graph, operator action or transient.\n'
        'Integral agreement is not pointwise interface-density equality, physical backend acceptance or path-kernel superiority.\n'
        'The conditional Cartesian ordinary-FV two-row baseline is specified for next work, not executed in this round.\n'
        'docs/RESEARCH_STATUS.md is a mutable snapshot outside the scientific seal. Stage 3 incomplete; fluid ORC disabled.\n'
    )
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in manifest:
            z.write(ROOT / rel, rel)
        z.writestr('review-manifest.json', json.dumps(manifest, indent=2) + '\n')
        z.writestr('DELIVERY_README.txt', note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert json.loads(z.read('review-manifest.json')) == manifest
        assert set(z.namelist()) == set(manifest) | {'review-manifest.json', 'DELIVERY_README.txt'}
        for rel, expected in manifest.items():
            assert hashlib.sha256(z.read(rel)).hexdigest() == expected, rel
        with tempfile.TemporaryDirectory(prefix='ltn-saved-path-review-', dir='/tmp') as directory:
            z.extractall(directory)
            task_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            completed = subprocess.run([
                sys.executable, str(Path(directory) / 'research/reconnect/saved_density_paths_study.py'),
                '--replay', 'runs/reconnect-saved-density-paths',
            ], cwd=directory, env=task_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
            assert completed.returncode == 0, completed.stdout
            replay_log = completed.stdout
    for rel, expected in read(seal).items():
        assert digest(ROOT / rel) == expected, rel
    info = {
        'review': {'file': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'review_only_not_full_history': True, 'standalone_current_saved_input_replay_included': True,
        'archive_CRC_and_all_included_SHA256_verified': True,
        'fresh_archive_extraction_replay_exit_code': completed.returncode,
        'fresh_archive_extraction_replay_log': replay_log,
        'review_workspace_entries': len(manifest), 'scientific_manifest_entries': len(entries),
        'source_scientific_manifest': source_seal,
        'source_scientific_manifest_sha256': digest(source / 'sha256.json'),
        'original_files_preserved': 782, 'historical_attachments_preserved': 15,
        'prior_formal_manifests_preserved': 57, 'new_localization_unit_checks_passed': 6,
        'previous_checks_rerun': 0, 'saved_path_observations': 3,
        'new_physical_or_flow_tests': 0, 'new_exact_algebra_certificate_groups': 0, 'recorded_checks_total': 378,
        'new_candidate_graphs': 0, 'new_edge_weights': 0, 'new_operator_quadrature_evaluations': 0,
        'new_candidate_transient_runs': 0, 'new_reference_transient_runs': 0,
        'finite_node_candidate_admitted': False, 'fluid_orc_enabled': False,
        'stage3_physical_acceptance': 'incomplete', 'published': False,
    }
    save(delivery, info)
    with (exports / 'ltn-saved-density-paths-downloads.md').open('x') as f:
        f.write('# 三条封存密度路径：实现与重放\n\n'
                '[实现、保存点输入、距离证书和三条观测](ltn-saved-density-paths-review.zip)。不含原完整图或历史附件。\n\n'
                f'SHA-256：`{info["review"]["sha256"]}`。\n\n'
                '[交付范围与哈希](ltn-saved-density-paths-delivery.json)。已在全新解压目录独立重放三条保存记录；完整历史审计仍需原工作区。\n\n'
                '本轮6项局部化单元检查通过，旧372项未重跑，累计378项。三条保存路径不另计物理测试。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
