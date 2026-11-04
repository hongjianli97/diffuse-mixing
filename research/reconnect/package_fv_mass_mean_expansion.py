"""Seal the theory-only interface expansion and check an isolated replay."""
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
    source = ROOT / 'runs/reconnect-FV-face-budget'
    out = ROOT / 'runs/reconnect-FV-interface-expansion'
    exports = ROOT / 'exports'
    archive = exports / 'ltn-FV-interface-expansion-review.zip'
    delivery = exports / 'ltn-FV-interface-expansion-delivery.json'
    seal = out / 'sha256.json'
    assert not any(path.exists() for path in [archive, delivery, seal]), 'never overwrite sealed evidence'
    execution = read(out / 'execution.json')
    assert execution['study_exit_code'] == execution['audit_exit_code'] == 0
    audit = read(out / 'implementation-audit.json')
    assert audit['prior_formal_manifests_verified'] == 60
    assert audit['new_exact_algebra_certificate_groups'] == 4
    prior = read(source / 'sha256.json')
    for rel, expected in prior.items():
        assert digest(ROOT / rel) == expected, rel
    new_files = ['research/reconnect/' + name for name in [
        'FV_MASS_MEAN_INTERFACE_EXPANSION.md', 'fv_mass_mean_expansion.py',
        'fv_mass_mean_expansion_study.py', 'audit_fv_mass_mean_expansion.py',
        'package_fv_mass_mean_expansion.py',
    ]]
    source_seal = str((source / 'sha256.json').relative_to(ROOT))
    entries = dict(prior)
    entries[source_seal] = digest(source / 'sha256.json')
    for rel in new_files:
        entries[rel] = digest(ROOT / rel)
    for path in out.iterdir():
        assert path.is_file(), path
        entries[str(path.relative_to(ROOT))] = digest(path)
    save(seal, dict(sorted(entries.items())))
    selected = set(new_files)
    selected.update(item['path'] for item in read(out / 'evidence-index.json')['sources'] if not item['path'].endswith('.npz'))
    selected.update(str(path.relative_to(ROOT)) for path in out.iterdir())
    selected.update([source_seal, 'research/reconnect/audit_route_decision.py', 'docs/RESEARCH_STATUS.md'])
    manifest = {rel: digest(ROOT / rel) for rel in sorted(selected)}
    note = (
        'LTN ordinary FV mass-mean interface expansion. REVIEW ONLY. No transport correction applied.\n'
        'True strip/cell averages, explicit curvature-jump leading defect and analytic third-derivative remainder bound.\n'
        'Original analytic background is extra information; derivatives are not candidate runtime inputs. No fitting.\n'
        'Saved g, mass, c0 and reference fluxes are unchanged; mathematical-formula versus saved-binary bridges are retained.\n'
        'Standalone standard-library replay: python -S research/reconnect/fv_mass_mean_expansion_study.py --replay runs/reconnect-FV-interface-expansion\n'
        'Exact coefficients only: python -S research/reconnect/fv_mass_mean_expansion_study.py --certificates-only runs/reconnect-FV-interface-expansion/analysis-inputs.json\n'
        'Four exact algebra certificate groups are separate from the unchanged 388 recorded checks. No new numerical or flow tests.\n'
        'Decimal precision controls are not interval arithmetic; the remainder inequality is proved analytically.\n'
        'This archive excludes the full snapshot and historical attachments. Full preservation audit requires the historical workspace.\n'
        'review-manifest.json covers all included workspace files. The cumulative scientific seal also covers unarchived history.\n'
        '782 originals, 15 attachments, 60 prior formal manifests preserved. Four historical diagnostic differences separately retained.\n'
        'Original c0 diagnosis retained despite initial higher-order compatibility issue. No transient or global acceptance claim.\n'
        'Next four-mean reconstruction admissibility review is predeclared and not executed here.\n'
        'docs/RESEARCH_STATUS.md is a mutable snapshot outside the scientific seal. Stage 3 incomplete; fluid ORC disabled.\n'
        'No new weights, graph, matrix, reference integration, transient, PDF, release, commit, push or publication.\n'
    )
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in manifest:
            z.write(ROOT / rel, rel)
        z.writestr('review-manifest.json', json.dumps(manifest, indent=2) + '\n')
        z.writestr('DELIVERY_README.txt', note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(manifest) | {'review-manifest.json', 'DELIVERY_README.txt'}
        for rel, expected in manifest.items():
            assert hashlib.sha256(z.read(rel)).hexdigest() == expected, rel
        with tempfile.TemporaryDirectory(prefix='ltn-FV-interface-expansion-', dir='/tmp') as directory:
            z.extractall(directory)
            completed = subprocess.run([
                sys.executable, '-S', str(Path(directory) / 'research/reconnect/fv_mass_mean_expansion_study.py'),
                '--replay', 'runs/reconnect-FV-interface-expansion',
            ], cwd=directory, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                capture_output=True, text=True, timeout=60)
            assert completed.returncode == 0, completed.stdout + completed.stderr
    for rel, expected in read(seal).items():
        assert digest(ROOT / rel) == expected, rel
    info = {
        'review': {'file': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'review_only_not_full_history': True,
        'archive_CRC_and_all_included_SHA256_verified': True,
        'fresh_archive_python_no_site_packages_replay_exit_code': completed.returncode,
        'fresh_archive_replay_log': completed.stdout,
        'review_workspace_entries': len(manifest), 'scientific_manifest_entries': len(entries),
        'source_scientific_manifest': source_seal,
        'source_scientific_manifest_sha256': digest(source / 'sha256.json'),
        'original_files_preserved': 782, 'historical_attachments_preserved': 15,
        'prior_formal_manifests_preserved': 60, 'new_exact_algebra_certificate_groups': 4,
        'new_numerical_or_flow_tests': 0, 'recorded_checks_total': 388,
        'new_edge_weights': 0, 'new_candidate_graphs': 0, 'new_reference_integrations': 0,
        'new_candidate_transient_runs': 0, 'new_reference_transient_runs': 0,
        'fluid_orc_enabled': False, 'stage3_physical_acceptance': 'incomplete', 'published': False,
    }
    save(delivery, info)
    with (exports / 'ltn-FV-interface-expansion-downloads.md').open('x') as f:
        f.write('# 物质平均界面展开：审阅与重放\n\n'
                '[推导、精确输入、算术控制与审计包](ltn-FV-interface-expansion-review.zip)。\n\n'
                f'SHA-256：`{info["review"]["sha256"]}`。\n\n'
                '[交付范围与哈希](ltn-FV-interface-expansion-delivery.json)。全新解压目录python -S重放通过。\n\n'
                '4组代数证书单独记录；累计软件检查仍388。本轮无新运输权重或求解。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
