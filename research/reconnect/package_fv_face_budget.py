"""Seal the saved face budget and verify a standard-library-only replay."""
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
    source = ROOT / 'runs/reconnect-cartesian-fv-rows'
    out = ROOT / 'runs/reconnect-FV-face-budget'
    exports = ROOT / 'exports'
    archive = exports / 'ltn-FV-face-budget-review.zip'
    delivery = exports / 'ltn-FV-face-budget-delivery.json'
    seal = out / 'sha256.json'
    assert not any(p.exists() for p in [archive, delivery, seal]), 'never overwrite sealed evidence'
    execution = read(out / 'execution.json')
    assert execution['tests_exit_code'] == execution['study_exit_code'] == execution['audit_exit_code'] == 0
    assert read(out / 'implementation-audit.json')['four_new_budget_software_checks_passed']
    prior = read(source / 'sha256.json')
    for rel, expected in prior.items():
        assert digest(ROOT / rel) == expected, rel
    new_files = [
        'research/reconnect/FV_SAVED_FACE_ERROR_BUDGET.md',
        'research/reconnect/saved_fv_face_budget.py',
        'research/reconnect/test_saved_fv_face_budget.py',
        'research/reconnect/fv_face_budget_study.py',
        'research/reconnect/audit_fv_face_budget.py',
        'research/reconnect/package_fv_face_budget.py',
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
    selected = set(new_files)
    selected.update(item['path'] for item in read(out / 'evidence-index.json')['sources'] if not item['path'].endswith('.npz'))
    selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([source_seal, 'research/reconnect/audit_route_decision.py',
                     'research/reconnect/periodic_density_path.py', 'docs/RESEARCH_STATUS.md'])
    manifest = {rel: digest(ROOT / rel) for rel in sorted(selected)}
    note = (
        'LTN signed saved-face error budget of the unchanged ordinary FV two-row c0 control. REVIEW ONLY.\n'
        'Contains exact parameters, selected reference-face bytes, source receipts, full signed Gram matrix, tests and audits.\n'
        'Standalone standard-library checks: python research/reconnect/test_saved_fv_face_budget.py\n'
        'Standalone standard-library replay: python -S research/reconnect/fv_face_budget_study.py --replay runs/reconnect-FV-face-budget\n'
        'NumPy is needed only to inspect the reference NPZ or reproduce full historical source audits, not the JSON replay.\n'
        'Original stencil, c0, fluxes and normalization unchanged. No FV construction module was executed.\n'
        'Reference face q=+rho*D*grad(c) is extra information, not a transport-runtime input.\n'
        'Reference and display roundoff are retained. Component norms are not additive causal percentages.\n'
        'Full historical preservation audit needs the original workspace, 782 originals and 15 attachments.\n'
        'The original full graph snapshot is not included. review-manifest.json covers included files; cumulative scientific SHA also covers unarchived history.\n'
        'Four new bookkeeping checks passed first run; old 384 checks not rerun. Recorded total 388.\n'
        'No new weights, graph, matrix, integration, flow/reference solve, transient, PDF or version.\n'
        'The next mass-mean curvature expansion is predeclared, not completed here.\n'
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
        with tempfile.TemporaryDirectory(prefix='ltn-FV-face-budget-', dir='/tmp') as directory:
            z.extractall(directory)
            task_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            completed = subprocess.run([
                sys.executable, '-S', str(Path(directory) / 'research/reconnect/fv_face_budget_study.py'),
                '--replay', 'runs/reconnect-FV-face-budget',
            ], cwd=directory, env=task_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
            assert completed.returncode == 0, completed.stdout
            replay_log = completed.stdout
    for rel, expected in read(seal).items():
        assert digest(ROOT / rel) == expected, rel
    info = {
        'review': {'file': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'review_only_not_full_history': True, 'standard_library_only_replay_included': True,
        'archive_CRC_and_all_included_SHA256_verified': True,
        'fresh_archive_python_no_site_packages_replay_exit_code': completed.returncode,
        'fresh_archive_replay_log': replay_log,
        'review_workspace_entries': len(manifest), 'scientific_manifest_entries': len(entries),
        'source_scientific_manifest': source_seal,
        'source_scientific_manifest_sha256': digest(source / 'sha256.json'),
        'original_files_preserved': 782, 'historical_attachments_preserved': 15,
        'prior_formal_manifests_preserved': 59, 'new_budget_software_checks_passed': 4,
        'previous_checks_rerun': 0, 'existing_two_row_budget_diagnostics': 1,
        'new_exact_algebra_certificate_groups': 0, 'recorded_checks_total': 388,
        'new_edge_weights': 0, 'new_candidate_graphs': 0, 'new_global_matrices': 0,
        'new_operator_quadrature_evaluations': 0, 'new_reference_integrations': 0,
        'new_candidate_transient_runs': 0, 'new_reference_transient_runs': 0,
        'finite_node_candidate_admitted': False, 'fluid_orc_enabled': False,
        'stage3_physical_acceptance': 'incomplete', 'published': False,
    }
    save(delivery, info)
    with (exports / 'ltn-FV-face-budget-downloads.md').open('x') as f:
        f.write('# 已存面通量误差预算：审阅与重放\n\n'
                '[预算、精确参数、原面值与审计包](ltn-FV-face-budget-review.zip)。不含完整历史图或原附件。\n\n'
                f'SHA-256：`{info["review"]["sha256"]}`。\n\n'
                '[交付范围与哈希](ltn-FV-face-budget-delivery.json)。已在全新解压目录用python -S验证标准库独立重放。\n\n'
                '本轮4项预算软件检查通过，旧384项未重跑，累计388项。没有改动权重或状态。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
