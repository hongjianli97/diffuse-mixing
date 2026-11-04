"""Seal the limited FV control, including the retained initial reporting output."""
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
    source = ROOT / 'runs/reconnect-saved-density-paths'
    out = ROOT / 'runs/reconnect-cartesian-fv-rows'
    initial = ROOT / 'runs/reconnect-cartesian-fv-rows-initial-float-norm'
    exports = ROOT / 'exports'
    archive = exports / 'ltn-cartesian-fv-rows-review.zip'
    delivery = exports / 'ltn-cartesian-fv-rows-delivery.json'
    seal = out / 'sha256.json'
    assert not any(p.exists() for p in [archive, delivery, seal]), 'never overwrite sealed evidence'
    execution = read(out / 'execution.json')
    assert execution['tests_exit_code'] == execution['initial_study_exit_code'] == execution['final_study_exit_code'] == execution['audit_exit_code'] == 0
    assert read(out / 'implementation-audit.json')['six_new_FV_row_unit_checks_passed']
    prior = read(source / 'sha256.json')
    for rel, expected in prior.items():
        assert digest(ROOT / rel) == expected, rel
    new_files = [
        'research/reconnect/CARTESIAN_FV_TWO_ROW_CONTROL.md',
        'research/reconnect/cartesian_material_fv_rows.py',
        'research/reconnect/test_cartesian_material_fv_rows.py',
        'research/reconnect/cartesian_fv_rows_study.py',
        'research/reconnect/cartesian_fv_rows_study_initial_float_norm.py',
        'research/reconnect/audit_cartesian_fv_rows.py',
        'research/reconnect/package_cartesian_fv_rows.py',
    ]
    source_seal = str((source / 'sha256.json').relative_to(ROOT))
    entries = dict(prior)
    entries[source_seal] = digest(source / 'sha256.json')
    for rel in new_files:
        entries[rel] = digest(ROOT / rel)
    for folder in [out, initial]:
        for p in folder.iterdir():
            assert p.is_file(), p
            entries[str(p.relative_to(ROOT))] = digest(p)
    save(seal, dict(sorted(entries.items())))
    selected = set(new_files)
    selected.update(item['path'] for item in read(out / 'evidence-index.json')['sources'] if not item['path'].endswith('.npz'))
    for folder in [out, initial]:
        selected.update(str(p.relative_to(ROOT)) for p in folder.iterdir())
    selected.update([source_seal, 'research/reconnect/audit_route_decision.py',
                     'research/reconnect/periodic_density_path.py', 'docs/RESEARCH_STATUS.md'])
    manifest = {rel: digest(ROOT / rel) for rel in sorted(selected)}
    note = (
        'LTN ordinary central-resistance Cartesian material-FV two-row control. REVIEW ONLY.\n'
        'Primary result: original c0 and original common global PDE norm, targets 63/64 only. NOT an all-row FV error.\n'
        'Explicit static Cartesian material-cell prior; no general point-trajectory backend acceptance.\n'
        'Includes seven local shared FV conductances, point/mass/c0/reference inputs, two original cached CSR rows, unit checks and audits.\n'
        'Standalone software checks: python research/reconnect/test_cartesian_material_fv_rows.py\n'
        'Standalone replay (NumPy): OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python research/reconnect/cartesian_fv_rows_study.py --replay runs/reconnect-cartesian-fv-rows\n'
        'Initial auxiliary reporting output and original script are retained under initial-float-norm names.\n'
        'Those auxiliary display-zero norms are superseded: final residuals subtract exact fractions before conversion.\n'
        'Inputs, conductances, raw fluxes, primary c0 comparison and thresholds were unchanged by that reporting correction.\n'
        'Original full graph/NPZ and attachments are NOT included; their provenance is recorded.\n'
        'Full preservation audit requires historical workspace, 782 originals and 15 attachments.\n'
        'review-manifest.json covers included files; cumulative scientific SHA manifest also covers unarchived history.\n'
        'Six new software checks passed first run; old 378 checks not rerun. Recorded total 384.\n'
        'One fixed-background two-row diagnostic; no new global graph, matrix, reference solve/integration, flow solve or transient.\n'
        'Auxiliary quadratic agreement does not accept the original c0 residual or Stage 3.\n'
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
        with tempfile.TemporaryDirectory(prefix='ltn-FV-rows-review-', dir='/tmp') as directory:
            z.extractall(directory)
            task_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
            completed = subprocess.run([
                sys.executable, str(Path(directory) / 'research/reconnect/cartesian_fv_rows_study.py'),
                '--replay', 'runs/reconnect-cartesian-fv-rows',
            ], cwd=directory, env=task_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
            assert completed.returncode == 0, completed.stdout
            replay_log = completed.stdout
    for rel, expected in read(seal).items():
        assert digest(ROOT / rel) == expected, rel
    info = {
        'review': {'file': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'review_only_not_full_history': True, 'standalone_current_FV_replay_included': True,
        'archive_CRC_and_all_included_SHA256_verified': True,
        'fresh_archive_extraction_replay_exit_code': completed.returncode,
        'fresh_archive_extraction_replay_log': replay_log,
        'retained_initial_reporting_output_included': True,
        'review_workspace_entries': len(manifest), 'scientific_manifest_entries': len(entries),
        'source_scientific_manifest': source_seal,
        'source_scientific_manifest_sha256': digest(source / 'sha256.json'),
        'original_files_preserved': 782, 'historical_attachments_preserved': 15,
        'prior_formal_manifests_preserved': 58, 'new_FV_row_unit_checks_passed': 6,
        'previous_checks_rerun': 0, 'fixed_background_two_row_diagnostics': 1,
        'new_exact_algebra_certificate_groups': 0, 'recorded_checks_total': 384,
        'new_local_FV_edge_conductances': 7, 'new_edge_weights': 7, 'new_kernel_edge_weights': 0,
        'new_candidate_graphs': 0, 'new_global_matrices': 0, 'new_operator_quadrature_evaluations': 0,
        'new_candidate_transient_runs': 0, 'new_reference_transient_runs': 0,
        'finite_node_candidate_admitted': False, 'fluid_orc_enabled': False,
        'stage3_physical_acceptance': 'incomplete', 'published': False,
    }
    save(delivery, info)
    with (exports / 'ltn-cartesian-fv-rows-downloads.md').open('x') as f:
        f.write('# 普通中心阻力FV两行冻结对照：审阅与重放\n\n'
                '[实现、原状态输入、两行结果与审计包](ltn-cartesian-fv-rows-review.zip)。不含原完整核图或历史附件；含首次辅助记录及修正说明。\n\n'
                f'SHA-256：`{info["review"]["sha256"]}`。\n\n'
                '[交付范围与哈希](ltn-cartesian-fv-rows-delivery.json)。已在全新解压目录独立重放最终基线和辅助控制。\n\n'
                '本轮6项软件检查通过，旧378项未重跑，累计384项。主结果只覆盖两个目标，不是全域FV误差。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
