"""Seal the primitive and create a standalone-unit-check, review-only ZIP."""
import hashlib
import json
import zipfile
from audit_route_decision import ROOT, digest, read, save


def main():
    source = ROOT / 'runs/reconnect-point-quadrature'
    out = ROOT / 'runs/reconnect-density-path'
    exports = ROOT / 'exports'
    archive = exports / 'ltn-density-path-review.zip'
    delivery = exports / 'ltn-density-path-delivery.json'
    seal = out / 'sha256.json'
    assert not any(p.exists() for p in [archive, delivery, seal]), 'never overwrite sealed evidence'
    execution = read(out / 'execution.json')
    assert execution['tests_exit_code'] == execution['study_exit_code'] == execution['audit_exit_code'] == 0
    audit = read(out / 'implementation-audit.json')
    assert audit['nine_new_software_unit_checks_passed'] and audit['recorded_checks_total'] == 372
    prior = read(source / 'sha256.json')
    for rel, expected in prior.items():
        assert digest(ROOT / rel) == expected, rel
    new_files = [
        'research/reconnect/PERIODIC_DENSITY_PATH_PRIMITIVE.md',
        'research/reconnect/periodic_density_path.py',
        'research/reconnect/test_periodic_density_path.py',
        'research/reconnect/density_path_study.py',
        'research/reconnect/audit_density_path.py',
        'research/reconnect/package_density_path.py',
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
    selected.update(item['path'] for item in read(out / 'evidence-index.json')['sources'])
    selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([source_seal, 'research/reconnect/audit_route_decision.py', 'docs/RESEARCH_STATUS.md'])
    manifest = {rel: digest(ROOT / rel) for rel in sorted(selected)}
    note = (
        'LTN periodic nearest-point inverse-density path primitive. REVIEW ONLY.\n'
        'Includes standard-library implementation, nine unit checks, exact geometry examples and audits.\n'
        'Standalone checks after extraction: python research/reconnect/test_periodic_density_path.py\n'
        'Unit checks use an independent periodic halfspace-intersection oracle, not the production lower hull.\n'
        'No mass, concentration, true material faces, F/C or PDE reference is a primitive input.\n'
        'No historical NPZ arrays, original attachments or complete execution environment are included.\n'
        'Full preservation-audit reproduction requires the historical workspace, 782 originals and 15 attachments.\n'
        'review-manifest.json covers included files; the cumulative scientific SHA manifest also covers unarchived history.\n'
        'Nine new primitive software checks passed on the first run; old 363 checks were not rerun. Recorded total 372.\n'
        'No new flow tests or proof groups, graph, conductance, numerical operator quadrature, LP, transient, PDF or version.\n'
        'Saved-path integration and endpoint-ball filtering are predeclared next tasks, not completed in this round.\n'
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
    for rel, expected in read(seal).items():
        assert digest(ROOT / rel) == expected, rel
    info = {
        'review': {'file': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'review_only_not_full_history': True, 'standalone_primitive_unit_checks_included': True,
        'archive_CRC_and_all_included_SHA256_verified': True,
        'review_workspace_entries': len(manifest), 'scientific_manifest_entries': len(entries),
        'source_scientific_manifest': source_seal,
        'source_scientific_manifest_sha256': digest(source / 'sha256.json'),
        'original_files_preserved': 782, 'historical_attachments_preserved': 15,
        'prior_formal_manifests_preserved': 56, 'new_primitive_unit_checks_passed': 9,
        'previous_checks_rerun': 0, 'new_physical_or_flow_tests': 0,
        'new_exact_algebra_certificate_groups': 0, 'recorded_checks_total': 372,
        'new_candidate_graphs': 0, 'new_edge_weights': 0, 'new_operator_quadrature_evaluations': 0,
        'new_candidate_transient_runs': 0, 'new_reference_transient_runs': 0,
        'finite_node_candidate_admitted': False, 'fluid_orc_enabled': False,
        'stage3_physical_acceptance': 'incomplete', 'published': False,
    }
    save(delivery, info)
    with (exports / 'ltn-density-path-downloads.md').open('x') as f:
        f.write('# 周期密度路径原语：审阅与独立单元检查\n\n'
                '[实现、检查、精确几何记录和审计包](ltn-density-path-review.zip)。不含历史NPZ或完整运行环境。\n\n'
                f'SHA-256：`{info["review"]["sha256"]}`。\n\n'
                '[交付范围与哈希](ltn-density-path-delivery.json)。可独立运行本轮test_periodic_density_path.py；完整保留性审计仍需历史工作区。\n\n'
                '本轮9项原语单元检查通过，历史363项未重跑，累计记录372项。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
