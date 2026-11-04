"""Seal new continuous density-bound evidence; create a review-only ZIP."""
import hashlib
import json
import zipfile
from audit_route_decision import ROOT, digest, read, save


def main():
    source = ROOT / 'runs/reconnect-route-decision'
    out = ROOT / 'runs/reconnect-density-weak'
    exports = ROOT / 'exports'
    archive = exports / 'ltn-density-weak-review.zip'
    delivery = exports / 'ltn-density-weak-delivery.json'
    seal = out / 'sha256.json'
    assert not any(p.exists() for p in [archive, delivery, seal]), 'never overwrite sealed evidence'
    execution = read(out / 'execution.json')
    assert execution['exit_code'] == execution['independent_process_exit_code'] == execution['audit_exit_code'] == 0
    assert read(out / 'proof-audit.json')['new_three_exact_algebra_groups_recomputed']
    prior = read(source / 'sha256.json')
    for rel, expected in prior.items():
        assert digest(ROOT / rel) == expected, rel
    new_files = [
        'research/reconnect/DENSITY_RECONSTRUCTION_WEAK_BOUND.md',
        'research/reconnect/density_weak_certificates.py',
        'research/reconnect/audit_density_weak.py',
        'research/reconnect/package_density_weak.py',
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
        'LTN conditional continuous density-reconstruction weak perturbation proof.\n'
        'REVIEW ONLY: current proof, standard-library certificates, audits and cited reports/JSON.\n'
        'Standalone exact algebra: python research/reconnect/density_weak_certificates.py --certificates-only\n'
        'Full source/preservation audit requires the historical workspace, 782 original files and 15 attachments.\n'
        'No historical NPZ or complete execution environment is included.\n'
        'review-manifest.json lists every included workspace file. The cumulative scientific SHA manifest also covers unarchived history.\n'
        'Integral/regularity proofs are in the report; exact algebra checks are not an automated theorem prover.\n'
        'Three new exact-algebra groups; no new numerical/flow tests. Recorded historical checks remain 363.\n'
        'Continuous weak O(epsilon+h) is conditional on the stated flat phase-constant transmission model.\n'
        'No finite-node, strong-local or transient acceptance; no new graphs, weights, quadrature, LPs, PDF or software version.\n'
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
        'review_only_not_full_history': True, 'archive_CRC_and_all_included_SHA256_verified': True,
        'review_workspace_entries': len(manifest), 'scientific_manifest_entries': len(entries),
        'source_scientific_manifest': source_seal,
        'source_scientific_manifest_sha256': digest(source / 'sha256.json'),
        'original_files_preserved': 782, 'historical_attachments_preserved': 15,
        'prior_formal_manifests_preserved': 54, 'new_exact_algebra_certificate_groups': 3,
        'new_numerical_tests': 0, 'recorded_checks_total': 363,
        'new_candidate_graphs': 0, 'new_edge_weights': 0, 'new_operator_quadrature_evaluations': 0,
        'new_candidate_transient_runs': 0, 'new_reference_transient_runs': 0,
        'finite_node_candidate_admitted': False, 'fluid_orc_enabled': False,
        'stage3_physical_acceptance': 'incomplete', 'published': False,
    }
    save(delivery, info)
    with (exports / 'ltn-density-weak-downloads.md').open('x') as f:
        f.write('# 连续密度重构弱扰动界：审阅材料\n\n'
                '[报告、标准库代数核验和来源审计包](ltn-density-weak-review.zip)。审阅包不含历史 NPZ 或完整环境。\n\n'
                f'SHA-256：`{info["review"]["sha256"]}`。\n\n'
                '[交付范围与哈希](ltn-density-weak-delivery.json)。独立分数核验可用 certificates-only；完整保留性审计需历史工作区。\n\n'
                '本轮3组精确代数核验另列，无新增数值测试，累计记录363项。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
