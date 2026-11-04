"""Seal the route review and create a small, explicitly review-only archive."""
from pathlib import Path
import hashlib
import json
import zipfile
from audit_route_decision import digest, read, save, ROOT, SOURCE, CURRENT


def main():
    exports = ROOT / 'exports'
    archive = exports / 'ltn-route-decision-review.zip'
    delivery = exports / 'ltn-route-decision-delivery.json'
    seal = CURRENT / 'sha256.json'
    assert not any(p.exists() for p in [archive, delivery, seal]), 'never overwrite sealed delivery'
    execution = read(CURRENT / 'execution.json')
    assert execution['audit_exit_code'] == 0
    audit = read(CURRENT / 'review-audit.json')
    assert audit['all_indexed_sources_previously_sealed']
    assert audit['recorded_checks_total'] == 363 and not audit['fluid_orc_enabled']
    prior = read(SOURCE / 'sha256.json')
    for rel, expected in prior.items():
        assert digest(ROOT / rel) == expected, rel
    new_sources = [
        'research/reconnect/CANDIDATE_ROUTE_DECISION.md',
        'research/reconnect/CONDITIONAL_DENSITY_CLOSURE.md',
        'research/reconnect/audit_route_decision.py',
        'research/reconnect/package_route_decision.py',
    ]
    entries = dict(prior)
    source_seal = str((SOURCE / 'sha256.json').relative_to(ROOT))
    entries[source_seal] = digest(SOURCE / 'sha256.json')
    for rel in new_sources:
        entries[rel] = digest(ROOT / rel)
    for p in CURRENT.iterdir():
        assert p.is_file(), p
        entries[str(p.relative_to(ROOT))] = digest(p)
    save(seal, dict(sorted(entries.items())))

    evidence = read(CURRENT / 'evidence-index.json')
    selected = set(new_sources)
    selected.update(item['path'] for item in evidence['sources'])
    selected.update(str(p.relative_to(ROOT)) for p in CURRENT.iterdir())
    selected.update([source_seal, str((SOURCE / 'stage3-status.json').relative_to(ROOT)),
                     'docs/RESEARCH_STATUS.md'])
    manifest = {rel: digest(ROOT / rel) for rel in sorted(selected)}
    note = (
        'LTN sealed-evidence route decision and conditional density-closure contract.\n'
        'REVIEW ONLY: contains current documents/metadata/audit code and the indexed prior reports/JSON records.\n'
        'Does not contain historical NPZ arrays, all historical scientific files, original attachments, or an executable full environment.\n'
        'The cumulative scientific SHA manifest covers history outside this archive. It is not the archive contents manifest.\n'
        'review-manifest.json covers every archived workspace file and can be checked independently with SHA-256.\n'
        'Full preservation-audit reproduction requires the historical workspace, the 782 original files and 15 attachments.\n'
        'docs/RESEARCH_STATUS.md is a mutable snapshot, excluded from the scientific seal.\n'
        'No new numerical tests or proofs. Recorded historical checks: 363. Old v0.8.0/246 not recovered.\n'
        'No new graphs, weights, quadrature, LPs, transients, PDF, release, push or publication.\n'
        'The conditional density-perturbation proof is pending; Stage 3 incomplete, fluid ORC disabled.\n'
    )
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in manifest:
            z.write(ROOT / rel, rel)
        z.writestr('review-manifest.json', json.dumps(manifest, indent=2) + '\n')
        z.writestr('DELIVERY_README.txt', note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        saved = json.loads(z.read('review-manifest.json'))
        assert saved == manifest
        assert set(z.namelist()) == set(manifest) | {'review-manifest.json', 'DELIVERY_README.txt'}
        for rel, expected in saved.items():
            assert hashlib.sha256(z.read(rel)).hexdigest() == expected, rel
    for rel, expected in read(seal).items():
        assert digest(ROOT / rel) == expected, rel
    info = {
        'review': {'file': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'review_only_not_full_history': True,
        'archive_CRC_and_all_included_SHA256_verified': True,
        'review_workspace_entries': len(manifest), 'scientific_manifest_entries': len(entries),
        'source_scientific_manifest': source_seal,
        'source_scientific_manifest_sha256': digest(SOURCE / 'sha256.json'),
        'original_files_preserved': 782, 'historical_attachments_preserved': 15,
        'prior_formal_manifests_preserved': 53,
        'new_numerical_tests': 0, 'new_exact_algebra_certificate_groups': 0,
        'recorded_checks_total': 363, 'fluid_orc_enabled': False,
        'stage3_physical_acceptance': 'incomplete', 'published': False,
    }
    save(delivery, info)
    with (exports / 'ltn-route-decision-downloads.md').open('x') as f:
        f.write('# 候选路线决定：审阅材料\n\n'
                '[本轮报告与来源审阅包](ltn-route-decision-review.zip)。包含输入契约、来源报告/JSON、状态和审计记录；不含历史 NPZ 或完整运行环境。\n\n'
                f'SHA-256：`{info["review"]["sha256"]}`。\n\n'
                '[交付清单](ltn-route-decision-delivery.json)。完整历史来源审计仍需原工作区；本包内 review-manifest.json 可独立核验所含文件。\n\n'
                '本轮没有新增数值测试或证明；累计记录363项。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
