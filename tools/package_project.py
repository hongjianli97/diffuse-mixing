#!/usr/bin/env python3
"""Create a small, verifiable source bundle; never claims full data recovery."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'exports/LTN_project_code_20261107.zip'
PREFIX = 'LTN_project_code_20261107/'


def main():
    originals = json.loads((ROOT / 'backup/original-handoff-sha256.json').read_text())
    files = set(originals)
    for directory in ['research', 'docs', 'tools', 'reference_baseline', 'prototypes', 'legacy_sources', 'audit']:
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'}:
                files.add(path.relative_to(ROOT).as_posix())
    files.update(['START_HERE.md', 'GITHUB_HANDOFF.md', 'backup/original-handoff-sha256.json',
                  'backup/requirements-replay.txt'])
    report = ROOT / 'reports/project-20261107'
    for path in report.rglob('*'):
        if path.is_file() and path.suffix in {'.md', '.tex', '.lua', '.png', '.json'}:
            files.add(path.relative_to(ROOT).as_posix())
    digests = {}
    for relative in sorted(files):
        with (ROOT / relative).open('rb') as stream:
            digests[relative] = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert all(digests[p] == digest for p, digest in originals.items())
    manifest = {'date': '2026-11-07', 'kind': 'research-source-package',
                'full_workspace_backup': False, 'original_files': 782,
                'omitted': ['runs directory including result JSON and raw arrays', '15 original attachments', 'old v080', 'Git history', 'chat'],
                'stage3': 'incomplete', 'fluid_orc_enabled': False, 'files': digests}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in sorted(files):
            archive.write(ROOT / relative, PREFIX + relative)
        archive.writestr(PREFIX + 'delivery-manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(OUTPUT), 'files': len(files), 'bytes': OUTPUT.stat().st_size,
                      'sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
