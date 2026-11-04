"""Verify only files included in this GitHub backup and all 782 originals."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'backup/github-backup-manifest.json').read_text())
    originals = json.loads((ROOT / 'backup/original-handoff-sha256.json').read_text())
    assert len(originals) == 782
    for label, files in [('backup', manifest['files']), ('originals', originals)]:
        for rel, expected in files.items():
            path = ROOT / rel
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected:
                raise SystemExit(f'{label} SHA-256 mismatch: {rel}')
    print(f"Verified {len(manifest['files'])} backed-up files and 782 unchanged originals.")
    print('Omitted historical runs and attachments are not claimed as recovered or verified.')


if __name__ == '__main__':
    main()
