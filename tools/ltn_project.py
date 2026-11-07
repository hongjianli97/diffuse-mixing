#!/usr/bin/env python3
"""Navigation, integrity verification and bounded checks of unchanged LTN code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify():
    delivery = ROOT / 'delivery-manifest.json'
    selected = delivery if delivery.exists() else ROOT / 'backup/github-backup-manifest.json'
    manifest = json.loads(selected.read_text())
    originals = json.loads((ROOT / 'backup/original-handoff-sha256.json').read_text())
    if len(originals) != 782:
        raise ValueError('expected 782 original files')
    for group, entries in [('selected', manifest['files']), ('originals', originals)]:
        for relative, digest in entries.items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError(f'path outside package: {relative}')
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != digest:
                raise ValueError(f'{group} hash mismatch: {relative}')
    print(f'Verified {len(manifest["files"])} selected files and 782 originals.')
    print('This does not verify omitted historical runs, attachments or old v080.')


def checks(output):
    destination = Path(output).expanduser().resolve()
    if destination == ROOT or destination.is_relative_to(ROOT / 'runs'):
        raise ValueError('use a new output directory outside sealed runs')
    destination.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    groups = [
        ('graph', [sys.executable, str(ROOT / 'research/reconnect/test_trajectory_chain.py')]),
        ('material_fv', [sys.executable, str(ROOT / 'research/material_fv/check_material_fv.py'),
                         '--output', str(destination / 'material-fv')]),
    ]
    records = []
    for name, command in groups:
        result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (destination / f'{name}.log').write_text(result.stdout)
        print(result.stdout, end='')
        records.append({'group': name, 'returncode': result.returncode})
        if result.returncode:
            break
    receipt = {'groups': records, 'passed': len(records) == 2 and all(r['returncode'] == 0 for r in records),
               'existing_checks_rerun': 15 if len(records) == 2 and all(r['returncode'] == 0 for r in records) else None,
               'new_physical_experiments': 0, 'historical_recorded_checks': 508,
               'stage3': 'incomplete', 'fluid_orc_enabled': False}
    (destination / 'checks-summary.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if not receipt['passed']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='show current entry points')
    commands.add_parser('verify', help='verify the selected recovery manifest and originals')
    check = commands.add_parser('checks', help='rerun 15 existing bounded software checks')
    check.add_argument('--output', required=True, help='new directory; existing paths are refused')
    args = parser.parse_args()
    if args.command == 'verify':
        verify()
    elif args.command == 'checks':
        checks(args.output)
    else:
        for relative in ['docs/PROJECT_OVERVIEW.md', 'docs/CODE_GUIDE.md', 'docs/NEXT_STEPS.md',
                         'research/material_fv/material_fv.py', 'research/reconnect/trajectory_chain.py',
                         'research/reconnect/periodic_voronoi_fv.py', 'research/reconnect/voronoi_time_action.py']:
            print(relative)
        print('Stage3 incomplete; fluid ORC OFF. No physical driver runs automatically.')


if __name__ == '__main__':
    main()
