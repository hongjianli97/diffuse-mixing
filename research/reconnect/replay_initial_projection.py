"""Replay initial-only analytic accounting from saved files."""
from pathlib import Path
import argparse
from audit_route_decision import read,digest
from initial_projection_review import analyze


def main(directory):
    out=Path(directory).resolve();r,v=analyze(out/'source')
    assert r==read(out/'results.json') and v==read(out/'diagnostic-vectors.json')
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==item['sha256']
    print('Initial-only boundary defects and rounding accounting reproduced; no PDE, candidate or terminal-state evaluation.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
