"""Recompute saved comparison/projections; do not rerun the reference PDE."""
from pathlib import Path
import argparse
import numpy as np
from audit_route_decision import read,digest
from becker_reference_check import arrays,overlap,LEVELS,key
from becker_reference_completion import analyze


def main(directory):
    out=Path(directory).resolve();assert analyze(out)==read(out/'results.json');col=arrays(out/'source/column.npz')
    for n,k in LEVELS:
        z=arrays(out/f'reference-{key(n,k)}.npz')
        np.testing.assert_array_equal(overlap(z['mass_edges'],z['c'],col['mass_edges']),z['projected'])
        np.testing.assert_array_equal(overlap(z['mass_edges'],z['c0'],col['mass_edges']),z['initial_projected'])
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==item['sha256']
    for rel,h in read(out/'pre-completion-sha256.json').items():assert digest(out/rel)==h
    print('Saved refined-reference projections and temporal comparison reproduced exactly; no PDE or candidate solve.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
