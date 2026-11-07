"""Replay scientific paths plus append-only diagnostic completion, without new geometry."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_route_decision import read,digest
from becker_point_study import replay,load_all
from becker_point_completion import density_diagnostics
from becker_point_core import Becker


def main(directory):
    out=Path(directory).resolve();replay(out);data,col,runs,paths,rm=load_all(out)
    values=density_diagnostics(out,Becker(delta=.08),np.load(out/'source-mass-labels.npy',allow_pickle=False),col,read(out/'pure-inputs.json'))
    assert values==read(out/'density-volume-diagnostics.json')
    for rel,h in read(out/'pre-completion-files-sha256.json').items():assert digest(out/rel)==h,rel
    assert read(out/'failed-execution.json')['type']=='ValueError'
    print('Append-only rational-area diagnostics reproduced; all pre-completion files and original parser-failure evidence unchanged.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
