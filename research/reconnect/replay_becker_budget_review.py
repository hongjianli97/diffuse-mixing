"""Reproduce saved-budget accounting without any geometry, action or PDE solver."""
from pathlib import Path
import argparse,csv
from audit_route_decision import read,digest
from becker_budget_review import analyze


def main(directory):
    out=Path(directory).resolve();result,rows=analyze(out/'source');assert result==read(out/'results.json')
    saved=list(csv.DictReader((out/'budget-components.csv').open()));assert len(saved)==len(rows)
    for a,b in zip(saved,rows):assert a=={key:str(value) for key,value in b.items()}
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==item['sha256']
    assert not read(out/'next-gate-spec.json')['already_executed']
    assert not result['source_case_passed'] and not result['new_candidate_or_reference_states_generated']
    print('Saved positive budgets, original H and unresolved decision reproduced; no geometry/action/PDE or scientific state regenerated.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
