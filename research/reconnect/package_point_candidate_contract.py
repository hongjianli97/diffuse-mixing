"""Seal the read-only contract review and replay its declarations only."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

from audit_route_decision import ROOT,digest,read,save


def main():
    source=ROOT/'runs/reconnect-FV-contact-spatial';out=ROOT/'runs/reconnect-point-contract-review'
    exports=ROOT/'exports';archive=exports/'ltn-point-contract-review.zip'
    delivery=exports/'ltn-point-contract-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed evidence'
    execution=read(out/'execution.json');audit=read(out/'review-audit.json')
    assert execution['review_exit_code']==execution['audit_exit_code']==0
    assert audit['prior_formal_manifests_verified']==66 and audit['new_edge_weights']==audit['new_numerical_cases']==0
    prior=read(source/'sha256.json')
    for rel,expected in prior.items():assert digest(ROOT/rel)==expected,rel
    files=['research/reconnect/'+name for name in ['POINT_CANDIDATE_INFORMATION_REVIEW.md',
        'point_candidate_contract_review.py','audit_point_candidate_contract.py','package_point_candidate_contract.py']]
    entries=dict(prior);source_seal=str((source/'sha256.json').relative_to(ROOT));entries[source_seal]=digest(source/'sha256.json')
    for rel in files:entries[rel]=digest(ROOT/rel)
    for p in out.iterdir():
        assert p.is_file(),p
        entries[str(p.relative_to(ROOT))]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'] if not item['path'].endswith('.npz'))
    selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([source_seal,'research/reconnect/audit_route_decision.py',
        'research/reconnect/test_periodic_density_path.py','research/reconnect/test_endpoint_density_path.py','docs/RESEARCH_STATUS.md'])
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=(
        'LTN point-input contract review and one future frozen-pair gate. REVIEW ONLY.\n'
        'No candidate implementation, weights, graph, operator evaluation, proof rerun or numerical case executed.\n'
        'Older next-work statements are reconciled with later completed conditional density and point-pair proofs.\n'
        'Limited FV geometry baseline, continuous oracle path operator and finite-node path candidate remain distinct.\n'
        'Selected future gate: original N128 snapshot, h=.04, D=.02, complete support rows 63/64, original mass-average c0.\n'
        'Fixed rule K=D/(integral 1/rho_hat), g=2*b*q_i*q_j*K*B_h. No scalar fit or true geometry input.\n'
        'The gate is NOT executed here; even pair improvement cannot admit a full graph, transient or ORC.\n'
        'Standard-library contract-record replay: python -S research/reconnect/point_candidate_contract_review.py --replay runs/reconnect-point-contract-review\n'
        'This replay checks the saved contract and records; it does not evaluate the future candidate.\n'
        'Includes relevant source reports, primitive code and existing tests for inspection; old tests were not rerun this round.\n'
        'Recorded checks remain 403, with zero new numerical/software tests or algebra certificate groups.\n'
        '782 originals, 15 attachments and 66 prior formal manifests preserved; four historical diagnostic differences retained separately.\n'
        'Full source audit needs historical workspace; full old snapshots and attachments are omitted.\n'
        'review-manifest.json covers included files. Cumulative scientific seal also covers unarchived history.\n'
        'docs/RESEARCH_STATUS.md is a mutable snapshot outside the scientific seal.\n'
        'No FV grid expansion, clipping, renormalization, fitted F/C, threshold change, PDF, release or publication.\n'
        'Stage 3 incomplete; fluid ORC disabled; no old v0.8.0 recovery claim.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,expected in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==expected,rel
        with tempfile.TemporaryDirectory(prefix='ltn-point-contract-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,'-S',str(Path(directory)/'research/reconnect/point_candidate_contract_review.py'),
                '--replay','runs/reconnect-point-contract-review'],cwd=directory,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
            assert completed.returncode==0,completed.stdout+completed.stderr
    for rel,expected in read(seal).items():assert digest(ROOT/rel)==expected,rel
    info={'review':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'review_only_not_full_history':True,'archive_CRC_and_all_included_SHA256_verified':True,
        'fresh_standard_library_contract_record_replay_exit_code':completed.returncode,
        'fresh_contract_record_replay_log':completed.stdout,
        'review_workspace_entries':len(manifest),'scientific_manifest_entries':len(entries),
        'source_scientific_manifest':source_seal,'source_scientific_manifest_sha256':digest(source/'sha256.json'),
        'selected_next_gate':'reconstructed_density_path_full_frozen_pair','gate_executed':False,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':66,
        'new_software_checks':0,'recorded_checks_total':403,'previous_checks_rerun':0,
        'new_exact_algebra_certificate_groups':0,'new_edge_weights':0,'new_candidate_graphs':0,
        'new_numerical_cases':0,'new_candidate_transient_runs':0,'new_reference_time_solutions':0,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete','published':False}
    save(delivery,info)
    with (exports/'ltn-point-contract-downloads.md').open('x') as f:
        f.write('# 点输入契约复核：审阅与门槛声明\n\n'
            '[复核、来源、假设清单与未执行的门槛规范](ltn-point-contract-review.zip)。\n\n'
            f'SHA-256：`{info["review"]["sha256"]}`。\n\n'
            '[交付范围与哈希](ltn-point-contract-delivery.json)。全新解压目录的标准库契约记录重放通过，未执行候选门槛。\n\n'
            '检查记录仍403；本轮新权重、图和数值算例均0。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
