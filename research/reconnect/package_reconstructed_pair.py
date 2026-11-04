"""Seal one pair experiment and verify its portable exact replay."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

from audit_route_decision import ROOT,read,save,digest


def main():
    source=ROOT/'runs/reconnect-point-contract-review'
    out=ROOT/'runs/reconnect-reconstructed-pair'
    aborted=ROOT/'runs/reconnect-reconstructed-pair-initial-norm-check'
    exports=ROOT/'exports';archive=exports/'ltn-reconstructed-pair.zip'
    delivery=exports/'ltn-reconstructed-pair-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed artifacts'
    status=read(out/'stage3-status.json');audit=read(out/'receipt-audit.json');execution=read(out/'execution.json')
    assert status['recorded_checks_total']==410 and audit['prior_formal_manifests_verified']==67
    assert execution['study_exit_code']==execution['audit_exit_code']==execution['tests_exit_code']==0
    assert execution['initial_study_exit_code']==1
    prior=read(source/'sha256.json')
    for rel,expected in prior.items():assert digest(ROOT/rel)==expected,rel
    files=['research/reconnect/'+name for name in ['reconstructed_path_rows.py','test_reconstructed_path_rows.py',
        'reconstructed_pair_study.py','reconstructed_pair_study_initial_norm_assert.py',
        'audit_reconstructed_pair.py','package_reconstructed_pair.py','RECONSTRUCTED_PATH_PAIR_CONTROL.md']]
    entries=dict(prior);source_seal=str((source/'sha256.json').relative_to(ROOT));entries[source_seal]=digest(source/'sha256.json')
    for rel in files:entries[rel]=digest(ROOT/rel)
    for directory in [out,aborted]:
        for p in directory.iterdir():
            assert p.is_file(),p
            entries[str(p.relative_to(ROOT))]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)
    for directory in [out,aborted]:selected.update(str(p.relative_to(ROOT)) for p in directory.iterdir())
    selected.update([source_seal,'runs/reconnect-point-contract-review/gate-spec.json',
        'research/reconnect/POINT_CANDIDATE_INFORMATION_REVIEW.md','research/reconnect/periodic_density_path.py',
        'research/reconnect/endpoint_density_path.py','research/reconnect/positive_kernel_2d.py',
        'research/reconnect/audit_route_decision.py','docs/RESEARCH_STATUS.md'])
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'] if not item['path'].endswith('.npz'))
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=(
        'LTN one predeclared full-support frozen pair: original N128, IDs63/64, D=.02, h=.04.\n'
        'Fixed nearest-point density path conductance, original material-mean c0, original reference and fixed norm.\n'
        '663 shared edges, 332 neighbors per row. This is NOT a full graph or finite-time result.\n'
        'Seven new adapter checks passed; recorded checks 403 -> 410. No historical suite rerun.\n'
        'First attempt aborted after edge construction because a recomputed BLAS norm differed by one ulp.\n'
        'Initial source, copied inputs, log and abort record retained. Correction uses the protocol-locked denominator.\n'
        'No physical input, formula, kernel, target, h, density, c0 or acceptance threshold was changed.\n'
        'Standalone replay (NumPy/SciPy): python research/reconnect/reconstructed_pair_study.py --replay runs/reconnect-reconstructed-pair\n'
        'Replay reconstructs exactly the same pair and verifies all edge receipts and saved comparison; not a new flow case.\n'
        'pair-manifest.json verifies included files. Cumulative seal also refers to omitted historical workspace files.\n'
        'Full source audit needs the historical workspace and attachment manifests, not included here.\n'
        '782 originals, 15 attachments, 67 prior formal manifests preserved. Four historical diagnostic differences retained.\n'
        'docs/RESEARCH_STATUS.md is mutable and excluded from cumulative science seal.\n'
        'No clipping, renormalization, fitting, scalar tuning, full graph, transient, new PDE solve, PDF, release or publication.\n'
        'Stage 3 incomplete, finite-node candidate unadmitted, fluid ORC disabled. Old v0.8.0 not recovered.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('pair-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'pair-manifest.json','DELIVERY_README.txt'}
        for rel,expected in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==expected,rel
        with tempfile.TemporaryDirectory(prefix='ltn-reconstructed-pair-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/reconstructed_pair_study.py'),
                '--replay','runs/reconnect-reconstructed-pair'],cwd=directory,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=600)
            assert completed.returncode==0,completed.stdout+completed.stderr
    for rel,expected in read(seal).items():assert digest(ROOT/rel)==expected,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_same_pair_replay_exit_code':completed.returncode,
        'fresh_same_pair_replay_log':completed.stdout,'archive_entries':len(manifest),'scientific_manifest_entries':len(entries),
        'source_scientific_manifest':source_seal,'new_adapter_checks':7,'recorded_checks_total':410,
        'previous_checks_rerun':0,'new_distinct_edge_weights':663,'new_target_rows':2,'new_fixed_snapshot_diagnostics':1,
        'initial_aborted_norm_check_preserved':True,'construction_attempts_before_replay':2,
        'new_full_candidate_graphs':0,'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':67,
        'decision':status['decision'],'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete',
        'fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with (exports/'ltn-reconstructed-pair-downloads.md').open('x') as f:
        f.write('# 原快照完整两行路径候选诊断\n\n[代码、原输入副本、逐边记录、对照及审计包](ltn-reconstructed-pair.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[交付范围与校验](ltn-reconstructed-pair-delivery.json)。全新解压目录同一两行完整重建与比较重放通过。\n\n'
            '新增7项软件检查，总记录410；仅一个原快照两行诊断，没有全图/瞬态/PDE新求解。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
