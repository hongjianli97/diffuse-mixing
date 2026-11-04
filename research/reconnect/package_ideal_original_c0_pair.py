"""Seal and independently extract/replay the same two-rule oracle reference."""
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
    source=ROOT/'runs/reconnect-global-residual-review';out=ROOT/'runs/reconnect-ideal-original-c0-pair'
    archive=ROOT/'exports/ltn-ideal-original-c0-pair.zip'
    delivery=ROOT/'exports/ltn-ideal-original-c0-pair-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed evidence'
    execution=read(out/'execution.json');audit=read(out/'reference-audit.json');status=read(out/'stage3-status.json')
    assert execution['study_exit_code']==execution['tests_exit_code']==0
    assert read(out/'audit-execution.json')['audit_exit_code']==0
    assert audit['prior_formal_manifests_verified']==71 and status['recorded_checks_total']==423
    assert status['new_candidate_weights']==status['new_candidate_transient_runs']==0
    prior=read(source/'sha256.json')
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+name for name in ['ideal_material_pair_reference.py','test_ideal_material_pair_reference.py',
        'ideal_original_c0_pair_study.py','audit_ideal_original_c0_pair.py','package_ideal_original_c0_pair.py','IDEAL_ORIGINAL_C0_PAIR_REFERENCE.md']]
    source_seal=str((source/'sha256.json').relative_to(ROOT));entries=dict(prior);entries[source_seal]=digest(source/'sha256.json')
    for rel in files:entries[rel]=digest(ROOT/rel)
    for p in out.iterdir():
        assert p.is_file(),p
        entries[str(p.relative_to(ROOT))]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files);selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([source_seal,'research/reconnect/audit_route_decision.py','docs/RESEARCH_STATUS.md'])
    selected.update('research/reconnect/'+name for name in ['oracle_path_reference.py','contact_continuous.py',
        'contact_frozen_probe.py','positive_kernel_2d.py','trajectory_chain.py'])
    selected.update(v['path'] for v in read(out/'evidence-index.json')['sources'] if not v['path'].endswith('.npz'))
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=(
        'LTN original-c0 ideal material-pair two-row reference, extra known geometry/full density ORACLE ONLY.\n'
        'Original IDs63/64, N128, D=.02, h=.04, unchanged material means/masses and saved comparators.\n'
        'Fixed 4D tensor Gauss8/16, fine16 primary; all 420 neighbor material cells per row, 58,490,880 node pairs.\n'
        'No candidate weights, full graph, FV replacement, probability normalization or scalar propagation.\n'
        'Point-pair/projection terms oppose with correlation -.99839557936; terms are not global causal shares.\n'
        'Ideal-reference pair error is higher than current point pair; no improvement claim or coefficient repair follows.\n'
        'Replay (NumPy/SciPy): python research/reconnect/ideal_original_c0_pair_study.py --replay runs/reconnect-ideal-original-c0-pair\n'
        'Replay recomputes only the same two predeclared rules and rows, verifies raw quadrature records and signed budget.\n'
        'Independent cell-major audit reassembles the same points with a different loop/reduction, not a new physical case.\n'
        'Six new reference implementation checks passed; recorded checks 417 -> 423, old suites not rerun.\n'
        '782 originals, 15 attachments, 71 prior formal manifests preserved; four historical diagnostic differences retained.\n'
        'Included original input copies support portable numerical replay. Full source/attachment audit needs historical workspace.\n'
        'reference-manifest.json covers included files; cumulative science seal also references omitted historical files.\n'
        'docs/RESEARCH_STATUS.md is mutable and excluded from cumulative scientific seal.\n'
        'No h/grid/state/order expansion, clipping, fitting, threshold changes, PDF, release, commit/push, environment change or publication.\n'
        'Stage 3 incomplete, finite-node candidate unadmitted, fluid ORC disabled. Old v0.8.0 not recovered.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('reference-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'reference-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-ideal-c0-pair-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/ideal_original_c0_pair_study.py'),
                '--replay','runs/reconnect-ideal-original-c0-pair'],cwd=directory,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=600)
            assert completed.returncode==0,completed.stdout+completed.stderr
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    result=read(out/'results.json')
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_same_reference_replay_exit_code':completed.returncode,
        'fresh_same_reference_replay_log':completed.stdout,'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),
        'source_scientific_manifest':source_seal,'new_software_checks':6,'recorded_checks_total':423,'previous_checks_rerun':0,
        'reference_only_fixed_configurations':1,'reference_rows':2,'tensor_Gauss_orders':[8,16],'reference_kernel_node_pairs':58490880,
        'point_quadrature_projection_correlation':result['point_quadrature_projection_correlation'],
        'new_candidate_weights':0,'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':71,
        'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with (ROOT/'exports/ltn-ideal-original-c0-pair-downloads.md').open('x') as f:
        f.write('# 原c0两行理想单元对参考\n\n'
            '[报告、原输入、8/16阶完整记录、代码与审计](ltn-ideal-original-c0-pair.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[交付范围与校验](ltn-ideal-original-c0-pair-delivery.json)。新解压目录同一两行两规则参考重放通过。\n\n'
            '新增6项软件检查，记录423；原候选权重/全域结果不变，无传播或新PDE。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
