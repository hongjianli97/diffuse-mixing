"""Seal the read-only route review and replay its unexecuted gate records."""
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
    source=ROOT/'runs/reconnect-ideal-original-c0-pair';out=ROOT/'runs/reconnect-post-ideal-route'
    archive=ROOT/'exports/ltn-post-ideal-route-review.zip'
    delivery=ROOT/'exports/ltn-post-ideal-route-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed evidence'
    execution=read(out/'execution.json');audit=read(out/'review-audit.json');status=read(out/'stage3-status.json')
    assert execution['review_exit_code']==execution['audit_exit_code']==0
    assert audit['prior_formal_manifests_verified']==72 and status['recorded_checks_total']==423
    assert status['point_kernel_finite_time_result']=='not_computed' and status['new_candidate_transient_runs']==0
    prior=read(source/'sha256.json')
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+name for name in ['post_ideal_route_review.py','audit_post_ideal_route.py',
        'package_post_ideal_route.py','POST_IDEAL_ROUTE_DECISION.md']]
    source_seal=str((source/'sha256.json').relative_to(ROOT));entries=dict(prior);entries[source_seal]=digest(source/'sha256.json')
    for rel in files:entries[rel]=digest(ROOT/rel)
    for p in out.iterdir():
        assert p.is_file(),p
        entries[str(p.relative_to(ROOT))]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files);selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([source_seal,'research/reconnect/audit_route_decision.py','docs/RESEARCH_STATUS.md'])
    selected.update(v['path'] for v in read(out/'evidence-index.json')['sources'] if not v['path'].endswith('.npz'))
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=(
        'LTN read-only post-ideal route review. NO FINITE-TIME POINT RESULT COMPUTED.\n'
        'Stop ideal-reference order/global expansion; preserve point coefficients, ordinary FV baseline and all failures.\n'
        'Two-row opposing terms are not global causal shares or evidence of production/evolution admission.\n'
        'Matched existing T=1/20 PDE reference and FV state to all original IDs, masses, c0, positions and density by bytes.\n'
        'Next separate research gate: same existing point graph, original full c0, one T=.05 endpoint and resolved-material I/H.\n'
        'All 128 discrete transverse modes and raw eigensystems retained; no concentration/mean/eigenvalue repair.\n'
        'Independent uniformization plan Lambda=168, terms0..47, exact a priori omitted-tail bound; no Poisson normalization.\n'
        'Research diagnostic is not physical transient or backend acceptance; Stage 3 incomplete, fluid ORC disabled.\n'
        'Replay (NumPy only): python research/reconnect/post_ideal_route_review.py --replay runs/reconnect-post-ideal-route\n'
        'Replay verifies declarations, copied reference records and tail plan; no FFT, eigen solve, kernel, graph action or propagation.\n'
        'New software checks 0, recorded checks423; no old suite/proof rerun, new case, operator or quadrature.\n'
        '782 originals, 15 attachments, 72 prior formal manifests preserved; four historical diagnostic differences retained.\n'
        'Includes small matched original/reference/FV arrays and relevant saved coefficient cache; full history/source audit needs original workspace.\n'
        'review-manifest.json covers included files; cumulative science seal also references omitted historical files.\n'
        'docs/RESEARCH_STATUS.md is mutable and outside the cumulative scientific seal.\n'
        'No PDF, version, commit/push, environment modification or publication. Old v0.8.0 not recovered.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-post-ideal-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/post_ideal_route_review.py'),
                '--replay','runs/reconnect-post-ideal-route'],cwd=directory,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
            assert completed.returncode==0,completed.stdout+completed.stderr
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_route_record_replay_exit_code':completed.returncode,
        'fresh_route_record_replay_log':completed.stdout,'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),
        'source_scientific_manifest':source_seal,'new_software_checks':0,'recorded_checks_total':423,'previous_checks_rerun':0,
        'future_point_time_gate_executed':False,'new_candidate_weights':0,'new_operator_actions':0,
        'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,'new_numerical_or_flow_cases':0,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':72,
        'finite_node_candidate_admitted':False,'contact_transient_physically_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with (ROOT/'exports/ltn-post-ideal-route-downloads.md').open('x') as f:
        f.write('# 理想两行后的路线审查\n\n'
            '[路线报告、匹配参考、来源与未执行时间门槛](ltn-post-ideal-route-review.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[交付范围与校验](ltn-post-ideal-route-delivery.json)。全新解压目录的路线/参考记录及尾项规划重放通过，未传播点方法。\n\n'
            '本轮新软件检查/权重/算子作用/物理案例均0，记录423；Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
