"""Seal a read-only endpoint interpretation and replay its saved-data algebra."""
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
    source=ROOT/'runs/reconnect-point-fixed-time';out=ROOT/'runs/reconnect-point-endpoint-review'
    archive=ROOT/'exports/ltn-point-endpoint-review.zip'
    delivery=ROOT/'exports/ltn-point-endpoint-review-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed evidence'
    execution=read(out/'execution.json');audit=read(out/'review-audit.json');status=read(out/'stage3-status.json')
    assert execution['review_exit_code']==execution['audit_exit_code']==0 and execution['first_attempts_passed']
    assert audit['prior_formal_manifests_verified']==74 and status['recorded_checks_total']==428
    assert not status['new_numerical_gate_selected'] and status['new_candidate_transient_runs']==0
    assert not status['fluid_orc_enabled'] and status['stage3_physical_acceptance']=='incomplete'
    prior=read(source/'sha256.json')
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+name for name in ['point_endpoint_review.py','audit_point_endpoint_review.py',
        'package_point_endpoint_review.py','POINT_ENDPOINT_ROUTE_REVIEW.md']]
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
        'LTN READ-ONLY endpoint variance interpretation and route closure. NO NEW PHYSICAL CASE OR PROPAGATION.\n'
        'The unfavorable point result is unchanged: endpoint error1.1899023366912084e-3, 43.6759 times saved ordinary FV.\n'
        'Exact algebra on saved binary values: DeltaE=<reference-mean(reference),error>+0.5||error||_m^2-0.5M*(Delta mean)^2.\n'
        'Point H bias -0.001102552267316775; signed cross contribution -0.0011011363997459117; square contribution -1.415867570863198e-6.\n'
        'This is an endpoint observable identity, NOT a causal split of quadrature/projection mechanisms.\n'
        'Original means, masks, endpoints, masses and c0 are copied unchanged; no mass/mean/state renormalization or fit.\n'
        'H is resolved material-mean mixing, not full continuous subcell dissipation or its time integral.\n'
        'Duhamel discussion states missing residual history; no new time points, graph actions or reference evaluation.\n'
        'Pause this fixed-contact point propagation line; keep limited FV baseline and all unfavorable results.\n'
        'Next science: a distinct material-average closure/input/defect proposal, not repeated packaging or h/grid/weight repair.\n'
        'No new numerical gate selected. General positive exchange not universally ruled out; no new feasible formula claimed.\n'
        'Replay needs standard Python plus NumPy (recorded NumPy2.3.5); no SciPy, graph or time solver imported.\n'
        'PYTHONDONTWRITEBYTECODE=1 python research/reconnect/point_endpoint_review.py --replay runs/reconnect-point-endpoint-review\n'
        'Replay verifies exact saved endpoint algebra/regions and route records only.\n'
        '0 new software checks, recorded428; prior suites not rerun. Exact data identity audit is not a new physical test or proof certificate.\n'
        '782 originals,15 attachments,74 prior formal scientific manifests preserved; four historical diagnostic differences retained.\n'
        'review-manifest.json covers included files; cumulative scientific seal also references omitted history.\n'
        'Full preservation audit requires original workspace. Mutable docs/RESEARCH_STATUS.md is outside science seal.\n'
        'No PDF, release, commit/push, environment configuration change or publication. Old v0.8.0 not recovered.\n'
        'Stage3 incomplete; finite-node/contact-transient physical admission false; fluid ORC disabled.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-endpoint-review-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/point_endpoint_review.py'),
                '--replay','runs/reconnect-point-endpoint-review'],cwd=directory,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
            assert completed.returncode==0,completed.stdout+completed.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    assert digest(ROOT/'docs/RESEARCH_STATUS.md')==manifest['docs/RESEARCH_STATUS.md']
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_saved_data_algebra_replay_exit_code':completed.returncode,
        'fresh_saved_data_algebra_replay_log':completed.stdout,'fresh_replay_files_unchanged':True,
        'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'source_scientific_manifest':source_seal,
        'new_software_checks':0,'recorded_checks_total':428,'previous_checks_rerun':0,
        'new_candidate_weights':0,'new_operator_actions':0,'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,
        'new_numerical_or_flow_cases':0,'new_exact_algebra_certificate_groups':0,
        'fixed_contact_point_propagation_line_paused':True,'new_numerical_gate_selected':False,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':74,
        'finite_node_candidate_admitted':False,'contact_transient_physically_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with (ROOT/'exports/ltn-point-endpoint-review-downloads.md').open('x') as f:
        f.write('# 有限时间终态解释与路线收束\n\n'
            '[报告、原终态小包、精确有符号预算及可重放代码](ltn-point-endpoint-review.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[范围与校验](ltn-point-endpoint-review-delivery.json)。全新解压的保存数据代数重放通过，没有运行传播。\n\n'
            '暂停当前固定配置的接触点图传播；下一项先给出不同物质平均闭合与理论契约，未选择新数值门槛。\n\n'
            '本轮新软件检查/物理案例均0，记录428；Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
