"""Seal one unfavorable saved-point time case and replay a fresh extraction."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile

from audit_route_decision import ROOT,read,save,digest


def main():
    source=ROOT/'runs/reconnect-post-ideal-route';out=ROOT/'runs/reconnect-point-fixed-time'
    archive=ROOT/'exports/ltn-point-fixed-time.zip'
    delivery=ROOT/'exports/ltn-point-fixed-time-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed evidence'
    execution=read(out/'execution.json');audit=read(out/'time-audit.json')
    audit_run=read(out/'audit-execution.json');status=read(out/'stage3-status.json');result=read(out/'results.json')
    assert execution['tests_exit_code']==execution['study_exit_code']==audit_run['audit_exit_code']==0
    assert audit_run['audit_attempts']==2 and not audit_run['first_audit_attempt_passed']
    assert not audit_run['scientific_main_rerun'] and not audit_run['saved_weights_or_outputs_modified']
    assert audit['prior_formal_manifests_verified']==73 and status['recorded_checks_total']==428
    assert status['gate_executed'] and status['new_saved_point_graph_research_time_cases']==1
    assert result['comparison_decision']=='point_endpoint_worse_than_saved_FV_resolvably'
    assert not status['finite_node_candidate_admitted'] and not status['fluid_orc_enabled']
    assert not status['contact_transient_physically_admitted'] and status['stage3_physical_acceptance']=='incomplete'
    prior=read(source/'sha256.json')
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+name for name in ['saved_exchange_time.py','test_saved_exchange_time.py',
        'point_fixed_time_study.py','audit_point_fixed_time.py','package_point_fixed_time.py',
        'POINT_PATH_FIXED_TIME_COMPARISON.md']]
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
        'LTN saved point graph, original material-mean c0, ONE exact T=1/20 research endpoint.\n'
        'UNFAVORABLE RESULT: point error1.1899023366912084e-3; saved ordinary FV2.724389373882817e-5; ratio43.675927828.\n'
        'Resolved material-mean H: point0.10288659526472646; PDE0.10398914753204323. Not full subcell dissipation.\n'
        'Same original IDs/mass/c0/geometry/rho and existing PDE reference; copied original conductances, no new kernel/path/weights.\n'
        'All128 original transverse modes, raw65 eigensystems and raw positive numerical zero eigenvalue retained.\n'
        'Independent full-stencil uniformization Lambda168, n0..47; raw weights and inventory NOT renormalized.\n'
        'Solver/reference indicators are not certified total physical error bounds; initial bridge needs stated constant-cell lift.\n'
        'No old harmonic endpoint baseline; no finite-time improvement claim over it. No new h/grid/time/weight scan or repair.\n'
        '5 new software checks passed first attempt; recorded428, previous423 not rerun. Software count is not physics acceptance.\n'
        'Main solve passed first attempt. First audit failure retained: alternate parenthesization of Poisson recurrence differed in floating point.\n'
        'Only audit was corrected; both recurrence forms checked; no scientific solve rerun or saved weight/state modification.\n'
        'Replay requires Python3.12 with NumPy/SciPy (recorded2.3.5/1.17.0); no network, kernel build or new PDE solve.\n'
        'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/point_fixed_time_study.py --replay runs/reconnect-point-fixed-time\n'
        'Replay rebuilds same65 blocks and both actions, checks endpoint via fixed machine formula and exactly replays saved output metrics.\n'
        'Fresh replay repeats the same case, not another physical case. Delivery metadata stores actual exit code and output.\n'
        'review-manifest.json hashes every included file. Cumulative science seal also references omitted historical files.\n'
        '782 originals,15 attachments,73 prior formal manifests preserved; four known historical diagnostics retained.\n'
        'Full historical preservation audit needs original workspace; docs/RESEARCH_STATUS.md is mutable outside science seal.\n'
        'Next work: read-only interpretation/route review of saved evidence, no time/grid/h/weight repair.\n'
        'Stage3 incomplete; finite-node and contact-transient physical admission false; fluid ORC disabled.\n'
        'No PDF, release, commit/push, environment configuration change or publication. Old v0.8.0 not recovered.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-point-time-',dir='/tmp') as directory:
            z.extractall(directory)
            started=time.monotonic()
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/point_fixed_time_study.py'),
                '--replay','runs/reconnect-point-fixed-time'],cwd=directory,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),
                capture_output=True,text=True,timeout=600)
            replay_elapsed=time.monotonic()-started
            assert completed.returncode==0,completed.stdout+completed.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    assert digest(ROOT/'docs/RESEARCH_STATUS.md')==manifest['docs/RESEARCH_STATUS.md']
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_full_time_action_replay_exit_code':completed.returncode,
        'fresh_full_time_action_replay_seconds':replay_elapsed,'fresh_full_time_action_replay_log':completed.stdout,
        'fresh_full_time_action_replay_stderr':completed.stderr,'fresh_replay_input_and_saved_output_files_unchanged':True,
        'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),
        'source_scientific_manifest':source_seal,'new_software_checks':5,'recorded_checks_total':428,'previous_checks_rerun':0,
        'new_saved_point_graph_research_time_cases':1,'fresh_replay_does_not_add_physical_case':True,
        'new_candidate_weights':0,'new_kernel_builds':0,'new_density_path_builds':0,'new_PDE_reference_solves':0,
        'point_endpoint_relative_error':result['point_endpoint_relative_error'],
        'saved_FV_endpoint_error_context':result['saved_FV_endpoint_error_context'],
        'point_over_saved_FV_error_ratio':result['point_over_saved_FV_error_ratio'],
        'comparison_decision':result['comparison_decision'],'main_solve_first_attempt_passed':True,
        'audit_attempts':2,'first_audit_failure_preserved':True,'scientific_main_rerun_for_audit':False,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':73,
        'finite_node_candidate_admitted':False,'contact_transient_physically_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with (ROOT/'exports/ltn-point-fixed-time-downloads.md').open('x') as f:
        f.write('# 同一保存点图的固定时间对照\n\n'
            '[报告、原状态、原电导、完整块和终态、审计及可重放代码](ltn-point-fixed-time.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[范围与校验](ltn-point-fixed-time-delivery.json)。全新解压目录的两种时间作用和原始指标重放通过。\n\n'
            '点终态误差1.1899e-3，为保存普通FV的43.6759倍；保留不利结果，不调参修补。\n\n'
            '新增5项软件检查，记录428项；首次审计括号顺序失败保留，科学主计算未改。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
