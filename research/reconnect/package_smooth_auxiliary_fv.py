"""Seal one smooth FV comparison and replay the same coefficients and state."""
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
    source=ROOT/'runs/reconnect-capacity-defect';out=ROOT/'runs/reconnect-smooth-auxiliary-FV-frozen'
    archive=ROOT/'exports/ltn-smooth-auxiliary-FV-frozen.zip'
    delivery=ROOT/'exports/ltn-smooth-auxiliary-FV-frozen-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed evidence'
    execution=read(out/'execution.json');audit=read(out/'frozen-audit.json');audit_run=read(out/'audit-execution.json')
    status=read(out/'stage3-status.json');result=read(out/'results.json');gate=read(out/'next-gate-spec.json')
    assert execution['tests_exit_code']==execution['study_exit_code']==audit_run['audit_exit_code']==0
    assert execution['test_attempts']==audit_run['audit_attempts']==2
    assert not execution['first_test_attempt_passed'] and not audit_run['first_audit_attempt_passed']
    assert execution['main_study_first_attempt_passed'] and not audit_run['scientific_main_rerun']
    assert audit['prior_formal_manifests_verified']==78 and status['recorded_checks_total']==433
    assert result['comparison_decision']=='FV_better_than_saved_kernel_resolvably'
    assert not gate['already_executed'] and gate['one_saved_state_max']==1 and gate['shock_extension_deferred']
    assert not status['general_nontensor_smooth_flow_admitted'] and not status['fluid_orc_enabled']
    prior=read(source/'sha256.json')
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+name for name in ['auxiliary_tensor_fv.py','test_auxiliary_tensor_fv.py',
        'smooth_auxiliary_fv_study.py','audit_smooth_auxiliary_fv.py','package_smooth_auxiliary_fv.py',
        'SMOOTH_AUXILIARY_FV_FROZEN.md']]
    source_seal=str((source/'sha256.json').relative_to(ROOT));entries=dict(prior);entries[source_seal]=digest(source/'sha256.json')
    for rel in files:entries[rel]=digest(ROOT/rel)
    for p in out.iterdir():
        assert p.is_file(),p
        entries[str(p.relative_to(ROOT))]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files);selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([source_seal,'research/reconnect/audit_route_decision.py','docs/RESEARCH_STATUS.md'])
    selected.update(v['path'] for v in read(out/'evidence-index.json')['sources'])
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=(
        'LTN SMOOTH NONSHOCK FIRST: one original N128 compression midpoint frozen-state auxiliary FV comparison.\n'
        'FAVORABLE RESULT: FV relative rate error2.1887199442374976e-4; saved same-state kernel6.874702103311877e-3; ratio31.409692781.\n'
        'Same original material mass means, actual m, IDs, positions/rho, D.02 and periodic boundary. No new PDE/reference or flow definition.\n'
        'Normalization is the same full PDE rate norm0.15851796857948153, including density drift, not an endpoint fluctuation norm.\n'
        'Coefficients depend only on allowed points/IDs/rho/D/period/thickness and were sealed before scalar/reference loading.\n'
        '32768 shared positive auxiliary midpoint faces, ordinary central resistance; no true material face inputs or mass replacement.\n'
        'Exact Fraction action versus independent directional float control difference2.494495190413887e-16 in mass norm.\n'
        'Error gain0.006655830108888127 exceeds reference/arithmetic indicator2.710688820034055e-10; not a certified full physical bound.\n'
        'Exact shared mass and energy pairing verified; raw float mass residual -6.313888710349739e-20 retained.\n'
        'No scalar propagation or I/H prediction; one frozen tensor snapshot is NOT general smooth-flow or Stage3 acceptance.\n'
        '5 new checks ultimately passed; recorded433, previous428 not rerun.\n'
        'Initial test failed only on expected ValueError versus actual forbidden-c0 signature TypeError; no backend change.\n'
        'Initial audit had a parenthesis syntax error; corrected, both failure logs retained, main scientific run unchanged.\n'
        'Main study first attempt3.8405s/93.4219MiB within600s/512MiB; replay repeats same case, not another physical case.\n'
        'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/smooth_auxiliary_fv_study.py --replay runs/reconnect-smooth-auxiliary-FV-frozen\n'
        'Replay requires NumPy (recorded2.3.5), rebuilds exact same faces/coefficients, exact and independent float actions and metrics.\n'
        'Current builder supports complete periodic tensor sets only; nonproduct shear requires general point-defined periodic Voronoi geometry.\n'
        'Next gate UNEXECUTED: existing N128 smooth shear t.05/s.25, one fixed state, full mixed/shear/density reference.\n'
        'Lock deterministic geometry and controls first; no point jitter, deletion, area/mass/rho normalization or weight clipping.\n'
        'Includes original current and future-shear source NPZ files unchanged; future shear candidate has NOT been evaluated.\n'
        'Shock extension deferred per user; original complete Stage3 conditions unchanged and fluid ORC disabled.\n'
        '782 originals,15 attachments,78 prior formal scientific manifests preserved; four known historical diagnostic differences retained.\n'
        'review-manifest.json covers included files; cumulative scientific seal also references omitted history.\n'
        'Full preservation audit needs original workspace. Mutable docs/RESEARCH_STATUS.md is outside science seal.\n'
        'No PDF, release, commit/push, environment change or publication; old v0.8.0 not recovered.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-smooth-FV-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/smooth_auxiliary_fv_study.py'),
                '--replay','runs/reconnect-smooth-auxiliary-FV-frozen'],cwd=directory,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),
                capture_output=True,text=True,timeout=600)
            assert completed.returncode==0,completed.stdout+completed.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    assert digest(ROOT/'docs/RESEARCH_STATUS.md')==manifest['docs/RESEARCH_STATUS.md']
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_action_replay_exit_code':completed.returncode,
        'fresh_action_replay_log':completed.stdout,'fresh_replay_files_unchanged':True,
        'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'source_scientific_manifest':source_seal,
        'FV_relative_rate_error':result['FV_relative_rate_error'],'saved_kernel_relative_rate_error':result['saved_kernel_relative_rate_error_recomputed'],
        'saved_kernel_over_FV_error_ratio':result['saved_kernel_over_FV_error_ratio'],
        'comparison_decision':result['comparison_decision'],'new_software_checks':5,'recorded_checks_total':433,
        'previous_checks_rerun':0,'test_attempts':2,'audit_attempts':2,'initial_failures_preserved':True,
        'scientific_main_first_attempt_passed':True,'new_actual_frozen_state_cases':1,'new_candidate_graphs':1,
        'new_shared_conductances':32768,'independent_float_action_controls':1,
        'new_scalar_transient_runs':0,'new_PDE_reference_solves':0,'new_physical_flow_definitions':0,
        'next_nonproduct_shear_gate_executed':False,'shock_extension_deferred':True,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':78,
        'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with (ROOT/'exports/ltn-smooth-auxiliary-FV-frozen-downloads.md').open('x') as f:
        f.write('# 光滑流辅助FV首次空间作用\n\n'
            '[报告、原状态/参考、原始电导/作用、检查与可重放代码](ltn-smooth-auxiliary-FV-frozen.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[范围与校验](ltn-smooth-auxiliary-FV-frozen-delivery.json)。全新解压同一候选作用重放通过。\n\n'
            'FV变化率误差2.18872e-4，保存核6.87470e-3，约低31.4倍；仅一个张量快照，不代表一般光滑流或瞬态验收。\n\n'
            '下一项为原二维周期剪切非张量点云，未执行；含激波推广后置。新检查5项，累计433，Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
