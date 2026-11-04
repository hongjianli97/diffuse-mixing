"""Seal a saved-vector review and replay its budget and future declarations."""
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
    source=ROOT/'runs/reconnect-reconstructed-global-frozen';out=ROOT/'runs/reconnect-global-residual-review'
    archive=ROOT/'exports/ltn-global-residual-review.zip'
    delivery=ROOT/'exports/ltn-global-residual-review-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed evidence'
    execution=read(out/'execution.json');audit=read(out/'review-audit.json');status=read(out/'stage3-status.json')
    assert execution['review_exit_code']==execution['scope_decision_exit_code']==execution['audit_exit_code']==0
    assert audit['prior_formal_manifests_verified']==70 and status['recorded_checks_total']==417
    assert status['ideal_reference_result']=='not_computed' and status['new_weights']==0
    prior=read(source/'sha256.json')
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+name for name in ['global_residual_review.py','global_residual_scope_decision.py',
        'audit_global_residual_review.py','package_global_residual_review.py','RECONSTRUCTED_PATH_RESIDUAL_SCOPE.md']]
    source_seal=str((source/'sha256.json').relative_to(ROOT));entries=dict(prior)
    entries[source_seal]=digest(source/'sha256.json')
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
        'LTN saved-vector residual scope review ONLY. No new candidate or ideal reference executed.\n'
        'Same original mass-average state, IDs, observation, PDE rate, point positions/density and regions match by member hashes.\n'
        'Fixed split: (saved point - saved fine continuous oracle) + (oracle - PDE) + uncorrected arithmetic closure.\n'
        'Global component norms .005311758993169808 and .031229167742325044; positive correlation .5937649258670508.\n'
        'These norms are not additive error percentages; combined point-pair/projection remainder is not pure projection error.\n'
        'Next gate NOT RUN: original c0 IDs63/64 ideal material-pair aggregate reference, extra geometry/density oracle only.\n'
        'Fixed tensor Gauss8/16, full 420 neighbor material cells per target, candidate coefficients and all old results untouched.\n'
        'Vector replay (NumPy only): python research/reconnect/global_residual_review.py --replay runs/reconnect-global-residual-review\n'
        'No kernel/path/graph/quadrature/propagation or old proof/test suite is called by that replay.\n'
        'New software checks 0, recorded checks 417; no new algebra certificate groups or flow/numerical cases.\n'
        '782 originals, 15 attachments and 70 prior formal manifests preserved; four historical diagnostic differences retained.\n'
        'Included saved vectors permit independent budget replay; full historical source/attachment audit requires original workspace.\n'
        'review-manifest.json covers included files; cumulative science seal also references omitted history.\n'
        'docs/RESEARCH_STATUS.md is a mutable snapshot outside the cumulative scientific seal.\n'
        'No clipping, renormalization, fitting, threshold changes, PDF, release, commit/push, environment change or publication.\n'
        'Stage 3 incomplete, finite-node candidate unadmitted, fluid ORC disabled. Old v0.8.0 not recovered.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-residual-review-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/global_residual_review.py'),
                '--replay','runs/reconnect-global-residual-review'],cwd=directory,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
            assert completed.returncode==0,completed.stdout+completed.stderr
            check=subprocess.run([sys.executable,'-S','-c',
                'import sys,json;sys.path.insert(0,"research/reconnect");from global_residual_scope_decision import next_gate;from pathlib import Path;assert next_gate()==json.loads(Path("runs/reconnect-global-residual-review/next-gate-spec.json").read_text());print("Future reference gate declarations replay exactly; gate not executed.")'],
                cwd=directory,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
            assert check.returncode==0,check.stdout+check.stderr
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_saved_vector_replay_exit_code':completed.returncode,
        'fresh_saved_vector_replay_log':completed.stdout,'fresh_gate_declaration_replay_exit_code':check.returncode,
        'fresh_gate_declaration_replay_log':check.stdout,'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),
        'source_scientific_manifest':source_seal,'new_software_checks':0,'recorded_checks_total':417,'previous_checks_rerun':0,
        'new_weights':0,'new_operator_or_quadrature_evaluations':0,'new_numerical_or_flow_cases':0,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,
        'next_ideal_pair_reference_executed':False,'original_files_preserved':782,'historical_attachments_preserved':15,
        'prior_formal_manifests_preserved':70,'finite_node_candidate_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with (ROOT/'exports/ltn-global-residual-review-downloads.md').open('x') as f:
        f.write('# 已存全域残差：只读预算与下一项参考\n\n'
            '[报告、保存向量、审计与未执行门槛](ltn-global-residual-review.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[交付范围与校验](ltn-global-residual-review-delivery.json)。独立保存向量重放和标准库门槛声明核对通过。\n\n'
            '本轮未运行核/路径/图/求积或传播，记录仍417；Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
