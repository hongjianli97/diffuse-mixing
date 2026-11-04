"""Seal the capacity theorem, one input audit and the original Stage3 criteria map."""
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
    source=ROOT/'runs/reconnect-laguerre-mass-feasibility';out=ROOT/'runs/reconnect-capacity-defect'
    archive=ROOT/'exports/ltn-capacity-defect-stage3.zip'
    delivery=ROOT/'exports/ltn-capacity-defect-stage3-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed evidence'
    execution=read(out/'execution.json');audit=read(out/'capacity-audit.json');status=read(out/'stage3-status.json')
    assert execution['study_exit_code']==execution['standalone_algebra_exit_code']==execution['audit_exit_code']==0
    assert execution['first_attempts_passed'] and audit['prior_formal_manifests_verified']==77
    assert status['recorded_checks_total']==428 and status['new_exact_algebra_certificate_groups']==8
    assert status['new_saved_input_diagnostics']==1 and not status['candidate_PDE_accuracy_evaluated']
    gate=read(out/'next-gate-spec.json');ledger=read(out/'acceptance-ledger.json')
    assert not gate['already_executed'] and gate['candidate_state_actions_max']==1
    assert not ledger['full_stage3_pass_asserted'] and not ledger['new_numeric_physical_threshold_defined']
    prior=read(source/'sha256.json')
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+name for name in ['capacity_defect_theory.py','capacity_defect_study.py',
        'audit_capacity_defect.py','package_capacity_defect.py','CAPACITY_DEFECT_SEMIGROUP_BOUND.md',
        'CAPACITY_INPUT_AND_STAGE3.md']]
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
        'LTN capacity defect theory, ONE existing-input diagnostic, and source-based Stage3 acceptance ledger.\n'
        'Actual m/rho/points unchanged. Auxiliary m_tilde is comparison only, never installed into an actual solver.\n'
        'Same fixed L: A-B=diag(m_tilde/m-1)*B; uniform time difference <= eta*(2+R)*original initial fluctuation norm.\n'
        'This controls capacity only; NOT full PDE error, moving-cell transport, or a new physical pass tolerance.\n'
        'Existing N128 compression midpoint, index1/t.025 only. eta2.4530545699940223e-5; R1.000048423202082.\n'
        'Capacity comparison coefficient7.359282494739227e-5; raw auxiliary-original total mass -3.174332799260325e-6 retained.\n'
        'Current input screen reads only original ids/m/positions/rho/time. No c0/PDE response, true face/area/J used in the screen.\n'
        '8 new exact algebra groups, separate from software record428. One saved input screen, no new physical flow case.\n'
        'No conductances, graph actions, scalar propagation, reference solve or partition optimizer executed.\n'
        'python -S research/reconnect/capacity_defect_theory.py --replay runs/reconnect-capacity-defect\n'
        'Replay uses standard Python only. NumPy was needed only to extract/read selected original NPZ members.\n'
        'Includes original midpoint-observations.npz byte-for-byte for traceability and the FUTURE fixed-state comparison.\n'
        'That original NPZ also contains old c0/reference fields; those were NOT read by the current capacity screen.\n'
        'next-gate-spec.json is UNEXECUTED: one same-snapshot auxiliary-face FV action, original m/c0/D, saved independent rate.\n'
        'Coefficients must be finalized before accessing scalar/reference; no mass replacement, full matrix, new PDE or time scan.\n'
        'Stage3 criteria mapped to unchanged docs/METHOD.md and CODEX_TASK.md; same backend T1-T4 plus density-source comparison.\n'
        'Original at least3 spatial/time levels apply to full acceptance; this input diagnostic does not count as convergence.\n'
        'T5/T6 large DNS and dynamic ORC are later-stage work, not added as Stage3 prerequisites. Unsupported scope remains explicit.\n'
        'No reliable calendar completion date; no numerical physical threshold invented or changed.\n'
        'Finite milestones: one fixed action, then controlled same-backend physics if warranted, then density-source/acceptance closure.\n'
        'Adverse or unresolved candidate result stops the route without parameter tuning or another indefinite theory branch.\n'
        '782 originals,15 attachments,77 prior formal manifests preserved; four historical diagnostic differences retained.\n'
        'review-manifest.json covers included files; cumulative science seal also references omitted history.\n'
        'Full preservation audit needs original workspace; mutable docs/RESEARCH_STATUS.md is outside science seal.\n'
        'Old v0.8.0 not recovered. No PDF, release, commit/push, environment change or publication.\n'
        'Stage3 incomplete; finite-node/contact-transient admission false; fluid ORC disabled.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-capacity-stage3-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,'-S',str(Path(directory)/'research/reconnect/capacity_defect_theory.py'),
                '--replay','runs/reconnect-capacity-defect'],cwd=directory,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
            assert completed.returncode==0,completed.stdout+completed.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    assert digest(ROOT/'docs/RESEARCH_STATUS.md')==manifest['docs/RESEARCH_STATUS.md']
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_standard_library_replay_exit_code':completed.returncode,
        'fresh_standard_library_replay_log':completed.stdout,'fresh_replay_files_unchanged':True,
        'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'source_scientific_manifest':source_seal,
        'new_software_checks':0,'recorded_checks_total':428,'previous_checks_rerun':0,'new_exact_algebra_certificate_groups':8,
        'new_saved_input_diagnostics':1,'new_numerical_or_flow_cases':0,'new_candidate_weights':0,
        'new_operator_actions':0,'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,
        'original_Stage3_criteria_indexed_without_new_physical_threshold':True,'next_fixed_action_executed':False,
        'original_midpoint_source_NPZ_included_unchanged':True,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':77,
        'finite_node_candidate_admitted':False,'contact_transient_physically_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with (ROOT/'exports/ltn-capacity-defect-stage3-downloads.md').open('x') as f:
        f.write('# 容量预算、保存输入与Stage 3路径\n\n'
            '[理论、输入、精确预算、原验收规范与未执行单次作用协议](ltn-capacity-defect-stage3.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[范围与校验](ltn-capacity-defect-stage3-delivery.json)。全新解压标准库重放通过，未执行候选作用。\n\n'
            '下一项为原N128压缩中点的一次辅助FV冻结作用判别；完整Stage3仍需同后端物理与密度来源验收。\n\n'
            '软件记录428，另列8组代数和一个保存输入诊断；Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
