"""Seal the read-only coverage ledger and verify a fresh standard-library receipt."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure
from t3_coverage_review import SOURCE,ZERO_FIELDS


def main():
    out=ROOT/'runs/reconnect-t3-coverage';failed=ROOT/'runs/reconnect-t3-coverage-attempt1'
    archive=ROOT/'exports/ltn-t3-coverage.zip';delivery=ROOT/'exports/ltn-t3-coverage-delivery.json';downloads=ROOT/'exports/ltn-t3-coverage-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)),'closed evidence never overwritten'
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'review-audit.json');status=read(out/'stage3-status.json');r=read(out/'coverage-review.json')
    assert ex['review_exit_code']==ar['audit_exit_code']==0 and ex['final_review_passed'] and ar['audit_first_attempt_passed']
    assert not ex['review_first_attempt_passed'] and ex['review_failed_attempts_preserved']==1
    assert audit['receipt_consistent'] and audit['prior_formal_manifests_verified']==96
    assert status['recorded_checks_total']==483 and all(status[key]==0 for key in ZERO_FIELDS)
    assert not status['general_T3_admitted'] and not status['fluid_orc_enabled'] and not status['next_adapter_or_experiment_executed']
    gate=read(out/'next-gate-spec.json');assert not gate['already_executed'] and gate['new_candidate_graphs']==28 and not gate['new_geometry_implemented_this_round']
    assert digest(ROOT/'exports/ltn-t3-affine-spatial.zip')=='a886cc0d90428ab3266a925b3df40260afcb7469ec7713bc0efa314d2744d3d6'
    prior=read(SOURCE/'sha256.json');assert len(prior)==5450
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+name for name in ('t3_coverage_protocol.json','t3_coverage_review.py','audit_t3_coverage.py','document_t3_coverage.py',
      'package_t3_coverage.py','T3_COVERAGE_REVIEW.md','AFFINE_SHEAR_DOMAIN_CONTRACT.md')]
    entries=dict(prior);source_seal=str((SOURCE/'sha256.json').relative_to(ROOT));entries[source_seal]=digest(SOURCE/'sha256.json')
    for rel in files:assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for folder in (out,failed):
        for p in folder.rglob('*'):
            if p.is_file():
                rel=str(p.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(p)
    # Preserve the historical indirectly SHA-bound file as a direct content entry now,
    # without changing its bytes or rewriting its historical parent manifest.
    for item in read(out/'evidence-index.json')['sources']:
        if not item['cumulative_content_entry_present']:
            assert item['path']=='docs/PHASE3_REBUILD.md' and item['path'] not in entries
            entries[item['path']]=digest(ROOT/item['path']);assert entries[item['path']]==item['sha256']
    save(seal,dict(sorted(entries.items())))
    selected=set(files)|dependency_closure([rel for rel in files if rel.endswith('.py')])
    selected.update(str(p.relative_to(ROOT)) for folder in (out,failed) for p in folder.rglob('*') if p.is_file())
    selected.update([source_seal,'docs/RESEARCH_STATUS.md'])
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'])
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note='''LTN T3 COVERAGE REVIEW: original criteria, immutable source lineage and saved results only.
Decision: retain_supported_controls_select_one_affine_shear_sliding_domain_control.
Seven implementation locks confirm identical original geometry/ordinary resistance/chronological action sources.
T1, three registered T2 controls, T3 orthogonal affine compression and nonlinear periodic sine shear positive evidence retained.
Original N32 inconclusive H decision retained; no old result rewritten or numerical path rerun.
Existing sine shear contains sin(2pi b) and is not an executed affine shear. Original METHOD T3 affine shear lacks a registered run in this reviewed chain.
Current domain API is axis-aligned / exact orthogonal only; inverse shear plus unchanged label Euclidean distances would change the PDE.
Original orthogonal direction and constantD/Gamma controls are already covered; no arbitrary-angle/flow scan added.
T4 Becker and stationary aligned contact FV baselines remain useful conditional references, not final point-pipeline acceptance.
Same final-backend oracle/reconstruction/ledger density-source comparison remains incomplete.
Research chronological exponential action is distinguished from original production implicit-Euler default and CLI/schema migration.
No new DNS, dynamic-ORC or old-v080 recovery prerequisite. Stage3 incomplete, fluid ORC off, kappa not computed.
Written affine-shear contract gives global B=[(1,0),(s,1)],s=5t,T=.05,0<=s<=.25,world Euclidean geometry and two tilted-mode references.
This is written design, not an implementation, generated conductance, numerical test or machine-verified algebra certificate.
Next gate only registered: opt-in exact sliding-periodic geometry and one N32/rho1/D=.02/K4,8,16 affine shear control.
28 graphs and3 actual/3 full independent paths, four software checks are FUTURE PLAN, NOT EXECUTED.
Original m/c0/IDs reused; all new coefficients must be sealed before scalar/reference access; old H/tail/algebra/seed retained.
No inverse-shear metric shortcut, F/C fit, clipping, repair, positive-face deletion, normalization, threshold relaxation or automatic scan.
Current new points/graphs/actions/paths/reference states/PDEs/software checks/algebra groups/LP solves all0; recorded checks remain483.
First review failed on assuming every historical input had a direct cumulative entry; PHASE3_REBUILD was bound by the immutable parent gate source SHA.
Failure code/lock/log/partial sources preserved; final receipt verifies this unique indirect chain. No source or scientific criterion changed.
782 originals,15 attachments,96 prior formal manifests and previous T3 spatial delivery preserved; four historical diagnostic differences retained.
Standard-library receipt:
PYTHONDONTWRITEBYTECODE=1 python -S research/reconnect/audit_t3_coverage.py --directory runs/reconnect-t3-coverage --receipt-only
Fresh receipt validates original quotes,source hashes,seven locks,AST scope and preserved decisions; no numerical backend import or replay.
review-manifest.json covers included files; cumulative science seal references omitted history. Full preservation audit needs original workspace.
Original README/METHOD/CODEX_TASK/PHASE,attachments,old code/results/reports unchanged; only mutable RESEARCH_STATUS updated.
No PDF, version, commit/push, environment change or publication. 2026-11-02 Asia/Shanghai.
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-t3-coverage-receipt-',dir='/tmp') as directory:
            z.extractall(directory)
            result=subprocess.run([sys.executable,'-S',str(Path(directory)/'research/reconnect/audit_t3_coverage.py'),
              '--directory','runs/reconnect-t3-coverage','--receipt-only'],cwd=directory,
              env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
            assert result.returncode==0,result.stdout+result.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'archive_CRC_and_all_included_SHA256_verified':True,'fresh_standard_library_receipt_exit_code':result.returncode,
      'fresh_standard_library_receipt_log':result.stdout,'fresh_receipt_files_unchanged':True,'old_numerical_paths_replayed':False,
      'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'review_decision':r['review_decision'],
      'same_backend_locks_verified':7,'acceptance_ledger_rows':len(r['acceptance_ledger']),
      'affine_compression_and_sine_shear_positive_evidence_retained':True,'original_N32_inconclusive_preserved':True,
      'current_affine_shear_execution_missing':True,'new_acceptance_criteria_added':False,'new_DNS_or_dynamicORC_prerequisite':False,
      'review_first_attempt_passed':False,'failed_review_code_sources_and_log_preserved':True,
      **dict.fromkeys(ZERO_FIELDS,0),'recorded_checks_total':483,'previous_checks_rerun':0,
      'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':96,
      'old_T3_spatial_delivery_unchanged':True,'next_gate_executed':False,'next_geometry_implemented':False,
      'next_gate_planned_new_graphs':28,'next_gate_planned_actual_paths':3,'next_gate_planned_software_checks':4,
      'proofs_are_written_not_machine_verified':True,'original_backends_changed':False,'general_T3_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# T3覆盖复核与仿射剪切滑移周期域契约\n\n[验收账本、逐行原要求、代码来源链、书面契约和未执行下一项](ltn-t3-coverage.zip)。\n\n'
          f'SHA-256：`{info["archive"]["sha256"]}`。\n\n[交付核验](ltn-t3-coverage-delivery.json)。CRC/SHA及全新解压标准库收据核验通过，无数值重放。\n\n'
          '已存T1/T2/仿射压缩/正弦剪切证据保留。仿射剪切的滑移周期域控制仅登记，28图/3路径尚未执行；Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
