"""Seal a read-only budget review and verify a freshly extracted standard-library receipt."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure
from t3_affine_budget_review import SOURCE,ZERO_FIELDS


def main():
    out=ROOT/'runs/reconnect-t3-affine-review';failed=ROOT/'runs/reconnect-t3-affine-review-attempt1'
    archive=ROOT/'exports/ltn-t3-affine-review.zip';delivery=ROOT/'exports/ltn-t3-affine-review-delivery.json';downloads=ROOT/'exports/ltn-t3-affine-review-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)),'sealed artifacts are never overwritten'
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'review-audit.json');status=read(out/'stage3-status.json');r=read(out/'results.json')
    assert ex['review_exit_code']==ar['audit_exit_code']==0 and ex['final_review_passed'] and ar['audit_first_attempt_passed']
    assert not ex['review_first_attempt_passed'] and ex['review_failed_attempts_preserved']==1
    assert audit['receipt_consistent'] and audit['prior_formal_manifests_verified']==94
    assert status['recorded_checks_total']==480 and all(status[k]==0 for k in ZERO_FIELDS)
    assert not status['original_T3_case_passed'] and not status['fluid_orc_enabled']
    gate=read(out/'next-gate-spec.json');assert not gate['already_executed'] and not gate['next_experiment_executed']
    assert digest(ROOT/'exports/ltn-t3-affine.zip')=='f58a50adc6c9b71e42351655330e5dc87670e425ce0564d44fab42bc1d213205'
    prior=read(SOURCE/'sha256.json');assert len(prior)==4483
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+n for n in ['t3_affine_budget_review.py','t3_affine_budget_review_protocol.json','audit_t3_affine_budget_review.py',
      'plot_t3_affine_budget_review.py','document_t3_affine_budget_review.py','package_t3_affine_budget_review.py','T3_AFFINE_BUDGET_REVIEW.md']]
    entries=dict(prior);source_seal=str((SOURCE/'sha256.json').relative_to(ROOT));entries[source_seal]=digest(SOURCE/'sha256.json')
    for rel in files:assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for folder in (out,failed):
        for p in folder.rglob('*'):
            if p.is_file():
                rel=str(p.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)|dependency_closure([p for p in files if p.endswith('.py')])
    selected.update(str(p.relative_to(ROOT)) for folder in (out,failed) for p in folder.rglob('*') if p.is_file())
    selected.update([source_seal,'docs/RESEARCH_STATUS.md'])
    selected.update(v['path'] for v in read(out/'evidence-index.json')['sources'] if Path(v['path']).suffix!='.zip')
    manifest={p:digest(ROOT/p) for p in sorted(selected)}
    note='''LTN T3 AFFINE BUDGET REVIEW: saved-data accounting only, no new physical experiment.
Decision: saved_N32_spatial_budget_dominates_time_only_expansion_not_selected.
Original N32 T3_affine_model_response_inconclusive retained; Stage3 incomplete and fluid ORC off.
t=.025: spatial term96.95129% of the original absolute state indicator, not a fraction of actual H error.
Spatial-only original-formula diagnostic .00506070588758 exceeds actual H gap .00497045808719 and saved continuous-grid H gap .00497245233399.
Original binary64 H indicator .005219845797588792 unchanged. No tighter formula, signed cancellation or retroactive acceptance.
All positive components and retained sequential model totals linked; exact regrouping roundoff is separate, not a negative physical contribution.
Only stored mass/c0 and already stored physical/grid arrays are read; no graph, action, new state, reference clock or PDE is evaluated.
Independent NPY decoding and80/100-digit Decimal second-moment variance validate saved observables against primary exact centered variance.
First review failed on an unsupported <=1ulp squared-sigma/variance assumption (observed1.0441499899528233ulp).
Failure code, lock, sources and log retained. Final review uses existing exact variance and certified fluctuation upper bound; no physical tolerance change.
New scientific points/graphs/weights/actions/paths/PDEs/reference states/software checks/algebra groups/LP solves all0; recorded checks remain480.
Next gate registered only: same affine models at N16/32/64, reuse six N32 paths, add N16/N64 with original K4/8/16.
112 new graphs and12 actual/12 independent paths are PLANNED, NOT EXECUTED; raw original material input sources hashed only.
Same original H formula and all raw numerical/input/residue/reference controls; common original N16 material observations and bridges retained.
No automatic N128/K32, new flow/model or parameter matrix; preserve future failure/unresolved.
782 originals,15 attachments,94 prior formal scientific manifests and previous full T3 delivery verified unchanged.
Four historical diagnostic manifest differences retained; old v0.8.0/246 still not recovered.
Fresh standard-library receipt (no numerical path replay):
PYTHONDONTWRITEBYTECODE=1 python -S research/reconnect/audit_t3_affine_budget_review.py --directory runs/reconnect-t3-affine-review --receipt-only
Included files checked by review-manifest.json; cumulative science seal references omitted history. Full historical audit needs original workspace.
No original README/METHOD/CODEX_TASK/PHASE, attachment, code, result or report changed; mutable RESEARCH_STATUS alone updated.
No PDF, version, commit/push, environment change or publication. General T3/smooth flow/finite-node admission false; kappa not computed.
2026-11-02 Asia/Shanghai.
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-t3-budget-receipt-',dir='/tmp') as directory:
            z.extractall(directory)
            result=subprocess.run([sys.executable,'-S',str(Path(directory)/'research/reconnect/audit_t3_affine_budget_review.py'),
              '--directory','runs/reconnect-t3-affine-review','--receipt-only'],cwd=directory,
              env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
            assert result.returncode==0,result.stdout+result.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'archive_CRC_and_all_included_SHA256_verified':True,'fresh_standard_library_receipt_exit_code':result.returncode,
      'fresh_standard_library_receipt_log':result.stdout,'fresh_receipt_files_unchanged':True,'old_numerical_paths_replayed':False,
      'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'decision':r['decision'],
      'original_T3_decision_retained':r['original_decision_retained'],'original_T3_case_passed':False,
      'critical_checkpoint':{k:r['checkpoints'][1][k] for k in ('time','state_indicator_fractions','source_actual_H_gap','source_fixed_H_indicator','spatial_only_original_formula_diagnostic_lower','saved_continuous_grid_H_gap')},
      'review_first_attempt_passed':False,'failed_review_code_sources_and_log_preserved':True,
      **dict.fromkeys(ZERO_FIELDS,0),'recorded_checks_total':480,'previous_checks_rerun':0,
      'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':94,
      'old_T3_delivery_unchanged':True,'next_gate_executed':False,'next_gate_new_N_values':[16,64],
      'next_gate_planned_new_graphs':112,'next_gate_planned_new_actual_paths':12,'next_gate_reused_N32_paths':6,
      'original_backend_or_api_changed':False,'general_T3_admitted':False,'general_smooth_flow_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# T3仿射压缩：只读预算复核\n\n[报告、保存来源副本、精确预算、独立观测审计和下一项方案](ltn-t3-affine-review.zip)。\n\n'
          f'SHA-256：`{info["archive"]["sha256"]}`。\n\n[交付核验](ltn-t3-affine-review-delivery.json)。CRC/SHA及全新解压标准库收据核验通过，不重放旧数值路径。\n\n'
          '空间项占原状态指标96.95%，不继续只加时间步。下一项三档空间控制已登记未执行；原N32未决保留，Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
