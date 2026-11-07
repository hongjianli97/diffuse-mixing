"""Package the read-only review with immutable parent data; verify a fresh extraction."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest


def main():
    out=ROOT/'runs/reconnect-becker-budget-review';parent=ROOT/'runs/reconnect-becker-point';exports=ROOT/'exports'
    pdf=exports/'LTN_becker_budget_report_20261104.pdf';archive=exports/'LTN_becker_budget_code_results_20261104.zip'
    delivery=exports/'LTN_becker_budget_delivery_20261104.json';downloads=exports/'LTN_becker_budget_downloads_20261104.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (pdf,archive,delivery,downloads,seal))
    r=read(out/'results.json');audit=read(out/'accounting-audit.json');ex=read(out/'execution.json');status=read(out/'stage3-status.json')
    assert audit['audit_first_attempt_passed'] and ex['review_first_attempt_passed'] and status['new_reference_PDE_solves']==0
    assert r['source_decision_retained']=='moving_Becker_control_failed_or_unresolved' and not r['source_case_passed']
    assert not read(out/'next-gate-spec.json')['already_executed'] and not status['fluid_orc_enabled']
    entries=read(parent/'sha256.json');assert len(entries)==6756
    for rel,h in entries.items():assert digest(ROOT/rel)==h,rel
    info=subprocess.run(['pdfinfo',str(out/pdf.name)],capture_output=True,text=True,check=True).stdout;assert 'Pages:           2' in info
    log=(out/'LTN_becker_budget_report_20261104.log').read_text();assert 'Overfull' not in log and 'Missing character' not in log
    shutil.copyfile(out/pdf.name,pdf);shutil.copyfile('/tmp/ltn-becker-budget-review-xelatex-final.log',out/'PDF-build.log')
    save(out/'PDF-generation.json',{'pages':2,'sha256':digest(pdf),'all_pages_rendered_and_visually_reviewed':True,
      'overfull_boxes':0,'missing_characters':0,'original_unresolved_and_future_reference_not_executed_disclosed':True})
    names=['becker_budget_review.py','becker_budget_review_protocol.json','audit_becker_budget_review.py','document_becker_budget_review.py',
      'replay_becker_budget_review.py','package_becker_budget_review.py','BECKER_BUDGET_REVIEW.md']
    files=['research/reconnect/'+name for name in names];entries[str((parent/'sha256.json').relative_to(ROOT))]=digest(parent/'sha256.json')
    for rel in files:assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for p in out.rglob('*'):
        if p.is_file():
            rel=str(p.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(p)
    entries[str(pdf.relative_to(ROOT))]=digest(pdf);save(seal,dict(sorted(entries.items())))
    with zipfile.ZipFile(exports/'LTN_becker_point_code_results_20261104.zip') as z:selected=set(json.loads(z.read('review-manifest.json')))
    originals=read('/workspace/original-handoff-sha256.json');assert len(originals)==782;selected.update(originals)
    for folder in ('research','docs','prototypes','reference_baseline'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    selected.update(str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file())
    selected.update(i['path'] for i in read(out/'review-protocol.json')['sources'])
    selected.update(i['path'] for i in read(out/'next-gate-spec.json')['sources'])
    selected.update([str(pdf.relative_to(ROOT)),str((parent/'sha256.json').relative_to(ROOT)),'GITHUB_HANDOFF.md'])
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    for rel,h in originals.items():assert manifest[rel]==h
    note='''LTN saved Becker H/reference budget review,2026-11-04.
Saved reference space dominates:99.818674% of original state budget. Original moving_Becker_control_failed_or_unresolved retained.
Original H formula/threshold and all original states/references unchanged. No new graph/operator action/candidate/PDE/state/software test; recorded total remains500.
Independent exact positive bookkeeping and80/100-digit Decimal audit passed. Exact-overlap projection check is not a continuum error certificate.
Same-formula reference budget planning requires about8.50fold reduction; not a new acceptance threshold or retroactive pass.
NEXT ONLY REGISTERED,NOT RUN: independent reference L-shape(16384,8192),(32768,8192),(32768,16384),no candidate changes or reruns.
Formal second order is only resource/feasibility planning,not measured order,guaranteed error or Richardson state. Failure/unresolved stops without automatic expansion.
782 originals byteidentical,15 attachments and102 prior formal manifests verified in workspace;4 historical diagnostic differences retained.
Main prior scientific parser failure and109 pre-completion files remain unchanged.
Contains all research/docs,782 originals,parent28graph and3paired paths,reference evidence,old PDF and current2page PDF,review scripts/CSV/sourcecopies/audit.
Not full39GB workspace,15 uploaded attachments or unrecovered oldv080. Cumulative scientific manifest references omitted history.
Only local delivery: public GitHub push remains pending explicit authorization; no environment/software publication.
Replay accounting without a solver:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/replay_becker_budget_review.py --directory runs/reconnect-becker-budget-review
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-becker-budget-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            replay=subprocess.run([sys.executable,'research/reconnect/replay_becker_budget_review.py','--directory','runs/reconnect-becker-budget-review'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
            assert replay.returncode==0,replay.stdout+replay.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    result={'date':'2026-11-04','timezone':'Asia/Shanghai',
      'PDF':{'path':str(pdf.relative_to(ROOT)),'pages':2,'bytes':pdf.stat().st_size,'sha256':digest(pdf)},
      'code_and_results_zip':{'path':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'included_files':len(manifest),'all_CRC_SHA256_and_original782_verified':True,'fresh_replay_exit_code':0,'fresh_replay_log':replay.stdout,
      'fresh_replay_files_unchanged':True,'review_decision':r['decision'],'original_case_decision_retained':r['source_decision_retained'],
      'original_H_formula_and_threshold_retained':True,'source_case_passed':False,'new_graphs_actions_candidates_PDE_states_tests':0,'recorded_checks_total':500,
      'review_first_attempt_passed':True,'accounting_audit_first_attempt_passed':True,'prior_formal_manifests_preserved':102,'attachments_preserved_in_workspace':15,
      'next_reference_only_L_shape_registered':True,'next_reference_check_executed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
      'public_GitHub_push_attempted':False,'public_GitHub_push_pending_explicit_authorization':True,'environment_or_software_published':False}
    save(delivery,result)
    with downloads.open('x') as f:f.write('# 移动Becker预算复核交付\n\n[两页PDF](LTN_becker_budget_report_20261104.pdf) · [代码与结果ZIP](LTN_becker_budget_code_results_20261104.zip)\n\n'
      f'PDF SHA-256：`{digest(pdf)}`。\n\nZIP SHA-256：`{digest(archive)}`。\n\n原单例仍未决，Stage3未完成、fluid ORC关闭；本轮无新PDE，下一项参考L形核验未执行。仅本地交付，公开GitHub推送待明确授权。\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
