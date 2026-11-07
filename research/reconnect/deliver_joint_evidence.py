"""Selected read-only evidence recovery; cumulative seals are not file selections."""
from pathlib import Path
import json,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest

OUT=ROOT/'runs/reconnect-joint-evidence'
PARENT=ROOT/'runs/reconnect-compression-joint-control'
BASE='LTN_joint_evidence'
FILES=['joint_evidence_review.py','audit_joint_evidence_review.py','joint_evidence_protocol.json',
       'DENSITY_SPATIAL_SNAPSHOT_SCOPE.md','JOINT_EVIDENCE_REVIEW.md','deliver_joint_evidence.py']


def main():
    exports=ROOT/'exports';report=exports/f'{BASE}_report_20261107.md';archive=exports/f'{BASE}_code_results_20261107.zip'
    assert read(OUT/'audit.json')['passed'] and report.is_file() and not archive.exists() and not (OUT/'sha256.json').exists()
    protocol=read(OUT/'protocol.json')
    for name,h in protocol['sources'].items():assert digest(ROOT/name)==h,name
    # Full historical verification was completed in preservation.json. Keep its seal intact.
    entries=read(PARENT/'sha256.json');entries[str((PARENT/'sha256.json').relative_to(ROOT))]=digest(PARENT/'sha256.json')
    for name in FILES:entries['research/reconnect/'+name]=digest(ROOT/'research/reconnect'/name)
    for p in OUT.rglob('*'):
        if p.is_file():entries[str(p.relative_to(ROOT))]=digest(p)
    entries[str(report.relative_to(ROOT))]=digest(report);save(OUT/'sha256.json',dict(sorted(entries.items())))
    with zipfile.ZipFile(exports/'LTN_compression_joint_control_code_results_20261107.zip') as z:
        selected=set(json.loads(z.read('review-manifest.json')))
    selected.update(protocol['sources']);selected.update(read(OUT/'implementation-lock.json')['files'])
    selected.update('research/reconnect/'+name for name in FILES)
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file())
    selected.update(('GITHUB_HANDOFF.md','docs/RESEARCH_STATUS.md',str(report.relative_to(ROOT))))
    originals=read('/workspace/original-handoff-sha256.json');selected.update(originals)
    assert set(protocol['sources'])<=selected
    size=sum((ROOT/name).stat().st_size for name in selected)
    # Operational guard: do not accidentally expand to the cumulative20GB science manifest.
    assert size<2_500_000_000
    print('Selected files',len(selected),'uncompressed bytes',size,flush=True)
    manifest={name:digest(ROOT/name) for name in sorted(selected)}
    for name,h in originals.items():assert manifest[name]==h,name
    note='''LTN joint-control read-only evidence synthesis,2026-11-07 Asia/Shanghai.
Read GITHUB_HANDOFF.md and research/reconnect/JOINT_EVIDENCE_REVIEW.md.
New source-error and point-density diagnostics use saved data only.0new physics/tests;508recorded.
Old budgets,classifications and original requirements retained. Stage3 incomplete;fluid ORC OFF.
Next step is source/protocol definition for a single saved density snapshot;no new fields yet.
Selected recovery only,not fullworkspace/15attachments/unrecoveredv080/all historical runs/chat.
Verify review-manifest.json and original-handoff-sha256.json.
Read-only audit:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_joint_evidence_review.py --directory runs/reconnect-joint-evidence
Do not rerun any old physical driver or overwrite frozen runs.
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in manifest:z.write(ROOT/name,name)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    recovery=Path('/workspace/.ltn-recovery-checks');recovery.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        with tempfile.TemporaryDirectory(prefix='ltn-joint-evidence-',dir=recovery) as directory:
            z.extractall(directory)
            for name,h in manifest.items():assert digest(Path(directory)/name)==h,name
            for name,h in originals.items():assert digest(Path(directory)/name)==h,name
    receipt={'date':'2026-11-07','timezone':'Asia/Shanghai','report':{'path':str(report.relative_to(ROOT)),'sha256':digest(report)},
        'ZIP':{'path':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'included_files':len(manifest),'CRC_and_fresh_extraction_all_SHA_verified':True,'audit_sources_and_implementation_complete':True,
        'originals_verified':782,'attachments_verified_in_workspace':15,'prior_formal_manifests_verified':122,
        'saved_data_audit_passed':True,'numerical_audit_repeated_in_extraction':False,
        'reason':'same independently audited bytes verified;no duplicate scientific computation',
        'new_physical_fields_or_solves':0,'new_software_tests':0,'recorded_checks_total':508,
        'PDF_generated':False,'stage3':'incomplete','fluid_orc_enabled':False,'published_or_pushed':False}
    save(exports/f'{BASE}_delivery_20261107.json',receipt)
    with (exports/f'{BASE}_downloads_20261107.md').open('x') as f:
        f.write(f'# 联合控制后的证据归纳\n\n[报告]({report.name}) · [精选代码与结果ZIP]({archive.name})\n\n'
                f'ZIP SHA-256：`{digest(archive)}`\n\n本轮只读分析；Stage3未通过，fluid ORC关闭。精选包不是完整工作区，未发布/推送。\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
