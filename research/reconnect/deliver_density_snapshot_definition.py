"""Selected read-only evidence recovery; cumulative seals are not file selections."""
from pathlib import Path
import json,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest

OUT=ROOT/'runs/reconnect-density-snapshot-definition'
PARENT=ROOT/'runs/reconnect-joint-evidence'
BASE='LTN_density_snapshot_definition'
FILES=['density_snapshot_definition.py','audit_density_snapshot_definition.py',
       'density_snapshot_definition_protocol.json','DENSITY_SNAPSHOT_CONTRACT.md',
       'DENSITY_SNAPSHOT_DEFINITION.md','deliver_density_snapshot_definition.py']


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
    with zipfile.ZipFile(exports/'LTN_joint_evidence_code_results_20261107.zip') as z:
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
    note='''LTN saved density snapshot definition,2026-11-07 UTC.
Read GITHUB_HANDOFF.md and research/reconnect/DENSITY_SNAPSHOT_DEFINITION.md.
Three saved grids and clock/IDs/masses/neighbor/lift inputs verified;0new physical fields/tests;508recorded.
Four missing density fields are specified,not yet run. Stage3 incomplete;fluid ORC OFF.
Fixed h=2/N can retain lattice quadrature bias;no fitted multiplier or normalization.
Selected recovery only,not fullworkspace/15attachments/unrecoveredv080/all historical runs/chat.
Verify review-manifest.json and original-handoff-sha256.json.
Read-only audit:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_density_snapshot_definition.py --directory runs/reconnect-density-snapshot-definition
Do not rerun old physical drivers or overwrite frozen runs.
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in manifest:z.write(ROOT/name,name)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    recovery=Path('/workspace/.ltn-recovery-checks');recovery.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        with tempfile.TemporaryDirectory(prefix='ltn-density-snapshot-definition-',dir=recovery) as directory:
            z.extractall(directory)
            for name,h in manifest.items():assert digest(Path(directory)/name)==h,name
            for name,h in originals.items():assert digest(Path(directory)/name)==h,name
    receipt={'date':'2026-11-07','timezone':'UTC','report':{'path':str(report.relative_to(ROOT)),'sha256':digest(report)},
        'ZIP':{'path':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'included_files':len(manifest),'CRC_and_fresh_extraction_all_SHA_verified':True,'audit_sources_and_implementation_complete':True,
        'originals_verified':782,'attachments_verified_in_workspace':15,'prior_formal_manifests_verified':123,
        'saved_data_audit_passed':True,'numerical_audit_repeated_in_extraction':False,
        'reason':'same independently audited bytes verified;no duplicate scientific computation',
        'new_physical_fields_or_solves':0,'new_software_tests':0,'recorded_checks_total':508,
        'PDF_generated':False,'stage3':'incomplete','fluid_orc_enabled':False,'published_or_pushed':False}
    save(exports/f'{BASE}_delivery_20261107.json',receipt)
    with (exports/f'{BASE}_downloads_20261107.md').open('x') as f:
        f.write(f'# 单快照密度诊断定义\n\n[报告]({report.name}) · [精选代码与结果ZIP]({archive.name})\n\n'
                f'ZIP SHA-256：`{digest(archive)}`\n\n本轮输入与协议核验；四新字段尚未执行；Stage3未通过，fluid ORC关闭。精选包不是完整工作区，未发布/推送。\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
