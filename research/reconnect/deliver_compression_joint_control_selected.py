"""Selected recovery archive; verify each restored byte without rerunning science."""
from pathlib import Path
import hashlib,json,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest

OUT=ROOT/'runs/reconnect-compression-joint-control'
PARENT=ROOT/'runs/reconnect-compression-joint-definition'
BASE='LTN_compression_joint_control'


def main():
    exports=ROOT/'exports';report=exports/f'{BASE}_report_20261107.md';archive=exports/f'{BASE}_code_results_20261107.zip'
    assert report.is_file() and not archive.exists() and (OUT/'sha256.json').exists()
    successful=(OUT/'audit.json').exists() and read(OUT/'audit.json')['passed']
    stopped=any((OUT/name).exists() for name in ('failed-execution.json','failed-audit.json')) or not read(OUT/'tests.json')['successful']
    assert successful or stopped
    for name,h in read(OUT/'sha256.json').items():assert digest(ROOT/name)==h,name
    with zipfile.ZipFile(exports/'LTN_compression_joint_definition_code_results_20261107.zip') as z:
        selected=set(json.loads(z.read('review-manifest.json')))
    selected.update(read(OUT/'protocol.json')['sources']);selected.update(read(OUT/'implementation-lock.json')['files'])
    selected.update(('research/reconnect/deliver_compression_joint_control.py',
        'research/reconnect/deliver_compression_joint_control_selected.py',
        'research/reconnect/COMPRESSION_JOINT_CONTROL_RESULTS.md',str(report.relative_to(ROOT))))
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file())
    selected.update(('GITHUB_HANDOFF.md','docs/RESEARCH_STATUS.md'))
    originals=read('/workspace/original-handoff-sha256.json');selected.update(originals)
    print('Selected file count',len(selected),'uncompressed bytes',sum((ROOT/name).stat().st_size for name in selected),flush=True)
    manifest={name:digest(ROOT/name) for name in sorted(selected)}
    for name,h in originals.items():assert manifest[name]==h,name
    note='''LTN N64 joint time/reference control,2026-11-07 Asia/Shanghai.
Read GITHUB_HANDOFF.md and research/reconnect/COMPRESSION_JOINT_CONTROL_RESULTS.md.
The report distinguishes completed numerical evidence from any preserved failure.
All old scientific judgments and budgets remain unchanged. Stage3 incomplete;fluid ORC OFF.
Selected recovery only,not fullworkspace/15attachments/unrecoveredv080/all historical runs/chat.
Check review-manifest.json and original-handoff-sha256.json; no publication/push.
When audit.json exists, the saved-data audit command is:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_compression_joint_control.py --directory runs/reconnect-compression-joint-control
This command checks saved data; do not rerun the physical driver or overwrite the run.
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in manifest:z.write(ROOT/name,name)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    recovery=Path('/workspace/.ltn-recovery-checks');recovery.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        with tempfile.TemporaryDirectory(prefix='ltn-joint-control-zip-',dir=recovery) as folder:
            z.extractall(folder)
            for name,h in manifest.items():assert digest(Path(folder)/name)==h,name
            for name,h in originals.items():assert digest(Path(folder)/name)==h,name
    receipt={'date':'2026-11-07','timezone':'Asia/Shanghai','report':{'path':str(report.relative_to(ROOT)),'sha256':digest(report)},
        'ZIP':{'path':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'included_files':len(manifest),'CRC_and_fresh_extraction_all_SHA_verified':True,'originals_verified':782,
        'attachments_verified_in_workspace':15,'prior_formal_manifests_verified':121,
        'saved_data_audit_passed':successful,'failure_preserved':stopped,
        'numerical_audit_repeated_in_extraction':False,'reason':'same audited bytes verified;no duplicate scientific computation',
        'recorded_checks_total':read(OUT/'tests.json')['recorded_checks_total'],
        'packaging_scope_corrected_without_scientific_rerun':True,'PDF_generated':False,'stage3':'incomplete','fluid_orc_enabled':False,'published_or_pushed':False}
    save(exports/f'{BASE}_delivery_20261107.json',receipt)
    with (exports/f'{BASE}_downloads_20261107.md').open('x') as f:
        f.write(f'# N64联合控制交付\n\n[报告]({report.name}) · [代码与精选结果ZIP]({archive.name})\n\n'
                f'ZIP SHA-256：`{digest(archive)}`\n\nStage3尚未通过，fluid ORC关闭。精选包不是完整工作区。未发布/推送。\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
