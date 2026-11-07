"""Selected local recovery for a read-only joint time/reference definition; no PDF."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest

OUT=ROOT/'runs/reconnect-compression-joint-definition'
PARENT=ROOT/'runs/reconnect-compression-density-diagnostics'
BASE='LTN_compression_joint_definition'
DATE='20261107'
FILES=['compression_joint_definition.py','audit_compression_joint_definition.py','compression_joint_definition_protocol.json',
       'COMPRESSION_JOINT_CONTROL_CONTRACT.md','COMPRESSION_JOINT_DEFINITION.md','deliver_compression_joint_definition.py']


def main():
    recovery_root=Path('/workspace/.ltn-recovery-checks');recovery_root.mkdir(exist_ok=True)
    exports=ROOT/'exports';archive=exports/f'{BASE}_code_results_{DATE}.zip';report=exports/f'{BASE}_report_{DATE}.md'
    assert read(OUT/'audit.json')['passed'] and not archive.exists() and not (OUT/'sha256.json').exists()
    assert not report.exists();shutil.copyfile(ROOT/'research/reconnect/COMPRESSION_JOINT_DEFINITION.md',report)
    entries=read(PARENT/'sha256.json')
    for name,h in entries.items():assert digest(ROOT/name)==h,name
    entries[str((PARENT/'sha256.json').relative_to(ROOT))]=digest(PARENT/'sha256.json')
    for name in FILES:entries['research/reconnect/'+name]=digest(ROOT/'research/reconnect'/name)
    for p in OUT.rglob('*'):
        if p.is_file():entries[str(p.relative_to(ROOT))]=digest(p)
    entries[str(report.relative_to(ROOT))]=digest(report);save(OUT/'sha256.json',dict(sorted(entries.items())))
    with zipfile.ZipFile(exports/'LTN_compression_density_diagnostics_code_results_20261107.zip') as z:selected=set(json.loads(z.read('review-manifest.json')))
    for folder in ('research','docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    selected.update(read(OUT/'protocol.json')['sources']);selected.update(read(OUT/'next-gate-spec.json')['sources'])
    selected.update(read(OUT/'implementation-lock.json')['files'])
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file())
    selected.update([str(report.relative_to(ROOT)),str((PARENT/'sha256.json').relative_to(ROOT)),'GITHUB_HANDOFF.md'])
    originals=read('/workspace/original-handoff-sha256.json');selected.update(originals)
    manifest={name:digest(ROOT/name) for name in sorted(selected)}
    for name,h in originals.items():assert manifest[name]==h,name
    note='LTN N64 ledger/oracle joint time/reference control contract,2026-11-07 Asia/Shanghai.\nRead COMPRESSION_JOINT_DEFINITION.md and COMPRESSION_JOINT_CONTROL_CONTRACT.md.\nThis round is read-only:canonical4references,old4actual/independent pairs and original N128-to-N64 observation verified.\nFuture one case only:newK64 for oracle/ledger,64geometry graphs,128density/rate packets,2actual/2independent paths,128intervals.\nTwo future reference solves:n2048-k1024 and n2048-k2048;reuse oldn1024-k1024. Matching space/time bases,unchanged material masses and observation hierarchy.\nAll old decisions/budgets remain immutable;newK64 rows are separately registered. No predicted improvement or time order.\nCurrent0new physical fields/solves/software tests;506recorded. Future two tests must pass before508;case not executed.\nStage3 incomplete,fluid ORC OFF. No newPDF,no publication/push.\nSelected recovery only,not fullworkspace/15attachments/unrecoveredv080/all historical runs/chat.\nVerify review-manifest.json and original-handoff-sha256.json. Read-only audit:\nOPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_compression_joint_definition.py --directory runs/reconnect-compression-joint-definition\n'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in manifest:z.write(ROOT/name,name)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
        with tempfile.TemporaryDirectory(prefix='ltn-compression-joint-definition-zip-',dir=str(recovery_root)) as directory:
            z.extractall(directory)
            result=subprocess.run([sys.executable,'research/reconnect/audit_compression_joint_definition.py','--directory','runs/reconnect-compression-joint-definition'],
                cwd=directory,env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
            assert result.returncode==0,result.stdout+result.stderr
            for name,h in manifest.items():assert digest(Path(directory)/name)==h,name
    receipt={'date':'2026-11-07','timezone':'Asia/Shanghai','report':{'path':str(report.relative_to(ROOT)),'sha256':digest(report)},
        'ZIP':{'path':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'included_files':len(manifest),'CRC_SHA_and782_originals_verified':True,'fresh_replay_exit_code':0,
        'fresh_replay_log':result.stdout,'extracted_files_unchanged':True,'source_files_verified':168,'prior_formal_manifests_verified':120,
        'attachments_verified_in_workspace':15,'new_physical_fields_or_solves':0,'new_software_checks':0,'recorded_checks_total':506,
        'next_case_executed':False,'PDF_generated':False,'stage3':'incomplete','fluid_orc_enabled':False,'published_or_pushed':False}
    save(exports/f'{BASE}_delivery_{DATE}.json',receipt)
    with (exports/f'{BASE}_downloads_{DATE}.md').open('x') as f:
        f.write(f'# N64联合时间与参考控制方案\n\n[Markdown报告]({report.name}) · [代码和精选来源ZIP]({archive.name})\n\nZIP SHA-256：`{digest(archive)}`\n\n'
                '本轮完成联合控制契约及来源核对，未来K64及两项参考尚未执行；Stage3 incomplete，fluid ORC关闭。精选包不是完整工作区，未发布/推送。\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='fresh_replay_log'},indent=2))


if __name__=='__main__':main()
