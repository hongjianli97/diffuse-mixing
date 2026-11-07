"""Selected local recovery for a read-only compression density definition; no PDF."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest

OUT=ROOT/'runs/reconnect-compression-density-definition'
PARENT=ROOT/'runs/reconnect-t1-density-calibration'
BASE='LTN_compression_density_definition'
DATE='20261107'
FILES=['compression_density_definition.py','audit_compression_density_definition.py','compression_density_definition_protocol.json',
       'COMPRESSION_DENSITY_CONTRACT.md','COMPRESSION_DENSITY_DEFINITION.md','deliver_compression_density_definition.py']


def main():
    exports=ROOT/'exports';archive=exports/f'{BASE}_code_results_{DATE}.zip';report=exports/f'{BASE}_report_{DATE}.md'
    assert read(OUT/'audit.json')['passed'] and not archive.exists() and not (OUT/'sha256.json').exists()
    assert not report.exists();shutil.copyfile(ROOT/'research/reconnect/COMPRESSION_DENSITY_DEFINITION.md',report)
    entries=read(PARENT/'sha256.json')
    for name,h in entries.items():assert digest(ROOT/name)==h,name
    entries[str((PARENT/'sha256.json').relative_to(ROOT))]=digest(PARENT/'sha256.json')
    for name in FILES:entries['research/reconnect/'+name]=digest(ROOT/'research/reconnect'/name)
    for p in OUT.rglob('*'):
        if p.is_file():entries[str(p.relative_to(ROOT))]=digest(p)
    entries[str(report.relative_to(ROOT))]=digest(report);save(OUT/'sha256.json',dict(sorted(entries.items())))
    with zipfile.ZipFile(exports/'LTN_t1_density_calibration_code_results_20261107.zip') as z:selected=set(json.loads(z.read('review-manifest.json')))
    for folder in ('research','docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    selected.update(read(OUT/'protocol.json')['sources']);selected.update(read(OUT/'next-gate-spec.json')['sources'])
    selected.update(read(OUT/'implementation-lock.json')['files'])
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file())
    selected.update([str(report.relative_to(ROOT)),str((PARENT/'sha256.json').relative_to(ROOT)),'GITHUB_HANDOFF.md'])
    originals=read('/workspace/original-handoff-sha256.json');selected.update(originals)
    manifest={name:digest(ROOT/name) for name in sorted(selected)}
    for name,h in originals.items():assert manifest[name]==h,name
    note='''LTN smooth compression N64 density-source contract,2026-11-07 Asia/Shanghai.
Read COMPRESSION_DENSITY_CONTRACT.md,COMPRESSION_DENSITY_DEFINITION.md and GITHUB_HANDOFF.md.
Current work read-only:48 old geometries,33/65 saved frames and fixed reference/time budgets verified.
64 periodic x crossings require explicit integer lifts. Fixed initial ledger ring images must not be replaced by frame-dependent face image labels.
Nonlinear finite-ring secant stretch differs from point density. Nonuniform proxy anchor mass ratio is not exactly2.
Future one case only:h=1/32,N64,D=.02,T=.05,K16/32;96density fields/96coefficients;4actual/4independent paths,96intervals.
No new density,coefficient,lift,graph,state or software test this round.504recorded;Stage3 incomplete,fluid ORC OFF.
No new PDF. Previous T1 PDF is a previous physical result,not a moving-density result.
Selected recovery includes48 original geometries and source records;not full workspace,15attachments,all history,missingv080 or chat.
Verify review-manifest.json/original-handoff-sha256.json. Current read-only audit:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_compression_density_definition.py --directory runs/reconnect-compression-density-definition
Local only,no publication/push.
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in manifest:z.write(ROOT/name,name)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
        with tempfile.TemporaryDirectory(prefix='ltn-compression-density-definition-zip-',dir='/tmp') as directory:
            z.extractall(directory)
            result=subprocess.run([sys.executable,'research/reconnect/audit_compression_density_definition.py','--directory','runs/reconnect-compression-density-definition'],
                cwd=directory,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
            assert result.returncode==0,result.stdout+result.stderr
            for name,h in manifest.items():assert digest(Path(directory)/name)==h,name
    receipt={'date':'2026-11-07','timezone':'Asia/Shanghai','report':{'path':str(report.relative_to(ROOT)),'sha256':digest(report)},
        'ZIP':{'path':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'included_files':len(manifest),'CRC_SHA_and782_originals_verified':True,'fresh_replay_exit_code':0,
        'fresh_replay_log':result.stdout,'extracted_files_unchanged':True,'source_files_verified':136,'prior_formal_manifests_verified':117,
        'attachments_verified_in_workspace':15,'new_physical_arrays_or_solves':0,'new_software_checks':0,'recorded_checks_total':504,
        'next_case_executed':False,'PDF_generated':False,'stage3':'incomplete','fluid_orc_enabled':False,'published_or_pushed':False}
    save(exports/f'{BASE}_delivery_{DATE}.json',receipt)
    with (exports/f'{BASE}_downloads_{DATE}.md').open('x') as f:
        f.write(f'# 平滑压缩密度契约与来源下载\n\n[Markdown报告]({report.name}) · [代码和精选来源ZIP]({archive.name})\n\nZIP SHA-256：`{digest(archive)}`\n\n'
                '本轮仅定义与保存数据核验，未执行新密度算例；Stage3 incomplete，fluid ORC关闭。精选包不是完整工作区，未发布/推送。\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='fresh_replay_log'},indent=2))


if __name__=='__main__':main()
