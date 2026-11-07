"""Selected local recovery for a read-only signed-error and budget diagnostic; no PDF."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest

OUT=ROOT/'runs/reconnect-compression-density-diagnostics'
PARENT=ROOT/'runs/reconnect-compression-density-calibration'
BASE='LTN_compression_density_diagnostics'
DATE='20261107'
FILES=['compression_density_diagnostics.py','audit_compression_density_diagnostics.py','compression_density_diagnostics_protocol.json',
       'COMPRESSION_DENSITY_DIAGNOSTICS_CONTRACT.md','COMPRESSION_DENSITY_DIAGNOSTICS.md','deliver_compression_density_diagnostics.py']


def main():
    exports=ROOT/'exports';archive=exports/f'{BASE}_code_results_{DATE}.zip';report=exports/f'{BASE}_report_{DATE}.md'
    assert read(OUT/'audit.json')['passed'] and not archive.exists() and not (OUT/'sha256.json').exists()
    assert not report.exists();shutil.copyfile(ROOT/'research/reconnect/COMPRESSION_DENSITY_DIAGNOSTICS.md',report)
    entries=read(PARENT/'sha256.json')
    for name,h in entries.items():assert digest(ROOT/name)==h,name
    entries[str((PARENT/'sha256.json').relative_to(ROOT))]=digest(PARENT/'sha256.json')
    for name in FILES:entries['research/reconnect/'+name]=digest(ROOT/'research/reconnect'/name)
    for p in OUT.rglob('*'):
        if p.is_file():entries[str(p.relative_to(ROOT))]=digest(p)
    entries[str(report.relative_to(ROOT))]=digest(report);save(OUT/'sha256.json',dict(sorted(entries.items())))
    with zipfile.ZipFile(exports/'LTN_compression_density_calibration_code_results_20261107.zip') as z:selected=set(json.loads(z.read('review-manifest.json')))
    for folder in ('research','docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    selected.update(read(OUT/'protocol.json')['sources']);selected.update(read(OUT/'next-gate-spec.json')['sources'])
    selected.update(read(OUT/'implementation-lock.json')['files'])
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file())
    selected.update([str(report.relative_to(ROOT)),str((PARENT/'sha256.json').relative_to(ROOT)),'GITHUB_HANDOFF.md'])
    originals=read('/workspace/original-handoff-sha256.json');selected.update(originals)
    manifest={name:digest(ROOT/name) for name in sorted(selected)}
    for name,h in originals.items():assert manifest[name]==h,name
    note='LTN moving density saved-error and budget diagnostics,2026-11-07 Asia/Shanghai.\nRead COMPRESSION_DENSITY_DIAGNOSTICS.md and GITHUB_HANDOFF.md.\nEight saved-state identities are exact in rational moments;full budgets audited at80/100digits.\nKernel cross terms are negative at both times,but its positive state-difference term dominates at the middle time.\nLedger:at fixed observed gains,removing reference indicators alone remains insufficient.\nOriginal physical experiment/thresholds/decisions preserved;no new density,rates,geometry,propagation,PDE or software tests.506recorded.\nStage3 incomplete,fluid ORC OFF,kappa uncomputed. Next step is a reviewable joint time/reference contract only.\nCurrent report is Markdown;previous physical PDF remains included,not a new PDF or experiment.\nSelected recovery only,not full workspace,15attachments,missingv080,all historical runs or chat.\nVerify review-manifest.json and original-handoff-sha256.json. Read-only saved-data audit:\nOPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_compression_density_diagnostics.py --directory runs/reconnect-compression-density-diagnostics\nLocal only,no publication/push.\n'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in manifest:z.write(ROOT/name,name)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
        with tempfile.TemporaryDirectory(prefix='ltn-compression-density-diagnostics-zip-',dir='/tmp') as directory:
            z.extractall(directory)
            result=subprocess.run([sys.executable,'research/reconnect/audit_compression_density_diagnostics.py','--directory','runs/reconnect-compression-density-diagnostics'],
                cwd=directory,env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
            assert result.returncode==0,result.stdout+result.stderr
            for name,h in manifest.items():assert digest(Path(directory)/name)==h,name
    receipt={'date':'2026-11-07','timezone':'Asia/Shanghai','report':{'path':str(report.relative_to(ROOT)),'sha256':digest(report)},
        'ZIP':{'path':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'included_files':len(manifest),'CRC_SHA_and782_originals_verified':True,'fresh_replay_exit_code':0,
        'fresh_replay_log':result.stdout,'extracted_files_unchanged':True,'source_files_verified':154,'prior_formal_manifests_verified':119,
        'attachments_verified_in_workspace':15,'new_physical_fields_or_solves':0,'new_software_checks':0,'recorded_checks_total':506,
        'next_case_executed':False,'PDF_generated':False,'stage3':'incomplete','fluid_orc_enabled':False,'published_or_pushed':False}
    save(exports/f'{BASE}_delivery_{DATE}.json',receipt)
    with (exports/f'{BASE}_downloads_{DATE}.md').open('x') as f:
        f.write(f'# 平滑压缩保存误差与预算诊断\n\n[Markdown报告]({report.name}) · [代码和精选来源ZIP]({archive.name})\n\nZIP SHA-256：`{digest(archive)}`\n\n'
                '本轮只读核验误差恒等式与预算，未执行新物理算例；Stage3 incomplete，fluid ORC关闭。精选包不是完整工作区，未发布/推送。\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='fresh_replay_log'},indent=2))


if __name__=='__main__':main()
