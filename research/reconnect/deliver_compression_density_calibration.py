"""Package the audited single moving-density comparison for local recovery."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
OUT=ROOT/'runs/reconnect-compression-density-calibration'
PARENT=ROOT/'runs/reconnect-compression-density-definition'
BASE='LTN_compression_density_calibration'
DATE='20261107'
FILES=['compression_density_sources.py','compression_density_audit_core.py','test_compression_density.py',
 'compression_density_compare.py','compression_density_calibration.py','audit_compression_density_calibration.py',
 'compression_density_calibration_protocol.json','COMPRESSION_DENSITY_CALIBRATION.md','deliver_compression_density_calibration.py']


def main():
    exports=ROOT/'exports';archive=exports/f'{BASE}_code_results_{DATE}.zip';pdf=exports/f'{BASE}_report_{DATE}.pdf';report=exports/f'{BASE}_report_{DATE}.md'
    assert read(OUT/'audit.json')['passed'] and read(OUT/'PDF-visual-review.json')['all_pages_reviewed']
    assert not any(p.exists() for p in (archive,pdf,report,OUT/'sha256.json'))
    info=subprocess.check_output(['pdfinfo',str(OUT/pdf.name)],text=True);assert 'Pages:           2' in info
    log=(OUT/f'{BASE}_report_{DATE}.log').read_text();assert 'Overfull' not in log and 'Missing character' not in log
    shutil.copyfile(OUT/pdf.name,pdf);shutil.copyfile(ROOT/'research/reconnect/COMPRESSION_DENSITY_CALIBRATION.md',report)
    save(OUT/'PDF-generation.json',{'pages':2,'sha256':digest(pdf),'no_overfull_or_missing_characters':True,'all_pages_visually_reviewed':True})
    entries=read(PARENT/'sha256.json')
    for name,h in entries.items():assert digest(ROOT/name)==h,name
    entries[str((PARENT/'sha256.json').relative_to(ROOT))]=digest(PARENT/'sha256.json')
    for name in FILES:entries['research/reconnect/'+name]=digest(ROOT/'research/reconnect'/name)
    for p in OUT.rglob('*'):
        if p.is_file():entries[str(p.relative_to(ROOT))]=digest(p)
    for p in (pdf,report):entries[str(p.relative_to(ROOT))]=digest(p)
    save(OUT/'sha256.json',dict(sorted(entries.items())))
    with zipfile.ZipFile(exports/'LTN_compression_density_definition_code_results_20261107.zip') as z:selected=set(json.loads(z.read('review-manifest.json')))
    for folder in ('research','docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    selected.update(read(OUT/'protocol.json')['sources']);selected.update(read(OUT/'next-gate-spec.json')['sources']);selected.update(read(OUT/'implementation-lock.json')['files'])
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file())
    selected.update([str(pdf.relative_to(ROOT)),str(report.relative_to(ROOT)),str((PARENT/'sha256.json').relative_to(ROOT)),'GITHUB_HANDOFF.md'])
    originals=read('/workspace/original-handoff-sha256.json');selected.update(originals)
    manifest={name:digest(ROOT/name) for name in sorted(selected)}
    for name,h in originals.items():assert manifest[name]==h,name
    note='''LTN smooth moving compression N64 density-source calibration,2026-11-07 Asia/Shanghai.
Read COMPRESSION_DENSITY_CALIBRATION.md and GITHUB_HANDOFF.md. One frozen case completed.
Original material masses,c0,reference,48geometries and2oracle paths reused bytewise.
96newdensity fields/96rate packets,4actual/4complete independent paths,96intervals.
Fixed h=1/32,explicit integer periodic lifts and fixed initial rings,no fitted correction.
Any unresolved or adverse source decision is retained;no automatic extra grids/K/h/reference.
506recorded checks;Stage3 incomplete,fluid ORC OFF,kappa uncomputed.
Selected recovery,NOT full workspace/15attachments/all historical runs/missingv080/chat.
Verify review-manifest.json and original-handoff-sha256.json. Saved-data audit(no actual propagation):
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_compression_density_calibration.py --directory runs/reconnect-compression-density-calibration
Local only,no publication/push.
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in manifest:z.write(ROOT/name,name)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
        with tempfile.TemporaryDirectory(prefix='ltn-moving-density-zip-',dir='/tmp') as directory:
            z.extractall(directory)
            print('ZIP hash/CRC verified; running saved-data audit in fresh extraction',flush=True)
            result=subprocess.run([sys.executable,'research/reconnect/audit_compression_density_calibration.py','--directory','runs/reconnect-compression-density-calibration'],
                cwd=directory,env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=900)
            assert result.returncode==0,result.stdout+result.stderr
            for name,h in manifest.items():assert digest(Path(directory)/name)==h,name
    receipt={'date':'2026-11-07','timezone':'Asia/Shanghai','report':{'path':str(report.relative_to(ROOT)),'sha256':digest(report)},
        'PDF':{'path':str(pdf.relative_to(ROOT)),'pages':2,'bytes':pdf.stat().st_size,'sha256':digest(pdf)},
        'ZIP':{'path':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'included_files':len(manifest),'CRC_SHA_and782_originals_verified':True,'fresh_replay_exit_code':0,'fresh_replay_log':result.stdout,
        'extracted_files_unchanged':True,'source_files_verified':140,'prior_formal_manifests_verified':118,'attachments_verified_in_workspace':15,
        'new_density_fields':96,'new_coefficient_packets':96,'new_actual_paths':4,'new_independent_paths':4,'new_action_intervals':96,
        'new_geometry_graphs':0,'new_reference_PDE_solves':0,'old_paths_rerun':0,'new_software_checks':2,'recorded_checks_total':506,
        'stage3':'incomplete','fluid_orc_enabled':False,'published_or_pushed':False}
    save(exports/f'{BASE}_delivery_{DATE}.json',receipt)
    with (exports/f'{BASE}_downloads_{DATE}.md').open('x') as f:
        f.write(f'# 平滑压缩三密度单例\n\n[PDF]({pdf.name}) · [代码及精选结果ZIP]({archive.name}) · [Markdown]({report.name})\n\n'
            f'ZIP SHA-256：`{digest(archive)}`\n\nStage3 incomplete，fluid ORC关闭。仅本地保存，未发布/推送；精选包不等于完整工作区。\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='fresh_replay_log'},indent=2),flush=True)

if __name__=='__main__':main()
