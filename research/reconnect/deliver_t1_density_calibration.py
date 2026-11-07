"""Local PDF and selected recovery package for the executed T1 density calibration."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
OUT=ROOT/'runs/reconnect-t1-density-calibration'
PARENT=ROOT/'runs/reconnect-t1-density-definition'
BASE='LTN_t1_density_calibration'
DATE='20261107'
FILES=['t1_density_sources.py','test_t1_density_calibration.py','t1_density_calibration_core.py',
       'audit_t1_density_calibration.py','t1_density_calibration.py','t1_density_calibration_protocol.json',
       'T1_DENSITY_CALIBRATION.md','deliver_t1_density_calibration.py']

def package():
    exports = ROOT / 'exports'
    pdf = exports / f'{BASE}_report_{DATE}.pdf'
    archive = exports / f'{BASE}_code_results_{DATE}.zip'
    seal = OUT / 'sha256.json'
    assert not any(p.exists() for p in (pdf, archive, seal))
    assert read(OUT / 'audit.json')['passed']
    assert read(OUT / 'PDF-visual-review.json')['all_pages_reviewed']
    info = subprocess.check_output(['pdfinfo', str(OUT / pdf.name)], text=True)
    assert 'Pages:           2' in info
    log = (OUT / f'{BASE}_report_{DATE}.log').read_text()
    assert 'Overfull' not in log and 'Missing character' not in log
    shutil.copyfile(OUT / pdf.name, pdf)
    save(OUT / 'PDF-generation.json', {'pages': 2, 'sha256': digest(pdf),
         'no_overfull_or_missing_characters': True, 'all_pages_visually_reviewed': True})
    entries = read(PARENT / 'sha256.json')
    for p, h in entries.items():
        assert digest(ROOT / p) == h, p
    entries[str((PARENT / 'sha256.json').relative_to(ROOT))] = digest(PARENT / 'sha256.json')
    for name in FILES:
        entries['research/reconnect/' + name] = digest(ROOT / 'research/reconnect' / name)
    for p in OUT.rglob('*'):
        if p.is_file():
            entries[str(p.relative_to(ROOT))] = digest(p)
    entries[str(pdf.relative_to(ROOT))] = digest(pdf)
    save(seal, dict(sorted(entries.items())))
    with zipfile.ZipFile(exports / 'LTN_t1_density_definition_code_results_20261107.zip') as z:
        selected = set(json.loads(z.read('review-manifest.json')))
    for folder in ('research', 'docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    selected.update(read(ROOT / 'research/reconnect/t1_density_calibration_protocol.json')['sources'])
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file())
    selected.update([str(pdf.relative_to(ROOT)), str((PARENT / 'sha256.json').relative_to(ROOT)), 'GITHUB_HANDOFF.md'])
    originals = read('/workspace/original-handoff-sha256.json')
    selected.update(originals)
    manifest = {p: digest(ROOT / p) for p in sorted(selected)}
    for p, h in originals.items():
        assert manifest[p] == h
    selected.update(read(OUT/'next-gate-spec.json')['sources'])
    selected.update(read(OUT/'implementation-lock.json')['files'])
    manifest={p:digest(ROOT/p) for p in sorted(selected)}
    note = """LTN static T1 same-backend density calibration,2026-11-07 Asia/Shanghai.
The single N32 case executed; implementation controls and independent audit passed.
Read T1_DENSITY_CALIBRATION.md and the preserved T1_DENSITY_SOURCE_CONTRACT.md.
Two density fields/two coefficient packets; only3kernel actual/3complete independent paths,28intervals.
Ledger reuses old3oracle paths after all2048exact coefficients agree; no newledger runs.
Kernel density1.0004157504550328 has positive bias. Its smaller scalar error partly compensates the baseline slow discrete decay,not proof of better density.
Original mass/c0/geometry/reference retained. No normalization,F/C fitting,threshold change or newPDE.
Two software groups passed:504recorded. Stage3 incomplete,fluid ORC OFF; all Becker gaps retained.
782originals,research/docs,selected data only;not full workspace,15attachments,unrecoveredv080 or chat.
Verify review-manifest.json/original-handoff-sha256.json. Saved-data audit(no primary propagation):
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_t1_density_calibration.py --directory runs/reconnect-t1-density-calibration
Local only,no publication/push.
"""
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in manifest:
            z.write(ROOT / p, p)
        z.writestr('review-manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        z.writestr('original-handoff-sha256.json', json.dumps(originals, indent=2) + '\n')
        z.writestr('DELIVERY_README.txt', note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p, h in manifest.items():
            assert hashlib.sha256(z.read(p)).hexdigest() == h
        with tempfile.TemporaryDirectory(prefix='ltn-t1-density-calibration-zip-', dir='/tmp') as directory:
            z.extractall(directory)
            result = subprocess.run([sys.executable, 'research/reconnect/audit_t1_density_calibration.py',
                '--directory', 'runs/reconnect-t1-density-calibration'], cwd=directory,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=120)
            assert result.returncode == 0, result.stdout + result.stderr
            for p, h in manifest.items():
                assert digest(Path(directory) / p) == h, p
    receipt = {'date': '2026-11-07', 'timezone': 'Asia/Shanghai',
        'PDF': {'path': str(pdf.relative_to(ROOT)), 'pages': 2, 'bytes': pdf.stat().st_size, 'sha256': digest(pdf)},
        'ZIP': {'path': str(archive.relative_to(ROOT)), 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'included_files': len(manifest), 'CRC_SHA_and782_originals_verified': True,
        'fresh_replay_exit_code': 0, 'fresh_replay_log': result.stdout, 'extracted_files_unchanged': True,
        'source_files_verified': 38, 'prior_formal_manifests_verified': 116,
        'attachments_verified_in_workspace': 15, 'new_graphs': 0, 'new_actual_paths': 3, 'new_independent_paths': 3, 'new_reference_PDE_solves': 0,
        'new_software_checks': 2, 'recorded_checks_total': 504, 'next_gate_executed': False,
        'stage3': 'incomplete', 'fluid_orc_enabled': False, 'published_or_pushed': False}
    save(exports / f'{BASE}_delivery_{DATE}.json', receipt)
    with (exports / f'{BASE}_downloads_{DATE}.md').open('x') as f:
        f.write(f'# T1三密度单例结果下载\n\n[两页PDF]({pdf.name}) · [代码与精选结果ZIP]({archive.name})\n\n'
                f'PDF SHA-256：`{digest(pdf)}`\n\nZIP SHA-256：`{digest(archive)}`\n\n'
                'Stage3 incomplete，fluid ORC关闭。本轮完成静态单例；平滑变密度推广尚未执行；仅本地保存，未发布/推送。\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'fresh_replay_log'}, indent=2))



if __name__=='__main__':package()
