"""Seal the ordinary FV time comparison; verify both replay modes."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

from audit_route_decision import ROOT,digest,read,save


def main():
    source=ROOT/'runs/reconnect-contact-finite-time-reference';out=ROOT/'runs/reconnect-FV-contact-time'
    exports=ROOT/'exports';archive=exports/'ltn-FV-contact-time-review.zip'
    delivery=exports/'ltn-FV-contact-time-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed evidence'
    execution=read(out/'execution.json');audit=read(out/'implementation-audit.json')
    assert execution['tests_exit_code']==execution['study_exit_code']==execution['audit_exit_code']==0
    assert audit['new_software_checks_passed']==4 and audit['prior_formal_manifests_verified']==64
    prior=read(source/'sha256.json')
    for rel,expected in prior.items():assert digest(ROOT/rel)==expected,rel
    files=['research/reconnect/'+name for name in ['FV_CONTACT_FIXED_TIME_COMPARISON.md','cartesian_fv_modes.py',
        'test_cartesian_fv_modes.py','fv_contact_time_study.py','audit_fv_contact_time.py','package_fv_contact_time.py']]
    entries=dict(prior);source_seal=str((source/'sha256.json').relative_to(ROOT));entries[source_seal]=digest(source/'sha256.json')
    for rel in files:entries[rel]=digest(ROOT/rel)
    for p in out.iterdir():
        assert p.is_file(),p
        entries[str(p.relative_to(ROOT))]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'] if not item['path'].endswith('.npz'))
    selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([source_seal,'research/reconnect/audit_route_decision.py','research/reconnect/periodic_density_path.py',
        'research/reconnect/contact_spectral_reference.py','docs/RESEARCH_STATUS.md'])
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=(
        'LTN ordinary central-resistance FV fixed-time comparison. REVIEW ONLY.\n'
        'One unchanged N128 contact, target T=1/20, original mass means, all 16384 discrete mode slots retained.\n'
        'One normal ring, all transverse FFT modes. No full 16384-dimensional transport matrix.\n'
        'Semidiscrete exponential and BE512/1024 spectral-factor endpoints; not explicit full step loops.\n'
        'Raw negative zero-mode eigenvalue and inventory drift are retained; no clipping, mean repair or renormalization.\n'
        'Independent Pade exponential and full-grid last-step residual controls are recorded.\n'
        'Standard-library METRICS only: python -S research/reconnect/fv_contact_time_study.py --summary-replay runs/reconnect-FV-contact-time\n'
        'Complete solver replay REQUIRES NUMPY: OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python research/reconnect/fv_contact_time_study.py --replay runs/reconnect-FV-contact-time\n'
        'Solver replay uses an epsilon-based arithmetic tolerance fixed before the run, not a physical acceptance threshold.\n'
        'requirements-solver.txt pins NumPy; full historical independent audit additionally needs SciPy and the original workspace.\n'
        'Source/reference bytes, original masks and initial fluctuation norm are retained. Reference bridge remains conditional.\n'
        'Indicators are not certified total physical error bounds. No kernel finite-time comparison or candidate admission.\n'
        'Four new checks passed; historical 396 not rerun; recorded total 400.\n'
        '782 originals, 15 attachments and 64 prior formal manifests preserved. Four historical diagnostic differences retained separately.\n'
        'This archive excludes full old snapshots and attachments. Cumulative scientific seal includes unarchived history.\n'
        'review-manifest.json covers included files. docs/RESEARCH_STATUS.md is a mutable snapshot outside the scientific seal.\n'
        'Next single N256 common-observation spatial control is predeclared, not executed here.\n'
        'No PDF, release, commit, push, environment reconfiguration or publication. Stage 3 incomplete; fluid ORC disabled.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,expected in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==expected,rel
        with tempfile.TemporaryDirectory(prefix='ltn-FV-contact-time-',dir='/tmp') as directory:
            z.extractall(directory);script=str(Path(directory)/'research/reconnect/fv_contact_time_study.py')
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
            summary=subprocess.run([sys.executable,'-S',script,'--summary-replay','runs/reconnect-FV-contact-time'],
                cwd=directory,env=env,capture_output=True,text=True,timeout=60)
            assert summary.returncode==0,summary.stdout+summary.stderr
            complete=subprocess.run([sys.executable,script,'--replay','runs/reconnect-FV-contact-time'],
                cwd=directory,env=env,capture_output=True,text=True,timeout=60)
            assert complete.returncode==0,complete.stdout+complete.stderr
    for rel,expected in read(seal).items():assert digest(ROOT/rel)==expected,rel
    r=read(out/'results.json')
    info={'review':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'review_only_not_full_history':True,'archive_CRC_and_all_included_SHA256_verified':True,
        'fresh_standard_library_metric_replay_exit_code':summary.returncode,'fresh_metric_replay_log':summary.stdout,
        'fresh_NumPy_full_solver_replay_exit_code':complete.returncode,'fresh_solver_replay_log':complete.stdout,
        'review_workspace_entries':len(manifest),'scientific_manifest_entries':len(entries),
        'source_scientific_manifest':source_seal,'source_scientific_manifest_sha256':digest(source/'sha256.json'),
        'spatial_semidiscrete_relative_error':r['spatial_semidiscrete_relative_error'],
        'BE512_relative_error':r['outputs']['BE512']['relative_error_vs_PDE'],
        'BE1024_relative_error':r['outputs']['BE1024']['relative_error_vs_PDE'],
        'time_error_ratio_512_over_1024':r['time_error_ratio_512_over_1024'],
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':64,
        'new_software_checks_passed':4,'recorded_checks_total':400,'new_exact_algebra_certificate_groups':0,
        'new_FV_baseline_time_cases':1,'FV_numerical_time_outputs':3,
        'new_reference_time_solutions':0,'new_point_kernel_builds':0,'new_point_candidate_transient_runs':0,
        'new_full_2D_transport_matrices':0,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete','published':False}
    save(delivery,info)
    with (exports/'ltn-FV-contact-time-downloads.md').open('x') as f:
        f.write('# 普通FV有限时间对照：审阅与重放\n\n'
            '[代码、原状态、三个端点和审计包](ltn-FV-contact-time-review.zip)。\n\n'
            f'SHA-256：`{info["review"]["sha256"]}`。\n\n'
            '[交付范围与哈希](ltn-FV-contact-time-delivery.json)。全新解压目录标准库指标重放和NumPy完整求解重放均通过。\n\n'
            '半离散终态误差2.72439e-5，时间控制比1.99963。新增4项检查，累计400。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
