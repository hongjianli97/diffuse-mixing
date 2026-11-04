"""Seal the single common-observation spatial control and its replays."""
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
    source=ROOT/'runs/reconnect-FV-contact-time';out=ROOT/'runs/reconnect-FV-contact-spatial'
    exports=ROOT/'exports';archive=exports/'ltn-FV-contact-spatial-review.zip'
    delivery=exports/'ltn-FV-contact-spatial-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed evidence'
    execution=read(out/'execution.json');audit=read(out/'implementation-audit.json')
    assert execution['tests_exit_code']==execution['study_exit_code']==execution['audit_exit_code']==0
    assert audit['prior_formal_manifests_verified']==65 and audit['new_software_checks_passed']==3
    prior=read(source/'sha256.json')
    for rel,expected in prior.items():assert digest(ROOT/rel)==expected,rel
    files=['research/reconnect/'+name for name in ['FV_CONTACT_COMMON_OBSERVATION.md','nested_material_observation.py',
        'test_nested_material_observation.py','fv_contact_spatial_study.py','audit_fv_contact_spatial.py','package_fv_contact_spatial.py']]
    entries=dict(prior);source_seal=str((source/'sha256.json').relative_to(ROOT));entries[source_seal]=digest(source/'sha256.json')
    for rel in files:entries[rel]=digest(ROOT/rel)
    for p in out.iterdir():
        assert p.is_file(),p
        entries[str(p.relative_to(ROOT))]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'])
    selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([source_seal,'research/reconnect/audit_route_decision.py','research/reconnect/periodic_density_path.py',
        'research/reconnect/contact_spectral_reference.py','docs/RESEARCH_STATUS.md'])
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=(
        'LTN one N256 ordinary FV spatial control on original N128 material observation. REVIEW ONLY.\n'
        'Same physical contact, exact target T=1/20, original coarse mass/c0/reference/endpoint bytes retained.\n'
        'Fine inputs are newly derived true material integrals, NOT recovered original samples or coarse point interpolation.\n'
        'Original continuum spectral coefficients and tail bound reused; no PDE solve or projection fit.\n'
        'Exact Fraction mass restriction requires all child masses to equal original coarse mass; no repair or normalization.\n'
        'Initial/reference bridges, Jensen checks, raw zero-mode eigenvalue and inventory drift retained.\n'
        'One fine semidiscrete endpoint, no new time-step controls, grid scan or kernel build.\n'
        'Standard-library observation/metric replay: python -S research/reconnect/fv_contact_spatial_study.py --summary-replay runs/reconnect-FV-contact-spatial\n'
        'Complete NumPy replay: OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python research/reconnect/fv_contact_spatial_study.py --replay runs/reconnect-FV-contact-spatial\n'
        'Solver replay has a predeclared epsilon-based arithmetic tolerance, not a physical acceptance threshold.\n'
        'requirements-solver.txt pins NumPy. Full historical independent audit also needs SciPy and the original workspace.\n'
        'Three new checks passed; historical 400 not rerun; recorded total 403.\n'
        '782 originals, 15 attachments, 65 prior formal manifests preserved; four historical diagnostic differences retained separately.\n'
        'A two-level log2 ratio is an observation, not a general convergence theorem. Indicators are not certified total error bounds.\n'
        'This archive excludes full old snapshots and attachments. Scientific seal includes unarchived history.\n'
        'review-manifest.json covers included files. docs/RESEARCH_STATUS.md is a mutable snapshot outside the scientific seal.\n'
        'Next point-input information-contract review is predeclared, not executed here. Stop further FV grid expansion.\n'
        'No clipping, inventory repair, fitted F/C, PDF, release, commit, push, environment reconfiguration or publication.\n'
        'Stage 3 incomplete; fluid ORC disabled; point candidate transient unadmitted.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,expected in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==expected,rel
        with tempfile.TemporaryDirectory(prefix='ltn-FV-contact-spatial-',dir='/tmp') as directory:
            z.extractall(directory);script=str(Path(directory)/'research/reconnect/fv_contact_spatial_study.py')
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
            summary=subprocess.run([sys.executable,'-S',script,'--summary-replay','runs/reconnect-FV-contact-spatial'],
                cwd=directory,env=env,capture_output=True,text=True,timeout=60)
            assert summary.returncode==0,summary.stdout+summary.stderr
            complete=subprocess.run([sys.executable,script,'--replay','runs/reconnect-FV-contact-spatial'],
                cwd=directory,env=env,capture_output=True,text=True,timeout=60)
            assert complete.returncode==0,complete.stdout+complete.stderr
    for rel,expected in read(seal).items():assert digest(ROOT/rel)==expected,rel
    r=read(out/'results.json')
    info={'review':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'review_only_not_full_history':True,'archive_CRC_and_all_included_SHA256_verified':True,
        'fresh_standard_library_observation_replay_exit_code':summary.returncode,'fresh_observation_replay_log':summary.stdout,
        'fresh_NumPy_solver_replay_exit_code':complete.returncode,'fresh_solver_replay_log':complete.stdout,
        'review_workspace_entries':len(manifest),'scientific_manifest_entries':len(entries),
        'source_scientific_manifest':source_seal,'source_scientific_manifest_sha256':digest(source/'sha256.json'),
        'coarse128_error':r['coarse128_error'],
        'fine256_error_under_original128_observation':r['fine256_error_under_original128_observation'],
        'coarse_error_divided_by_observed_fine_error':r['coarse_error_divided_by_observed_fine_error'],
        'two_level_log2_ratio_is_not_a_general_order_theorem':True,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':65,
        'new_software_checks_passed':3,'recorded_checks_total':403,'new_exact_algebra_certificate_groups':0,
        'new_fine_FV_time_cases':1,'new_time_step_controls':0,'new_PDE_time_solutions_or_projection_fits':0,
        'new_point_kernel_builds':0,'new_point_candidate_transient_runs':0,'new_full_2D_transport_matrices':0,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete','published':False}
    save(delivery,info)
    with (exports/'ltn-FV-contact-spatial-downloads.md').open('x') as f:
        f.write('# 一档空间控制与共同物质观测：审阅与重放\n\n'
            '[代码、原粗观测、新细数据与审计包](ltn-FV-contact-spatial-review.zip)。\n\n'
            f'SHA-256：`{info["review"]["sha256"]}`。\n\n'
            '[交付范围与哈希](ltn-FV-contact-spatial-delivery.json)。全新解压目录标准库观测重放和NumPy完整求解重放均通过。\n\n'
            '共同观测误差：N128为2.72439e-5，N256为6.80651e-6，比4.00262。仅一档观察。新增3项检查，累计403。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
