"""Seal the one-state global FV comparison and independently extract/replay."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

from audit_route_decision import ROOT, digest, read, save


def main():
    source=ROOT/'runs/reconnect-four-mean-interface';out=ROOT/'runs/reconnect-FV-global-frozen'
    exports=ROOT/'exports';archive=exports/'ltn-FV-global-frozen-review.zip'
    delivery=exports/'ltn-FV-global-frozen-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed evidence'
    execution=read(out/'execution.json')
    assert execution['tests_exit_code']==execution['study_exit_code']==execution['audit_exit_code']==0
    audit=read(out/'implementation-audit.json')
    assert audit['prior_formal_manifests_verified']==62 and audit['new_software_checks_passed']==4
    prior=read(source/'sha256.json')
    for rel,expected in prior.items():assert digest(ROOT/rel)==expected,rel
    files=['research/reconnect/'+name for name in ['FV_GLOBAL_FROZEN_COMPARISON.md','cartesian_fv_stream.py',
        'test_cartesian_fv_stream.py','fv_global_frozen_study.py','audit_fv_global_frozen.py','package_fv_global_frozen.py']]
    entries=dict(prior);source_seal=str((source/'sha256.json').relative_to(ROOT))
    entries[source_seal]=digest(source/'sha256.json')
    for rel in files:entries[rel]=digest(ROOT/rel)
    for p in out.iterdir():
        assert p.is_file(),p
        entries[str(p.relative_to(ROOT))]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'] if not item['path'].endswith('.npz'))
    selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([source_seal,'research/reconnect/audit_route_decision.py','research/reconnect/periodic_density_path.py',
        'research/reconnect/FV_MASS_MEAN_INTERFACE_EXPANSION.md','docs/RESEARCH_STATUS.md'])
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=(
        'LTN global ordinary FV frozen comparison on one unchanged saved periodic density contact. REVIEW ONLY.\n'
        'Original material-cell mass means, original global PDE norm and original masks retained.\n'
        'Main action streams 32768 shared faces exactly. Seven old faces and two old rows are controls, not the full backend.\n'
        'No global matrix, new point-kernel build, reference integration, time propagation, clipping or normalization.\n'
        'True Cartesian material geometry is an extra-information baseline prior, not generic point-only runtime geometry.\n'
        'Standalone tests: python -S research/reconnect/test_cartesian_fv_stream.py\n'
        'Standalone full replay: python -S research/reconnect/fv_global_frozen_study.py --replay runs/reconnect-FV-global-frozen\n'
        'Contains complete JSON inputs, exact action, selected original observation NPZ, results and audits.\n'
        'NumPy and original CSR arrays are needed only for full historical/source audit; standalone replay uses neither.\n'
        'Four new software checks passed; historical 388 not rerun; recorded total 392. No new algebra certificate groups.\n'
        '782 originals, 15 attachments, 62 prior formal manifests preserved; four historical diagnostic differences retained separately.\n'
        'Uncertainty indicators are not a certified total physical error bound or a changed acceptance threshold.\n'
        'This package omits full raw graph and historical attachments. Cumulative scientific seal includes unarchived history.\n'
        'review-manifest.json covers included files. docs/RESEARCH_STATUS.md is a mutable snapshot outside the scientific seal.\n'
        'Four-mean correction not installed. No old v0.8.0 recovery claim, PDF, release, commit, push or publication.\n'
        'Next fixed-time independent PDE reference is predeclared; not evaluated here.\n'
        'Stage 3 incomplete; fluid ORC disabled; contact point-kernel transient not admitted.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,expected in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==expected,rel
        with tempfile.TemporaryDirectory(prefix='ltn-FV-global-frozen-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,'-S',str(Path(directory)/'research/reconnect/fv_global_frozen_study.py'),
                '--replay','runs/reconnect-FV-global-frozen'],cwd=directory,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
            assert completed.returncode==0,completed.stdout+completed.stderr
    for rel,expected in read(seal).items():assert digest(ROOT/rel)==expected,rel
    result=read(out/'results.json')
    info={'review':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'review_only_not_full_history':True,'archive_CRC_and_all_included_SHA256_verified':True,
        'fresh_archive_python_no_site_packages_replay_exit_code':completed.returncode,
        'fresh_archive_replay_log':completed.stdout,'review_workspace_entries':len(manifest),
        'scientific_manifest_entries':len(entries),'source_scientific_manifest':source_seal,
        'source_scientific_manifest_sha256':digest(source/'sha256.json'),
        'ordinary_FV_global_relative_error':result['ordinary_FV_global_relative_error'],
        'saved_kernel_global_relative_error':result['saved_kernel_global_relative_error'],
        'reduction_percent_of_saved_kernel_error':result['reduction_percent_of_saved_kernel_error'],
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':62,
        'new_software_checks_passed':4,'recorded_checks_total':392,'new_exact_algebra_certificate_groups':0,
        'main_FV_shared_face_evaluations':32768,'new_FV_global_frozen_actions':1,
        'new_point_kernel_builds':0,'new_global_matrices':0,'new_reference_integrations':0,
        'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete','published':False}
    save(delivery,info)
    with (exports/'ltn-FV-global-frozen-downloads.md').open('x') as f:
        f.write('# 原c0全域FV冻结对照：审阅与重放\n\n'
          '[代码、原观测、精确作用和审计包](ltn-FV-global-frozen-review.zip)。\n\n'
          f'SHA-256：`{info["review"]["sha256"]}`。\n\n'
          '[交付范围与哈希](ltn-FV-global-frozen-delivery.json)。全新解压目录python -S重放通过。\n\n'
          'FV全域冻结误差3.2630%，保存核14.1565%。4项新软件检查通过，累计392。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
