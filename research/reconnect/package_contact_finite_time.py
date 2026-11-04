"""Seal one fixed-time analytic PDE reference and verify an isolated replay."""
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
    source=ROOT/'runs/reconnect-FV-global-frozen';out=ROOT/'runs/reconnect-contact-finite-time-reference'
    exports=ROOT/'exports';archive=exports/'ltn-contact-finite-time-reference-review.zip'
    delivery=exports/'ltn-contact-finite-time-reference-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed evidence'
    execution=read(out/'execution.json');audit=read(out/'implementation-audit.json')
    assert execution['tests_exit_code']==execution['study_exit_code']==execution['audit_exit_code']==0
    assert audit['prior_formal_manifests_verified']==63 and audit['new_software_checks_passed']==4
    prior=read(source/'sha256.json')
    for rel,expected in prior.items():assert digest(ROOT/rel)==expected,rel
    files=['research/reconnect/'+name for name in ['CONTACT_PERIODIC_SPECTRAL_BASIS.md','CONTACT_FINITE_TIME_REFERENCE.md',
        'contact_spectral_reference.py','test_contact_spectral_reference.py','contact_finite_time_study.py',
        'audit_contact_finite_time.py','package_contact_finite_time.py']]
    entries=dict(prior);source_seal=str((source/'sha256.json').relative_to(ROOT));entries[source_seal]=digest(source/'sha256.json')
    for rel in files:entries[rel]=digest(ROOT/rel)
    for p in out.iterdir():
        assert p.is_file(),p
        entries[str(p.relative_to(ROOT))]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'] if not item['path'].endswith('.npz'))
    selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([source_seal,'research/reconnect/audit_route_decision.py','docs/RESEARCH_STATUS.md'])
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=(
        'LTN independent same-contact finite-time material-mean PDE reference at exact T=1/20. REVIEW ONLY.\n'
        'Complete weighted transmission basis proved before evaluation; nonzero normal constant mode retained.\n'
        'Four predeclared controls: cutoffs 64/128, output precision 60/90 digits. No parameter search.\n'
        'Reference projections do not accept saved c0, FV/kernel action or errors as inputs.\n'
        'All original mass/c0/ID/position/rho bytes retained in the new NPZ. Initial bridge and inventory gap retained.\n'
        'Bridge contraction is conditional on an explicit cellwise-constant perturbation lift, not arbitrary subcell data.\n'
        'Rigorous tail bound concerns analytic truncation only. Decimal checks and combined indicator are not interval certification.\n'
        'Standalone tests: python -S research/reconnect/test_contact_spectral_reference.py\n'
        'Standalone full replay: python -S research/reconnect/contact_finite_time_study.py --replay runs/reconnect-contact-finite-time-reference\n'
        'Includes JSON factors, exact tail certificates, reference NPZ, retained source arrays and audits. No NumPy needed for replay.\n'
        'The NPZ 0.05 time is a label only; exact reference time is 1/20.\n'
        '782 originals, 15 historical attachments and 63 prior formal manifests preserved; four historical diagnostic differences retained separately.\n'
        'Full historical source audit needs the original workspace. Full old snapshots and attachments are omitted.\n'
        'review-manifest.json covers included files. Cumulative scientific seal also covers unarchived history.\n'
        'docs/RESEARCH_STATUS.md is a mutable snapshot outside the scientific seal.\n'
        'Four new software checks passed; historical 392 not rerun; recorded total 396. No added algebra-group count.\n'
        'One analytic PDE time-reference case, four controls; no FV/candidate propagation or reference time stepping/quadrature.\n'
        'No clipping, renormalization, fitted F/C, thresholds, PDF, release, commit, push or publication.\n'
        'Next ordinary FV fixed-time comparison is predeclared, not executed here. Point-kernel transient remains unadmitted.\n'
        'Stage 3 incomplete; fluid ORC disabled.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,expected in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==expected,rel
        with tempfile.TemporaryDirectory(prefix='ltn-contact-reference-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,'-S',str(Path(directory)/'research/reconnect/contact_finite_time_study.py'),
                '--replay','runs/reconnect-contact-finite-time-reference'],cwd=directory,
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
        'reference_initial_bridge_serialization_and_precision_indicator_mass_norm':result['reference_initial_bridge_serialization_and_precision_indicator_mass_norm'],
        'indicator_not_a_certified_total_error_bound':True,'bridge_requires_explicit_perturbation_lift':True,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':63,
        'new_software_checks_passed':4,'recorded_checks_total':396,'new_exact_algebra_certificate_groups':0,
        'reference_physical_time_cases':1,'reference_control_evaluations':4,
        'new_candidate_or_FV_time_propagations':0,'new_reference_time_stepping_solves':0,
        'new_reference_numerical_integrations':0,'new_kernel_builds':0,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete','published':False}
    save(delivery,info)
    with (exports/'ltn-contact-finite-time-reference-downloads.md').open('x') as f:
        f.write('# 同一接触有限时间参考：审阅与重放\n\n'
            '[推导、代码、原初值、参考均值与审计包](ltn-contact-finite-time-reference-review.zip)。\n\n'
            f'SHA-256：`{info["review"]["sha256"]}`。\n\n'
            '[交付范围与哈希](ltn-contact-finite-time-reference-delivery.json)。全新解压目录python -S重放通过。\n\n'
            '固定T=1/20的独立PDE参考；没有FV/点核传播。新4项检查通过，累计396。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
