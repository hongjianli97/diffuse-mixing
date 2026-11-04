"""Seal the restricted four-mean decision and verify a fresh JSON replay."""
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
    source=ROOT/'runs/reconnect-FV-interface-expansion'
    out=ROOT/'runs/reconnect-four-mean-interface'
    exports=ROOT/'exports'
    archive=exports/'ltn-four-mean-interface-review.zip'
    delivery=exports/'ltn-four-mean-interface-delivery.json'
    seal=out/'sha256.json'
    assert not any(path.exists() for path in [archive,delivery,seal]),'never overwrite sealed science'
    execution=read(out/'execution.json')
    assert execution['study_exit_code'] == execution['audit_exit_code'] == 0
    audit=read(out/'implementation-audit.json')
    assert audit['prior_formal_manifests_verified'] == 61
    assert audit['new_exact_algebra_certificate_groups'] == 4 and not audit['four_mean_correction_admitted']
    prior=read(source/'sha256.json')
    for rel,expected in prior.items():
        assert digest(ROOT/rel) == expected,rel
    new_files=['research/reconnect/'+name for name in [
        'FOUR_MEAN_INTERFACE_ADMISSIBILITY.md','four_mean_interface_review.py',
        'four_mean_interface_study.py','audit_four_mean_interface.py','package_four_mean_interface.py']]
    source_seal=str((source/'sha256.json').relative_to(ROOT))
    entries=dict(prior)
    entries[source_seal]=digest(source/'sha256.json')
    for rel in new_files:
        entries[rel]=digest(ROOT/rel)
    for path in out.iterdir():
        assert path.is_file(),path
        entries[str(path.relative_to(ROOT))]=digest(path)
    save(seal,dict(sorted(entries.items())))
    selected=set(new_files)
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'])
    selected.update(str(path.relative_to(ROOT)) for path in out.iterdir())
    selected.update([source_seal,'research/reconnect/audit_route_decision.py',
                     'research/rebuild/CONTACT_TRANSMISSION.md','docs/RESEARCH_STATUS.md'])
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=(
        'LTN four-material-mean linear interface admissibility review. REVIEW ONLY.\n'
        'Formal coefficients g/4*[1,-7,7,-1] satisfy the four predeclared exactness conditions uniquely.\n'
        'Unchanged outer closures yield negative remote offdiagonals. This correction is NOT admitted or installed.\n'
        'The result is restricted to this support, linear shared face functional, exactness and unchanged other faces.\n'
        'Not a no-go for all positive methods, nonlinear methods, other stencils or weak/transient convergence.\n'
        'Original c0 and acceptance criteria retained. No fitted curvature, c0 error fit, clipping or renormalization.\n'
        'Standalone standard-library replay: python -S research/reconnect/four_mean_interface_study.py --replay runs/reconnect-four-mean-interface\n'
        'Exact certificates only: python -S research/reconnect/four_mean_interface_study.py --certificates-only runs/reconnect-four-mean-interface/review-inputs.json\n'
        'Four new algebra certificate groups are separate from the unchanged 388 recorded software checks.\n'
        '782 originals, 15 attachments and 61 prior formal manifests preserved; four historical diagnostic differences separately retained.\n'
        'Full preservation audit requires historical workspace. This package omits full raw snapshots and historical attachments.\n'
        'review-manifest.json covers included files. Cumulative scientific seal also covers unarchived history.\n'
        'docs/RESEARCH_STATUS.md is a mutable snapshot outside the scientific seal.\n'
        'Next: one same-case global ordinary FV frozen action; not executed here. No new kernel or contact transient admission.\n'
        'No new transport weights, graph, global transport matrix, numerical reference integration, flow, transient, PDF, release or publication.\n'
        'Stage 3 incomplete; fluid ORC disabled.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n')
        z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,expected in manifest.items():
            assert hashlib.sha256(z.read(rel)).hexdigest() == expected,rel
        with tempfile.TemporaryDirectory(prefix='ltn-four-mean-interface-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,'-S',str(Path(directory)/'research/reconnect/four_mean_interface_study.py'),
                '--replay','runs/reconnect-four-mean-interface'],cwd=directory,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
            assert completed.returncode == 0,completed.stdout+completed.stderr
    for rel,expected in read(seal).items():
        assert digest(ROOT/rel) == expected,rel
    info={'review':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'review_only_not_full_history':True,'archive_CRC_and_all_included_SHA256_verified':True,
        'fresh_archive_python_no_site_packages_replay_exit_code':completed.returncode,
        'fresh_archive_replay_log':completed.stdout,
        'review_workspace_entries':len(manifest),'scientific_manifest_entries':len(entries),
        'source_scientific_manifest':source_seal,'source_scientific_manifest_sha256':digest(source/'sha256.json'),
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':61,
        'new_exact_algebra_certificate_groups':4,'new_numerical_or_flow_tests':0,'recorded_checks_total':388,
        'new_edge_weights':0,'new_candidate_graphs':0,'new_global_matrices':0,'new_reference_integrations':0,
        'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'four_mean_correction_admitted':False,'fluid_orc_enabled':False,
        'stage3_physical_acceptance':'incomplete','published':False}
    save(delivery,info)
    with (exports/'ltn-four-mean-interface-downloads.md').open('x') as f:
        f.write('# 四物质均值线性重构：审阅与重放\n\n'
          '[推导、精确输入、符号证书与审计包](ltn-four-mean-interface-review.zip)。\n\n'
          f'SHA-256：`{info["review"]["sha256"]}`。\n\n'
          '[交付范围与哈希](ltn-four-mean-interface-delivery.json)。全新解压目录python -S重放通过。\n\n'
          '该线性修正未接纳、未安装。4组代数证书单独计数，软件检查仍388。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__ == '__main__':main()
