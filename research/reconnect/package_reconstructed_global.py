"""Seal one original-snapshot global frozen experiment and replay it."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

from audit_route_decision import ROOT,read,save,digest


def main():
    source=ROOT/'runs/reconnect-global-frozen-plan';out=ROOT/'runs/reconnect-reconstructed-global-frozen'
    archive=ROOT/'exports/ltn-reconstructed-global-frozen.zip'
    delivery=ROOT/'exports/ltn-reconstructed-global-frozen-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed artifacts'
    status=read(out/'stage3-status.json');audit=read(out/'receipt-audit.json');execution=read(out/'execution.json')
    assert execution['study_exit_code']==execution['tests_exit_code']==0
    assert read(out/'audit-execution.json')['audit_exit_code']==0
    assert audit['prior_formal_manifests_verified']==69 and status['recorded_checks_total']==417
    assert audit['independent_object_integer_periodic_shift_assembly_matches_every_primary_flux_numerator']
    prior=read(source/'sha256.json')
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    names=['cached_global_exchange.py','test_cached_global_exchange.py','reconstructed_global_study.py',
        'audit_reconstructed_global.py','package_reconstructed_global.py','RECONSTRUCTED_PATH_GLOBAL_FROZEN.md']
    files=['research/reconnect/'+name for name in names]
    source_seal=str((source/'sha256.json').relative_to(ROOT))
    entries=dict(prior);entries[source_seal]=digest(source/'sha256.json')
    for rel in files:entries[rel]=digest(ROOT/rel)
    for p in out.iterdir():
        assert p.is_file(),p
        entries[str(p.relative_to(ROOT))]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)
    selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([source_seal,'research/reconnect/audit_route_decision.py','research/reconnect/global_frozen_gate_plan.py',
        'research/reconnect/reconstructed_pair_study.py','research/reconnect/reconstructed_path_rows.py',
        'research/reconnect/periodic_density_path.py','research/reconnect/endpoint_density_path.py',
        'research/reconnect/positive_kernel_2d.py','docs/RESEARCH_STATUS.md'])
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'] if not item['path'].endswith('.npz'))
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=(
        'LTN one executed original N128 density-contact full-domain frozen-rate comparison.\n'
        'D=.02, h=.04, all original fixed material-mean c0/mass/IDs, saved PDE reference/regions and locked norm unchanged.\n'
        '132 exact representative integrals, 45 original kernel radii, 964 coefficient slots cover all 2,719,744 undirected edges.\n'
        'Integral-only reuse; original ID ownership receipts belong only to actual representative paths.\n'
        'All 663 old exact edge coefficients and both exact row actions matched before global evaluation.\n'
        'Exact common-integer shared action; independent object-integer periodic-shift audit agrees for all nodes and precision controls.\n'
        'Global frozen-rate relative error .14156491268558014 -> .034647731611281644 (75.52519833% reduction).\n'
        'Ordinary FV context .032630071085544846 is still lower; no global/transient superiority claim.\n'
        'This is a frozen-rate result, NOT a finite-time scalar or mixing acceptance. Stage 3 incomplete, fluid ORC disabled.\n'
        'Replay (NumPy/SciPy): python research/reconnect/reconstructed_global_study.py --replay runs/reconnect-reconstructed-global-frozen\n'
        'Replay rebuilds the same original cache, old pair control, exact global action and arithmetic controls; not a new physical case.\n'
        'Seven new implementation checks pass; recorded checks 410 -> 417. Historical suite not rerun.\n'
        '782 originals, 15 attachments and 69 prior formal manifests preserved; four historical diagnostic differences unchanged.\n'
        'Included original input copies permit standalone scientific replay; complete historical source/attachment audit needs the workspace.\n'
        'global-manifest.json covers included files; cumulative science seal also references omitted historical files.\n'
        'docs/RESEARCH_STATUS.md is mutable and outside the cumulative science seal.\n'
        'No clipping, renormalization, fitting, threshold change, new PDE, grid/h/time scan, transient, PDF, release or publication.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('global-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'global-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-global-frozen-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/reconstructed_global_study.py'),
                '--replay','runs/reconnect-reconstructed-global-frozen'],cwd=directory,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=600)
            assert completed.returncode==0,completed.stdout+completed.stderr
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    result=read(out/'results.json')
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_same_global_case_replay_exit_code':completed.returncode,
        'fresh_same_global_case_replay_log':completed.stdout,'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),
        'source_scientific_manifest':source_seal,'new_software_checks':7,'recorded_checks_total':417,'previous_checks_rerun':0,
        'new_fixed_global_frozen_physical_configurations':1,'logical_undirected_edges':2719744,
        'representative_path_integrals':132,'coefficient_cache_slots':964,'new_sparse_matrices':0,
        'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':69,
        'original_kernel_global_error':result['saved_kernel_global_error'],
        'reconstructed_global_error':result['reconstructed_global_error'],'decision':result['decision'],
        'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with (ROOT/'exports/ltn-reconstructed-global-frozen-downloads.md').open('x') as f:
        f.write('# 原N128路径候选：全域冻结变化率比较\n\n'
            '[报告、代码、原输入副本、精确作用与审计](ltn-reconstructed-global-frozen.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[交付范围与校验](ltn-reconstructed-global-frozen-delivery.json)。全新解压目录同一配置完整科学重放通过。\n\n'
            '新增7项软件检查，记录417；仅一档全域冻结作用，没有瞬态或新PDE求解。Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
