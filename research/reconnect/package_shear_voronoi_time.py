"""Seal and freshly replay the registered smooth-shear FV transient review bundle."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from audit_route_decision import ROOT, read, save, digest


def main():
    source=ROOT/'runs/reconnect-shear-voronoi-FV-frozen'
    out=ROOT/'runs/reconnect-shear-voronoi-time'
    archive=ROOT/'exports/ltn-shear-voronoi-time.zip'
    delivery=ROOT/'exports/ltn-shear-voronoi-time-delivery.json'
    downloads=ROOT/'exports/ltn-shear-voronoi-time-downloads.md'
    seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)), 'never overwrite sealed evidence'
    run=read(out/'execution.json');audit=read(out/'numerical-audit.json');audit_run=read(out/'audit-execution.json')
    result=read(out/'results.json');status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert run['tests_exit_code']==run['study_exit_code']==audit_run['audit_exit_code']==0
    assert run['scientific_main_first_attempt_passed'] and run['test_first_attempt_passed']
    assert audit['all12_geometry_caches_and_independent_whole_control_sequences_verified']
    assert audit['prior_formal_manifests_verified']==80 and status['recorded_checks_total']==445
    assert status['source_registered_smooth_shear_time_gate_executed'] and not gate['already_executed']
    assert status['new_candidate_graphs']==12 and status['new_PDE_reference_solves']==0
    assert not status['general_smooth_flow_admitted'] and not status['fluid_orc_enabled']
    prior=read(source/'sha256.json')
    assert len(prior)==1844
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+name for name in (
        'voronoi_time_protocol.json','SHEAR_VORONOI_TIME_CONTROL.md','voronoi_time_action.py',
        'test_voronoi_time_action.py','shear_voronoi_time_study.py','audit_shear_voronoi_time.py',
        'package_shear_voronoi_time.py','SHEAR_VORONOI_TIME_COMPARISON.md')]
    entries=dict(prior);source_seal=str((source/'sha256.json').relative_to(ROOT))
    entries[source_seal]=digest(source/'sha256.json')
    for rel in files:
        assert rel not in entries
        entries[rel]=digest(ROOT/rel)
    for p in out.iterdir():
        assert p.is_file(),p
        rel=str(p.relative_to(ROOT));assert rel not in entries
        entries[rel]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)
    selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update((source_seal,'docs/RESEARCH_STATUS.md','research/reconnect/audit_route_decision.py',
        'research/reconnect/periodic_voronoi_fv.py','research/reconnect/shear_voronoi_fv_study.py',
        'research/reconnect/audit_shear_voronoi_fv.py'))
    # Current byte-identical packets already contain all required original inputs
    # and references. Historical large NPZ/CSR containers stay in the workspace.
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources']
                    if Path(item['path']).suffix!='.npz')
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    final=result['observations'][-1]
    note=(
        'LTN SMOOTH NONSHOCK FIRST: one original N128/D.02/T.05 periodic shear actual scalar flow.\n'
        'Fixed4/8 chronological midpoint controls,12 exact geometry/conductance caches; original m/c0/IDs/point inputs and independent PDE reference.\n'
        f'Final FV relative concentration error:{final["FV_relative_concentration_error"]}; saved same-flow kernel:{final["saved_kernel_relative_concentration_error"]}.\n'
        f'Concentration comparison:{result["concentration_comparison_decision"]}; H improvement resolved at both checkpoints:{result["H_better_at_both_checkpoints_resolvably"]}.\n'
        'Error denominator is original initial mass-fluctuation norm, not the frozen-rate norm.\n'
        'All12 coefficient caches sealed before mass/scalar/reference loading; fixed material masses/averages never replaced by auxiliary Voronoi capacities.\n'
        'Exact Fraction periodic geometry, complete radius/bin search, no jitter/fixed-k truncation/positive-face deletion.\n'
        'Primary mass-similar symmetric sparse exponential action; independent complete chronological uniformization sequences.\n'
        'Exact ceil outgoing-rate Lambda, exact minimal Poisson-tail degree at1e-16 per interval; raw weights/rows/states never normalized.\n'
        'Independent audit reconstructs every vertex by bisectors, all shared faces by two half-resistances and all uniformization states bitwise.\n'
        'Six fixed IDs per graph check all other input points;256-bit global area enclosures.\n'
        'Original raw mass residuals, extrema, E/I/H, graph chi and Simpson energy checks retained; graph dissipation is not full continuous/subcell mixing.\n'
        'Two time levels and saved reference differences are empirical uncertainty indicators, not certified total errors or asymptotic convergence proof.\n'
        'Six new software checks passed first attempt; recorded445, previous439 not rerun. Original v0.8.0/246 not recovered.\n'
        'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/shear_voronoi_time_study.py --replay runs/reconnect-shear-voronoi-time\n'
        'Replay needs NumPy/SciPy, reconstructs sparse actions from exact sealed conductances and original masses; does not rebuild geometry or solve a reference PDE.\n'
        'Full numerical/provenance audit needs historical workspace. review-manifest.json covers included files; cumulative seal also references omitted history.\n'
        '782 originals,15 attachments,80 prior formal scientific manifests preserved; four known historical diagnostic differences retained.\n'
        'Next gate remains unexecuted; see next-gate-spec.json. No parameter matrix or automatic follow-on simulation.\n'
        'One smooth-flow time case does not admit general smooth flow, spatial convergence, shocks, reconstructed density or Stage3. Fluid ORC disabled.\n'
        'Mutable docs/RESEARCH_STATUS.md is included as a review snapshot outside the scientific seal.\n'
        'No PDF, software release, commit/push, environment configuration change or publication.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n')
        z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-shear-FV-time-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/shear_voronoi_time_study.py'),
                '--replay','runs/reconnect-shear-voronoi-time'],cwd=directory,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),
                capture_output=True,text=True,timeout=600)
            assert completed.returncode==0,completed.stdout+completed.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_transient_replay_exit_code':completed.returncode,
        'fresh_transient_replay_log':completed.stdout,'fresh_replay_files_unchanged':True,
        'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),
        'FV_final_relative_concentration_error':final['FV_relative_concentration_error'],
        'saved_kernel_final_relative_concentration_error':final['saved_kernel_relative_concentration_error'],
        'kernel_over_FV_error_ratio':final['kernel_over_FV_error_ratio'],
        'concentration_comparison_decision':result['concentration_comparison_decision'],
        'H_better_at_both_checkpoints_resolvably':result['H_better_at_both_checkpoints_resolvably'],
        'new_software_checks':6,'recorded_checks_total':445,'previous_checks_rerun':0,
        'new_actual_flow_cases':1,'new_candidate_midpoint_graphs':12,'new_candidate_transient_runs':2,
        'independent_uniformization_sequences':2,'new_PDE_reference_solves':0,'new_kernel_builds':0,
        'next_gate_executed':False,'shock_extension_deferred':True,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':80,
        'general_smooth_flow_admitted':False,'finite_node_candidate_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# 同一光滑剪切的辅助FV实际时间对照\n\n'
            '[报告、全部12张精确几何/电导、原输入与状态、审计和重放代码](ltn-shear-voronoi-time.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[交付校验](ltn-shear-voronoi-time-delivery.json)。全新解压后的同一实际时间序列重放通过。\n\n'
            f'结论：{result["concentration_comparison_decision"]}；仍不构成Stage3完整验收，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
