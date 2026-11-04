"""Seal the registered affine space control and replay only its twelve new paths."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure


def main():
    source=ROOT/'runs/reconnect-t3-affine-review';out=ROOT/'runs/reconnect-t3-affine-spatial'
    archive=ROOT/'exports/ltn-t3-affine-spatial.zip';delivery=ROOT/'exports/ltn-t3-affine-spatial-delivery.json'
    downloads=ROOT/'exports/ltn-t3-affine-spatial-downloads.md';seal=out/'sha256.json'
    assert not any(path.exists() for path in (archive,delivery,downloads,seal)),'sealed evidence is never overwritten'
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'numerical-audit.json');r=read(out/'results.json')
    status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert ex['tests_exit_code']==ex['study_exit_code']==ar['audit_exit_code']==0
    assert ex['main_first_attempt_passed'] and ex['tests_first_attempt_passed'] and ar['audit_first_attempt_passed']
    assert audit['all112_independent_world_geometries_exact_rates_and_full_raw_sequences_verified'] and audit['prior_formal_manifests_verified']==95
    assert status['recorded_checks_total']==483 and not status['fluid_orc_enabled'] and not status['general_T3_admitted']
    assert not gate['already_executed'] and gate['new_candidate_graphs']==gate['new_candidate_transient_runs']==0
    assert not r['per_grid_controls']['32']['T3_registered_affine_case_passed'] and not status['original_N32_case_passed']
    prior=read(source/'sha256.json');assert len(prior)==4545
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    assert digest(ROOT/'exports/ltn-t3-affine-review.zip')=='4e1639b16f73424dcdbba03a8b57f8391a2ad33b06ae535b89f6d3f67abf1482'
    assert digest(ROOT/'exports/ltn-t3-affine.zip')=='f58a50adc6c9b71e42351655330e5dc87670e425ce0564d44fab42bc1d213205'
    files=['research/reconnect/'+name for name in ['t3_affine_spatial_core.py','test_t3_affine_spatial.py','t3_affine_spatial_protocol.json',
      't3_affine_spatial_study.py','audit_t3_affine_spatial_geometry.py','audit_t3_affine_spatial.py','plot_t3_affine_spatial.py',
      'document_t3_affine_spatial.py','package_t3_affine_spatial.py','T3_AFFINE_SPATIAL_COMPARISON.md']]
    entries=dict(prior);source_seal=str((source/'sha256.json').relative_to(ROOT));entries[source_seal]=digest(source/'sha256.json')
    for rel in files:assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for path in out.rglob('*'):
        if path.is_file():
            rel=str(path.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(path)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)|{str(path.relative_to(ROOT)) for path in out.rglob('*') if path.is_file()}|dependency_closure([name for name in files if name.endswith('.py')])
    selected.update([source_seal,'docs/RESEARCH_STATUS.md','requirements-reference.txt'])
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'] if Path(item['path']).suffix!='.zip')
    selected.update(item['path'] for item in gate['sources'] if Path(item['path']).suffix not in ('.npz','.zip'))
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=f'''LTN T3 AFFINE COMPRESSION: one registered three-space-level same-backend control.
Decision: {r['comparison_decision']}. Stage3 incomplete; fluid ORC off.
Original N16/32/64 fixed material masses, finite-cell initial means and identities preserved.
Same two physical PDEs: a=1-5t, periods(a,1), rho_exact=1/a; constant D=.02 vs constant Gamma with D_exact=Gamma0*a.
Raw rho/D float conversion defects retained and controlled by exact input Duhamel indicators.
112 new N16/N64 local/world graphs all sealed before actual material/scalar/reference or saved N32 path load.
12 new actual and12 complete independent paths at K4/8/16; all six old N32 paths copied and loaded only, not rerun.
Unchanged geometry/ordinary center resistance/time actions/exact orthogonal boundary adapter; no F/C fit or true internal face input.
Independent world bisectors/faces/half-resistances/areas/probes on all112 graphs; all exact rates/minimal1e-16 tails/raw weights/full independent sequences checked.
Independent80/100-digit initial-cell and both physical-clock references on both new grids passed; no new reference PDE solve.
Common observation uses original N16 fixed regions, direct child component integrals divided by original parent masses, with all initial/reference projection bridges retained.
At both times, all adjacent native/common concentration and native/common H gains are compared to fixed positive time/input/numerical/reference/bridge budgets.
Spatial H comparison uses the preregistered original T1 structure of direct absolute time H differences plus original quadratic H conversion of numerical/input/reference allowances.
Fine N64 model H response uses the unchanged original full model indicator including spatial terms; no tighter H formula or threshold change.
Original N32 t=.025 H model-response decision remains inconclusive; no retroactive pass.
429 common snapshots and2145 fixed exact parent-integral probes independently checked; no mass, density, state, row or weight normalization.
Three new software checks passed first attempt; recorded483; old480 not rerun; old v0.8.0/246 still not recovered.
Main384.685s /371.65MiB within900s/1024MiB; independent world/numerical audit resources in their receipts.
782 original files,15 historical attachments,95 prior formal scientific manifests and both previous T3 deliveries verified unchanged.
Four historical diagnostic differences retained. No original README/METHOD/CODEX_TASK/PHASE, attachment, old code/result/report changed; only mutable RESEARCH_STATUS updated.
Fresh replay:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/t3_affine_spatial_study.py --replay runs/reconnect-t3-affine-spatial
Replay rebuilds only twelve new actual/independent paths, checks all endpoints/halfstates/rawCSR/weights, references, common integrals, budgets and retained old N32 decision.
No geometry, PDE or old candidate rerun. Old N32 data are loaded only. Full historical preservation audit needs original workspace.
review-manifest.json covers included files; cumulative science seal also references omitted history.
Next gate only registered: read-only original-criteria/same-backend T1-T4/T3 direction/shear coverage review; zero new numerical tests, unexecuted.
No automatic further affine grid/time/model scan. No clipping, point repair, F/C fit, tolerance relaxation, PDF, version, commit/push, environment change or publication.
General T3/smooth flow/finite-node admission false; no kappa computed. 2026-11-02 Asia/Shanghai.
'''
    print(f'sealed {len(entries)} cumulative entries; packing {len(manifest)} review files',flush=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    print('archive written; checking CRC, every included SHA and fresh twelve-path replay',flush=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-t3-spatial-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            result=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/t3_affine_spatial_study.py'),'--replay','runs/reconnect-t3-affine-spatial'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=600)
            assert result.returncode==0,result.stdout+result.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'archive_CRC_and_all_included_SHA256_verified':True,'fresh_replay_exit_code':result.returncode,'fresh_replay_log':result.stdout,'fresh_replay_files_unchanged':True,
      'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'comparison_decision':r['comparison_decision'],
      'T3_registered_three_space_level_affine_control_passed':r['T3_registered_three_space_level_affine_control_passed'],
      'original_N32_inconclusive_preserved':True,'old_N32_paths_rerun':False,
      'final_native_error_own_initial_fluctuation':{str(N):{model:r['per_grid_observations'][f'n{N}_{model}'][-1]['native_error_native_denominator'] for model in ('constant_D','constant_Gamma')} for N in (16,32,64)},
      'fine_N64_model_comparisons':r['per_grid_controls']['64']['model_comparisons'],
      'new_software_checks':3,'recorded_checks_total':483,'previous_checks_rerun':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':True,'audit_first_attempt_passed':True,
      'new_candidate_graphs':112,'new_candidate_transient_runs':12,'new_complete_independent_uniformization_sequences':12,'reused_N32_saved_paths':6,
      'new_raw_action_intervals':112,'old_candidate_reruns':0,'new_reference_PDE_solves':0,'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':95,
      'next_gate_executed':False,'next_gate_new_numerical_runs':0,'original_backends_changed':False,'general_T3_admitted':False,'general_smooth_flow_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# T3仿射压缩：三档空间对照\n\n[代码、报告、112张图、12条实际/独立新路径、原N32保存路径、参考和完整审计](ltn-t3-affine-spatial.zip)。\n\n'
          f'SHA-256：`{info["archive"]["sha256"]}`。\n\n[交付核验](ltn-t3-affine-spatial-delivery.json)。CRC/逐文件SHA及全新解压12条新路径与半步状态逐位重放通过；N32只加载，不重跑。\n\n'
          '本次登记的仿射压缩三档对照通过，原N32单例未决结论保持。完整Stage3未通过，fluid ORC关闭；下一项只读覆盖复核未执行。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
