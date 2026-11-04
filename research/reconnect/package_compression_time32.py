"""Seal the single necessary32-interval control and replay a fresh archive."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure


def main():
    source=ROOT/'runs/reconnect-compression-voronoi-time';out=ROOT/'runs/reconnect-compression-time32'
    archive=ROOT/'exports/ltn-compression-time32.zip';delivery=ROOT/'exports/ltn-compression-time32-delivery.json'
    downloads=ROOT/'exports/ltn-compression-time32-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)), 'never overwrite closed evidence'
    ex=read(out/'execution.json');audit=read(out/'numerical-audit.json');ar=read(out/'audit-execution.json');r=read(out/'results.json')
    status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert ex['tests_exit_code']==ex['study_exit_code']==ar['audit_exit_code']==0
    assert ex['main_first_attempt_passed'] and not ex['tests_first_attempt_passed']
    assert read(out/'tests-first-attempt.json')['failures']==1 and read(out/'tests.json')['successful']
    assert audit['all32_geometries_exact_rates_and_complete_raw_sequence_verified'] and audit['prior_formal_manifests_verified']==85
    assert status['recorded_checks_total']==463 and not status['fluid_orc_enabled'] and not status['general_smooth_flow_admitted'] and not gate['already_executed']
    prior=read(source/'sha256.json');assert len(prior)==2500
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+n for n in ['compression_time32_study.py','test_compression_time32.py','compression_time32_protocol.json',
      'COMPRESSION_TIME32_CONTROL.md','audit_compression_time32_geometry.py','audit_compression_time32.py',
      'package_compression_time32.py','plot_compression_time32.py','COMPRESSION_TIME32_COMPARISON.md']]
    source_seal=str((source/'sha256.json').relative_to(ROOT));entries=dict(prior);entries[source_seal]=digest(source/'sha256.json')
    for rel in files:
        assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for p in out.rglob('*'):
        if p.is_file():
            rel=str(p.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files);selected.update(str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file())
    selected.update(dependency_closure([p for p in files if p.endswith('.py')]));selected.update([source_seal,'docs/RESEARCH_STATUS.md'])
    for row in read(out/'evidence-index.json')['sources']:
        if not row['path'].endswith('.npz'):selected.add(row['path'])
    manifest={p:digest(ROOT/p) for p in sorted(selected)}
    note=(
      'LTN: SINGLE REGISTERED NECESSARY32-INTERVAL SMOOTH COMPRESSION CONTROL.\n'
      f'Decision:{r["comparison_decision"]}; Stage3 incomplete, fluid ORC off.\n'
      'Same original N128 material mass averages,masses,c0,IDs,D=.02,T=.05 and independent reference.\n'
      'All old4/8/16 pure time/positions/rho records and nested32 subsets match bitwise before geometry.\n'
      'All32 exact ordinary periodic Voronoi face-resistance graphs sealed before mass,c0,reference or old-state loading.\n'
      'One new actual32 path and one independent complete uniformization path; raw weights, original1e-16 tail per interval and machine formula.\n'
      'Old4/8/16 actual and independent states reused byte-identically, no original candidate or PDE rerun.\n'
      'Fixed positive-sum16/32,time/reference/initial/new-and-old numerical controls retained; no extrapolation,fitting,clipping,point repair or normalization.\n'
      'Original kernel4/8 local direct/two-half checks are local controls, not full independent paths or certified total error bounds.\n'
      'Independent all32 geometry/exact-rate and complete raw uniformization audits; original input producer and both indicators independently checked.\n'
      'Two new software checks ultimately passed, recorded463; old461 not rerun.\n'
      'First test had an incorrect positive expectation for an inconclusive fixture; initial failure retained, expectation corrected and a positive fixture added. No scientific criterion/backend change.\n'
      'Scientific main and independent audit first attempts passed; oldv0.8.0/246 not recovered.\n'
      '782 originals,15 attachments and85 prior formal manifests preserved; four historical diagnostic differences retained.\n'
      'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/compression_time32_study.py --replay runs/reconnect-compression-time32\n'
      'Replay rebuilds sparse actions from32 sealed exact conductances, actual32/full independent paths and all comparisons; no geometry,PDE or old candidate reruns.\n'
      'review-manifest.json covers included files; cumulative science seal also refers to omitted history. Full preservation audit needs original workspace.\n'
      'Next gate unexecuted; no automaticK64 or grid/flow/parameter scan. One oracle flow does not admit general smooth flow, density reconstruction, shocks or Stage3.\n'
      '2026-11-02 Asia/Shanghai; no PDF,release,commit/push,configuration change or publication.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-compression-time32-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/compression_time32_study.py'),
              '--replay','runs/reconnect-compression-time32'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=900)
            assert completed.returncode==0,completed.stdout+completed.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    last=r['comparison_observations'][-1];time_row=r['time_controls'][-1]
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'archive_CRC_and_all_included_SHA256_verified':True,'fresh_time32_replay_exit_code':completed.returncode,
      'fresh_time32_replay_log':completed.stdout,'fresh_replay_files_unchanged':True,'scientific_manifest_entries':len(entries),
      'archive_entries':len(manifest),'comparison_decision':r['comparison_decision'],'final_FV32_error':last['FV32_error'],
      'final_FV16_32_difference':time_row['FV16_32_state_difference'],'final_state_difference_ratio':time_row['state_new_difference_ratio'],
      'final_difference_over_FV32_error':time_row['new_difference_over_FV32_error'],'final_state_indicator':last['state_indicator'],
      'new_software_checks':2,'recorded_checks_total':463,'initial_test_expectation_failure_retained':True,'previous_checks_rerun':0,
      'new_candidate_graphs':32,'new_candidate_transient_runs':1,'new_independent_uniformization_sequences':1,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_spatial_levels':[],'next_gate_executed':False,
      'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':85,
      'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# 光滑压缩：必要32区间时间控制\n\n'
          '[报告、32张精确几何、状态、参考、审计与重放代码](ltn-compression-time32.zip)。\n\n'
          f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
          '[交付核验](ltn-compression-time32-delivery.json)。全新解压实际32和独立完整路径及比较重放通过。\n\n'
          f'结论：{r["comparison_decision"]}。Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
