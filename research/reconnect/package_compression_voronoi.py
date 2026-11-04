"""Seal one actual compression case; fresh archive replay of all three full paths."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure


def main():
    source=ROOT/'runs/reconnect-shear-union64';out=ROOT/'runs/reconnect-compression-voronoi-time'
    archive=ROOT/'exports/ltn-compression-voronoi-time.zip';delivery=ROOT/'exports/ltn-compression-voronoi-time-delivery.json'
    downloads=ROOT/'exports/ltn-compression-voronoi-time-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)), 'never overwrite closed evidence'
    ex=read(out/'execution.json');audit=read(out/'numerical-audit.json');ar=read(out/'audit-execution.json');r=read(out/'results.json')
    status=read(out/'stage3-status.json');next_gate=read(out/'next-gate-spec.json')
    assert ex['tests_exit_code']==ex['study_exit_code']==ar['audit_exit_code']==0
    assert ex['main_first_attempt_passed'] and audit['all28_geometries_exact_rates_and_complete_raw_sequences_verified']
    assert audit['prior_formal_manifests_verified']==84 and status['recorded_checks_total']==461
    assert not status['fluid_orc_enabled'] and not status['general_smooth_flow_admitted'] and not next_gate['already_executed']
    prior=read(source/'sha256.json');assert len(prior)==2331
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+n for n in ['compression_voronoi_core.py','test_compression_voronoi.py','compression_voronoi_protocol.json',
      'COMPRESSION_VORONOI_TIME_CONTROL.md','compression_voronoi_study.py','audit_compression_voronoi_geometry.py',
      'audit_compression_voronoi.py','package_compression_voronoi.py','plot_compression_voronoi.py','COMPRESSION_VORONOI_TIME_COMPARISON.md']]
    source_seal=str((source/'sha256.json').relative_to(ROOT));entries=dict(prior);entries[source_seal]=digest(source/'sha256.json')
    for rel in files:
        assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for p in out.rglob('*'):
        if p.is_file():
            rel=str(p.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files);selected.update(str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file())
    selected.update(dependency_closure([p for p in files if p.endswith('.py')]))
    selected.update([source_seal,'docs/RESEARCH_STATUS.md'])
    for row in read(out/'evidence-index.json')['sources']:
        if not row['path'].endswith('.npz'):selected.add(row['path'])
    manifest={p:digest(ROOT/p) for p in sorted(selected)}
    note=(
      'LTN: ONE GIVEN SMOOTH COMPRESSION, SAME ORDINARY PERIODIC VORONOI FV.\n'
      f'Decision:{r["comparison_decision"]}; Stage3 incomplete, fluid ORC off.\n'
      'OriginalN128 material masses,mean initial fractions,IDs,given point density and independent PDE reference unchanged.\n'
      'Pure given map producer reproduces all originalK4/8 time/position/rho records and newK16 nested subsets bitwise.\n'
      'All28 exact geometric/ordinary-center-resistance graphs sealed before mass,c0,reference or saved kernel states loaded.\n'
      'Actual chronological K4/8/16 and independent complete uniformization sequences, raw weights, original capacities and1e-16 per-interval exact tail.\n'
      'Same unchanged action backend as prior shear; no tensor-specialized replacement.\n'
      'Original saved kernelK8 background only; original kernelK4/8 time difference and all direct/two-half local solver controls retained.\n'
      'Local saved checks do not constitute independent complete sequences or certified remaining error.\n'
      'Fixed positive-sum state/H comparison includes time,reference,initial bridges and numerical indicators; raw4/8 and8/16 differences retained.\n'
      'No extrapolated/fitted state,clipping,point repair,mass/rho/row/state/weight normalization,F/C fitting or threshold relaxation.\n'
      'Independent all28 geometry/exact-rate audits, complete raw uniformization sequences bitwise verified; original producer and local controls independently recomputed.\n'
      'Four new software checks, recorded461; old457 not rerun, oldv0.8.0/246 not recovered.\n'
      '782 original files,15 attachments and84 prior formal manifests preserved; four known historical diagnostic differences retained.\n'
      'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/compression_voronoi_study.py --replay runs/reconnect-compression-voronoi-time\n'
      'Replay rebuilds sparse actions from sealed exact conductances; all three actual paths, independent sequences and fixed comparisons reproduced.\n'
      'No geometry,old candidate or PDE solve in replay; complete historical preservation audit needs original workspace.\n'
      'Included files verified by review-manifest.json; cumulative science seal references omitted historical evidence; mutable status outside science seal.\n'
      'Next gate remains unexecuted; no automatic grid,K32 or parameter matrix.\n'
      'One oracle compression and prior shear do not prove general smooth flow, reconstructed density, shock/contact or Stage3 acceptance.\n'
      'Date2026-11-02 Asia/Shanghai; no PDF,release,commit/push,configuration change or publication.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-compression-Voronoi-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/compression_voronoi_study.py'),
              '--replay','runs/reconnect-compression-voronoi-time'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=900)
            assert completed.returncode==0,completed.stdout+completed.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    last=r['comparison_observations'][-1]
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'archive_CRC_and_all_included_SHA256_verified':True,'fresh_compression_replay_exit_code':completed.returncode,
      'fresh_compression_replay_log':completed.stdout,'fresh_replay_files_unchanged':True,
      'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'comparison_decision':r['comparison_decision'],
      'final_FV_errors':{K:v[-1]['relative_concentration_error'] for K,v in r['observations_by_intervals'].items()},
      'final_saved_kernel8_error':last['saved_kernel8_error'],'final_state_indicator':last['state_indicator'],
      'final_H_indicator':last['H_indicator'],'both_state_and_H_time_differences_decreased':r['both_state_and_H_time_differences_decreased'],
      'new_software_checks':4,'recorded_checks_total':461,'previous_checks_rerun':0,'new_candidate_graphs':28,
      'new_candidate_transient_runs':3,'new_independent_uniformization_sequences':3,'old_candidate_reruns':0,
      'new_reference_PDE_solves':0,'new_spatial_levels':[],'next_gate_executed':False,
      'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':84,
      'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# 同一光滑压缩：普通Voronoi FV实际时间推进\n\n'
          '[报告、28张完整精确几何、原始状态、参考、审计与重放代码](ltn-compression-voronoi-time.zip)。\n\n'
          f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
          '[交付核验与范围](ltn-compression-voronoi-time-delivery.json)。全新解压实际4/8/16路径及独立完整序列与比较重放通过。\n\n'
          f'结论：{r["comparison_decision"]}。Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
