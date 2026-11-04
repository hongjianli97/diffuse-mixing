"""Seal the bounded union control and freshly replay the actual N64 sequences."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure


def main():
    source=ROOT/'runs/reconnect-shear-voronoi-time16';out=ROOT/'runs/reconnect-shear-union64'
    archive=ROOT/'exports/ltn-shear-union64.zip';delivery=ROOT/'exports/ltn-shear-union64-delivery.json';seal=out/'sha256.json'
    downloads=ROOT/'exports/ltn-shear-union64-downloads.md'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)), 'closed evidence is never overwritten'
    run=read(out/'execution.json');audit=read(out/'numerical-audit.json');ar=read(out/'audit-execution.json');r=read(out/'results.json')
    status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert run['tests_exit_code']==run['study_exit_code']==ar['audit_exit_code']==0
    assert run['main_first_attempt_passed'] and audit['all24_geometries_exact_rates_and_complete_raw_sequences_verified']
    assert audit['prior_formal_manifests_verified']==83 and status['recorded_checks_total']==457
    assert not status['fluid_orc_enabled'] and not status['general_smooth_flow_admitted'] and not gate['already_executed']
    prior=read(source/'sha256.json');assert len(prior)==2174
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+n for n in ['material_union.py','test_material_union.py','union64_protocol.json',
      'SHEAR_UNION64_CONTROL.md','shear_union64_study.py','audit_shear_union64_geometry.py','audit_shear_union64.py',
      'package_shear_union64.py','plot_shear_union64.py','SHEAR_UNION64_COMPARISON.md']]
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
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=(
      'LTN: ONE N64 MATERIAL-UNION CONTROL, SAME SMOOTH NONSHOCK SHEAR.\n'
      f'Comparison decision:{r["comparison_decision"]}; Stage3 incomplete, fluid ORC off.\n'
      'N64 physical cells are originalN128 2x2 unions, new mass=sum original mass, new c0=sum original m*c0/union mass.\n'
      'OriginalN128/N256 data untouched; all24 ordinary-resistance graphs sealed before scalar/capacity/reference load.\n'
      'Only new actualN64 K8/16 sequences; originalN128/N256 K8/16 paths and independent references reused byte-identically.\n'
      'Pure seed label centers and original prescribed map; given point density verified bitwise on both old grids.\n'
      'CommonN64 observations use direct ratio2/4 original component sums divided by physical union masses. No intermediate normalization.\n'
      'Native errors are required as well: coarse observations can hide oscillatory error.\n'
      'Both adjacent pairs at both nonzero times require common concentration, native concentration and commonH improvements above fixed positive-sum indicators.\n'
      'Actual K16 paths only, no Richardson or fitted states; observed ratios do not certify general order or remaining total error.\n'
      'No clipping, moving/deleting original points, m/rho/state/weight/row rescaling, F/C fitting, threshold relaxation or ORC.\n'
      'All24 pure exact-geometry audits, exact local rates and complete raw independent uniformization sequences verified.\n'
      'Physical union integrals checked with Fractions; ratio2/4 projections independently checked with bincount.\n'
      'Four new software checks, recorded457; old453 not rerun. Oldv0.8.0/246 not recovered.\n'
      '782 original files,15 attachments and83 prior formal scientific manifests preserved.\n'
      'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/shear_union64_study.py --replay runs/reconnect-shear-union64\n'
      'Replay rebuilds N64 sparse actions from sealed exact conductances and raw physical union capacities; old candidate/PDE/geometry not rerun.\n'
      'Included file checksums are in review-manifest.json; cumulative scientific seal references omitted historical evidence.\n'
      'Full historical preservation audit requires original workspace. Mutable research status is outside the scientific seal.\n'
      'Next gate remains unexecuted; no automatic spatial/time/parameter matrix.\n'
      'Report date2026-11-02 uses Asia/Shanghai client time. No PDF, version, commit/push, publication or environment configuration change.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-union64-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/shear_union64_study.py'),
              '--replay','runs/reconnect-shear-union64'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=300)
            assert completed.returncode==0,completed.stdout+completed.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'archive_CRC_and_all_included_SHA256_verified':True,'fresh_union64_replay_exit_code':completed.returncode,
      'fresh_union64_replay_log':completed.stdout,'fresh_replay_files_unchanged':True,
      'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'comparison_decision':r['comparison_decision'],
      'final_grid_observations':{N:v[-1] for N,v in r['per_grid_observations'].items()},
      'final_adjacent_comparisons':[v for v in r['adjacent_comparisons'] if v['time']==.05],
      'new_software_checks':4,'recorded_checks_total':457,'previous_checks_rerun':0,'new_candidate_graphs':24,
      'new_candidate_transient_runs':2,'new_independent_uniformization_sequences':2,'old_candidate_reruns':0,
      'new_reference_PDE_solves':0,'new_spatial_levels':[64],'next_gate_executed':False,
      'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':83,
      'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# 同一光滑剪切N64物质并集控制\n\n'
          '[报告、24张完整精确几何、原始状态、参考、审计与重放代码](ltn-shear-union64.zip)。\n\n'
          f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
          '[交付核验与范围](ltn-shear-union64-delivery.json)。全新解压的实际N64 K8/16序列及三档比较重放通过。\n\n'
          f'比较结论：{r["comparison_decision"]}。Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
