"""Seal the three-grid material-union control, fresh replay and optional download parts."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure


def main():
    source=ROOT/'runs/reconnect-compression-spatial-fv';out=ROOT/'runs/reconnect-compression-union64'
    archive=ROOT/'exports/ltn-compression-union64.zip';delivery=ROOT/'exports/ltn-compression-union64-delivery.json'
    downloads=ROOT/'exports/ltn-compression-union64-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)), 'never overwrite closed evidence'
    ex=read(out/'execution.json');audit=read(out/'numerical-audit.json');ar=read(out/'audit-execution.json');r=read(out/'results.json')
    status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert ex['tests_exit_code']==ex['study_exit_code']==ar['audit_exit_code']==0 and ex['main_first_attempt_passed'] and ex['tests_first_attempt_passed']
    assert audit['all48_geometries_exact_rates_and_complete_raw_sequences_verified'] and audit['prior_formal_manifests_verified']==87
    assert status['recorded_checks_total']==468 and not status['fluid_orc_enabled'] and not status['general_smooth_flow_admitted'] and not gate['already_executed']
    prior=read(source/'sha256.json');assert len(prior)==3090
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+n for n in ['compression_union64_core.py','test_compression_union64.py',
      'compression_union64_study.py','compression_union64_protocol.json','COMPRESSION_UNION64_CONTROL.md',
      'audit_compression_union64_geometry.py','audit_compression_union64.py','package_compression_union64.py',
      'plot_compression_union64.py','COMPRESSION_UNION64_COMPARISON.md']]
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
      'LTN: ONE N64 PHYSICAL MATERIAL-UNION CONTROL, SAME SMOOTH COMPRESSION.\n'
      f'Decision:{r["comparison_decision"]}; Stage3 incomplete, fluid ORC off.\n'
      'D=.02,T=.05; physicalN64 cells are originalN128 2x2 unions, mass=sum original mass,c0=sum original m*c0/union mass.\n'
      'OriginalN128/N256 masses,c0,IDs,point records,references and16/32 paths untouched and reused byte-identically.\n'
      'Pure new label centers advected by original map; given density formula checked bitwise on both old grids.\n'
      'All48 new exact ordinary-resistance geometry caches sealed before mass,c0,reference or old state load.\n'
      'No geometry,ordinary resistance,sampler,projection or action backend changes. Auxiliary capacities do not replace physical masses.\n'
      'Only newN64 actual16/32 and two complete independent raw uniformization sequences. No new reference PDE or old candidate rerun.\n'
      'Common64 observations are directratio2/4 component sums divided by same true union masses; native errors also required.\n'
      'Both adjacent spatial pairs at both nonzero times compare common,native concentration and commonH gains against fixed positive-sum controls.\n'
      'Original16/32 time,reference space/time,initial/projection,machine/tail and observed whole-sequence indicators retained.\n'
      'H reference space uses originaln512-k256 vsn1024-k256; time usesn1024-k512 vsbestn1024-k1024.\n'
      'No clipping,point repair,mass/rho/state/row/weight normalization,F/C fit,threshold relaxation or fitted states.\n'
      'Three observed spacings do not certify general order or fullStage3; oracleconstantD experiment is not reconstructed density,constantGamma or shock acceptance.\n'
      'All48 pure geometry audits,exact rates,minimal raw tails,complete independent sequences,Fraction union andbincount projections verified.\n'
      'Two new software checks passed,recorded468; old466 not rerun; oldv0.8.0/246 not recovered.\n'
      '782 originals,15 attachments and87 prior formal manifests preserved; four known historical diagnostic differences retained.\n'
      'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/compression_union64_study.py --replay runs/reconnect-compression-union64\n'
      'Replay rebuilds actions from48 exact caches and replays actualN64/independent paths bitwise; old128/256 paths reused.\n'
      'Included files verified by review-manifest.json; cumulative science seal references omitted history. Full historical audit requires original workspace.\n'
      'Next gate unexecuted; no automatic further grid,K64,D/flow changes. No PDF,release,commit/push,configuration change or publication.\n'
      '2026-11-02 Asia/Shanghai. Optional64MiB parts and actually verified restoration supplied if archive>128MiB.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-compression-spatial-FV-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/compression_union64_study.py'),
              '--replay','runs/reconnect-compression-union64'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=1800)
            assert completed.returncode==0,completed.stdout+completed.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    parts=[];restore_checked=False
    if archive.stat().st_size>128*1024**2:
        folder=ROOT/'exports/ltn-compression-union64-parts';folder.mkdir(exist_ok=False)
        with archive.open('rb') as f:
            i=0
            while block:=f.read(64*1024**2):
                i+=1;p=folder/f'part-{i:03d}.bin'
                with p.open('xb') as g:g.write(block)
                parts.append({'file':p.name,'bytes':len(block),'sha256':digest(p)})
        save(folder/'parts.json',{'archive':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive),'parts':parts})
        restore='''from pathlib import Path
import argparse,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
base=Path(__file__).resolve().parent;spec=json.loads((base/'parts.json').read_text());whole=hashlib.sha256();size=0
with Path(a.output).open('xb') as out:
 for item in spec['parts']:
  h=hashlib.sha256();count=0
  with (base/item['file']).open('rb') as source:
   while block:=source.read(1024**2):out.write(block);h.update(block);whole.update(block);count+=len(block);size+=len(block)
  assert count==item['bytes'] and h.hexdigest()==item['sha256'],item['file']
assert size==spec['bytes'] and whole.hexdigest()==spec['sha256']
print('All parts and restored archive SHA-256 verified.')
'''
        with (folder/'restore.py').open('x') as f:f.write(restore)
        with tempfile.TemporaryDirectory(prefix='ltn-spatial-parts-restore-',dir='/tmp') as directory:
            restored=Path(directory)/archive.name
            check=subprocess.run([sys.executable,str(folder/'restore.py'),'--output',str(restored)],capture_output=True,text=True,timeout=180)
            assert check.returncode==0,check.stdout+check.stderr
            assert digest(restored)==digest(archive) and restored.stat().st_size==archive.stat().st_size;restore_checked=True
    last=[v for v in r['adjacent_comparisons'] if v['time']==.05]
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'archive_CRC_and_all_included_SHA256_verified':True,'fresh_spatial_replay_exit_code':completed.returncode,
      'fresh_spatial_replay_log':completed.stdout,'fresh_replay_files_unchanged':True,'scientific_manifest_entries':len(entries),
      'archive_entries':len(manifest),'comparison_decision':r['comparison_decision'],'final_adjacent_comparisons':last,'final_grid_observations':{N:v[-1] for N,v in r['per_grid_observations'].items()},
      'new_software_checks':2,'recorded_checks_total':468,'previous_checks_rerun':0,'new_candidate_graphs':48,
      'new_candidate_transient_runs':2,'new_independent_uniformization_sequences':2,'old_candidate_reruns':0,
      'new_reference_PDE_solves':0,'next_gate_executed':False,'original_files_preserved':782,'historical_attachments_preserved':15,
      'prior_formal_manifests_preserved':87,'download_parts':parts,'download_parts_restore_helper_verified':restore_checked,
      'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# 同一光滑压缩N64物质并集：三档原生与共同区域控制\n\n'
          '[完整报告、48张精确几何、输入/状态、审计与重放代码](ltn-compression-union64.zip)。\n\n'
          f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
          '[交付核验](ltn-compression-union64-delivery.json)。全新解压实际N64的16/32及完整独立序列重放通过。\n\n')
        if parts:
            f.write('可下载以下64MiB分卷、[清单](ltn-compression-union64-parts/parts.json)和[恢复脚本](ltn-compression-union64-parts/restore.py)，保存到同一目录：\n\n')
            for item in parts:f.write(f'- [{item["file"]}](ltn-compression-union64-parts/{item["file"]})\n')
            f.write('\n运行 `python restore.py --output restored.zip`；恢复脚本已实际往返核验。\n\n')
        f.write(f'结论：{r["comparison_decision"]}。Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
