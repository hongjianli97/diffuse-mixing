"""Seal the single registered T1 control and verify a fresh complete replay."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure


def main():
    source=ROOT/'runs/reconnect-compression-union64';out=ROOT/'runs/reconnect-t1-uniform'
    archive=ROOT/'exports/ltn-t1-uniform.zip';delivery=ROOT/'exports/ltn-t1-uniform-delivery.json';downloads=ROOT/'exports/ltn-t1-uniform-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)),'closed evidence is never overwritten'
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'numerical-audit.json');r=read(out/'results.json')
    status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert ex['tests_exit_code']==ex['study_exit_code']==ar['audit_exit_code']==0
    assert ex['main_first_attempt_passed'] and ex['tests_first_attempt_passed'] and ar['audit_first_attempt_passed']
    assert audit['all3_geometries_exact_rates_and_all84_complete_raw_intervals_verified'] and audit['prior_formal_manifests_verified']==88
    assert status['recorded_checks_total']==470 and not status['fluid_orc_enabled'] and not status['general_smooth_flow_admitted'] and not gate['already_executed']
    prior=read(source/'sha256.json');assert len(prior)==3342
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+n for n in ['t1_uniform_core.py','test_t1_uniform.py','t1_uniform_study.py','t1_uniform_protocol.json',
      'T1_UNIFORM_CONTROL.md','audit_t1_uniform_geometry.py','audit_t1_uniform.py','plot_t1_uniform.py','package_t1_uniform.py','T1_UNIFORM_COMPARISON.md']]
    source_seal=str((source/'sha256.json').relative_to(ROOT));entries=dict(prior);entries[source_seal]=digest(source/'sha256.json')
    for rel in files:assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for path in out.rglob('*'):
        if path.is_file():
            rel=str(path.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(path)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)|{str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()}|dependency_closure([p for p in files if p.endswith('.py')])
    selected.update([source_seal,'docs/RESEARCH_STATUS.md']);selected.update(v['path'] for v in read(out/'evidence-index.json')['sources'])
    manifest={p:digest(ROOT/p) for p in sorted(selected)}
    note=(
      'LTN T1: ONE REGISTERED UNIFORM STATIC MATERIAL-MEAN SINE, SAME ORDINARY VORONOI BACKEND.\n'
      f'Decision:{r["comparison_decision"]}; Stage3 incomplete,fluid ORC off.\n'
      'rho1,D=.02,T=.05,N16/32/64,K4/8/16; physical uniform cells,m=1/N^2,c0 includes sinc(pi/N)^2 cell mean.\n'
      'All3 exact periodic Voronoi ordinary-resistance graphs sealed before mass,c0,reference or symbol generation. No tensor shortcut.\n'
      'Three static graphs reused across9 actual and9 complete raw independent uniformization paths,84 intervals.\n'
      'Original geometry,ordinary resistance,projection and action backends unchanged; no old candidate or reference PDE reruns.\n'
      'Analytic continuous decay lambda=8pi^2D; postsealed independent discrete lambda_h=8DN^2sin(pi/N)^2.\n'
      '80/100 decimal digit rounded references agree bitwise; main formulas retained,not replaced or fitted.\n'
      'CommonN16 direct component integrals and native errors both required; fixed time,analytic evaluation,projection/initial and numerical controls retained.\n'
      'Static generator means exact interval-count independence; observed near-zero time differences do not establish general time order.\n'
      'No clipping,point repair,mass/rho/state/row/weight normalization,F/C or symbol fitting,threshold relaxation or extrapolated states.\n'
      'All3 pure geometry audits,exact rates,minimal original Poisson tails and84 raw independent intervals verified.\n'
      'Two new software checks passed,recorded470; old468 not rerun; oldv0.8.0/246 not recovered.\n'
      '782 originals,15 attachments,88 prior formal manifests preserved; four known historical diagnostic differences retained.\n'
      'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/t1_uniform_study.py --replay runs/reconnect-t1-uniform\n'
      'Replay rebuilds actions and reproduces9 actual/9 complete independent paths bitwise and all comparisons from3 exact caches.\n'
      'Replay needs pinnednumpy/scipy; independent analytic audit also usesmpmath. See requirements-replay.txt.\n'
      'Included files verified by review-manifest.json; cumulative seal references omitted history. Full historical preservation audit requires original workspace.\n'
      'T1 registered case only; uniformrho does not distinguishconstantD/Gamma; T2/T3/T4 anddensity-source acceptance incomplete.\n'
      'NextT2 translation gate unexecuted; no further points,graphs or paths generated.\n'
      '2026-11-02 Asia/Shanghai; no PDF,version,commit/push,configuration change or publication.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-t1-fresh-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            result=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/t1_uniform_study.py'),'--replay','runs/reconnect-t1-uniform'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=600)
            assert result.returncode==0,result.stdout+result.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'archive_CRC_and_all_included_SHA256_verified':True,'fresh_replay_exit_code':result.returncode,'fresh_replay_log':result.stdout,'fresh_replay_files_unchanged':True,
      'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'comparison_decision':r['comparison_decision'],
      'T1_registered_case_passed':r['T1_registered_case_passed'],'final_grid_observations':{N:v[-1] for N,v in r['per_grid_observations'].items()},
      'new_software_checks':2,'recorded_checks_total':470,'previous_checks_rerun':0,'new_candidate_graphs':3,'new_candidate_transient_runs':9,
      'new_independent_uniformization_sequences':9,'new_raw_action_intervals':84,'old_candidate_reruns':0,'new_reference_PDE_solves':0,
      'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':88,'next_gate_executed':False,
      'general_temporal_order_claimed':False,'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# T1均匀介质：同一普通Voronoi后端解析基准\n\n[报告、三张精确图、九条实际/独立路径、参考、审计和重放代码](ltn-t1-uniform.zip)。\n\n'
          f'SHA-256：`{info["archive"]["sha256"]}`。\n\n[交付核验](ltn-t1-uniform-delivery.json)。全新解压，九条实际和九条完整独立路径逐位重放通过。\n\n'
          '本例T1控制通过；Stage3 incomplete，fluid ORC关闭。下一项T2刚体平移仅登记，尚未执行。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
