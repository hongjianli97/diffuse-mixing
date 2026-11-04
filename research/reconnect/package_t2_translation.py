"""Seal the single registered T1 control and verify a fresh complete replay."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure


def main():
    source=ROOT/'runs/reconnect-t1-uniform';out=ROOT/'runs/reconnect-t2-translation'
    archive=ROOT/'exports/ltn-t2-translation.zip';delivery=ROOT/'exports/ltn-t2-translation-delivery.json';downloads=ROOT/'exports/ltn-t2-translation-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)),'closed evidence is never overwritten'
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'numerical-audit.json');r=read(out/'results.json')
    status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert ex['tests_exit_code']==ex['study_exit_code']==ar['audit_exit_code']==0
    assert ex['main_first_attempt_passed'] and (not ex['tests_first_attempt_passed'] and ex['tests_failed_attempts_preserved']==1) and ar['audit_first_attempt_passed']
    assert audit['all16_geometries_exact_rates_and_complete_raw_intervals_verified'] and audit['prior_formal_manifests_verified']==89
    assert status['recorded_checks_total']==472 and not status['fluid_orc_enabled'] and not status['general_smooth_flow_admitted'] and not gate['already_executed']
    prior=read(source/'sha256.json');assert len(prior)==3592
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+n for n in ['t2_translation_core.py','test_t2_translation.py','t2_translation_study.py','t2_translation_protocol.json',
      'T2_TRANSLATION_CONTROL.md','audit_t2_translation_geometry.py','audit_t2_translation.py','plot_t2_translation.py','package_t2_translation.py','T2_TRANSLATION_COMPARISON.md']]
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
      'LTN T2: ONE REGISTERED POSITIVE-D RIGID TRANSLATION, SAME ORDINARY VORONOI BACKEND.\n'
      f'Decision:{r["comparison_decision"]}; Stage3 incomplete,fluid ORC off.\n'
      'OriginalT1 N32 fixed materialm,c0,IDs,analytic reference,static16 actual/independent paths reused byte-identically.\n'
      'D=.02,T=.05,v=(.2,-.1),K16; only16 new moving geometries andone actual/one complete raw independent path.\n'
      'All16 ordinary exact periodic Voronoi graphs sealed before mass,c0,reference,static geometry or old state load.\n'
      'Onlypositions,IDs,givenpointdensity1,D,period enter original unmodified builder; no tensor shortcut,true faces,F/C or scalar input.\n'
      'Exact shared-coefficient differences retained; topology andoriginal1e-10 relative coefficient scale independently required.\n'
      'Duhamel uses2max_i sum_j abs(delta g_ij)/m_i andupward-rounded exact initialfluctuation; no mass/state/mean repair.\n'
      'Fixed state/H controls retainbothoriginal machine/tails,observedcomplete sequences,referenceevaluation andinitialbridges.\n'
      'Both complete sequences must independently pass theiroriginal machine/tail allowance; anomalous differences cannot donate budget.\n'
      'First software attempt hadone missing comparison-entry sequence guard; corrected before any scientific points/run.\n'
      'First failedJSON/log preserved; finaltwo checks pass,recorded472; old470 not rerun; no backend or threshold change.\n'
      'All16 pure geometry audits,exact rates,minimal raw Poisson tails,complete independent path,coefficient/Duhamel/state/H checks verified.\n'
      'No clipping,point repair,mass/rho/state/row/weight normalization,F/C fitting,threshold relaxation or extrapolated states.\n'
      '782 originals,15 attachments and89 priorformalmanifests preserved; fourknownhistorical diagnostic differences retained.\n'
      'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/t2_translation_study.py --replay runs/reconnect-t2-translation\n'
      'Fresh replay rebuilds actions and reproduces movingactual/independent paths bitwise; staticT1 path reused,all exactperturbation andstate/H results reproduced.\n'
      'Included files verified byreview-manifest.json;cumulative seal references omittedhistory;fullhistorical preservation needs originalworkspace.\n'
      'OnlypositiveDtranslation tested; rotation,D0motion,kappa,generalT2/T3/T4 remain unaccepted.\n'
      'NextD0nonrigid materialtransport gate unexecuted; no nextpoints,graphs orstates generated.\n'
      '2026-11-02 Asia/Shanghai; no PDF,version,commit/push,configuration change orpublication.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-t1-fresh-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            result=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/t2_translation_study.py'),'--replay','runs/reconnect-t2-translation'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=600)
            assert result.returncode==0,result.stdout+result.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'archive_CRC_and_all_included_SHA256_verified':True,'fresh_replay_exit_code':result.returncode,'fresh_replay_log':result.stdout,'fresh_replay_files_unchanged':True,
      'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'comparison_decision':r['comparison_decision'],
      'T2_registered_translation_case_passed':r['T2_registered_translation_case_passed'],'observations':r['observations'],
      'new_software_checks':2,'recorded_checks_total':472,'previous_checks_rerun':0,'new_candidate_graphs':16,'new_candidate_transient_runs':1,
      'new_independent_uniformization_sequences':1,'new_raw_action_intervals':16,'old_candidate_reruns':0,'new_reference_PDE_solves':0,
      'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':89,'next_gate_executed':False,
      'first_test_attempt_passed':False,'first_failed_test_checks':1,'first_failed_test_logs_preserved':True,'general_T2_admitted':False,'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# T2刚体平移：同一普通Voronoi后端客观性对照\n\n[报告、16张精确图、移动/静止状态、原始参考、审计和重放代码](ltn-t2-translation.zip)。\n\n'
          f'SHA-256：`{info["archive"]["sha256"]}`。\n\n[交付核验](ltn-t2-translation-delivery.json)。全新解压，移动实际与完整独立路径逐位重放通过，静止路径复用。\n\n'
          '本例T2平移控制通过；Stage3 incomplete，fluid ORC关闭。下一项D0非刚体物质输运仅登记，尚未执行。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
