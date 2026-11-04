"""Seal the single registered T1 control and verify a fresh complete replay."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure


def main():
    source=ROOT/'runs/reconnect-t2-translation';out=ROOT/'runs/reconnect-t2-d0'
    archive=ROOT/'exports/ltn-t2-d0.zip';delivery=ROOT/'exports/ltn-t2-d0-delivery.json';downloads=ROOT/'exports/ltn-t2-d0-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)),'closed evidence is never overwritten'
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'numerical-audit.json');r=read(out/'results.json')
    status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert ex['tests_exit_code']==ex['study_exit_code']==ar['audit_exit_code']==0
    assert ex['main_first_attempt_passed'] and ex['tests_first_attempt_passed'] and ar['audit_first_attempt_passed']
    assert audit['all16_zero_geometry_audits_and_original_zero_action_branches_verified'] and audit['prior_formal_manifests_verified']==90
    assert status['recorded_checks_total']==474 and not status['fluid_orc_enabled'] and not status['general_smooth_flow_admitted'] and not gate['already_executed']
    prior=read(source/'sha256.json');assert len(prior)==3711
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+n for n in ['t2_d0_core.py','test_t2_d0.py','t2_d0_study.py','t2_d0_protocol.json',
      'T2_D0_CONTROL.md','audit_d0_geometry_core.py','audit_t2_d0_geometry.py','audit_t2_d0.py','plot_t2_d0.py','package_t2_d0.py','T2_D0_COMPARISON.md']]
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
      'LTN T2: ONE REGISTERED D=0 NONRIGID MATERIAL-TRANSPORT CONTROL.\n'
      f'Decision:{r["comparison_decision"]}; Stage3 incomplete,fluid ORC off.\n'
      'OriginalT1 N32 true fixedm,c0 andIDs reused byte-identically; D0,T=.05,K16,original given compression sampler withoracle rho0=1,rho=rho0/J.\n'
      'Only16 new exact ordinary Voronoi graphs,one actual andone complete independent path; no newPDE oroldcandidate rerun.\n'
      'All16 coefficients sealed before actualm/c0 load. Onlypositions,IDs,givenrho,D0,period enter unchanged builder.\n'
      'Identity reference isoriginalmaterialc0,not savedpositiveDdecay. No tensor shortcut,true materialfaces,F/C orscalarinput.\n'
      'All exactg=0,Lambda0,A0,P=I,original rawweight[1],exacttail0; original zero-action branch called unchanged.\n'
      'All17 actual/independent endpoints and16 halfstates haveidentical dtype,shape andC-orderbytes tooriginalc0.\n'
      'Rawinventory/variance unchanged,I1,H0,chi0,energy loss0; no acceptance tolerance used to relax identity.\n'
      'Explicit zero andsignedzero sparse entries retained; positiveoriginalmachineindicator retained,not used to accept statechange.\n'
      'New independent zero-limit geometry auditor retains full bisector,reciprocalface,probe andarea checks without division byzero; oldpositiveD auditor unchanged.\n'
      'All16 geometry/zero-action audits,sparse identity,weights,states andrawmetrics independently verified.\n'
      'Two new checks passed first attempt,recorded474; old472 not rerun; oldv0.8.0/246 not recovered.\n'
      '782 originals,15attachments and90 priorformalmanifests preserved; fourknownhistorical diagnostic differences retained.\n'
      'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/t2_d0_study.py --replay runs/reconnect-t2-d0\n'
      'Fresh replay rebuilds theoriginalzeroaction from16 sealedcaches and reproduces allstates/metrics bitwise; no geometry,PDE oroldcandidate rerun.\n'
      'Includedfiles verified byreview-manifest.json;cumulativeseal references omittedhistory;fullhistoricalpreservation needs originalworkspace.\n'
      'No clipping,pointrepair,mass/rho/state/row/weightnormalization,F/Cfit,thresholdchange,PDF,version,commit/push,configurationchange orpublication.\n'
      'Next gate isonlyunexecuted read-only rotation-domain contract,with0newpoints/graphs/paths/PDE/tests. No automatic angularscan orAPIchange.\n'
      'PositiveDrotation,originalT3models/directions,T4 anddensitysources remain incomplete; no kappa orfluidORC run.\n'
      '2026-11-02 Asia/Shanghai.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-t1-fresh-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            result=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/t2_d0_study.py'),'--replay','runs/reconnect-t2-d0'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=600)
            assert result.returncode==0,result.stdout+result.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'archive_CRC_and_all_included_SHA256_verified':True,'fresh_replay_exit_code':result.returncode,'fresh_replay_log':result.stdout,'fresh_replay_files_unchanged':True,
      'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'comparison_decision':r['comparison_decision'],
      'T2_registered_D0_case_passed':r['T2_registered_D0_case_passed'],'final_observation':r['observations'][-1],'all_states_bitwise_initial':r['state_arrays_bitwise_initial'],
      'new_software_checks':2,'recorded_checks_total':474,'previous_checks_rerun':0,'new_candidate_graphs':16,'new_candidate_transient_runs':1,
      'new_independent_uniformization_sequences':1,'new_raw_action_intervals':16,'old_candidate_reruns':0,'new_reference_PDE_solves':0,
      'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':90,'next_gate_executed':False,
      'kappa_computed':False,'general_T2_admitted':False,'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# T2零扩散：非刚体物质输运恒等检查\n\n[报告、16张精确图、原输入、恒等参考、实际/独立状态、审计和重放代码](ltn-t2-d0.zip)。\n\n'
          f'SHA-256：`{info["archive"]["sha256"]}`。\n\n[交付核验](ltn-t2-d0-delivery.json)。全新解压，实际/完整独立路径和半步状态逐位重放通过。\n\n'
          '本例D0物质恒等控制通过；Stage3 incomplete，fluid ORC关闭。下一项只读旋转域契约审查仅登记，尚未执行。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
