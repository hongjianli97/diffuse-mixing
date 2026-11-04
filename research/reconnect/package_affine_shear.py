"""Seal the affine-shear cache plus reporting failure and verify a fresh full-path replay."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure


def main():
    source=ROOT/'runs/reconnect-t3-coverage';out=ROOT/'runs/reconnect-affine-shear'
    archive=ROOT/'exports/ltn-affine-shear.zip';delivery=ROOT/'exports/ltn-affine-shear-delivery.json';downloads=ROOT/'exports/ltn-affine-shear-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)),'sealed files never overwritten'
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'numerical-audit.json');r=read(out/'results-completed.json')
    status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert ex['tests_exit_code']==ex['completion_exit_code']==ar['audit_exit_code']==0
    assert ex['original_study_exit_code']==1 and not ex['main_first_attempt_passed'] and ex['tests_first_attempt_passed'] and ar['audit_first_attempt_passed']
    assert ex['all_pre_completion_files_unchanged'] and ex['no_scientific_path_rerun_during_completion']
    assert audit['all28_independent_world_geometries_exact_rates_and_full_raw_sequences_verified'] and audit['prior_formal_manifests_verified']==97
    assert status['recorded_checks_total']==487 and not status['general_T3_admitted'] and not status['fluid_orc_enabled']
    assert r['registered_affine_shear_numerical_control_passed'] and not gate['already_executed']
    for rel,h in read(out/'pre-completion-files-sha256.json').items():assert digest(out/rel)==h,rel
    prior=read(source/'sha256.json');assert len(prior)==5540
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    assert digest(ROOT/'exports/ltn-t3-coverage.zip')=='dac706856f97f44be0cb037b85b5476908baca6bb0fc473d9e64032a0588ace0'
    files=['research/reconnect/'+name for name in ('sliding_periodic_voronoi.py','audit_sliding_world_geometry.py','affine_shear_core.py',
      'test_affine_shear.py','affine_shear_protocol.json','affine_shear_study.py','audit_affine_shear_geometry.py','affine_shear_completion.py',
      'audit_affine_shear.py','plot_affine_shear.py','document_affine_shear.py','package_affine_shear.py','AFFINE_SHEAR_COMPARISON.md')]
    entries=dict(prior);source_seal=str((source/'sha256.json').relative_to(ROOT));entries[source_seal]=digest(source/'sha256.json')
    for rel in files:assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for p in out.rglob('*'):
        if p.is_file():
            rel=str(p.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)|dependency_closure([rel for rel in files if rel.endswith('.py')])
    selected.update(str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file())
    selected.update([source_seal,'docs/RESEARCH_STATUS.md','requirements-reference.txt'])
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'] if Path(item['path']).suffix!='.zip')
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note='''LTN T3 AFFINE SHEAR: one registered N32 given sliding-periodic world-Euclidean FV control.
Decision: T3_affine_shear_control_passed. Single-space-level case only; Stage3 incomplete and fluid ORC off.
Given B columns(1,0),(s,1),s=5t,T=.05,rho1,D=.02; constantD/Gamma coincide here,one model only.
Original material masses/finite-cell c0/IDs reused byte-for-byte; no scalar point-resampling or ideal-mode state replacement.
New opt-in sliding Euclidean Voronoi geometry; fractional coordinates only canonicalize/index. Original rectangle/isometry/ordinary resistance/time code unchanged.
s0 unit dispatch returns original rectangle packet exactly. Scientific midpoints all positive s.
All28 geometry/coefficient caches closed before mass/c0/reference load. No true material faces,F/C fit or precomputed analytic g supplied to builder.
All world faces, reciprocal half resistances, areas and fixed all-point9-image probes audited independently.
All cells additionally checked against every possible world halfplane using independent finer bins and rebuilt final radii; no hidden fixed-neighbor truncation.
Three positive g directions and cell second moments verified after sealing; no positive-face deletion, coefficient/state/weight normalization or clipping repair.
Independent80/100-digit true finite-cell/two-clock/frozen-continuous-grid references passed within fixed original allowance.
Analytic rational-phase sine zeros/unit values evaluated exactly only in independent references; actual NumPy weights/modes/states kept unchanged.
Three K4/8/16 actual and complete independent paths; exact rates/minimal1e-16 tails/raw weights and full independent sequences verified.
State/H reference controls,time decrease and two raw tilted-mode response contrasts pass original fixed positive indicators at both nonzero times.
Raw mode observables and initial changes are not fitted, normalized or used to replace actual states. Original c0-to-two-mode bridge retained.
FOUR new software checks passed first attempt; recorded487; old483 not rerun; oldv080/246 not recovered.
IMPORTANT REPORTING FAILURE: original scientific generator exits1 after all scientific caches/references are saved because a NumPy bool cannot be JSON serialized.
The original incomplete results.json,failed-execution.json and log are retained. Do not parse that incomplete file as current results.
Valid authoritative result: runs/reconnect-affine-shear/results-completed.json.
Append-only affine_shear_completion.py loads saved data and converts JSON scalar types without changing frozen source/criteria or rerunning scientific paths.
197 pre-completion files remain byte-identical, verified by pre-completion-files-sha256.json; no claim that the original main first attempt completed successfully.
Use the completion entry for fresh replay:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/affine_shear_completion.py --directory runs/reconnect-affine-shear --replay
Replay rebuilds three actual/full independent paths from28 sealed caches,compares all endpoints/halfstates/rawCSR/weights,references,bridge,raw observables and decisions bitwise.
Replay also verifies the retained original reporting failure and incomplete JSON bytes; no geometry,PDE,old candidate or ORC run.
782 originals,15 attachments,97 prior formal manifests and previous coverage delivery verified unchanged; four historical diagnostic differences retained.
Next gate only registered: same affine shear N16/32/64 space control,reuse three N32 paths,add only N16/N64 K4/8/16.
Planned56 new graphs,6 actual/6 independent paths and3 software checks are NOT EXECUTED in this archive. No automatic further grid/time/angle/model scan.
review-manifest.json covers included files; cumulative science seal references omitted history. Full historical preservation audit needs original workspace.
Original README/METHOD/CODEX_TASK/PHASE,attachments,old code/data/reports unchanged; only mutable RESEARCH_STATUS updated.
No PDF,version,commit/push,environment change or publication. General T3/smooth flow/nonorthogonal domains/finite-node admission false;kappa not computed.
2026-11-02 Asia/Shanghai.
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-affine-shear-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            result=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/affine_shear_completion.py'),
              '--directory','runs/reconnect-affine-shear','--replay'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=600)
            assert result.returncode==0,result.stdout+result.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'archive_CRC_and_all_included_SHA256_verified':True,'fresh_replay_exit_code':result.returncode,'fresh_replay_log':result.stdout,'fresh_replay_files_unchanged':True,
      'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'comparison_decision':r['comparison_decision'],
      'registered_affine_shear_control_passed':True,'final_observation':r['observations'][-1],'direction_observables':r['direction_observables'],
      'original_study_exit_code':1,'main_first_attempt_passed':False,'reporting_only_completion_exit_code':0,
      'original_reporting_failure_and_incomplete_JSON_preserved':True,'pre_completion_files_unchanged':197,'scientific_paths_rerun_during_completion':0,
      'valid_result_file':'runs/reconnect-affine-shear/results-completed.json','new_software_checks':4,'recorded_checks_total':487,
      'previous_checks_rerun':0,'tests_first_attempt_passed':True,'audit_first_attempt_passed':True,
      'new_candidate_graphs':28,'new_candidate_transient_runs':3,'new_complete_independent_uniformization_sequences':3,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':97,
      'original_backends_changed':False,'next_gate_executed':False,'next_gate_planned_new_graphs':56,'next_gate_planned_new_paths':6,
      'single_N_case_not_new_spatial_convergence':True,'general_T3_admitted':False,'general_nonorthogonal_domains_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# 仿射剪切：给定滑移周期域N32判别\n\n[新几何/独立审计、28图、3条实际/独立路径、参考、原始失败与补全结果](ltn-affine-shear.zip)。\n\n'
          f'SHA-256：`{info["archive"]["sha256"]}`。\n\n[交付核验](ltn-affine-shear-delivery.json)。CRC/SHA及全新解压三条路径与半步状态逐位重放通过，全部补全前文件保持。\n\n'
          '有效结果为results-completed.json；原不完整results.json与序列化失败记录保留。N32判别通过；下一项三空间档仅登记未执行。Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
