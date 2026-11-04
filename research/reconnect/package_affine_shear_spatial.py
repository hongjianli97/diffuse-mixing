"""Seal new affine-shear space evidence and verify six fresh cache-based path replays."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure


def main():
    source=ROOT/'runs/reconnect-affine-shear';out=ROOT/'runs/reconnect-affine-shear-spatial'
    archive=ROOT/'exports/ltn-affine-shear-spatial.zip';delivery=ROOT/'exports/ltn-affine-shear-spatial-delivery.json';downloads=ROOT/'exports/ltn-affine-shear-spatial-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)),'sealed files never overwritten'
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'numerical-audit.json');r=read(out/'results.json');status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert ex['tests_exit_code']==ex['study_exit_code']==ar['audit_exit_code']==0 and ex['main_first_attempt_passed'] and ar['audit_first_attempt_passed']
    assert not ex['tests_first_attempt_passed'] and ex['tests_second_attempt_passed']
    assert audit['all56_independent_world_geometries_exact_rates_and_full_raw_sequences_verified'] and audit['prior_formal_manifests_verified']==98
    assert status['recorded_checks_total']==490 and not status['general_T3_admitted'] and not status['fluid_orc_enabled'] and not gate['already_executed']
    prior=read(source/'sha256.json');assert len(prior)==5830
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    assert digest(ROOT/'exports/ltn-affine-shear.zip')=='470fd77d903b240f0752bc43238df967ccdb809b052487b4550de8c676f47c0e'
    files=['research/reconnect/'+name for name in ('affine_shear_spatial_core.py','affine_shear_spatial_study.py','test_affine_shear_spatial.py',
      'affine_shear_spatial_protocol.json','audit_affine_shear_spatial_geometry.py','audit_affine_shear_spatial.py',
      'plot_affine_shear_spatial.py','document_affine_shear_spatial.py','package_affine_shear_spatial.py','AFFINE_SHEAR_SPATIAL_COMPARISON.md')]
    entries=dict(prior);source_seal=str((source/'sha256.json').relative_to(ROOT));entries[source_seal]=digest(source/'sha256.json')
    for rel in files:assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for p in out.rglob('*'):
        if p.is_file():
            rel=str(p.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)|dependency_closure([rel for rel in files if rel.endswith('.py')])
    selected.update(str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file());selected.update([source_seal,'docs/RESEARCH_STATUS.md','requirements-reference.txt'])
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'] if Path(item['path']).suffix!='.zip')
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=f'''LTN T3 AFFINE SHEAR: registered N16/32/64 space control.
Decision: {r['comparison_decision']}. Stage3 incomplete; fluid ORC off.
Same rho1,D originalfloat.02,T=.05,s=5t,K4/8/16; add only N16/N64. No angle/flow/model scan.
56 new world-Euclidean graphs sealed before any true mass/c0/reference/reused states parsed.
Original sliding geometry,rectangle/isometry,ordinary half resistance andtime/action backends unchanged.
Six actual and six full independent paths; three existing N32 paths loaded only,not rerun.
Original fixed material masses andfinite-cell c0 preserved; no re-sampling/normalization/clipping/F_Cfit.
Independent world faces/reciprocal resistances/areas/all-point9image probes andALL-cell halfplanes verified.
Independent80/100digit finite-cell/two-clock/frozen-continuous symbols andallraw control sequences verified.
Common original N16 material mass observations retain capacity factors andallinitial/reference projection bridges.
All native/common concentration andnative/common H space gains use fixedpositive time/numerical/reference/projection budgets.
All grid state/H timecontrols andN64 raw tilted-mode response verified; no signed cancellation orRichardson state.
First software check attempt had one incorrect inherited compression-case expectation; original test source/log/JSON retained.
Only two expected booleans changedFalse toTrue; formulas/thresholds andadverse fullsequence rejection unchanged.
Three checks passed second attempt;490 recordedunique checks; old487 not rerun. Main andaudit first attempt passed.
Parent incomplete JSON/serialization failure/completed result andallolddata untouched; not retroactively called successful.
Replay in a fresh extracted directory:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/affine_shear_spatial_study.py --replay runs/reconnect-affine-shear-spatial
Replay reuses56 saved graphs and checks six new actual/full independent paths,allhalfstates,rawCSR/weights bitwise.
N32 paths are loaded only; allreferences,rawobservables,commonintegrals,bridges,positivebudgets andN32 decisions reproduced.
782 originals,15 attachments,98 prior formal manifests andparentZIP verified unchanged; four historical diagnostics differences retained.
Next gate NOT EXECUTED: one same-final-point-geometry/action N32 periodic contact withoracle densities,one staticgraph reused,K16.
Next trueN32means/reference newly evaluated fromoriginalcontinuousformula/stored128mode90digit coefficients,not recoveredv080 orinterpolatedoldmeans.
No T4 input/reference/graph/state generated this round; moving shock andthree-density-source closure remain.
T3 registered affine coverage completed ifdecisionpasses; this isnot generalnonorthogonal pointcloud theorem.
No automatic extraT3N/angles/models. Stage3 incomplete; generalT3/generalnonorthogonal/finite-node admission false; kappa notcomputed.
No PDF,release/version,commit/push,environmentchange orpublication. OriginalREADME/METHOD/CODEX_TASK/PHASE/attachments unchanged; mutableRESEARCH_STATUS updated.
review-manifest.json covers included files;cumulative scientific seal references omittedhistory; fullpreservationaudit needs originalworkspace.
2026-11-02 Asia/Shanghai.
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-affine-shear-spatial-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            replay=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/affine_shear_spatial_study.py'),
              '--replay','runs/reconnect-affine-shear-spatial'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=600)
            assert replay.returncode==0,replay.stdout+replay.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},'archive_CRC_and_all_included_SHA256_verified':True,
      'fresh_replay_exit_code':replay.returncode,'fresh_replay_log':replay.stdout,'fresh_replay_files_unchanged':True,'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),
      'comparison_decision':r['comparison_decision'],'registered_three_space_level_affine_shear_control_passed':r['registered_three_space_level_affine_shear_control_passed'],
      'per_grid_final_observations':{N:rows[-1] for N,rows in r['per_grid_observations'].items()},
      'adjacent_spatial_comparisons':[row for row in r['adjacent_spatial_comparisons'] if row['time']>0],
      'fine_direction_observables':r['per_grid_controls']['64']['direction_observables'],
      'new_software_checks':3,'recorded_checks_total':490,'previous_checks_rerun':0,'tests_first_attempt_passed':False,'tests_second_attempt_passed':True,
      'test_expectation_correction_and_original_failure_preserved':True,'main_first_attempt_passed':True,'audit_first_attempt_passed':True,
      'new_candidate_graphs':56,'new_candidate_transient_runs':6,'new_complete_independent_uniformization_sequences':6,'N32_paths_loaded_only_not_replayed':3,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':98,
      'original_backends_changed':False,'parent_reporting_failure_and_completed_results_unchanged':True,
      'next_gate_type':gate['type'],'next_gate_executed':False,'next_T4_inputs_graphs_references_states_generated':False,
      'general_T3_admitted':False,'general_nonorthogonal_domains_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:f.write('# 仿射剪切：同一普通FV三档物质空间对照\n\n[代码、56图、六条新路径、原N32复用、参考、审计与失败记录](ltn-affine-shear-spatial.zip)。\n\n'
      f'SHA-256：`{info["archive"]["sha256"]}`。\n\n[交付核验](ltn-affine-shear-spatial-delivery.json)。CRC/SHA与全新解压六条新路径逐位重放通过，原N32三路径只加载。\n\n'
      f'决定：{r["comparison_decision"]}。下一项同后端周期接触oracle仅登记未执行。Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
