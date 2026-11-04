"""Seal and independently extract/replay the single registered N256 spatial case."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest


def main():
    source=ROOT/'runs/reconnect-shear-voronoi-time';out=ROOT/'runs/reconnect-shear-voronoi-spatial'
    archive=ROOT/'exports/ltn-shear-voronoi-spatial.zip';delivery=ROOT/'exports/ltn-shear-voronoi-spatial-delivery.json'
    downloads=ROOT/'exports/ltn-shear-voronoi-spatial-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)), 'never overwrite sealed evidence'
    run=read(out/'execution.json');audit=read(out/'numerical-audit.json');audit_run=read(out/'audit-execution.json')
    result=read(out/'results.json');status=read(out/'stage3-status.json');next_gate=read(out/'next-gate-spec.json')
    assert run['tests_exit_code']==run['study_exit_code']==audit_run['audit_exit_code']==0
    assert run['scientific_main_first_attempt_passed'] and run['tests_first_attempt_passed']
    assert audit['all12_geometries_and_whole_control_sequences_verified']
    assert audit['prior_formal_manifests_verified']==81 and status['recorded_checks_total']==449
    assert status['registered_N256_spatial_gate_executed'] and status['new_candidate_graphs']==12
    assert not next_gate['already_executed'] and not status['fluid_orc_enabled']
    assert status['new_PDE_reference_solves']==0 and not status['general_smooth_flow_admitted']
    prior=read(source/'sha256.json');assert len(prior)==1905
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+name for name in ['voronoi_spatial_projection.py','test_voronoi_spatial_projection.py',
        'voronoi_spatial_protocol.json','SHEAR_VORONOI_SPATIAL_CONTROL.md','shear_voronoi_spatial_study.py',
        'audit_shear_voronoi_spatial.py','audit_shear_voronoi_spatial_geometry.py','package_shear_voronoi_spatial.py','SHEAR_VORONOI_SPATIAL_COMPARISON.md']]
    source_seal=str((source/'sha256.json').relative_to(ROOT));entries=dict(prior);entries[source_seal]=digest(source/'sha256.json')
    for rel in files:
        assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for p in out.iterdir():
        assert p.is_file(),p
        rel=str(p.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files);selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([source_seal,'docs/RESEARCH_STATUS.md'])
    selected.update('research/reconnect/'+name for name in ['audit_route_decision.py','periodic_voronoi_fv.py',
        'shear_voronoi_fv_study.py','voronoi_time_action.py','shear_voronoi_time_study.py',
        'audit_shear_voronoi_fv.py','audit_shear_voronoi_time.py','variable_parent_observation.py','trajectory_chain.py'])
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'] if Path(item['path']).suffix!='.npz')
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    final=result['observations'][-1]
    note=(
        'LTN SMOOTH NONSHOCK FIRST: one original N256 spatial control against saved N128, same D.02,T.05 and4/8 intervals.\n'
        f'Final common-parent concentration errors:coarse={final["coarse_error_common"]},fine={final["fine_error_common"]}.\n'
        f'Fixed overall comparison decision:{result["comparison_decision"]}; raw smaller error alone is not resolution of space/time uncertainty.\n'
        'Original fine material m/c0/IDs, midpoint positions/rho, parent mapping and independent reference reused without modification.\n'
        'All12 exact Voronoi face/conductance caches sealed before mass/scalar/reference loading.\n'
        'Unchanged central face resistance and chronological symmetric exponential action; independent complete uniformization sequences.\n'
        'No true material faces,F/C,velocity gradient,c0 or PDE responses enter geometry/coefficient builder.\n'
        'Original material masses remain actual capacities; auxiliary rho*Voronoi area never substitutes them.\n'
        'Raw sum of child component masses divided by original parent mass, not by child mass sum. Native fine and common parent metrics labeled separately.\n'
        'Original child/parent mass rounding, three initial bridges and reference projection differences retained; no state or mass normalization.\n'
        '1e-16 exact per-interval Poisson tail budget, unscaled raw weights/rows and original numerical tolerances.\n'
        'Fixed process seed20261101 only for reproducible numerical norm estimation, not physical/geometry selection.\n'
        'Independent audit reconstructs all vertices from bisectors and shared conductances from reciprocal half resistances.\n'
        'Whole control sequences and raw Poisson weights verified bitwise; parent projection independently regrouped.\n'
        'Original PDE reference and coarse candidate not recomputed; no new kernel build or physical-flow definition.\n'
        'Four new identity/observation checks passed first attempt; recorded449, previous445 not rerun; old v0.8.0/246 not recovered.\n'
        'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/shear_voronoi_spatial_study.py --replay runs/reconnect-shear-voronoi-spatial\n'
        'Replay reconstructs sparse actions from the exact sealed conductances and original masses, repeats both fine sequences and common observation.\n'
        'Replay does not rebuild geometry, run coarse candidate or solve a reference PDE; full provenance audit needs historical workspace.\n'
        'Full12 exact geometry caches are included. Cumulative scientific seal additionally references omitted historical files.\n'
        'review-manifest.json covers included files; mutable docs/RESEARCH_STATUS.md is a review snapshot outside scientific seal.\n'
        '782 originals,15 historical attachments and81 prior formal scientific manifests preserved; four known historical diagnostic differences retained.\n'
        'No automatic added K/grid or threshold/weight changes. Next gate unexecuted; see next-gate-spec.json.\n'
        'Two space/time levels do not prove asymptotic order or certify remaining error. Stage3 incomplete, general smooth flow unadmitted, fluid ORC off.\n'
        'Shock extension deferred; no PDF, release, commit/push, environment configuration change or publication.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-shear-FV-space-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/shear_voronoi_spatial_study.py'),
                '--replay','runs/reconnect-shear-voronoi-spatial'],cwd=directory,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),
                capture_output=True,text=True,timeout=600)
            assert completed.returncode==0,completed.stdout+completed.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_spatial_replay_exit_code':completed.returncode,
        'fresh_spatial_replay_log':completed.stdout,'fresh_replay_files_unchanged':True,
        'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'comparison_decision':result['comparison_decision'],
        'final_coarse_error_common':final['coarse_error_common'],'final_fine_error_common':final['fine_error_common'],
        'final_state_indicator':final['combined_state_indicator'],'final_state_decision':final['state_decision'],
        'final_H_decision':final['H_decision'],'new_software_checks':4,'recorded_checks_total':449,'previous_checks_rerun':0,
        'new_candidate_midpoint_graphs':12,'new_candidate_transient_runs':2,'new_independent_uniformization_sequences':2,
        'new_reference_PDE_solves':0,'coarse_candidate_rerun':False,'new_physical_flow_definitions':0,
        'next_gate_executed':False,'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':81,
        'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# 光滑剪切辅助FV：N256共同物质区域空间控制\n\n'
            '[报告、完整12张精确几何/电导、原始输入/状态、审计与重放代码](ltn-shear-voronoi-spatial.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[交付范围与校验](ltn-shear-voronoi-spatial-delivery.json)。全新解压完整细档序列与共同观测重放通过。\n\n'
            f'固定比较结论：{result["comparison_decision"]}。Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
