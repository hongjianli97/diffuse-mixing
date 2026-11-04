"""Seal the registered two-grid time control and freshly replay its full archive."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest


def dependency_closure(paths):
    base=ROOT/'research/reconnect';queue=list(paths);seen=set()
    while queue:
        rel=queue.pop()
        if rel in seen:continue
        seen.add(rel);tree=ast.parse((ROOT/rel).read_text())
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.ImportFrom) and node.level==0 and node.module:modules=[node.module]
            elif isinstance(node,ast.Import):modules=[v.name for v in node.names]
            for module in modules:
                target=base/(module.replace('.','/')+'.py')
                if target.is_file():queue.append(str(target.relative_to(ROOT)))
    return seen


def main():
    source=ROOT/'runs/reconnect-shear-voronoi-spatial';out=ROOT/'runs/reconnect-shear-voronoi-time16'
    archive=ROOT/'exports/ltn-shear-voronoi-time16.zip';delivery=ROOT/'exports/ltn-shear-voronoi-time16-delivery.json'
    downloads=ROOT/'exports/ltn-shear-voronoi-time16-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)), 'never overwrite sealed evidence'
    run=read(out/'execution.json');audit=read(out/'numerical-audit.json');audit_run=read(out/'audit-execution.json')
    r=read(out/'results.json');status=read(out/'stage3-status.json');next_gate=read(out/'next-gate-spec.json')
    assert run['tests_exit_code']==run['study_exit_code']==audit_run['audit_exit_code']==0
    assert run['main_first_attempt_passed'] and run['tests_first_attempt_passed']
    assert audit['all32_geometries_and_exact_rates_verified'] and audit['prior_formal_manifests_verified']==82
    assert status['recorded_checks_total']==453 and status['registered16_interval_control_executed']
    assert not next_gate['already_executed'] and not status['fluid_orc_enabled'] and not status['general_smooth_flow_admitted']
    prior=read(source/'sha256.json');assert len(prior)==1993
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+name for name in ['voronoi_time16_core.py','test_voronoi_time16.py','voronoi_time16_protocol.json',
        'SHEAR_VORONOI_TIME16_CONTROL.md','shear_voronoi_time16_study.py','audit_shear_voronoi_time16_geometry.py',
        'audit_shear_voronoi_time16.py','package_shear_voronoi_time16.py','plot_shear_voronoi_time16.py','SHEAR_VORONOI_TIME16_COMPARISON.md']]
    source_seal=str((source/'sha256.json').relative_to(ROOT));entries=dict(prior);entries[source_seal]=digest(source/'sha256.json')
    for rel in files:
        assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for p in out.rglob('*'):
        if p.is_file():
            rel=str(p.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files);selected.update(str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file())
    selected.update([source_seal,'docs/RESEARCH_STATUS.md'])
    selected.update(dependency_closure([rel for rel in files if rel.endswith('.py')]))
    selected.update(v['path'] for v in read(out/'evidence-index.json')['sources'] if Path(v['path']).suffix!='.npz')
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)};last=r['observations'][-1]
    note=(
        'LTN SMOOTH NONSHOCK FIRST: registered necessary16-interval time control on the originalN128/N256 material grids.\n'
        f'Final common-parent actual16 errors:coarse={last["coarse16_error_common"]},fine={last["fine16_error_common"]}.\n'
        f'Fixed comparison decision:{r["comparison_decision"]}; Stage3 remains incomplete.\n'
        'Original masses/initial mass averages, IDs, parent mapping and independent reference reused unchanged.\n'
        'New pure point sampling uses original shear_map and original labels/rho0; all old4/8 full records and nested subsets match bitwise before geometry.\n'
        'All32 exact Voronoi/ordinary-face-resistance caches sealed before any mass/c0/reference load.\n'
        'Unchanged symmetric sparse exponential action and independent complete chronological uniformization, fixed1e-16 tail per interval and original machine allowance.\n'
        'Raw coefficients, Poisson weights, rows, means, masses, initial bridges and references never normalized or fitted.\n'
        'Same original2x2 child component sums divided by original parent masses; native and parent observations labeled separately.\n'
        'Fixed process seed20261101 is for numerical norm-estimator reproducibility only.\n'
        'Old4/8 actual results and inconclusive spatial decision unchanged; new8/16 measured controls use same nonnegative comparison formula.\n'
        'All4/8 and8/16 difference vectors retained; no Richardson or fitted state, ratios do not certify total error/order.\n'
        'Independent bisector/half-resistance audits on all32 graphs, whole raw uniformization sequences bitwise checked, parent grouping independently verified.\n'
        'Four new checks passed first attempt, recorded453, old449 not rerun; old v0.8.0/246 not recovered.\n'
        'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/shear_voronoi_time16_study.py --replay runs/reconnect-shear-voronoi-time16\n'
        'Replay reconstructs sparse actions from sealed exact conductances and original masses, both actual16 and independent sequences plus parent observation.\n'
        'No geometry/PDE/old4_8 candidate rerun during replay. Full historical preservation audit needs original workspace.\n'
        'All32 full exact geometry caches,原始 states/actions/weights, common observations, old controls and original references included.\n'
        'review-manifest.json covers included files; cumulative science seal also references omitted history. Mutable progress index outside science seal.\n'
        '782 originals,15 historical attachments and82 prior formal scientific manifests preserved; four known historical diagnostic differences retained.\n'
        'No automaticK32 or new spatial/flow/parameter scan; next gate unexecuted, see next-gate-spec.json.\n'
        'One given oracle smooth flow with two spatial grids does not admit general smooth flow, shocks, reconstructed density or Stage3. Fluid ORC off.\n'
        'No PDF, release, commit/push, environment configuration change or publication. New report dates use Asia/Shanghai client time.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-shear-FV-time16-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/shear_voronoi_time16_study.py'),
                '--replay','runs/reconnect-shear-voronoi-time16'],cwd=directory,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),
                capture_output=True,text=True,timeout=900)
            assert completed.returncode==0,completed.stdout+completed.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_time16_replay_exit_code':completed.returncode,
        'fresh_time16_replay_log':completed.stdout,'fresh_replay_files_unchanged':True,
        'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'comparison_decision':r['comparison_decision'],
        'final_coarse16_error_common':last['coarse16_error_common'],'final_fine16_error_common':last['fine16_error_common'],
        'final_state_indicator':last['combined_state_indicator'],'final_state_decision':last['state_decision'],'final_H_decision':last['H_decision'],
        'new_software_checks':4,'recorded_checks_total':453,'previous_checks_rerun':0,'new_candidate_graphs':32,'new_candidate_transient_runs':2,
        'new_independent_uniformization_sequences':2,'new_reference_PDE_solves':0,'old4_8_candidate_reruns':0,'new_spatial_levels':0,
        'old_inconclusive4_8_result_unchanged':True,'next_gate_executed':False,'original_files_preserved':782,
        'historical_attachments_preserved':15,'prior_formal_manifests_preserved':82,'general_smooth_flow_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# 同一光滑剪切：必要16区间时间控制\n\n'
            '[完整报告、32张精确几何/电导、原始状态与输入、审计及重放代码](ltn-shear-voronoi-time16.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[交付校验与范围](ltn-shear-voronoi-time16-delivery.json)。全新解压完整时序与共同观察重放通过。\n\n'
            f'固定比较结论：{r["comparison_decision"]}。Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
