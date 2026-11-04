"""Seal the unresolved affine pair and verify fresh standalone full-path replay."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure


def main():
    source=ROOT/'runs/reconnect-t2-rotation';out=ROOT/'runs/reconnect-t3-affine'
    archive=ROOT/'exports/ltn-t3-affine.zip';delivery=ROOT/'exports/ltn-t3-affine-delivery.json';downloads=ROOT/'exports/ltn-t3-affine-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)),'closed evidence is never overwritten'
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'numerical-audit.json');r=read(out/'results.json')
    status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert ex['tests_exit_code']==ex['study_exit_code']==ar['audit_exit_code']==0
    assert ex['main_first_attempt_passed'] and ar['final_audit_passed']
    assert not ex['tests_first_attempt_passed'] and not ar['audit_first_attempt_passed']
    assert ex['tests_failed_attempts_preserved']==ar['audit_failed_attempts_preserved']==1
    assert audit['all56_independent_world_geometries_exact_rates_and_full_raw_sequences_verified'] and audit['prior_formal_manifests_verified']==93
    assert status['recorded_checks_total']==480 and not status['fluid_orc_enabled'] and not status['general_smooth_flow_admitted']
    assert r['comparison_decision']=='T3_affine_model_response_inconclusive' and not r['T3_registered_affine_case_passed']
    assert not gate['already_executed'] and gate['new_candidate_graphs']==gate['new_candidate_transient_runs']==0
    prior=read(source/'sha256.json');assert len(prior)==4015
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    assert digest(ROOT/'exports/ltn-t2-rotation.zip')=='007d1037de1eac12ccfe43b8a6202825d0751abcf52682b6a78183e7240e011d'
    files=['research/reconnect/'+n for n in ['affine_compression_inputs.py','t3_affine_core.py','t3_affine_study.py',
      'test_t3_affine.py','t3_affine_protocol.json','T3_AFFINE_CONTROL.md','audit_t3_affine_geometry.py','audit_t3_affine.py',
      'plot_t3_affine.py','document_t3_affine.py','package_t3_affine.py','T3_AFFINE_COMPARISON.md']]
    entries=dict(prior);source_seal=str((source/'sha256.json').relative_to(ROOT));entries[source_seal]=digest(source/'sha256.json')
    for rel in files:assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for path in out.rglob('*'):
        if path.is_file():
            rel=str(path.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(path)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)|{str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()}|dependency_closure([p for p in files if p.endswith('.py')])
    selected.update([source_seal,'docs/RESEARCH_STATUS.md','requirements-reference.txt'])
    selected.update(v['path'] for v in read(out/'evidence-index.json')['sources'] if Path(v['path']).suffix not in ('.npz','.zip'))
    manifest={p:digest(ROOT/p) for p in sorted(selected)}
    note='''LTN T3: one registered planar affine compression pair, N32, T=.05.
Decision: T3_affine_model_response_inconclusive. Stage3 incomplete; fluid ORC off.
Physical model laws: a=1-5t, points=(a*label_x,label_y), periods=(a,1), rho_exact=1/a.
constant_D: D0=.02. constant_Gamma: D_exact=Gamma0*a. They are distinct PDEs.
Original fixed material masses and initial finite-cell mass averages reused byte-for-byte.
56 fresh ordinary-resistance graphs sealed before material/scalar/reference load; six K4/8/16 actual and six full independent paths.
Unchanged original geometry, flux, time and exact orthogonal boundary adapter; no F/C fit or hidden internal material faces.
Raw rho/D float conversions and exact Gamma product defects retained; Duhamel indicators recorded without correction.
Exact directional gx=Gamma_wire/a and gy=Gamma_wire*a checked on every graph and independently audited world faces.
Separate analytic physical clocks for both models; independent 80/100-digit true finite-cell mean evaluations included.
Frozen discrete eigenmode controls, original algebra residue guards, exact minimal 1e-16 tails, raw weights and full independent sequences verified.
Both time controls and both checkpoint state model responses resolved; t=.05 H response resolved.
t=.025 H difference=.004970458087194651 < fixed indicator=.005219845797588792: retained INCONCLUSIVE.
No tightened H formula, changed threshold or extra N/K/model; no claim of general spatial convergence or full T3 admission.
Three new software checks finally pass; recorded total480; old477 not rerun; old v0.8.0/246 not recovered.
First unit-test failure was an incorrect synthetic coarse-grid H pass expectation; expectation corrected before science, formula unchanged.
First numerical-audit stop was Python3.12 compensated versus registered sequential sum, one ulp; only audit arithmetic order corrected.
Both failure logs/JSON retained; prior partial audit files checked without overwriting; scientific results and criteria unchanged.
782 original files, 15 attachments and 93 prior formal manifests verified; four historical diagnostic differences retained.
Previous rotation ZIP SHA256 007d1037de1eac12ccfe43b8a6202825d0751abcf52682b6a78183e7240e011d unchanged.
Replay (dependencies in requirements-reference.txt; recorded Python/Numpy/Scipy in execution.json):
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/t3_affine_study.py --replay runs/reconnect-t3-affine
Replay reconstructs six actual/independent actions from56 sealed caches and checks all endpoints/halfstates/CSR/weights, residues, references and decisions.
No geometry, reference PDE solve or old candidate rerun during replay. Full historical preservation audit needs original workspace.
review-manifest.json covers included files; cumulative science seal also references omitted history.
Next gate is read-only saved-control/error-budget review only, unexecuted; zero new graphs/paths/PDEs, no retroactive acceptance.
No clipping, repair, mass/rho/state/row/weight normalization, PDF, version, commit/push, environment change or publication.
2026-11-02 Asia/Shanghai. General T3/smooth flow/finite-node admission false; kappa not computed.
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-t3-affine-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            result=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/t3_affine_study.py'),'--replay','runs/reconnect-t3-affine'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=600)
            assert result.returncode==0,result.stdout+result.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'archive_CRC_and_all_included_SHA256_verified':True,'fresh_replay_exit_code':result.returncode,'fresh_replay_log':result.stdout,
      'fresh_replay_files_unchanged':True,'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),
      'comparison_decision':r['comparison_decision'],'T3_registered_affine_case_passed':False,
      'per_model_final_observations':{M:v[-1] for M,v in r['per_model_observations'].items()},'model_comparisons':r['model_comparisons'],
      'new_software_checks':3,'recorded_checks_total':480,'previous_checks_rerun':0,'tests_first_attempt_passed':False,
      'unit_test_and_audit_failed_attempts_preserved':True,'new_candidate_graphs':56,'new_candidate_transient_runs':6,
      'new_independent_uniformization_sequences':6,'new_raw_action_intervals':56,'old_candidate_reruns':0,'new_reference_PDE_solves':0,
      'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':93,
      'next_gate_executed':False,'next_gate_new_graphs_or_paths':0,'original_backends_changed':False,
      'general_T3_admitted':False,'general_smooth_flow_admitted':False,'kappa_computed':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# T3平面仿射压缩：恒D/恒Γ对照\n\n[代码、报告、56图、原始状态与动作、两模型参考和独立审计](ltn-t3-affine.zip)。\n\n'
          f'SHA-256：`{info["archive"]["sha256"]}`。\n\n[交付核验](ltn-t3-affine-delivery.json)。CRC/逐文件SHA通过，全新解压6条实际/独立路径及半步状态逐位重放通过。\n\n'
          't=.025 H模型差未超过固定指标，本例保留未决；终点H差和两时刻状态差已分辨。下一项只读复核，未执行；Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
