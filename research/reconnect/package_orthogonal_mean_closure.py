"""Seal theory, seven scalar algebra groups and a standalone no-solver replay."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

from audit_route_decision import ROOT,read,save,digest


def main():
    source=ROOT/'runs/reconnect-point-endpoint-review';out=ROOT/'runs/reconnect-orthogonal-mean-closure'
    archive=ROOT/'exports/ltn-orthogonal-mean-closure.zip'
    delivery=ROOT/'exports/ltn-orthogonal-mean-closure-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed evidence'
    execution=read(out/'execution.json');audit=read(out/'theory-audit.json');status=read(out/'stage3-status.json')
    assert execution['definition_exit_code']==execution['standalone_algebra_exit_code']==execution['audit_exit_code']==0
    assert execution['first_attempts_passed'] and audit['prior_formal_manifests_verified']==75
    assert status['recorded_checks_total']==428 and status['new_exact_algebra_certificate_groups']==7
    assert not status['candidate_graph_implemented'] and not status['new_numerical_gate_selected']
    assert not status['general_allowed_input_support'] and not status['full_transient_error_estimate_proved']
    scope=read(out/'next-theory-scope.json')
    assert scope['no_automatic_reuse_of_cellwise_constant_Voronoi_resistance_formula']
    assert not scope['numerical_gate_selected'] and not scope['fluid_orc_enabled']
    prior=read(source/'sha256.json')
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+name for name in ['orthogonal_mean_closure.py','audit_orthogonal_mean_closure.py',
        'package_orthogonal_mean_closure.py','ORTHOGONAL_MATERIAL_MEAN_CLOSURE.md']]
    source_seal=str((source/'sha256.json').relative_to(ROOT));entries=dict(prior);entries[source_seal]=digest(source/'sha256.json')
    for rel in files:entries[rel]=digest(ROOT/rel)
    for p in out.iterdir():
        assert p.is_file(),p
        entries[str(p.relative_to(ROOT))]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files);selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([source_seal,'research/reconnect/audit_route_decision.py','docs/RESEARCH_STATUS.md'])
    selected.update(v['path'] for v in read(out/'evidence-index.json')['sources'])
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=(
        'LTN THEORY ONLY: point-defined orthogonal surrogate cells and fixed material-mean closure.\n'
        'Conditional formula: shared conductance b*face_length / ((distance/2)/(rho_i D)+(distance/2)/(rho_j D)); D0 gives zero.\n'
        'Surrogate Voronoi cells are NOT identified true material cells. Requires every original m_i=b*rho_i*surrogate_area_i.\n'
        'Original mass, point rho and positions must not be adjusted to enforce compatibility.\n'
        'General valid point inputs FAIL this compatibility: exact within-true-cell column displacement witness retained.\n'
        'Material-mean weak defect explicitly retains centroid offset and second moments/covariance.\n'
        'Bound covers means versus nodal form only, not complete PDE consistency or transient convergence.\n'
        'True/surrogate observation mismatch bounded conditionally via mass-weighted W1; true geometry is theory only, not candidate input.\n'
        'Four transient terms retained; full surrogate discretization and density-evolution estimates remain unproved.\n'
        '7 new exact scalar algebra groups, NOT automated geometry/PDE proofs or new software/physical tests.\n'
        'No mesh, conductance, graph, operator action, quadrature, flow case or propagation evaluated. Old proofs/tests not rerun.\n'
        'Replay needs standard Python only: python -S research/reconnect/orthogonal_mean_closure.py --replay runs/reconnect-orthogonal-mean-closure\n'
        'Algebra only: python -S research/reconnect/orthogonal_mean_closure.py --certificates-only\n'
        'Software check record remains428. 7 new algebra groups counted once despite standalone/replay execution.\n'
        'Next THEORY scope: mass-constrained Laguerre feasibility, total fixed-density mass, seed inclusion and positive face distances.\n'
        'Keep nearest-point density fixed when changing auxiliary cells; cut density interfaces need new resistance justification.\n'
        'No new numerical gate or backend selected; do not restart old point graph h/time/grid/weight retuning.\n'
        '782 original files,15 attachments,75 prior formal scientific manifests preserved; four historical diagnostic differences retained.\n'
        'review-manifest.json covers included files; cumulative science seal also references omitted history.\n'
        'Full preservation audit needs original workspace. Mutable docs/RESEARCH_STATUS.md is outside science seal.\n'
        'No PDF, release, commit/push, environment configuration change or publication. Old v0.8.0 not recovered.\n'
        'Ordinary central-resistance FV baseline retained; mass-average-resistance flux experimental.\n'
        'Stage3 incomplete; finite-node and contact-transient physical admission false; fluid ORC disabled.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-orthogonal-theory-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,'-S',str(Path(directory)/'research/reconnect/orthogonal_mean_closure.py'),
                '--replay','runs/reconnect-orthogonal-mean-closure'],cwd=directory,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
            assert completed.returncode==0,completed.stdout+completed.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    assert digest(ROOT/'docs/RESEARCH_STATUS.md')==manifest['docs/RESEARCH_STATUS.md']
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_standard_library_replay_exit_code':completed.returncode,
        'fresh_standard_library_replay_log':completed.stdout,'fresh_replay_files_unchanged':True,
        'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'source_scientific_manifest':source_seal,
        'new_software_checks':0,'recorded_checks_total':428,'previous_checks_rerun':0,'new_exact_algebra_certificate_groups':7,
        'new_candidate_weights':0,'new_operator_actions':0,'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,
        'new_numerical_or_flow_cases':0,'candidate_graph_implemented':False,'general_allowed_input_support':False,
        'full_transient_error_estimate_proved':False,'new_numerical_gate_selected':False,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':75,
        'finite_node_candidate_admitted':False,'contact_transient_physically_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with (ROOT/'exports/ltn-orthogonal-mean-closure-downloads.md').open('x') as f:
        f.write('# 点定义辅助面闭合理论\n\n'
            '[推导、输入契约、7组精确代数、审计与可重放代码](ltn-orthogonal-mean-closure.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[范围与校验](ltn-orthogonal-mean-closure-delivery.json)。全新解压后标准库重放通过。\n\n'
            '候选有质量相容/辅助几何条件，不能覆盖一般合法点输入；未运行或安装新图。\n\n'
            '软件记录428，另列7组新代数；Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
