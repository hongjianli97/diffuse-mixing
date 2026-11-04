"""Seal fixed-density mass/interiority theory and replay without numerical solvers."""
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
    source=ROOT/'runs/reconnect-orthogonal-mean-closure';out=ROOT/'runs/reconnect-laguerre-mass-feasibility'
    archive=ROOT/'exports/ltn-laguerre-mass-feasibility.zip'
    delivery=ROOT/'exports/ltn-laguerre-mass-feasibility-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed evidence'
    execution=read(out/'execution.json');audit=read(out/'theory-audit.json');status=read(out/'stage3-status.json')
    assert execution['definition_exit_code']==execution['standalone_algebra_exit_code']==execution['audit_exit_code']==0
    assert execution['first_attempts_passed'] and audit['prior_formal_manifests_verified']==76
    assert status['recorded_checks_total']==428 and status['new_exact_algebra_certificate_groups']==8
    assert status['mass_partition_existence_proved_in_report']
    assert not status['general_seed_interiority_and_positive_face_distance_guarantee']
    assert not status['general_Laguerre_TPFA_backend_selected'] and not status['new_numerical_gate_selected']
    scope=read(out/'next-theory-scope.json');assert scope['no_mass_density_position_or_response_normalization']
    assert not scope['new_numerical_gate_selected'] and not scope['new_candidate_graph_selected']
    prior=read(source/'sha256.json')
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+name for name in ['laguerre_mass_feasibility.py','audit_laguerre_mass_feasibility.py',
        'package_laguerre_mass_feasibility.py','LAGUERRE_MASS_INTERIORITY_REVIEW.md']]
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
        'LTN THEORY ONLY: fixed-density periodic Laguerre mass feasibility and seed-interiority obstruction.\n'
        'Existence proof in report: finite distinct sites, fixed bounded positive density, positive target masses with matching total.\n'
        'Concave dual has gauge-coercive maximum and gradient target mass minus cell mass. No general weight uniqueness claimed.\n'
        'Existence does NOT imply seeds inside assigned cells or positive two-sided face distances.\n'
        'Exact analytic witness: sites0 and1/4 in periodic x, masses1/4 and3/4, weight difference3/32.\n'
        'Cell0=[-5/16,-1/16] times periodic y; mass exact but seed0 outside; right face distance -1/16.\n'
        'Closed seed inclusion permits cell0 mass at least1/3, hence target mass residual at least1/12 for this geometry only.\n'
        'This is NOT a PDE error bound, physical threshold, new flow case or universal no-go for positive exchange.\n'
        'No clipping/absolute-value repair of negative distances; no mass/density/point normalization or scalar/PDE fitting.\n'
        'Fixed nearest-point density must remain fixed under geometric weights; new cells may cut its density interfaces.\n'
        'Old half-distance resistance and unweighted moments cannot be reused generally. 2D cut-interface flux remains unjustified.\n'
        'Positive-distance Laguerre+ordinary TPFA is not selected as a general backend. More general methods remain open.\n'
        '8 new exact algebra groups; analytic existence/interiority proof in report is not an automatic optimization or PDE proof.\n'
        'Software record remains428; no old suite/proof program rerun and no numerical mesh optimizer, graph or propagation.\n'
        'python -S research/reconnect/laguerre_mass_feasibility.py --replay runs/reconnect-laguerre-mass-feasibility\n'
        'Replay needs only standard Python, checks new algebra/analytic witness and fixed contract; does not construct a partition.\n'
        'Next theory scope: explicit capacity mismatch with original m and fixed rho; auxiliary mass is comparison only, not replacement.\n'
        'No new numerical gate selected; no old path graph h/grid/time/weight scan.\n'
        '782 originals,15 attachments,76 prior formal scientific manifests preserved; four historical diagnostic differences retained.\n'
        'review-manifest.json covers included files; cumulative scientific seal also references omitted historical files.\n'
        'Full preservation audit needs original workspace; mutable docs/RESEARCH_STATUS.md is outside science seal.\n'
        'No PDF, release, commit/push, environment change or publication. Old v0.8.0 not recovered.\n'
        'Ordinary central-resistance FV baseline retained; mass-average-resistance flux experimental.\n'
        'Stage3 incomplete; finite-node/contact-transient physical admission false; fluid ORC disabled.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-laguerre-theory-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,'-S',str(Path(directory)/'research/reconnect/laguerre_mass_feasibility.py'),
                '--replay','runs/reconnect-laguerre-mass-feasibility'],cwd=directory,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
            assert completed.returncode==0,completed.stdout+completed.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    assert digest(ROOT/'docs/RESEARCH_STATUS.md')==manifest['docs/RESEARCH_STATUS.md']
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_standard_library_replay_exit_code':completed.returncode,
        'fresh_standard_library_replay_log':completed.stdout,'fresh_replay_files_unchanged':True,
        'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'source_scientific_manifest':source_seal,
        'new_software_checks':0,'recorded_checks_total':428,'previous_checks_rerun':0,'new_exact_algebra_certificate_groups':8,
        'new_candidate_weights':0,'new_operator_actions':0,'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,
        'new_numerical_or_flow_cases':0,'numerical_partition_optimizer_runs':0,
        'mass_partition_existence_proved_in_report':True,'general_seed_interiority_guarantee':False,
        'general_Laguerre_TPFA_backend_selected':False,'new_numerical_gate_selected':False,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':76,
        'finite_node_candidate_admitted':False,'contact_transient_physically_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with (ROOT/'exports/ltn-laguerre-mass-feasibility-downloads.md').open('x') as f:
        f.write('# 质量约束分区与正面距限制\n\n'
            '[存在性推导、精确几何证书、输入契约与标准库重放](ltn-laguerre-mass-feasibility.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[范围与校验](ltn-laguerre-mass-feasibility-delivery.json)。全新解压重放通过，未执行分区求解器。\n\n'
            '存在质量匹配分区不保证种子包含或正面距；选定解析几何的包含约束留下至少1/12质量残差。\n\n'
            '软件记录428，另列8组新代数；Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
