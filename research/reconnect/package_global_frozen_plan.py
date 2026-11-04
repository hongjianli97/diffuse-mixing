"""Seal and replay geometry/cost declarations, never the future global gate."""
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
    source=ROOT/'runs/reconnect-reconstructed-pair';out=ROOT/'runs/reconnect-global-frozen-plan'
    archive=ROOT/'exports/ltn-global-frozen-plan.zip'
    delivery=ROOT/'exports/ltn-global-frozen-plan-delivery.json';seal=out/'sha256.json'
    assert not any(p.exists() for p in [archive,delivery,seal]),'never overwrite sealed artifacts'
    status=read(out/'stage3-status.json');audit=read(out/'plan-audit.json');execution=read(out/'execution.json')
    assert execution['planning_exit_code']==execution['audit_exit_code']==0
    assert audit['prior_formal_manifests_verified']==68 and status['recorded_checks_total']==410
    assert status['global_candidate_result']=='not_computed' and status['new_edge_weights']==0
    prior=read(source/'sha256.json')
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+name for name in ['global_frozen_gate_plan.py','audit_global_frozen_plan.py',
        'package_global_frozen_plan.py','GLOBAL_FROZEN_GATE_SCOPE_AND_COST.md']]
    source_seal=str((source/'sha256.json').relative_to(ROOT))
    entries=dict(prior);entries[source_seal]=digest(source/'sha256.json')
    for rel in files:entries[rel]=digest(ROOT/rel)
    for p in out.iterdir():
        assert p.is_file(),p
        entries[str(p.relative_to(ROOT))]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)
    selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update([source_seal,'research/reconnect/audit_route_decision.py','docs/RESEARCH_STATUS.md'])
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'] if not item['path'].endswith('.npz'))
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    note=(
        'LTN read-only original-snapshot global frozen gate scope/cost review. NOT AN EXECUTED GLOBAL RESULT.\n'
        'Geometry and given-density validation permits integral-only reuse: 132 future primitive requests, 45 radius keys, 964 coefficient slots.\n'
        'All 2,719,744 undirected edges must still be covered by the future action. No edge dropping or global sparse matrix required.\n'
        'Fixed-ID ownership receipts cannot be transplanted; only a.e. inverse-density integral values are reusable under verified input pattern.\n'
        'Original mass-average state, D/h, kernel, saved reference/region masks and locked original norm remain fixed.\n'
        'Future exact common-denominator arithmetic is a representation of unchanged values, never physical renormalization.\n'
        'Mandatory old 663-edge and two exact-row agreement before a future global comparison.\n'
        'Replay (NumPy only): python research/reconnect/global_frozen_gate_plan.py --replay runs/reconnect-global-frozen-plan\n'
        'Replay checks planning point inputs, request classifications, costs and gate declarations only.\n'
        'No path primitive, kernel, conductance, operator, flow, reference solve, old suite or new software tests run this round.\n'
        'Recorded software checks remain 410. Geometry/audit assertions are not counted as new unit tests.\n'
        '782 originals, 15 attachments and 68 prior formal manifests preserved; four historical diagnostic differences unchanged.\n'
        'Included point arrays support independent planning replay; full historical NPZ/attachments/source audit are not included.\n'
        'plan-manifest.json covers included files; cumulative scientific seal also covers unarchived history.\n'
        'docs/RESEARCH_STATUS.md is a mutable snapshot outside the cumulative scientific seal.\n'
        'No PDF, release, commit/push, environment modification or publication. Stage 3 incomplete; fluid ORC disabled.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('plan-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'plan-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-global-plan-',dir='/tmp') as directory:
            z.extractall(directory)
            completed=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/global_frozen_gate_plan.py'),
                '--replay','runs/reconnect-global-frozen-plan'],cwd=directory,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
            assert completed.returncode==0,completed.stdout+completed.stderr
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    cost=read(out/'cost-and-reuse.json')
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified':True,'fresh_geometry_cost_declaration_replay_exit_code':completed.returncode,
        'fresh_geometry_cost_declaration_replay_log':completed.stdout,'future_global_gate_executed':False,
        'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),
        'source_scientific_manifest':source_seal,'new_software_checks':0,'recorded_checks_total':410,'previous_checks_rerun':0,
        'planned_global_unordered_edges':cost['undirected_edge_iterations_planned'],
        'planned_primitive_requests':132,'planned_radius_classes':45,'planned_coefficient_cache_slots':964,
        'new_path_integrals':0,'new_edge_weights':0,'new_operator_actions':0,'new_numerical_cases':0,
        'new_candidate_graphs':0,'new_global_matrices':0,'new_candidate_transient_runs':0,'new_reference_solves':0,
        'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':68,
        'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with (ROOT/'exports/ltn-global-frozen-plan-downloads.md').open('x') as f:
        f.write('# 同一快照全域冻结门槛：范围与成本审阅\n\n'
            '[报告、规划代码、点输入及未执行门槛](ltn-global-frozen-plan.zip)。\n\n'
            f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
            '[交付范围与校验](ltn-global-frozen-plan-delivery.json)。全新解压目录的几何/成本/契约重放通过，未运行全域候选。\n\n'
            '本轮新权重/算子作用/数值算例/软件检查均0，记录仍410；Stage 3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
