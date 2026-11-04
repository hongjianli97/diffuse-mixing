"""Provenance/receipt audit only; no numerical cases, test runs or machine proof claims."""
from pathlib import Path
import argparse,ast,json
from audit_route_decision import ROOT,read,save,digest,preservation
from t2_rotation_domain_review import SOURCE,REPORT,ZERO_FIELDS,inspect_backend


def verify_receipt(directory):
    out=Path(directory).resolve();config=read(out/'config.json');status=read(out/'stage3-status.json');ex=read(out/'execution.json')
    assert config['parent_manifest_entries_verified']==3825 and config['recorded_checks_total']==status['recorded_checks_total']==474
    for key in ZERO_FIELDS:assert config[key]==status[key]==0,key
    assert ex['review_exit_code']==0 and ex['review_first_attempt_passed'] and ex['standard_library_only']
    assert not ex['NPZ_arrays_loaded'] and ex['new_numerical_runs']==0
    lock=read(out/'review-lock.json')
    for rel,h in lock['files'].items():assert digest(ROOT/rel)==h,rel
    assert digest(out/'contract-at-review.md')==digest(REPORT)
    assert digest(out/'review-protocol.json')==digest(ROOT/'research/reconnect/t2_rotation_domain_protocol.json')
    capability=read(out/'backend-capability.json');assert capability==inspect_backend(ROOT/capability['path'])
    assert not capability['world_periodic_basis_argument_present'] and not capability['api_or_backend_changed']
    decision=read(out/'rotation-domain-decision.json')
    assert decision['decision']==status['decision']=='boundary_representation_gap_with_minimal_isometry_adapter_proposed'
    assert not decision['adapter_implemented'] and not decision['backend_or_api_changed'] and not decision['rotation_test_executed']
    assert not decision['axis_wrapping_of_general_rotated_worldpoints_is_valid_objectivity_case']
    assert not decision['quarterturn_static_test_completes_continuous_rotation'] and not decision['ordinary_center_resistance_disproved']
    assert not decision['global_boundary_orientation_is_local_F_C_fit']
    derivation=read(out/'derivation-index.json')
    assert [v['id'] for v in derivation['written_derivations']]==['P1','P2','P3','P4','P5','P6']
    assert not derivation['machine_proof_verification_performed'] and derivation['new_exact_algebra_certificate_groups']==0
    assert derivation['report_sha256']==digest(REPORT)
    ledger=read(out/'acceptance-ledger.json');assert ledger['recorded_checks_total']==474 and ledger['new_software_checks']==0
    for obj in (status,ledger):
        assert not obj['general_T2_admitted'] and not obj['general_smooth_flow_admitted'] and not obj['finite_node_candidate_admitted']
        assert obj['stage3_physical_acceptance']=='incomplete' and not obj['fluid_orc_enabled']
    gate=read(out/'next-gate-spec.json');assert not gate['already_executed']
    assert gate['N']==32 and gate['K']==16 and gate['D']==.02 and gate['new_candidate_graphs']==16
    assert gate['new_candidate_transient_runs']==gate['new_complete_independent_uniformization_sequences']==1
    assert gate['new_reference_PDE_solves']==gate['old_candidate_reruns']==gate['new_material_definitions']==0
    assert gate['planned_new_software_checks']==len(gate['check_categories'])==3
    assert gate['main_wall_seconds_cap']==300 and gate['main_peak_RSS_MiB_cap']==1024
    assert not gate['adapter_implemented_this_round'] and not gate['rotation_points_graphs_states_generated_this_round']
    assert not gate['general_float_world_trajectories_or_nonorthogonal_lattices_admitted']
    assert not gate['kappa_computed'] and not gate['fluid_orc_enabled']
    assert len(gate['static_action_sources'])==16
    assert not list(out.rglob('*.npz')) and not list(out.rglob('geometry*.json')) and not list(out.rglob('action-step*.npz'))
    code=ROOT/'research/reconnect/t2_rotation_domain_review.py';tree=ast.parse(code.read_text());imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(v.name for v in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append(node.module)
        elif isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id not in {'sample_points','native_run','operators','build_periodic_faces','exponential','uniformization'}
    assert not any(v.split('.')[0] in {'numpy','scipy','mpmath','sympy'} for v in imports)
    return {'receipt_consistent':True,'new_numerical_runs':0,'new_software_checks':0,'recorded_checks_total':474,
      'rotation_test_executed':False,'adapter_implemented':False,'next_gate_executed':False,'proofs_are_written_derivations_not_machine_verification':True,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}


def full_audit(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    result=verify_receipt(out);prior=read(SOURCE/'sha256.json');assert len(prior)==3825
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    assert read(out/'source-gate-spec.json')==read(SOURCE/'next-gate-spec.json')
    for item in read(out/'evidence-index.json')['sources']:assert digest(ROOT/item['path'])==item['sha256'],item['path']
    gate=read(out/'next-gate-spec.json')
    for item in list(gate['sources'].values())+list(gate['static_action_sources'].values()):assert digest(ROOT/item['path'])==item['sha256'],item['path']
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    previous=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
      {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(prior)}]
    for item in previous:
        if item['manifest'] in known:continue
        path=ROOT/item['manifest'];assert digest(path)==item['manifest_sha256'];entries=read(path);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==91;save(out/'preservation.json',preserved)
    old=read(ROOT/'exports/ltn-t2-d0-delivery.json');assert digest(ROOT/'exports/ltn-t2-d0.zip')==old['archive']['sha256']=='56530e4f41e65e66b6580a0803205469aed1ad65be70073a35785e1aa086c47e'
    result.update({'parent_manifest_entries_verified':3825,'original_files_verified':782,'historical_attachments_verified':15,
      'prior_formal_manifests_verified':91,'previous_D0_archive_sha256_verified':old['archive']['sha256'],
      'four_known_historical_diagnostic_differences_unchanged':True,'source_API_backend_and_prior_results_unchanged':True})
    save(out/'review-audit.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--receipt-only',action='store_true');a=p.parse_args()
    result=verify_receipt(a.directory) if a.receipt_only else full_audit(a.directory)
    print(json.dumps(result,indent=2),flush=True)
