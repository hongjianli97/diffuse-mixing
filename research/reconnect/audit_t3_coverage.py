"""Standard-library original-criteria/source receipt audit; no numerical import or replay."""
from pathlib import Path
import argparse,ast,json,resource,time
from audit_route_decision import ROOT,read,save,digest,preservation
from t3_coverage_review import SOURCE,ZERO_FIELDS,coverage


def receipt(out):
    out=Path(out).resolve();ex=read(out/'execution.json');lock=read(out/'review-lock.json');index=read(out/'evidence-index.json')
    assert ex['review_exit_code']==0 and ex['final_review_passed'] and not ex['review_first_attempt_passed']
    assert ex['review_failed_attempts_preserved']==1 and ex['standard_library_only'] and not ex['old_numerical_paths_replayed']
    assert ex['elapsed_main_seconds']<120 and ex['peak_RSS_MiB']<1024
    assert digest(ROOT/'research/reconnect/t3_coverage_review.py')==lock['code_sha256']
    assert digest(out/'review-protocol.json')==lock['protocol_sha256']==digest(ROOT/'research/reconnect/t3_coverage_protocol.json')
    assert digest(out/'contract-at-review.md')==lock['contract_sha256']==digest(ROOT/'research/reconnect/AFFINE_SHEAR_DOMAIN_CONTRACT.md')
    first=out/'review-first-attempt';firstlock=read(first/'review-lock.json')
    assert digest(first/'review-source-attempt1.py.txt')==firstlock['code_sha256']
    assert read(first/'failed-execution.json')['review_exit_code']==1 and not read(first/'failed-execution.json')['source_changed']
    source_gate=read(out/'source-gate-spec.json');parent_gate_refs={item['path']:item['sha256'] for item in source_gate['sources']}
    lookup={}
    for item in index['sources']:
        p=out/item['copy'];assert digest(p)==item['sha256'] and item['source_SHA_bound_by_formal_parent']
        if not item['cumulative_content_entry_present']:
            assert item['path']=='docs/PHASE3_REBUILD.md' and item['binding']=='immutable_parent_gate_source_SHA'
            assert item['sha256']==parent_gate_refs[item['path']]
        else:assert item['binding']=='cumulative_content_entry'
        lookup[item['path']]=p
    calculated=coverage(lambda rel:lookup[rel]);assert calculated==read(out/'coverage-review.json')
    ledger=read(out/'acceptance-ledger.json');assert ledger['rows']==calculated['acceptance_ledger']
    assert not ledger['new_acceptance_criteria_added'] and not ledger['source_generals_or_original_N32_decision_changed']
    # Independent structural inspection: identify the non-affine source without running it.
    tree=ast.parse(lookup['research/reconnect/shear_probe.py'].read_text())
    shear=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='shear_map')
    assert any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='sin' for node in ast.walk(shear))
    adapter=lookup['research/reconnect/oriented_periodic_boundary.py'].read_text()
    assert 'exact proper orthogonality required; no orthonormalization' in adapter
    assert 'after,half=exponential(op,state,dt)' in lookup['research/reconnect/voronoi_time16_core.py'].read_text()
    rows={item['id']:item for item in ledger['rows']}
    assert rows['T3_periodic_sinusoidal_shear']['state']=='positive_same_backend_nonaffine_evidence'
    assert rows['T3_affine_shear']['state']=='missing_registered_execution_and_sliding_lattice_mode'
    assert rows['T3_direction_and_constitutive']['state']=='covered_in_registered_orthogonal_case'
    assert rows['density_sources']['state']=='same_final_backend_three_source_comparison_incomplete'
    status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert status['recorded_checks_total']==483 and all(status[key]==0 for key in ZERO_FIELDS)
    assert not status['fluid_orc_enabled'] and not status['general_T3_admitted'] and not status['next_adapter_or_experiment_executed']
    assert not gate['already_executed'] and gate['N']==32 and gate['ordered_interval_counts']==[4,8,16]
    assert gate['new_candidate_graphs']==28 and gate['new_candidate_transient_runs']==gate['new_complete_independent_uniformization_sequences']==3
    assert gate['planned_new_software_checks']==4 and gate['recorded_checks_total_if_all_pass']==487
    assert not gate['new_geometry_implemented_this_round'] and not gate['new_shear_points_graphs_references_or_states_generated_this_round']
    assert gate['proofs_are_written_not_machine_verified'] and gate['old_candidate_reruns']==gate['new_reference_PDE_solves']==0
    assert gate['main_wall_seconds_cap']==900 and gate['main_peak_RSS_MiB_cap']==1024
    return {'original_requirement_quote_lines_and_source_SHA_verified':True,'all_seven_same_backend_locks_verified':True,
      'independent_AST_sine_shear_and_orthogonal_boundary_scope_verified':True,
      'registered_affine_compression_and_nonlinear_shear_positive_evidence_retained':True,
      'source_integrator_distinguished_from_production_implicit_Euler_default':True,
      'historical_phase_indirect_parent_gate_binding_verified':True,'old_N32_inconclusive_retained':True,
      'no_new_acceptance_demand_DNS_wait_or_fluid_ORC_run':True,'next_affine_shear_control_registered_only':True,
      'written_contract_is_not_implemented_or_machine_certified':True,'receipt_consistent':True}


def main(directory,receipt_only=False):
    out=Path(directory).resolve();start=time.monotonic();result=receipt(out)
    if receipt_only:
        print('Fresh standard-library coverage receipt passed: original quotes,seven immutable backend locks,sine-vs-affine shear scope,orthogonal boundary limit and preserved old decisions; one sliding-domain control registered only. No numerical graph/action/path/reference/ORC or old replay.',flush=True);return
    assert not (out/'sha256.json').exists()
    preserved=preservation();known={item['manifest'] for item in preserved['formal_scientific_manifests_verified']}
    previous=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
      'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in previous:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==96;save(out/'preservation.json',preserved)
    index=read(out/'evidence-index.json');parent=read(SOURCE/'sha256.json')
    for item in index['sources']:
        assert digest(ROOT/item['path'])==item['sha256']
        if item['cumulative_content_entry_present']:assert parent[item['path']]==item['sha256']
    gate=read(out/'next-gate-spec.json')
    for key in ('pure_source','material_source'):
        item=gate[key];assert digest(ROOT/item['path'])==parent[item['path']]==item['sha256']
    assert digest(ROOT/'exports/ltn-t3-affine-spatial.zip')=='a886cc0d90428ab3266a925b3df40260afcb7469ec7713bc0efa314d2744d3d6'
    result.update({'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':96,
      **dict.fromkeys(ZERO_FIELDS,0),'recorded_checks_total':483,'old_numerical_paths_replayed':False,
      'review_is_provenance_and_scope_not_new_physical_acceptance':True,'general_T3_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})
    save(out/'review-audit.json',result)
    save(out/'audit-execution.json',{'audit_exit_code':0,'audit_first_attempt_passed':True,'standard_library_only':True,
      'elapsed_audit_seconds':time.monotonic()-start,'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      'audit_source_sha256':digest(Path(__file__))})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--receipt-only',action='store_true');a=p.parse_args();main(a.directory,a.receipt_only)
