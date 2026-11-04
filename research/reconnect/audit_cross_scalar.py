"""Cross-phase scalar theory: exact certificate and preservation audit."""
from pathlib import Path
import argparse,json
import hashlib
import numpy as np
from cross_scalar_certificate import certificates

_hash_cache={}
def digest(path):
    path=Path(path);stat=path.stat();key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
    if key not in _hash_cache:
        with path.open('rb') as f:_hash_cache[key]=hashlib.file_digest(f,'sha256').hexdigest()
    return _hash_cache[key]


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-interface-pair'
    old=json.loads((source/'preservation.json').read_text());originals=json.loads(Path('/workspace/original-handoff-sha256.json').read_text())
    for rel,h in originals.items():assert digest(root/rel)==h,rel
    attachments=json.loads(Path('/workspace/setup-ltn/attachment-manifest.json').read_text())
    for item in attachments:
        p=Path('/workspace')/item['path'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],str(p)
    formal=old['formal_scientific_manifests_verified']+[{'manifest':str((source/'sha256.json').relative_to(root)),
        'manifest_sha256':digest(source/'sha256.json'),'entries_verified':len(json.loads((source/'sha256.json').read_text()))}]
    for item in formal:
        p=root/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=json.loads(p.read_text());assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(root/rel)==h,(item['manifest'],rel)
    historical=old['historical_diagnostic_manifests_not_claimed_as_current_hash_matches']
    for item in historical:
        p=root/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=json.loads(p.read_text());assert len(entries)==item['entries']
        differences=[{'path':rel,'saved':h,'current':digest(root/rel)} for rel,h in entries.items() if digest(root/rel)!=h]
        assert sorted(differences,key=lambda v:v['path'])==sorted(item['historical_paths_differing_from_later_formal_versions'],key=lambda v:v['path'])
    cfg=json.loads((out/'config.json').read_text());r=json.loads((out/'proof-certificates.json').read_text())
    protocol=json.loads((source/'stage3-status.json').read_text())['next_cross_phase_scalar_compatibility_review_predeclared']
    assert cfg['protocol']==protocol and cfg['protocol_source_sha256']==digest(source/'stage3-status.json')
    snapshot=root/protocol['source_snapshot'];assert digest(snapshot)==protocol['source_snapshot_sha256']==cfg['source_snapshot_sha256']
    assert digest(source/'results.json')==cfg['source_results_sha256']
    expected,original_arrays,derived=certificates(snapshot)
    assert expected==r and r['verified_exact_algebra_groups']==7 and r['recorded_checks_total']==363
    with np.load(out/'cross-scalar.npz') as z:
        assert set(z.files)==set(original_arrays)|set(derived)
        for key,value in original_arrays.items():np.testing.assert_array_equal(z[key],value)
        for key,value in derived.items():np.testing.assert_array_equal(z[key],value)
    assert json.loads((out/'standalone-certificates.json').read_text())==r
    previous=json.loads((source/'results.json').read_text());controls=json.loads((out/'raw-controls.json').read_text())
    for side in ['left','right']:
        oldrow=previous['rows'][side]['row'];c=controls[side]
        assert c['previous_raw_CSR_response_per_s']==oldrow['raw_CSR_quadratic_response_per_s']
        assert c['previous_centered_response_per_s']==oldrow['centered_difference_response_per_s']
        assert c['previous_raw_minus_exact_binary_raw_per_s']==oldrow['raw_CSR_quadratic_response_per_s']-r['rows'][side]['exact_raw_CSR_on_cached_binary_means']['decimal_for_reading_only']
        assert c['previous_centered_minus_exact_A_plus_C_per_s']==oldrow['centered_difference_response_per_s']-r['rows'][side]['exact_centered_original_response']['decimal_for_reading_only']
        assert c['phase_float_sum_minus_previous_centered_per_s']==r['rows'][side]['floating_reductions']['sum_phase_contributions_per_s']-oldrow['centered_difference_response_per_s']
    assert r['parameter_values_evaluated_on_a_graph']==r['new_candidate_graphs']==r['new_edge_weights']==r['new_operator_quadrature_evaluations']==r['new_candidate_transient_runs']==r['new_reference_transient_runs']==0
    assert not r['finite_node_candidate_admitted'] and not r['fluid_orc_enabled']
    checks={'predeclared_theory_only_shared_scalar_scope_verified':True,'source_snapshot_all_cached_arrays_unchanged':True,
        'seven_exact_rational_certificate_groups_recomputed':True,'independent_process_certificates_identical':True,
        'original_raw_CSR_controls_and_all_float_residuals_retained':True,'no_parameter_graph_runs_or_edge_changes':True,
        'new_numerical_tests_added':0,'exact_algebra_certificate_groups':7,'recorded_checks_total':363,'fluid_orc_enabled':False}
    (out/'proof-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    preservation={'checked_original_files':len(originals),'checked_historical_attachments':len(attachments),
        'formal_scientific_manifests_verified':formal,'all_formal_scientific_entries_unchanged':True,
        'historical_diagnostic_manifests_not_claimed_as_current_hash_matches':historical,
        'historical_nonformal_path_differences_unchanged_from_source_audit':True,'historical_note':old['historical_note'],
        'old_candidate_code_and_results_unchanged':True,'mutable_index_intentionally_excluded':'docs/RESEARCH_STATUS.md'}
    (out/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    print(json.dumps({'original_files_verified':len(originals),'historical_attachments_verified':len(attachments),
        'prior_formal_manifests_verified':len(formal),'unchanged_historical_diagnostic_manifests':len(historical),**checks},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
