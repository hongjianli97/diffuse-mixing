"""One saved point row: raw CSR and signed-budget audit, no graph rebuild."""
from pathlib import Path
import argparse,json
import hashlib
import numpy as np
from point_row_response import row_response,signed_budget

_hash_cache={}
def digest(path):
    path=Path(path);stat=path.stat();key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
    if key not in _hash_cache:
        with path.open('rb') as f:_hash_cache[key]=hashlib.file_digest(f,'sha256').hexdigest()
    return _hash_cache[key]


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-ideal-row'
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
    cfg=json.loads((out/'config.json').read_text());r=json.loads((out/'results.json').read_text())
    protocol=json.loads((source/'stage3-status.json').read_text())['next_saved_point_row_review_predeclared']
    assert cfg['protocol']==protocol and cfg['protocol_source_sha256']==digest(source/'stage3-status.json')
    graph=root/protocol['source_graph'];assert digest(graph)==protocol['source_graph_sha256']==cfg['source_graph_sha256']
    target=protocol['target_index'];ell=1/protocol['N_per_direction'];h=protocol['h_m'];D=protocol['D_m2_per_s']
    with np.load(out/'point-row.npz') as z,np.load(graph) as original:
        ptr=original['L_indptr'];start,stop=map(int,ptr[target:target+2])
        cols=original['L_indices'][start:stop];values=original['L_data'][start:stop]
        np.testing.assert_array_equal(z['row_columns'],cols);np.testing.assert_array_equal(z['saved_L_row'],values)
        assert cfg['CSR_source_row_start_stop']==[start,stop] and cfg['CSR_shape']==original['L_shape'].tolist()
        assert cfg['source_row_columns_dtype']==str(cols.dtype) and cfg['source_row_values_dtype']==str(values.dtype)
        for key in ['mass','ids','c0','material_label_centers','rho0_point','quadrature_area','true_cell_area_reference_only','reference_rate','kernel_rate']:
            np.testing.assert_array_equal(z['original_'+key],original[key][cols])
            assert str(z['original_'+key].dtype)==cfg['original_array_dtypes'][key]
        at=int(np.flatnonzero(cols==target)[0]);mass=float(z['original_mass'][at]);center=z['original_material_label_centers'][at]
        assert cfg['target_id']==int(z['original_ids'][at])==protocol['target_id'] and cfg['target_mass_kg']==mass
        assert cfg['original_c0_at_target']==float(z['original_c0'][at])
        np.testing.assert_array_equal(z['original_rho0_point'],1.)
        np.testing.assert_array_equal(z['original_mass']/z['original_rho0_point'],z['original_quadrature_area'])
        np.testing.assert_array_equal(z['original_quadrature_area'],z['original_true_cell_area_reference_only'])
        np.testing.assert_array_equal(z['original_quadrature_area'],ell**2)
        dx=z['original_material_label_centers']-center;dx-=np.rint(dx)
        np.testing.assert_array_equal(z['periodic_row_displacement'],dx)
        assert np.all(np.linalg.norm(dx,axis=1)<=2*h)
        assert r['maximum_saved_row_neighbor_distance_m']==float(np.max(np.linalg.norm(dx,axis=1)))
        arrays,metrics=row_response(values,cols,z['original_material_label_centers'][:,0],target,mass,ell)
        for key,value in arrays.items():np.testing.assert_array_equal(z[key],value)
        assert metrics==r['row']
    ideal_path=root/protocol['saved_ideal_response_source'];assert digest(ideal_path)==cfg['source_ideal_results_sha256']
    ideal=json.loads(ideal_path.read_text())['rules']['fine']['raw_direct_aggregate_rate_per_s']
    point=r['row']['raw_CSR_quadratic_response_per_s'];budget=signed_budget(point,ideal,D)
    assert budget==r['signed_budget'] and r['saved_ideal_direct_response_per_s']==ideal
    assert r['raw_point_response_per_s']==point and r['PDE_unit_curvature_response_per_s']==D
    assert r['point_relative_error_percent']==100*budget['point_minus_PDE_relative_to_PDE']
    assert r['point_relative_error_ppm']==1e6*budget['point_minus_PDE_relative_to_PDE']
    assert r['raw_CSR_value_selected_before_inspection'] and not r['row']['row_sum_or_moments_repaired']
    assert r['saved_point_row_evaluations']==1 and r['original_c0_saved_cosine_PDE_and_candidate_unchanged']
    assert r['new_candidate_graphs']==r['new_edge_weights']==r['new_operator_quadrature_evaluations']==r['new_candidate_transient_runs']==r['new_reference_transient_runs']==0
    assert not r['finite_node_candidate_admitted'] and not r['fluid_orc_enabled']
    tests=json.loads((out/'tests.json').read_text());assert tests['exit_code']==0 and tests['checks_passed']==4 and tests['recorded_total']==360
    checks={'predeclared_one_saved_CSR_row_scope_verified':True,'source_hash_row_bytes_indices_and_original_neighbor_arrays_unchanged':True,
        'raw_CSR_main_action_centered_control_and_moments_recomputed':True,
        'same_pure_phase_original_mass_and_area_assumptions_verified':True,
        'all_signed_terms_and_float_identity_residual_recomputed':True,
        'no_row_sum_or_moment_repair':True,'new_checks_passed':4,'recorded_checks_total':360,
        'new_candidate_graphs':0,'new_edge_weights':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'fluid_orc_enabled':False}
    (out/'numerical-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
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
