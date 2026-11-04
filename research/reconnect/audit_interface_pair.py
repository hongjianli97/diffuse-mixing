"""Two saved interface rows: exact material references and raw-row audit."""
from pathlib import Path
import argparse,json
import hashlib
import numpy as np
from point_row_response import row_response
from interface_pair_probe import quadratic_cell_reference,region_summary

_hash_cache={}
def digest(path):
    path=Path(path);stat=path.stat();key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
    if key not in _hash_cache:
        with path.open('rb') as f:_hash_cache[key]=hashlib.file_digest(f,'sha256').hexdigest()
    return _hash_cache[key]


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-point-row'
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
    protocol=json.loads((source/'stage3-status.json').read_text())['next_saved_interface_pair_predeclared']
    assert cfg['protocol']==protocol and cfg['protocol_source_sha256']==digest(source/'stage3-status.json')
    graph=root/protocol['source_graph'];assert digest(graph)==protocol['source_graph_sha256']==cfg['source_graph_sha256']
    ell=1/protocol['N_per_direction'];h=protocol['h_m'];D=protocol['D_m2_per_s'];origin=protocol['contact_origin_a_m']
    keys=['mass','ids','c0','material_label_centers','rho0_point','quadrature_area','true_cell_area_reference_only','reference_rate','kernel_rate']
    masses=[];rates=[];refs=[];state_values=[];cross_values=[]
    with np.load(out/'interface-pair.npz') as z,np.load(graph) as original:
        ptr=original['L_indptr'];all_values=original['L_data'];all_cols=original['L_indices']
        original_arrays={key:original[key] for key in keys}
        assert cfg['CSR_shape']==original['L_shape'].tolist()
        for side,target in zip(['left','right'],protocol['target_indices']):
            c=cfg['rows'][side];v=r['rows'][side]
            start,stop=map(int,ptr[target:target+2]);cols=all_cols[start:stop];values=all_values[start:stop]
            assert c['source_start_stop']==[start,stop] and c['source_row_columns_dtype']==str(cols.dtype) and c['source_row_values_dtype']==str(values.dtype)
            np.testing.assert_array_equal(z[side+'_row_columns'],cols);np.testing.assert_array_equal(z[side+'_saved_L_row'],values)
            saved={key:arr[cols] for key,arr in original_arrays.items()}
            for key,value in saved.items():
                np.testing.assert_array_equal(z[side+'_original_'+key],value)
                assert c['original_array_dtypes'][key]==str(value.dtype)
            at=int(np.flatnonzero(cols==target)[0]);mass=float(saved['mass'][at]);center=saved['material_label_centers'][at]
            assert c['target_index']==target==v['target_index'] and c['original_c0_at_target']==float(saved['c0'][at])
            assert v['target_id']==int(saved['ids'][at])==target and v['target_rho']==float(saved['rho0_point'][at])
            np.testing.assert_array_equal(saved['mass']/saved['rho0_point'],saved['quadrature_area'])
            np.testing.assert_array_equal(saved['quadrature_area'],saved['true_cell_area_reference_only'])
            np.testing.assert_array_equal(saved['quadrature_area'],ell**2)
            assert set(saved['rho0_point'])=={1.,4.}
            dx=saved['material_label_centers']-center;dx-=np.rint(dx)
            np.testing.assert_array_equal(z[side+'_periodic_row_displacement'],dx)
            assert np.all(np.linalg.norm(dx,axis=1)<=2*h)
            assert v['maximum_row_neighbor_distance_m']==float(np.max(np.linalg.norm(dx,axis=1)))
            arrays,metrics=row_response(values,cols,saved['material_label_centers'][:,0],target,mass,ell,origin=origin)
            for key,value in arrays.items():np.testing.assert_array_equal(z[side+'_'+key],value)
            assert metrics==v['row'] and not metrics['row_sum_or_moments_repaired']
            independent=[quadratic_cell_reference(float(saved['material_label_centers'][k,0]),ell,float(saved['rho0_point'][k]),float(saved['mass'][k]),D,origin) for k in range(len(cols))]
            exact_means=np.array([item['values']['mass_average'] for item in independent])
            np.testing.assert_array_equal(z[side+'_independent_integrated_quadratic_mass_means'],exact_means)
            target_ref=independent[at]
            assert v['independent_reference']==target_ref and target_ref['original_mass_matches_constant_phase_square']
            assert target_ref['values']['PDE_mass_average_rate_per_s']==v['PDE_response_per_s']==D
            face_key='right_q_normal' if side=='left' else 'left_q_normal'
            assert target_ref['exact'][face_key]=='0'
            raw=metrics['raw_CSR_quadratic_response_per_s'];integrated=-float(values@exact_means)/mass
            assert v['response_error_per_s']==raw-D and v['relative_error']==(raw-D)/D and v['relative_error_percent']==100*((raw-D)/D)
            assert v['independent_integrated_state_action_per_s']==integrated
            assert v['integrated_state_action_minus_preselected_CSR_per_s']==integrated-raw
            assert v['integrated_mean_minus_formula_max_abs']==float(np.max(abs(exact_means-arrays['probe_quadratic_mass_means'])))
            assert v['opposite_phase_entries']==int(np.sum(saved['rho0_point']!=saved['rho0_point'][at]))
            other=64 if side=='left' else 63;cross_values.append(float(values[cols==other][0]))
            masses.append(mass);rates.append(raw);refs.append(D);state_values.append(metrics['target_probe_mass_average'])
    assert region_summary(masses,rates,refs)==r['two_cell_region']
    assert cross_values[0]==cross_values[1]==r['saved_shared_offdiagonal_L_left_to_right_kg_per_s']==r['saved_shared_offdiagonal_L_right_to_left_kg_per_s']
    assert r['two_target_quadratic_mass_mean_difference']==state_values[1]-state_values[0]
    assert r['exact_interface_left_and_right_normal_flux']==0. and r['probe_origin_m']==origin
    assert r['same_probe_correct_PDE_response_both_sides_per_s']==D and r['saved_row_evaluations']==2
    assert r['same_saved_graph_and_original_data_unchanged'] and r['raw_CSR_main_value_selected_before_inspection']
    assert not r['pure_phase_ideal_eta_used_as_correction']
    assert r['new_candidate_graphs']==r['new_edge_weights']==r['new_operator_quadrature_evaluations']==r['new_candidate_transient_runs']==r['new_reference_transient_runs']==0
    assert not r['finite_node_candidate_admitted'] and not r['fluid_orc_enabled']
    tests=json.loads((out/'tests.json').read_text());assert tests['exit_code']==0 and tests['checks_passed']==3 and tests['recorded_total']==363
    checks={'predeclared_two_saved_interface_rows_scope_verified':True,'source_hash_raw_rows_indices_and_original_neighbor_arrays_unchanged':True,
        'raw_CSR_actions_and_all_moment_controls_recomputed':True,
        'independent_exact_material_state_and_face_flux_references_recomputed':True,
        'mass_weighted_region_and_arithmetic_diagnostic_kept_distinct':True,
        'shared_pair_symmetry_and_equal_target_means_verified':True,'no_row_sum_moment_or_pure_phase_eta_correction':True,
        'new_checks_passed':3,'recorded_checks_total':363,'new_candidate_graphs':0,'new_edge_weights':0,
        'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'fluid_orc_enabled':False}
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
