"""Read-only provenance, coupled PDE and vector-order evidence audit."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_kernel2d import digest
from compression_probe import make_compression_samples,analytic_material_rate,compression_geometry,compression_gauss_reference
from compression_midpoint import load_laplacian,commutator_action,weighted_norm


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-kernel2d-compression'
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
    assert cfg['protocol_source_sha256']==digest(source/'stage3-status.json')
    assert cfg['protocol']==json.loads((source/'stage3-status.json').read_text())['next_midpoint_probe_predeclared']
    for rel,h in cfg['endpoint_sources_sha256'].items():assert digest(root/rel)==h
    with np.load(out/'compression-midpoint.npz') as z,np.load(source/'compression-kernel2d.npz') as end:
        for key in ['mass','c0','ids','material_label_centers']:np.testing.assert_array_equal(z[key],end[key])
        m=z['mass'];c=z['c0'];labels=z['material_label_centers'];rho0=end['rho0_point']
        data,g,a=make_compression_samples(labels,z['ids'],m,c,rho0,z['time'])
        for key,value in [('positions',data.positions),('rho',data.rho),*g.items()]:np.testing.assert_array_equal(z[key],value)
        assert data.rho[1].min()<1<data.rho[1].max()
        ref,split=analytic_material_rate(128,.25,.125,.02,m,c,labels)
        for key,value in [('reference_rate',ref),*split.items(),*compression_geometry(128,labels,.125,.025).items()]:np.testing.assert_array_equal(z[key],value)
        for order in [8,16]:
            control=compression_gauss_reference(128,.25,.125,.02,labels,m,order)
            for key,value in control.items():np.testing.assert_array_equal(z[f'gauss{order}_{key}'],value)
        np.testing.assert_array_equal(-(load_laplacian(z)@c)/m,z['kernel_rate'])
        np.testing.assert_array_equal(z['rate_error'],z['kernel_rate']-ref)
        np.testing.assert_array_equal(z['quadrature_area'],m/z['rho'][1])
        assert np.max(abs(z['quadrature_area']/z['true_cell_area_from_unwrapped_faces']-1))>1e-6
        norm=weighted_norm(m,ref);err=lambda v:weighted_norm(m,v)/norm
        for key,v in [('relative_mass_weighted_instantaneous_rate_error',z['rate_error']),
            ('analytic_face_vs_Gauss16_reference_relative_difference',ref-z['gauss16_rate']),
            ('Gauss8_vs_16_reference_relative_difference',z['gauss8_rate']-z['gauss16_rate']),
            ('density_drift_weighted_norm_relative_to_full_reference',z['gauss16_drift_rate'])]:
            np.testing.assert_allclose(err(v),r[key],atol=1e-18,rtol=1e-13)
        assert np.all(z['conductances']>0) and r['sparse_graph_audit']['graph_algebra_eligible']
        with np.load(out/'midpoint-observations.npz') as small:
            for key in small.files:np.testing.assert_array_equal(small[key],z[key])
    start_path=root/cfg['protocol']['endpoint_commutator_diagnostic']['start_source']
    with np.load(start_path) as start,np.load(source/'compression-kernel2d.npz') as end,np.load(out/'endpoint-actions.npz') as saved:
        for key in ['mass','c0','ids']:np.testing.assert_array_equal(saved[key],end[key]);np.testing.assert_array_equal(start[key],end[key])
        actions=commutator_action(m,load_laplacian(start),load_laplacian(end),c)
        for key,value in actions.items():np.testing.assert_array_equal(saved[key],value)
        np.testing.assert_array_equal(saved['pair_commutator'],saved['pair_end_after_start']-saved['pair_start_after_end'])
        comm=actions['commutator'];order=r['endpoint_time_order_diagnostic']
        np.testing.assert_allclose(weighted_norm(m,comm),order['weighted_commutator_norm_kg_sqrt_per_s2'],atol=0.,rtol=0.)
        np.testing.assert_allclose(weighted_norm(m,saved['pair_commutator']-comm)/weighted_norm(m,comm),order['pair_scatter_vs_CSR_commutator_relative_difference'],atol=0.,rtol=0.)
        assert order['pair_scatter_vs_CSR_commutator_relative_difference']<1e-8
    assert not cfg['fluid_orc_enabled'] and cfg['new_transient_runs']==0 and not cfg['matrix_matrix_products']
    checks={'saved_mass_c0_ids_unchanged':True,'predeclared_single_midpoint_verified':True,
        'nonuniform_rho0_over_J_and_raw_area_discrepancy_retained':True,
        'complete_material_PDE_and_independent_Gauss_arrays_recomputed':True,
        'saved_sparse_action_and_error_metrics_recomputed':True,'small_observations_equal_raw_data':True,
        'endpoint_source_hashes_fixed_identity_and_vector_commutator_verified':True,
        'independent_pair_scatter_agrees_with_nested_CSR':True,'new_transient_runs':0,'fluid_orc_enabled':False}
    (out/'numerical-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    preservation={'checked_original_files':len(originals),'checked_historical_attachments':len(attachments),
        'formal_scientific_manifests_verified':formal,'all_formal_scientific_entries_unchanged':True,
        'historical_diagnostic_manifests_not_claimed_as_current_hash_matches':historical,
        'historical_nonformal_path_differences_unchanged_from_source_audit':True,'historical_note':old['historical_note'],
        'old_candidate_code_and_results_unchanged':True,'mutable_index_intentionally_excluded':'docs/RESEARCH_STATUS.md'}
    (out/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    print(json.dumps({'original_files_verified':len(originals),'historical_attachments_verified':len(attachments),
        'prior_formal_manifests_verified':len(formal),'unchanged_historical_diagnostic_manifests':len(historical),**checks},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
