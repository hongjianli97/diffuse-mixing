"""Inherited byte preservation and raw full-PDE frozen shear evidence audit."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_kernel2d import digest
from shear_probe import make_shear_samples,shear_gauss_reference,shear_face_reference,shear_geometry
from compression_midpoint import load_laplacian,weighted_norm
from sparse_exchange_audit import audit_laplacian


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-compression-spatial'
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
    assert cfg['protocol']==json.loads((source/'stage3-status.json').read_text())['next_shear_frozen_predeclared']
    assert cfg['state_source_sha256']==digest(Path(cfg['state_source']))
    with np.load(out/'shear-frozen.npz') as z,np.load(cfg['state_source']) as state:
        for key in ['mass','c0','ids','material_label_centers','rho0_point']:np.testing.assert_array_equal(z[key],state[key])
        m=z['mass'];c=z['c0'];labels=z['material_label_centers']
        data,arrays,audit=make_shear_samples(labels,z['ids'],m,c,z['rho0_point'],z['time'])
        np.testing.assert_array_equal(z['positions'],data.positions);np.testing.assert_array_equal(z['rho'],data.rho)
        for key,value in arrays.items():np.testing.assert_array_equal(z[key],value)
        for key,value in shear_geometry(128,labels,.25,.05).items():np.testing.assert_array_equal(z[key],value)
        for order in [8,16]:
            q=shear_gauss_reference(128,.25,.25,.02,labels,m,order)
            f=shear_face_reference(128,.25,.25,.02,labels,m,order)
            for prefix,control in [('gauss',q),('face',f)]:
                for key,value in control.items():np.testing.assert_array_equal(z[f'{prefix}{order}_{key}'],value)
        np.testing.assert_array_equal(z['reference_rate'],z['face16_rate'])
        L=load_laplacian(z);assert audit_laplacian(m,L)==r['sparse_graph_audit']
        np.testing.assert_array_equal(-(L@c)/m,z['kernel_rate']);np.testing.assert_array_equal(z['rate_error'],z['kernel_rate']-z['reference_rate'])
        np.testing.assert_array_equal(z['quadrature_area'],m/data.rho[1])
        norm=weighted_norm(m,z['reference_rate']);relative=lambda v:weighted_norm(m,v)/norm
        for key,v in [('relative_mass_weighted_instantaneous_rate_error',z['rate_error']),
            ('face16_vs_full_Eulerian_Gauss16_relative_difference',z['reference_rate']-z['gauss16_rate']),
            ('Gauss8_vs_Gauss16_relative_difference',z['gauss8_rate']-z['gauss16_rate']),
            ('omitting_mixed_and_shear_gradient_terms_reference_relative_change',z['gauss16_mixed_derivative']+z['gauss16_shear_gradient'])]:
            assert relative(v)==r[key]
        for key,value in r['PDE_component_norms_over_full_rate_norm'].items():assert relative(z['gauss16_'+key])==value
        assert np.all(z['conductances']>0) and not r['fluid_orc_enabled'] and r['new_frozen_graphs']==1 and r['new_transient_runs']==0
        with np.load(out/'shear-observations.npz') as small:
            for key in small.files:np.testing.assert_array_equal(small[key],z[key])
    checks={'predeclared_single_shear_frozen_case_verified':True,'saved_mass_c0_ids_and_oracle_input_unchanged':True,
        'periodic_shear_and_area_centroid_geometry_verified':True,
        'independent_material_faces_and_full_Eulerian_Gauss8_16_arrays_recomputed':True,
        'mixed_derivative_shear_gradient_and_density_drift_retained':True,
        'raw_graph_action_area_weights_and_all_reported_norms_recomputed':True,'small_observation_arrays_equal_raw_data':True,
        'initial_sub_ulp_periodicity_test_failure_log_preserved':(out/'tests-initial-periodicity.log').is_file(),
        'new_transient_runs':0,'full_P_constructed':False,'fluid_orc_enabled':False}
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
