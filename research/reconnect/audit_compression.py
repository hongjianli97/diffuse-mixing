"""Read-only provenance and saved compression-map/PDE/graph-action audit."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.sparse import csr_matrix
from audit_kernel2d import digest
from compression_probe import make_compression_samples,analytic_material_rate,compression_geometry


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-kernel2d-density-transient'
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
    assert cfg['protocol']==json.loads((source/'stage3-status.json').read_text())['next_compression_probe_predeclared']
    with np.load(out/'compression-kernel2d.npz') as z,np.load(source/'density-transient.npz') as prior:
        for key in ['mass','c0','ids','material_label_centers']:np.testing.assert_array_equal(z[key],prior[key])
        m=z['mass'];c=z['c0'];a=z['material_label_centers'];rho0=prior['rho'][0]
        d,g,diagnostic=make_compression_samples(a,z['ids'],m,c,rho0,z['time'],.25,.05)
        np.testing.assert_array_equal(z['positions'],d.positions);np.testing.assert_array_equal(z['rho'],d.rho);np.testing.assert_array_equal(z['J'],g['J'])
        np.testing.assert_array_equal(z['rho'],rho0[None,:]/z['J']);np.testing.assert_array_equal(z['rho'][1],1.)
        reference,split=analytic_material_rate(128,.25,.25,.02,m,c,a)
        np.testing.assert_array_equal(z['reference_rate'],reference)
        for key,value in split.items():np.testing.assert_array_equal(z[key],value)
        geometry=compression_geometry(128,a,.25,.05)
        for key,value in geometry.items():np.testing.assert_array_equal(z[key],value)
        L=csr_matrix((z['L_data'],z['L_indices'],z['L_indptr']),shape=tuple(z['L_shape']))
        np.testing.assert_allclose(-(L@c)/m,z['kernel_rate'],atol=0.,rtol=0.)
        np.testing.assert_array_equal(z['rate_error'],z['kernel_rate']-reference)
        np.testing.assert_array_equal(z['quadrature_area'],m/z['rho'][1])
        norm=np.sqrt(m@(reference*reference));err=lambda v:np.sqrt(m@(v*v))/norm
        for key,v in [('relative_mass_weighted_instantaneous_rate_error',z['rate_error']),
            ('analytic_face_vs_Gauss16_reference_relative_difference',reference-z['gauss16_rate']),
            ('Gauss8_vs_16_reference_relative_difference',z['gauss8_rate']-z['gauss16_rate'])]:
            np.testing.assert_allclose(err(v),r[key],atol=1e-18,rtol=1e-13)
        assert np.all(z['conductances']>=0) and not r['fluid_orc_enabled'] and r['new_transient_runs']==0
        with np.load(out/'compression-observations.npz') as small:
            for key in small.files:np.testing.assert_array_equal(small[key],z[key])
    checks={'fixed_compression_protocol_verified':True,'saved_mass_c0_ids_unchanged':True,
        'rho0_over_J_and_explicit_trajectory_verified':True,'true_area_and_raw_center_offset_verified':True,
        'analytic_material_reference_sparse_action_and_errors_recomputed':True,'small_observation_arrays_equal_to_raw_data':True,
        'initial_shape_test_failure_preserved':(out/'tests-initial-shape.log').is_file(),'new_transient_runs':0}
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
