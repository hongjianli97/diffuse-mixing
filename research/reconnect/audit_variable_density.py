"""Preservation and independent saved-input/target/error audit; no graph solve."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.sparse import csr_matrix
from audit_kernel2d import digest
from smooth_density_probe import analytic_reference


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-kernel2d-refinement'
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
    cfg=json.loads((out/'config.json').read_text());result=json.loads((out/'results.json').read_text());protocol=cfg['predeclared_protocol']
    assert cfg['protocol_source_sha256']==digest(source/'stage3-status.json')
    assert protocol==json.loads((source/'stage3-status.json').read_text())['next_probe_predeclared']
    reference=analytic_reference(protocol['N_per_direction'],protocol['density_amplitude'],protocol['D_m2_per_s'])
    with np.load(out/'variable-density-kernel2d.npz') as z:
        m=z['mass'];c=z['c0'];L=csr_matrix((z['L_data'],z['L_indices'],z['L_indptr']),shape=tuple(z['L_shape']))
        for saved,key in [('mass','mass'),('c0','state'),('reference_rate','rate'),('reference_laplacian_rate','laplacian_rate'),('reference_drift_rate','drift_rate'),('rho_cell_average','rho_average')]:
            np.testing.assert_array_equal(z[saved],reference[key])
        np.testing.assert_array_equal(z['rho'][1],reference['rho_point']);np.testing.assert_array_equal(z['rho'][0],reference['rho_point'])
        np.testing.assert_allclose(-(L@c)/m,z['kernel_rate'],atol=0.,rtol=0.)
        np.testing.assert_array_equal(z['rate_error'],z['kernel_rate']-z['reference_rate'])
        np.testing.assert_array_equal(z['quadrature_area'],m/z['rho'][1])
        np.testing.assert_array_equal(z['quadrature_area_relative_error'],z['quadrature_area']/z['true_cell_area']-1)
        norm=np.sqrt(m@(z['reference_rate']**2));err=lambda v:np.sqrt(m@(v*v))/norm
        for key,v in [('relative_mass_weighted_instantaneous_rate_error',z['rate_error']),
            ('missing_density_drift_reference_error',z['reference_laplacian_rate']-z['reference_rate']),
            ('candidate_distance_to_laplacian_only_diagnostic',z['kernel_rate']-z['reference_laplacian_rate']),
            ('analytic_vs_Gauss16_reference_relative_difference',z['reference_rate']-z['gauss16_rate']),
            ('Gauss8_vs_16_reference_relative_difference',z['gauss8_rate']-z['gauss16_rate'])]:
            np.testing.assert_allclose(err(v),result[key],atol=1e-15,rtol=1e-14)
        assert np.all(z['conductances']>=0) and not result['fluid_orc_enabled'] and result['new_fluid_transient_runs']==0 and result['new_propagators_constructed']==0
    checks={'fixed_protocol_verified':True,'analytic_fixed_mass_and_mass_average_state_verified':True,
        'full_PDE_drift_target_and_independent_saved_Gauss_controls_verified':True,
        'sparse_action_and_error_metrics_recomputed':True,'raw_point_density_weights_retained':True,'new_fluid_transient_runs':0}
    (out/'numerical-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    preservation={'checked_original_files':len(originals),'checked_historical_attachments':len(attachments),
        'formal_scientific_manifests_verified':formal,'all_formal_scientific_entries_unchanged':True,
        'historical_diagnostic_manifests_not_claimed_as_current_hash_matches':historical,
        'historical_nonformal_path_differences_unchanged_from_source_audit':True,'historical_note':old['historical_note'],
        'candidate_point_kernel_backend_unchanged':True,'mutable_index_intentionally_excluded':'docs/RESEARCH_STATUS.md'}
    (out/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    print(json.dumps({'original_files_verified':len(originals),'historical_attachments_verified':len(attachments),
        'prior_formal_manifests_verified':len(formal),'unchanged_historical_diagnostic_manifests':len(historical),**checks},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
