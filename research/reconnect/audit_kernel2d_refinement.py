"""Preservation plus independent saved-vector comparison audit, no new solve."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.sparse import csr_matrix
from audit_kernel2d import digest
from sparse_exchange_audit import restrict_equal_nested_squares


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-kernel2d-oracle'
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
    cfg=json.loads((out/'config.json').read_text());result=json.loads((out/'results.json').read_text());comparison=result['comparison']
    assert cfg['protocol_source_sha256']==digest(source/'stage3-status.json')
    assert cfg['second_level']==json.loads((source/'stage3-status.json').read_text())['next_control_predeclared']
    with np.load(out/'refined-kernel2d.npz') as z,np.load(out/'same-parent-comparison.npz') as p,np.load(source/'periodic-kernel2d.npz') as prior:
        m=z['mass'];c=z['c0'];L=csr_matrix((z['L_data'],z['L_indices'],z['L_indptr']),shape=tuple(z['L_shape']))
        np.testing.assert_allclose(-(L@c)/m,z['kernel_rate'],atol=0.,rtol=0.)
        np.testing.assert_allclose(z['reference_rate'],-8*np.pi**2*.02*(c-.5),atol=2e-15,rtol=2e-15)
        np.testing.assert_array_equal(z['finite_width_error'],z['continuum_finite_width_rate']-z['reference_rate'])
        np.testing.assert_array_equal(z['node_quadrature_error'],z['kernel_rate']-z['continuum_finite_width_rate'])
        norm=np.sqrt(m@(z['reference_rate']**2))
        for key,v in [('relative_mass_weighted_instantaneous_rate_error',z['kernel_rate']-z['reference_rate']),('finite_width_rate_bias_relative_to_local_PDE',z['finite_width_error']),('node_quadrature_rate_difference_relative_to_local_PDE',z['node_quadrature_error'])]:
            np.testing.assert_allclose(np.sqrt(m@(v*v))/norm,result[key],atol=1e-15)
        base_keys=['mass','c0','reference_rate','kernel_rate','continuum_finite_width_rate','finite_width_error','node_quadrature_error']
        for key in base_keys:np.testing.assert_array_equal(p['parent_'+key],prior[key])
        for key in base_keys[1:]:np.testing.assert_array_equal(p['restricted_'+key],restrict_equal_nested_squares(z[key],m,prior['mass']))
        np.testing.assert_array_equal(p['action_change'],p['restricted_kernel_rate']-p['parent_kernel_rate'])
        np.testing.assert_array_equal(p['continuous_change'],p['restricted_continuum_finite_width_rate']-p['parent_continuum_finite_width_rate'])
        np.testing.assert_array_equal(p['quadrature_change'],p['restricted_node_quadrature_error']-p['parent_node_quadrature_error'])
        norm0=np.sqrt(prior['mass']@(prior['reference_rate']**2));err=lambda v:np.sqrt(prior['mass']@(v*v))/norm0
        reduction=err(prior['kernel_rate']-prior['reference_rate'])-err(p['restricted_kernel_rate']-prior['reference_rate'])
        np.testing.assert_allclose(reduction,comparison['absolute_rate_error_reduction_on_same_parent_cells'],atol=1e-15)
        assert result['new_fluid_transient_runs']==0 and result['new_propagators_constructed']==0 and not result['fluid_orc_enabled']
        assert np.all(z['conductances']>=0) and result['sparse_only']
    audit={'saved_sparse_action_and_error_metrics_verified':True,'saved_coarse_arrays_bytewise_value_equal_to_source':True,
        'mass_restriction_and_signed_change_vectors_recomputed':True,'fixed_second_level_protocol_verified':True,'new_fluid_transient_runs':0}
    (out/'numerical-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    preservation={'checked_original_files':len(originals),'checked_historical_attachments':len(attachments),
        'formal_scientific_manifests_verified':formal,'all_formal_scientific_entries_unchanged':True,
        'historical_diagnostic_manifests_not_claimed_as_current_hash_matches':historical,
        'historical_nonformal_path_differences_unchanged_from_source_audit':True,'historical_note':old['historical_note'],
        'previous_point_kernel_backend_and_physical_problem_unchanged':True,'mutable_index_intentionally_excluded':'docs/RESEARCH_STATUS.md'}
    (out/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    print(json.dumps({'original_files_verified':len(originals),'historical_attachments_verified':len(attachments),
        'prior_formal_manifests_verified':len(formal),'unchanged_historical_diagnostic_manifests':len(historical),**audit},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
