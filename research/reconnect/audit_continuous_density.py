"""Verify immutable source evidence and saved fixed-comparator decomposition."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_kernel2d import digest
from continuous_density_kernel import split_budget


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-kernel2d-variable-density'
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
    cfg=json.loads((out/'config.json').read_text());result=json.loads((out/'results.json').read_text())
    assert cfg['reference_protocol_source_sha256']==digest(source/'stage3-status.json')
    assert cfg['reference_protocol']==json.loads((source/'stage3-status.json').read_text())['next_reference_predeclared']
    with np.load(out/'continuous-density-budget.npz') as z,np.load(source/'variable-density-kernel2d.npz') as original:
        for key in ['mass','c0','reference_rate','kernel_rate','rho','material_label_centers','quadrature_area','true_cell_area']:
            np.testing.assert_array_equal(z[key],original[key])
        components,budget=split_budget(z['mass'],z['reference_rate'],z['kernel_rate'],z['rule1_physical_rate'],z['rule1_input_closure_rate'])
        for key,value in budget.items():
            if isinstance(value,(list,float)):np.testing.assert_allclose(value,result['budget'][key],atol=1e-15,rtol=1e-13)
            else:assert value==result['budget'][key]
        for key,value in zip(['finite_width_error','closure_error','node_quadrature_error'],components):np.testing.assert_array_equal(z[key],value)
        np.testing.assert_array_equal(z['total_error'],z['kernel_rate']-z['reference_rate'])
        np.testing.assert_allclose(components.sum(axis=0),z['total_error'],atol=1e-16,rtol=0.)
        np.testing.assert_allclose(z['input_extension_state'],z['c0'],atol=2e-16,rtol=0.)
        np.testing.assert_array_equal(z['input_extension_weight_ratio'],z['quadrature_area']/z['true_cell_area'])
        _,low=split_budget(z['mass'],z['reference_rate'],z['kernel_rate'],z['rule0_physical_rate'],z['rule0_input_closure_rate'])
        norm=np.sqrt(z['mass']@(z['reference_rate']**2));err=lambda v:np.sqrt(z['mass']@(v*v))/norm
        np.testing.assert_allclose(err(z['rule1_physical_rate']-z['rule0_physical_rate']),result['reference_audit']['physical_reference_rule_difference_relative'],atol=1e-18)
        np.testing.assert_allclose(err(z['rule1_input_closure_rate']-z['rule0_input_closure_rate']),result['reference_audit']['input_closure_rule_difference_relative'],atol=1e-18)
        np.testing.assert_allclose(np.sum(budget['normalized_Gram']),budget['total_relative_error']**2,atol=1e-17)
    assert result['new_graphs']==0 and result['new_discrete_operator_applications']==0 and result['new_fluid_transient_runs']==0 and not result['fluid_orc_enabled']
    checks={'fixed_reference_protocol_verified':True,'saved_input_and_discrete_action_arrays_equal_to_source':True,
        'raw_signed_error_vectors_Gram_and_total_recomputed':True,'both_reference_rule_differences_verified':True,
        'initial_cartesian96_failure_preserved':(out/'tests-initial-cartesian96.log').is_file(),
        'new_graphs':0,'new_discrete_operator_applications':0,'new_fluid_transient_runs':0}
    (out/'numerical-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    preservation={'checked_original_files':len(originals),'checked_historical_attachments':len(attachments),
        'formal_scientific_manifests_verified':formal,'all_formal_scientific_entries_unchanged':True,
        'historical_diagnostic_manifests_not_claimed_as_current_hash_matches':historical,
        'historical_nonformal_path_differences_unchanged_from_source_audit':True,'historical_note':old['historical_note'],
        'candidate_and_all_saved_data_unchanged':True,'mutable_index_intentionally_excluded':'docs/RESEARCH_STATUS.md'}
    (out/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    print(json.dumps({'original_files_verified':len(originals),'historical_attachments_verified':len(attachments),
        'prior_formal_manifests_verified':len(formal),'unchanged_historical_diagnostic_manifests':len(historical),**checks},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
