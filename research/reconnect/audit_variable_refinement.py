"""Preserve old evidence and recheck the stored fine/parent numerical budget."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.sparse import csr_matrix
from audit_kernel2d import digest
from continuous_density_kernel import split_budget
from smooth_density_probe import analytic_reference
from variable_parent_observation import parent_mass_audit,observe_parent


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-kernel2d-continuous-density'
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
    assert cfg['protocol']==json.loads((source/'stage3-status.json').read_text())['next_control_predeclared']
    ref=analytic_reference(128,.25,.02)
    with np.load(out/'variable-density-refined.npz') as z,np.load(out/'same-parent-comparison.npz') as p,np.load(source/'continuous-density-budget.npz') as prior:
        for key in ['mass','c0','reference_rate','kernel_rate','rule1_physical_rate','rule1_input_closure_rate','finite_width_error','closure_error','node_quadrature_error']:
            np.testing.assert_array_equal(p['parent_'+key],prior[key])
        for saved,key in [('mass','mass'),('c0','state'),('reference_rate','rate'),('rho_cell_average','rho_average')]:np.testing.assert_array_equal(z[saved],ref[key])
        np.testing.assert_array_equal(z['rho'][1],ref['rho_point'])
        L=csr_matrix((z['L_data'],z['L_indices'],z['L_indptr']),shape=tuple(z['L_shape']))
        np.testing.assert_allclose(-(L@z['c0'])/z['mass'],z['kernel_rate'],atol=0.,rtol=0.)
        components,budget=split_budget(z['mass'],z['reference_rate'],z['kernel_rate'],z['rule1_physical_rate'],z['rule1_input_closure_rate'])
        for key,value in budget.items():
            if isinstance(value,(list,float)):np.testing.assert_allclose(value,r['budget'][key],atol=1e-15,rtol=1e-13)
            else:assert value==r['budget'][key]
        for key,value in zip(['finite_width_error','closure_error','node_quadrature_error'],components):np.testing.assert_array_equal(z[key],value)
        names={'c0':'c0','reference_rate':'reference_rate','kernel_rate':'kernel_rate','physical_rate':'rule1_physical_rate','input_rate':'rule1_input_closure_rate',
            'finite_width_error':'finite_width_error','closure_error':'closure_error','node_quadrature_error':'node_quadrature_error'}
        for current,saved in names.items():np.testing.assert_array_equal(p['restricted_'+current],observe_parent(z[saved],z['mass'],prior['mass']))
        sums,rounding,audit=parent_mass_audit(z['mass'],prior['mass']);np.testing.assert_array_equal(p['child_mass_sum'],sums);np.testing.assert_array_equal(p['mass_rounding_relative'],rounding)
        assert audit==r['parent_mass_roundoff_audit']
        np.testing.assert_array_equal(p['actual_change'],p['restricted_kernel_rate']-p['parent_kernel_rate'])
        np.testing.assert_array_equal(p['physical_change'],p['restricted_physical_rate']-p['parent_rule1_physical_rate'])
        np.testing.assert_allclose(p['actual_change'],p['physical_change']+p['closure_change']+p['quadrature_change'],atol=1e-16,rtol=0.)
        norm=np.sqrt(prior['mass']@(prior['reference_rate']**2));err=lambda v:np.sqrt(prior['mass']@(v*v))/norm
        olderr=err(prior['kernel_rate']-prior['reference_rate']);newerr=err(p['restricted_kernel_rate']-prior['reference_rate'])
        np.testing.assert_allclose(newerr,r['comparison']['new_total_error_saved_parent_cells'],atol=1e-15)
        np.testing.assert_allclose(1-newerr/olderr,r['comparison']['relative_error_reduction_fraction'],atol=1e-15)
        assert np.all(z['conductances']>=0) and not r['fluid_orc_enabled'] and r['new_fluid_transient_runs']==0
    checks={'fixed_refinement_protocol_verified':True,'saved_parent_arrays_equal_to_source':True,'analytic_fine_mass_and_state_verified':True,
        'sparse_rate_three_part_budget_and_parent_comparison_recomputed':True,'raw_parent_child_mass_rounding_preserved':True,
        'child_mass_renormalized':False,'new_fluid_transient_runs':0}
    (out/'numerical-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    preservation={'checked_original_files':len(originals),'checked_historical_attachments':len(attachments),
        'formal_scientific_manifests_verified':formal,'all_formal_scientific_entries_unchanged':True,
        'historical_diagnostic_manifests_not_claimed_as_current_hash_matches':historical,
        'historical_nonformal_path_differences_unchanged_from_source_audit':True,'historical_note':old['historical_note'],
        'candidate_backend_and_old_results_unchanged':True,'mutable_index_intentionally_excluded':'docs/RESEARCH_STATUS.md'}
    (out/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    print(json.dumps({'original_files_verified':len(originals),'historical_attachments_verified':len(attachments),
        'prior_formal_manifests_verified':len(formal),'unchanged_historical_diagnostic_manifests':len(historical),**checks},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
