"""Immutable-source and saved scalar/mixing audit, without another evolution."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.sparse import csr_matrix
from audit_kernel2d import digest
from trajectory_chain import mixing_metrics


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-kernel2d-variable-refinement'
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
    assert cfg['protocol']==json.loads((source/'stage3-status.json').read_text())['next_transient_research_predeclared']
    with np.load(out/'density-transient.npz') as z,np.load(source/'variable-density-refined.npz') as prior:
        for key in ['mass','c0','ids','material_label_centers']:np.testing.assert_array_equal(z[key],prior[key])
        m=z['mass'];c0=z['c0'];E0=mixing_metrics(m,c0,1.)['energy_kg'];norm=np.sqrt(2*E0);err=lambda v:np.sqrt(m@(v*v))/norm
        np.testing.assert_array_equal(z['candidate_states'][0],c0)
        expected=np.mod(z['material_label_centers'][None,...]+z['time'][:,None,None]*np.array([.2,-.1]),1.)
        np.testing.assert_array_equal(z['positions'],expected)
        np.testing.assert_array_equal(z['rho'],np.broadcast_to(prior['rho'][0],z['rho'].shape))
        L=csr_matrix((prior['L_data'],prior['L_indices'],prior['L_indptr']),shape=tuple(prior['L_shape']))
        for index,row in enumerate(r['runs']):
            actual=z['candidate_states'][index];reference=z['reference512_states'][index]
            np.testing.assert_allclose(err(actual-reference),row['relative_concentration_error_initial_fluctuation_norm'],atol=1e-15)
            np.testing.assert_allclose(err(z['reference256_states'][index]-reference),row['reference_256_vs_512_relative_difference'],atol=1e-18)
            for name,state in [('candidate_metrics',actual),('reference_metrics_same_saved_initial_E',reference)]:
                metrics=mixing_metrics(m,state,E0)
                for key,value in metrics.items():
                    if isinstance(value,str):assert value==row[name][key]
                    else:np.testing.assert_allclose(value,row[name][key],atol=1e-15)
            np.testing.assert_allclose(actual@(L@actual),row['energy_dissipation_rate_kg_per_s'],atol=1e-15)
            np.testing.assert_allclose(row['H_candidate_minus_reference'],row['candidate_metrics']['H']-row['reference_metrics_same_saved_initial_E']['H'],atol=0.)
        np.testing.assert_allclose(err(z['full_endpoint']-z['mass_symmetric_endpoint']),r['algorithm_controls']['Q_vs_mass_symmetric_initial_fluctuation_norm'],atol=1e-18)
    assert r['new_graphs']==0 and not r['full_P_constructed'] and not r['fluid_orc_enabled']
    checks={'fixed_transient_protocol_verified':True,'saved_mass_actual_c0_and_source_L_unchanged':True,
        'complete_analytic_trajectory_record_verified':True,'raw_state_mixing_and_error_metrics_recomputed':True,
        'independent_algorithm_arrays_and_reference_differences_verified':True,'initial_test_shape_failure_preserved':(out/'tests-initial-shape.log').is_file(),
        'new_graphs':0,'full_P_constructed':False,'new_finite_time_research_cases':1}
    (out/'numerical-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    preservation={'checked_original_files':len(originals),'checked_historical_attachments':len(attachments),
        'formal_scientific_manifests_verified':formal,'all_formal_scientific_entries_unchanged':True,
        'historical_diagnostic_manifests_not_claimed_as_current_hash_matches':historical,
        'historical_nonformal_path_differences_unchanged_from_source_audit':True,'historical_note':old['historical_note'],
        'candidate_and_source_data_unchanged':True,'mutable_index_intentionally_excluded':'docs/RESEARCH_STATUS.md'}
    (out/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    print(json.dumps({'original_files_verified':len(originals),'historical_attachments_verified':len(attachments),
        'prior_formal_manifests_verified':len(formal),'unchanged_historical_diagnostic_manifests':len(historical),**checks},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
