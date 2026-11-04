"""Inherited bytes, raw reference projections and uncertainty-metric audit."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_kernel2d import digest
from compression_reference import material_columns,project_to_saved_cells
from compression_reference_study import metrics


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-kernel2d-compression-midpoint'
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
    assert cfg['protocol']==json.loads((source/'stage3-status.json').read_text())['next_compression_reference_predeclared']
    with np.load(out/'compression-reference.npz') as z,np.load(source/'midpoint-observations.npz') as prior:
        for key in ['mass','c0','ids','material_label_centers']:np.testing.assert_array_equal(z[key],prior[key])
        m=z['mass'];c0=z['c0'];mean=m@c0/m.sum();norm=np.sqrt(m@((c0-mean)**2))
        relative=lambda v:float(np.sqrt(m@(v*v))/norm)
        for case in cfg['reference_cases']:
            M=case['columns'];steps=case['time_steps'];prefix=f'n{M}-k{steps}_';audit=r['reference_runs'][prefix[:-1]]
            a,mf,B=material_columns(M,.25)
            for key,value in [('axis',a),('mass_1d',mf),('initial_B',B)]:np.testing.assert_array_equal(z[prefix+key],value)
            snapshots=z[prefix+'B_snapshots'];times=z[prefix+'snapshot_time'];states=z[prefix+'projected_states']
            np.testing.assert_array_equal(snapshots[0],B)
            np.testing.assert_array_equal(z[prefix+'A_snapshots'],snapshots*np.exp(-.02*(2*np.pi)**2*times[:,None]))
            for index,(b,t) in enumerate(zip(snapshots,times)):
                c,child=project_to_saved_cells(b,mf,t,.02,128,m)
                np.testing.assert_array_equal(states[index],c);np.testing.assert_array_equal(z[prefix+'child_mass_1d'],child)
                assert metrics(m,c,c0)==audit['snapshot_metrics'][index]
                step=[0,steps//2,steps][index]
                np.testing.assert_allclose(mf@b,z[prefix+'B_inventory'][step],atol=0.,rtol=0.)
                np.testing.assert_allclose(.5*mf@(b*b),z[prefix+'B_energy'][step],atol=0.,rtol=0.)
            np.testing.assert_allclose(states[0],c0,atol=3e-16,rtol=0.)
            assert np.max(np.diff(z[prefix+'B_energy']))<0
            assert np.max(abs(z[prefix+'CN_energy_balance_residual']))<1e-15
            assert not audit['reference_is_a_positive_candidate_backend']
        differences={
            'space_difference':z['n512-k256_projected_states']-z['n1024-k256_projected_states'],
            'time_coarse_difference':z['n1024-k256_projected_states']-z['n1024-k512_projected_states'],
            'time_fine_difference':z['n1024-k512_projected_states']-z['n1024-k1024_projected_states']}
        controls=r['uncertainty_controls'];keys=['space_512_vs_1024_at_256_steps','time_256_vs_512_at_1024_columns','time_512_vs_1024_at_1024_columns']
        for key,(name,vector) in zip(keys,differences.items()):
            np.testing.assert_array_equal(z[name],vector)
            np.testing.assert_array_equal([relative(v) for v in vector],controls['state_differences_relative_to_original_mass_weighted_fluctuation'][key])
        np.testing.assert_array_equal(z['best_projected_states'],z['n1024-k1024_projected_states'])
        assert not controls['Richardson_extrapolated_state_used'] and not controls['indicator_is_a_rigorous_error_bound']
    checks={'original_saved_mass_c0_ids_unchanged':True,'predeclared_flow_and_reference_levels_verified':True,
        'raw_nested_component_projection_and_initial_state_verified':True,
        'snapshot_amplitudes_inventory_energy_and_saved_balance_diagnostics_verified':True,
        'all_state_difference_vectors_and_metrics_recomputed':True,'candidate_kernel_and_old_results_unchanged':True,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'fluid_orc_enabled':False}
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
