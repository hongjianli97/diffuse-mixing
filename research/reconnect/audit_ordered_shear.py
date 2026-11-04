"""Inherited bytes and saved chronological scalar/graph evidence audit."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_kernel2d import digest
from shear_probe import make_shear_samples
from compression_midpoint import load_laplacian
from compression_reference_study import metrics
from ordered_compression import interval_grid,dissipation
from sparse_exchange_audit import audit_laplacian


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-shear-reference'
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
    assert cfg['protocol']==json.loads((source/'stage3-status.json').read_text())['next_candidate_shear_transient_predeclared']
    graph_count=0;algorithm_max=0.;weighted_mass_max=0.
    with np.load(out/'ordered-shear.npz') as z,np.load(source/'shear-reference-observations.npz') as prior:
        for key in ['mass','c0','ids','material_label_centers']:np.testing.assert_array_equal(z[key],prior['N128_'+key])
        np.testing.assert_array_equal(z['reference_states'],prior['N128_best_reference_states'])
        np.testing.assert_array_equal(z['reference_times'],prior['snapshot_time'])
        z_saved={key:z[key] for key in ['mass','c0','ids','material_label_centers','rho0_point']}
        np.testing.assert_array_equal(z['reference_space_difference'],prior['N128_space_difference'])
        np.testing.assert_array_equal(z['reference_time_difference'],prior['N128_time_difference'])
        m=z['mass'];c0=z['c0'];mu0=m@c0/m.sum();sigma0=np.sqrt(m@((c0-mu0)**2));relative=lambda v:float(np.sqrt(m@(v*v))/sigma0)
        for K in [4,8]:
            boundaries,midpoints,times=interval_grid(K,.05);prefix=f'k{K}_';case=r['cases'][str(K)]
            np.testing.assert_array_equal(z[prefix+'time'],times);np.testing.assert_array_equal(z[prefix+'midpoint_time'],midpoints)
            d,g,a=make_shear_samples(z['material_label_centers'],z['ids'],m,c0,z['rho0_point'],times)
            np.testing.assert_array_equal(z[prefix+'positions'],d.positions);np.testing.assert_array_equal(z[prefix+'rho'],d.rho)
            states=z[prefix+'states'];np.testing.assert_array_equal(states[0],c0)
            for step,row in enumerate(case['steps']):
                path=out/row['saved_graph'];assert digest(path)==row['saved_graph_sha256']
                with np.load(path) as graph:
                    L=load_laplacian(graph);assert int(graph['frame'])==2*step+1 and float(graph['midpoint_time'])==midpoints[step]
                assert audit_laplacian(m,L)==row['sparse_graph_audit']
                before=states[step];half=z[prefix+'exponential_half_states'][step];after=states[step+1]
                assert metrics(m,before,c0)==row['start_metrics'] and metrics(m,after,c0)==row['end_metrics']
                np.testing.assert_array_equal([dissipation(L,c) for c in [before,half,after]],row['chi_start_mid_end_kg_per_s'])
                for key,saved_key in [('direct_vs_sampled_relative_difference','direct_endpoints'),('two_half_vs_sampled_relative_difference','two_half_endpoints')]:
                    v=relative(z[prefix+saved_key][step]-after)
                    np.testing.assert_allclose(v,row[key],atol=0.,rtol=0.);algorithm_max=max(algorithm_max,v)
                assert float(after.min())==row['minimum_state'] and float(after.max())==row['maximum_state']
                assert float(m@(after-before))==row['raw_component_mass_change_kg']
                loss=.5*(metrics(m,before,c0)['variance']-metrics(m,after,c0)['variance'])
                chi=row['chi_start_mid_end_kg_per_s'];simpson=.05/K*(chi[0]+4*chi[1]+chi[2])/6
                assert loss==row['half_variance_energy_loss_kg'] and simpson==row['Simpson_chi_integral_kg']
                assert loss-simpson==row['Simpson_energy_balance_difference_kg']
                drift=abs(float(m@(after-c0)));weighted_mass_max=max(weighted_mass_max,drift)
                assert drift<1e-10 and after.min()>=c0.min()-1e-10 and after.max()<=c0.max()+1e-10
                assert row['half_variance_energy_loss_kg']>=0. and min(row['chi_start_mid_end_kg_per_s'])>=0.
                graph_count+=1;del L
            observed=states[[0,K//2,K]]
            for index,row in enumerate(case['observations']):
                np.testing.assert_allclose(relative(observed[index]-z['reference_states'][index]),row['relative_concentration_error_initial_fluctuation_norm'],atol=0.,rtol=0.)
                assert metrics(m,observed[index],c0)==row['candidate_metrics']
                assert metrics(m,z['reference_states'][index],c0)==row['reference_metrics']
                assert row['H_candidate_minus_reference']==row['candidate_metrics']['H']-row['reference_metrics']['H']
                assert float(m@(observed[index]-c0))==row['raw_component_mass_change_from_c0_kg']
        difference=z['k4_states'][[0,2,4]]-z['k8_states'][[0,4,8]]
        np.testing.assert_array_equal(z['time_control_difference'],difference)
        np.testing.assert_array_equal([relative(v) for v in difference],r['time_and_reference_controls']['relative_state_difference_initial_fluctuation_norm'])
    with np.load(root/'runs/reconnect-shear-frozen/shear-frozen.npz') as original:
        for key in ['mass','c0','ids','material_label_centers','rho0_point']:np.testing.assert_array_equal(z_saved[key],original[key])
    assert graph_count==12 and not r['fluid_orc_enabled'] and not r['full_P_constructed']
    checks={'saved_shear_reference_mass_average_projections_and_differences_verified':True,'original_mass_c0_ids_and_saved_reference_unchanged':True,'predeclared_4_8_midpoint_inputs_verified':True,
        'raw_CSR_files_verified':graph_count,'saved_graph_algebra_and_chi_recomputed':True,
        'chronological_boundary_states_and_exponential_checks_verified':True,
        'actual_concentration_time_control_and_mixing_metrics_recomputed':True,
        'maximum_exponential_control_relative_difference':algorithm_max,
        'maximum_raw_component_mass_drift_from_initial_kg':weighted_mass_max,
        'reference_rerun':False,'full_P_constructed':False,'fluid_orc_enabled':False}
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
