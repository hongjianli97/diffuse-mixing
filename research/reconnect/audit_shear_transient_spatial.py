"""Saved fine graph, common material observations and inherited byte audit."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_kernel2d import digest
from compression_spatial_observation import restrict_path
from variable_parent_observation import parent_mass_audit
from shear_probe import make_shear_samples
from compression_midpoint import load_laplacian
from compression_reference_study import metrics
from ordered_compression import interval_grid,dissipation
from sparse_exchange_audit import audit_laplacian


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-ordered-shear'
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
    assert cfg['protocol']==json.loads((source/'stage3-status.json').read_text())['next_shear_spatial_transient_predeclared']
    reference_path=root/cfg['protocol']['reference_source']
    fine_path=root/cfg['protocol']['fine_mass_c0_ids_parent_source']
    assert cfg['fine_input_source_sha256']==digest(fine_path)
    assert cfg['reference_source_sha256']==digest(reference_path)
    graph_count=0;algorithm_max=0.;mass_drift_max=0.
    with np.load(out/'shear-transient-spatial.npz') as z,np.load(source/'ordered-shear.npz') as old_run,np.load(reference_path) as old_reference:
        for key in ['mass','c0','ids','material_label_centers','reference_times','reference_states']:
            np.testing.assert_array_equal(z['parent_'+key],old_run[key])
        m=z['mass'];c0=z['c0'];mp=z['parent_mass'];cp0=z['parent_c0']
        with np.load(fine_path) as prior_fine:
            for key in cfg['protocol']['fine_input_keys']:np.testing.assert_array_equal(z[key],prior_fine[key])
        child_sum,rounding,mass_audit=parent_mass_audit(m,mp)
        np.testing.assert_array_equal(z['child_mass_sum'],child_sum)
        np.testing.assert_array_equal(z['parent_mass_rounding_relative'],rounding)
        np.testing.assert_array_equal(z['parent_id_per_child'],z['parent_ids'][z['parent_index']])
        assert mass_audit==r['parent_mass_roundoff_audit']
        reference={key:old_reference[key+'_N256_states'] for key in ['M63-k128','M127-k128','M127-k256']}
        for key,value in reference.items():np.testing.assert_array_equal(z['reference_'+key+'_states'],value)
        ref=reference['M127-k256'];ref_parent=z['parent_reference_states']
        np.testing.assert_array_equal(ref_parent,old_reference['N128_best_reference_states'])
        np.testing.assert_array_equal(z['reference_space_difference_common'],old_reference['N128_space_difference'])
        np.testing.assert_array_equal(z['reference_time_difference_common'],old_reference['N128_time_difference'])
        np.testing.assert_array_equal(z['reference_parent_projection_difference'],restrict_path(ref,m,mp)-ref_parent)
        mu=m@c0/m.sum();mup=mp@cp0/mp.sum();sigma=np.sqrt(m@((c0-mu)**2));sigmap=np.sqrt(mp@((cp0-mup)**2))
        relative=lambda v:float(np.sqrt(m@(v*v))/sigma)
        common_relative=lambda v:float(np.sqrt(mp@(v*v))/sigmap)
        observed={};coarse={}
        for K in [4,8]:
            prefix=f'k{K}_';case=r['cases'][str(K)];boundaries,midpoints,times=interval_grid(K,.05)
            np.testing.assert_array_equal(z[prefix+'time'],times)
            np.testing.assert_array_equal(z[prefix+'boundary_time'],boundaries)
            np.testing.assert_array_equal(z[prefix+'midpoint_time'],midpoints)
            data,g,a=make_shear_samples(z['material_label_centers'],z['ids'],m,c0,z['rho0_point'],times)
            np.testing.assert_array_equal(z[prefix+'positions'],data.positions);np.testing.assert_array_equal(z[prefix+'rho'],data.rho)
            states=z[prefix+'states'];parent_states=z[prefix+'parent_states']
            np.testing.assert_array_equal(states[0],c0);np.testing.assert_array_equal(parent_states,restrict_path(states,m,mp))
            observed[K]=parent_states[[0,K//2,K]];coarse[K]=old_run[prefix+'states'][[0,K//2,K]]
            np.testing.assert_array_equal(z[f'coarse_k{K}_observations'],coarse[K])
            for step,row in enumerate(case['steps']):
                path=out/row['saved_graph'];assert digest(path)==row['saved_graph_sha256']
                with np.load(path) as graph:
                    L=load_laplacian(graph);assert L.shape==(65536,65536) and float(graph['midpoint_time'])==midpoints[step]
                    assert int(graph['frame'])==2*step+1
                assert row['left_time_s']==boundaries[step] and row['right_time_s']==boundaries[step+1]
                assert row['frozen_midpoint_time_s']==midpoints[step]
                assert audit_laplacian(m,L)==row['sparse_graph_audit']
                before=states[step];half=z[prefix+'exponential_half_states'][step];after=states[step+1]
                assert metrics(m,before,c0)==row['start_metrics'] and metrics(m,after,c0)==row['end_metrics']
                np.testing.assert_array_equal([dissipation(L,c) for c in [before,half,after]],row['chi_start_mid_end_kg_per_s'])
                for key,array in [('direct_vs_sampled_relative_difference','direct_endpoints'),('two_half_vs_sampled_relative_difference','two_half_endpoints')]:
                    value=relative(z[prefix+array][step]-after);assert value==row[key];algorithm_max=max(algorithm_max,value)
                assert float(after.min())==row['minimum_state'] and float(after.max())==row['maximum_state']
                assert float(m@(after-before))==row['raw_component_mass_change_kg']
                loss=.5*(metrics(m,before,c0)['variance']-metrics(m,after,c0)['variance'])
                chi=row['chi_start_mid_end_kg_per_s'];integral=.05/K*(chi[0]+4*chi[1]+chi[2])/6
                assert loss==row['half_variance_energy_loss_kg']
                assert loss-integral==row['Simpson_energy_balance_difference_kg']
                drift=abs(float(m@(after-c0)));mass_drift_max=max(mass_drift_max,drift)
                assert drift<1e-10 and after.min()>=c0.min()-1e-10 and after.max()<=c0.max()+1e-10
                assert row['half_variance_energy_loss_kg']>=0. and min(row['chi_start_mid_end_kg_per_s'])>=0.
                del L;graph_count+=1
            for index,row in enumerate(case['observations']):
                assert row['fine_own_cell_concentration_error']==relative(states[[0,K//2,K]][index]-ref[index])
                assert row['common_parent_concentration_error']==common_relative(observed[K][index]-ref_parent[index])
                assert row['common_parent_metrics']==metrics(mp,observed[K][index],cp0)
                assert row['fine_own_metrics']==metrics(m,states[[0,K//2,K]][index],c0)
                assert row['reference_common_metrics']==metrics(mp,ref_parent[index],cp0)
                assert row['common_H_minus_reference']==row['common_parent_metrics']['H']-row['reference_common_metrics']['H']
        np.testing.assert_array_equal(z['fine_time_difference_common'],observed[4]-observed[8])
        np.testing.assert_array_equal(z['coarse_time_difference_common'],coarse[4]-coarse[8])
        for index,row in enumerate(r['comparison_common_original_cells']):
            old_error=common_relative(coarse[8][index]-ref_parent[index]);new_error=common_relative(observed[8][index]-ref_parent[index])
            assert old_error==row['coarse_error_common_cells'] and new_error==row['fine_error_common_cells']
            assert old_error-new_error==row['absolute_error_reduction']
            vectors=[coarse[4]-coarse[8],observed[4]-observed[8],z['reference_space_difference_common'],z['reference_time_difference_common'],z['reference_parent_projection_difference']]
            for key,vector in zip(['coarse_4_vs_8','fine_4_vs_8','reference_63_vs_127','reference_fine_time','reference_projection_roundoff'],vectors):
                assert common_relative(vector[index])==row['empirical_difference_components'][key]
            assert sum(row['empirical_difference_components'].values())==row['combined_empirical_indicator']
            assert row['improvement_exceeds_empirical_indicator']==bool(old_error-new_error>row['combined_empirical_indicator'])
            assert row['improvement_over_indicator']==row['absolute_error_reduction']/row['combined_empirical_indicator']
            assert row['relative_error_reduction_fraction']==row['absolute_error_reduction']/old_error
            assert row['coarse_H']==metrics(mp,coarse[8][index],cp0)['H']
            assert row['fine_H_common_cells']==metrics(mp,observed[8][index],cp0)['H']
            assert row['reference_H']==metrics(mp,ref_parent[index],cp0)['H']
    assert graph_count==12 and not r['fluid_orc_enabled'] and not r['reference_rerun'] and not r['coarse_candidate_rerun']
    checks={'original_parent_mass_c0_ids_and_saved_coarse_states_unchanged':True,'saved_fine_mass_averages_identity_map_and_raw_roundoff_verified':True,
        'saved_2D_reference_mass_averages_reused_without_solve':True,'predeclared_joint_level_and_midpoint_inputs_verified':True,
        'raw_CSR_files_verified':graph_count,'saved_graph_algebra_dissipation_states_and_exponential_controls_verified':True,
        'raw_common_observation_and_paired_improvement_recomputed':True,
        'maximum_exponential_control_relative_difference':algorithm_max,'maximum_raw_component_mass_drift_kg':mass_drift_max,
        'reference_and_coarse_candidate_rerun':False,'fluid_orc_enabled':False,'full_P_constructed':False}
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
