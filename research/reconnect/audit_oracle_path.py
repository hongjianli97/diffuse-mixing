"""Oracle-only path reference: preserved bytes, paired quadratures and raw metrics."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_contact_constitutive import digest
from oracle_path_reference import normal_response
from contact_frozen_probe import point_fields,AMPLITUDE
from compression_midpoint import weighted_norm


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-contact-constitutive'
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
    protocol=json.loads((source/'stage3-status.json').read_text())['next_oracle_path_control_predeclared']
    assert cfg['protocol']==protocol and cfg['protocol_source_sha256']==digest(source/'stage3-status.json')
    data_path=root/protocol['source'];assert digest(data_path)==cfg['input_source_sha256']
    with np.load(out/'oracle-path.npz') as z,np.load(data_path) as old_data:
        for key in ['mass','c0','ids','material_label_centers','rho0_point','reference_rate','kernel_rate','near_contact','far_from_contact','distance_to_contact']:
            np.testing.assert_array_equal(z[key],old_data[key])
        for name in ['coarse','fine']:np.testing.assert_array_equal(z['harmonic_'+name+'_rate'],old_data[name+'_rate'])
        with np.load(root/'runs/reconnect-contact-frozen/contact-observations.npz') as ref:
            np.testing.assert_array_equal(z['PDE_face_volume16_difference'],ref['face_rate']-ref['volume16_rate'])
            np.testing.assert_array_equal(z['PDE_volume8_16_difference'],ref['volume8_rate']-ref['volume16_rate'])
        m=z['mass'];pde=z['reference_rate'];near=z['near_contact'];norm=weighted_norm(m,pde)
        for name,quad in zip(['coarse','fine'],protocol['fixed_quadrature_levels']):
            row=r['oracle_runs'][name];assert row['quadrature']==quad
            for key in ['outer_x','outer_weight','outer_column','y_cosine_integrals']:
                np.testing.assert_array_equal(z[name+'_'+key],old_data[name+'_'+key])
            x=z[name+'_outer_x'];w=z[name+'_outer_weight'];J=z[name+'_outer_normal_response'];column=z[name+'_outer_column']
            assert len(x)==row['outer_quadrature_points']
            integrals=np.zeros(128);cursor=0;order=quad['cell_x_order']
            while cursor<len(x):
                stop=cursor+order;index=int(column[cursor]);np.testing.assert_array_equal(column[cursor:stop],index)
                integrals[index]+=float(w[cursor:stop]@J[cursor:stop]);cursor=stop
            np.testing.assert_array_equal(z[name+'_normal_cell_integrals'],integrals)
            numerator=(2*AMPLITUDE*z[name+'_y_cosine_integrals'][:,None]*integrals[None,:]).ravel()
            np.testing.assert_array_equal(z[name+'_component_mass_rate_numerator'],numerator)
            rate=numerator/m;np.testing.assert_array_equal(z[name+'_rate'],rate)
            samples=sorted(set([0,len(x)-1,int(np.argmin(abs(x-.003))),int(np.argmin(abs(x-.497))),int(np.argmin(abs(x-.503))),int(np.argmin(abs(x-.75)))]))
            for i in samples:assert normal_response(float(x[i]),.04,.02,quad['normal_displacement_order'],quad['transverse_order'])==J[i]
            assert float(m@rate)==row['raw_mass_rate_kg_per_s']
            assert -float(m@(z['c0']*rate))==row['projected_chi_kg_per_s']
            chi=-AMPLITUDE**2*float(w@(np.cos(point_fields(x,0.,.02)['phase'])*J))
            assert chi==row['physical_nonlocal_chi_kg_per_s'] and chi>=0.
            for region,mask in [('all',np.ones(len(m),dtype=bool)),('near_contact',near),('far_from_contact',~near)]:
                assert weighted_norm(m[mask],(rate-pde)[mask])/norm==row['errors'][region]
        np.testing.assert_array_equal(z['best_oracle_rate'],z['fine_rate'])
        np.testing.assert_array_equal(z['oracle_error'],z['fine_rate']-pde)
        np.testing.assert_array_equal(z['oracle_minus_harmonic'],z['fine_rate']-z['harmonic_fine_rate'])
        np.testing.assert_array_equal(z['oracle_rule_difference'],z['coarse_rate']-z['fine_rate'])
        np.testing.assert_array_equal(z['harmonic_rule_difference'],z['harmonic_coarse_rate']-z['harmonic_fine_rate'])
        for region,mask in [('all',np.ones(len(m),dtype=bool)),('near_contact',near),('far_from_contact',~near)]:
            rel=lambda v:weighted_norm(m[mask],v[mask])/norm
            row=r['comparison'][region];old_error=rel(z['harmonic_fine_rate']-pde);new_error=rel(z['fine_rate']-pde)
            assert old_error==row['saved_harmonic_error'] and new_error==row['oracle_path_error']
            assert old_error-new_error==row['absolute_error_reduction']
            assert row['relative_error_reduction_fraction']==(old_error-new_error)/old_error
            assert rel(z['oracle_minus_harmonic'])==row['oracle_vs_harmonic_difference']
            for key,value in row['empirical_indicator_components'].items():assert rel(z[key])==value
            indicator=sum(row['empirical_indicator_components'].values());assert indicator==row['combined_empirical_indicator']
            assert row['reduction_over_indicator']==(old_error-new_error)/indicator
            assert row['reduction_exceeds_indicator']==bool(old_error-new_error>indicator)
    initial=(out/'initial-reference-source.py.txt').read_text();current=(root/'research/reconnect/oracle_path_reference.py').read_text()
    assert initial.replace('gamma=rho*D','gamma=np.asarray(rho*D)')==current
    assert r['reference_only'] and r['uses_additional_oracle_density_information'] and not r['finite_node_candidate_admitted']
    assert r['new_candidate_graphs']==r['new_candidate_transient_runs']==r['new_reference_transient_runs']==0
    checks={'predeclared_oracle_only_scope_and_two_rules_verified':True,'all_original_inputs_discrete_candidate_PDE_and_harmonic_rates_unchanged':True,
        'same_quadrature_nodes_weights_and_material_observation_verified':True,'raw_response_samples_and_all_mass_average_closures_recomputed':True,
        'all_region_errors_operator_differences_and_fixed_indicator_sums_recomputed':True,'raw_mass_and_positive_dissipation_recomputed':True,
        'initial_scalar_error_log_and_source_preserved':True,'pre_fix_source_diff_is_only_zero_dimensional_array_conversion':True,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'finite_node_candidate_admitted':False,'fluid_orc_enabled':False}
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
